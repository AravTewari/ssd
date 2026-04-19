import os
import torch
from time import perf_counter
from transformers import AutoTokenizer

from ssd.engine.sequence import Sequence
from ssd.engine.model_runner import ModelRunner
from ssd.utils.verify import verify
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, VerifierBase


class Verifier(VerifierBase):
    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        target_model_runner: ModelRunner,
        sampler_x: float | None = None,
        async_fan_out: int | None = None,
        jit_speculate: bool = False,
        tokenizer: AutoTokenizer = None,
        metrics: dict = None,
    ):
        super().__init__(lookahead, device)
        self.target_model_runner = target_model_runner
        self.sampler_x = sampler_x
        self.async_fan_out = async_fan_out
        self.jit_speculate = jit_speculate
        self.tokenizer = tokenizer
        self.metrics = metrics
        self._dflash_backends = {"dflash", "dflash_ssd", "ddtree", "ddtree_ssd"}

    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        result = self.target_model_runner.call("run", seqs, True)
        dflash = self.target_model_runner.config.draft_backend in self._dflash_backends
        if eagle:
            token_ids, eagle_acts = result
            dflash_target_features = None
        elif dflash:
            token_ids, dflash_features_flat = result
            dflash_target_features = []
        else:
            token_ids = result
            dflash_target_features = None

        offset = 0
        for seq, token_id in zip(seqs, token_ids):
            seq.recovery_token_id = token_id
            if eagle:
                seq_len = seq.num_prompt_tokens
                # this doesn't move acts onto cpu does it? 
                seq.last_target_hidden_state = eagle_acts[offset + seq_len - 1].clone()
                offset += seq_len
            elif dflash:
                seq_len = seq.num_prompt_tokens
                dflash_target_features.append(dflash_features_flat[offset:offset + seq_len].clone())
                offset += seq_len

        return VerifyResult(
            [], # no accepted tokens for prefill, just recovery tokens (first sampled token).
            [seq.recovery_token_id for seq in seqs],
            eagle_acts if eagle else None,
            dflash_target_features=dflash_target_features,
            dflash_target_features_full=None,
            target_verify_s=None,
        )

    def _verify_ddtree(self, seqs: list[Sequence], speculate_result: SpeculateResult) -> VerifyResult:
        if speculate_result.ddtree_entries is None:
            raise RuntimeError("DDTree verify requested without ddtree_entries")

        t0 = perf_counter()
        accepted_suffixes, recovery_tokens, committed_features, visited_node_counts, tree_node_counts, tree_compile_s = self.target_model_runner.call(
            "run_ddtree_verify",
            seqs,
            speculate_result.ddtree_entries,
        )
        elapsed = perf_counter() - t0
        self.metrics["target_verify_times"].append(elapsed)
        self.metrics["accepted_suffix_lens_with_recovery"].extend([len(suffix) for suffix in accepted_suffixes])
        if "ddtree_verified_node_counts" in self.metrics:
            self.metrics["ddtree_verified_node_counts"].extend(visited_node_counts)
        if "ddtree_tree_node_counts" in self.metrics:
            self.metrics["ddtree_tree_node_counts"].extend(tree_node_counts)
        if "ddtree_tree_compile_times" in self.metrics:
            self.metrics["ddtree_tree_compile_times"].append(tree_compile_s)
        if speculate_result.cache_hits is not None:
            cache_hits_cpu = speculate_result.cache_hits.cpu()
            self.metrics["cache_hits"].append(cache_hits_cpu.float().mean().item())
            for row_idx, suffix_len in enumerate([len(suffix) for suffix in accepted_suffixes]):
                if cache_hits_cpu[row_idx] == 1:
                    self.metrics["accepted_suffix_lens_on_hit"].append(suffix_len)
                else:
                    self.metrics["accepted_suffix_lens_on_miss"].append(suffix_len)

        return VerifyResult(
            new_suffixes=accepted_suffixes,
            recovery_tokens=[int(token) for token in recovery_tokens],
            eagle_acts=None,
            dflash_target_features=committed_features,
            dflash_target_features_full=None,
            target_verify_s=elapsed,
            ddtree_verified_node_counts=visited_node_counts,
        )

    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        """Verify speculative tokens using the target model."""
        if speculate_result.ddtree_entries is not None:
            return self._verify_ddtree(seqs, speculate_result)

        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        batch_size = len(seqs)

        if _prof:
            torch.cuda.synchronize()
            _vt0 = perf_counter()

        _pt = os.environ.get("SSD_PROFILE_TARGET", "0") == "1"
        _tv0 = perf_counter()
        result = self.target_model_runner.call("run", seqs, False, False, True)

        if _prof:
            torch.cuda.synchronize()
            _vt1 = perf_counter()

        if _pt:
            torch.cuda.synchronize()
            _vt_call = perf_counter()
            print(f"[PROFILE verifier] target_call={(_vt_call-_tv0)*1000:.2f}ms eagle={eagle} bs={batch_size}", flush=True)

        dflash = self.target_model_runner.config.draft_backend in self._dflash_backends
        if eagle:
            logits_p_flat, eagle_acts_flat = result
            dflash_features_flat = None
        elif dflash:
            logits_p_flat, dflash_features_flat = result
        else:
            logits_p_flat = result
            dflash_features_flat = None

        for s in seqs:
            s.num_cached_tokens += self.lookahead + 1

        logits_p = logits_p_flat.view(
            batch_size, self.lookahead + 1, -1)  # [b, k+1, v]

        # Build per-seq temps for target verify and draft q respectively.
        temps_target = [seq.temperature for seq in seqs]
        temps_draft = [
            seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
            for seq in seqs
        ]
        temperatures_target = torch.tensor(temps_target, dtype=torch.float32, device=self.device)
        temperatures_draft = torch.tensor(temps_draft, dtype=torch.float32, device=self.device)

        new_suffixes, recovery_tokens = verify(
            logits_p=logits_p,
            logits_q=speculate_result.logits_q,
            speculations=speculate_result.speculations,
            temperatures_target=temperatures_target,
            temperatures_draft=temperatures_draft,
            cache_hits=speculate_result.cache_hits,
            sampler_x=self.sampler_x,
            async_fan_out=self.async_fan_out,
            jit_speculate=self.jit_speculate,
        )

        self.metrics["target_verify_times"].append(perf_counter() - _tv0)

        if _prof:
            torch.cuda.synchronize()
            _vt2 = perf_counter()
            print(f"[PROFILE verify] target_fwd={(_vt1-_vt0)*1000:.2f}ms verify_compute={(_vt2-_vt1)*1000:.2f}ms", flush=True)


        # # Debug: print recovery tokens detokenized
        if __debug__ and recovery_tokens is not None and len(recovery_tokens) > 0:
            recovery_texts = []
            for token in recovery_tokens:
                try:
                    text = self.tokenizer.decode([token], skip_special_tokens=False)
                    recovery_texts.append(text)
                except Exception:
                    recovery_texts.append(f"<token_id:{token}>")
            print(f"[verify] recovery tokens: {recovery_texts}", flush=True)

        self.metrics["accepted_suffix_lens_with_recovery"].extend(
            [len(s) for s in new_suffixes])

        # For async mode, also track accepted suffix lengths only for cache hits
        if speculate_result.cache_hits is not None:
            _ch_cpu = speculate_result.cache_hits.cpu()
            self.metrics["cache_hits"].append(_ch_cpu.float().mean().item())
            for i, suffix_len in enumerate([len(s) for s in new_suffixes]):
                if _ch_cpu[i] == 1:
                    self.metrics["accepted_suffix_lens_on_hit"].append(suffix_len)
                else:
                    self.metrics["accepted_suffix_lens_on_miss"].append(suffix_len)

        # Print mean length of new suffixes for monitoring
        if __debug__ and new_suffixes:
            mean_suffix_len = sum([len(suffix) for suffix in new_suffixes]) / len(new_suffixes)
            print(f"[verify] mean new suffix length: {mean_suffix_len:.2f}", flush=True)

        eagle_acts = None
        if eagle:
            eagle_acts = eagle_acts_flat.view(batch_size, self.lookahead + 1, -1)

        dflash_target_features = None
        dflash_target_features_full = None
        if dflash and dflash_features_flat is not None:
            dflash_features = dflash_features_flat.view(batch_size, self.lookahead + 1, -1)
            dflash_target_features_full = dflash_features.clone()
            dflash_target_features = [
                dflash_features[i, :len(new_suffix)].clone()
                for i, new_suffix in enumerate(new_suffixes)
            ]
        
        return VerifyResult(
            new_suffixes=new_suffixes,
            recovery_tokens=recovery_tokens,
            eagle_acts=eagle_acts,
            dflash_target_features=dflash_target_features,
            dflash_target_features_full=dflash_target_features_full,
            target_verify_s=perf_counter() - _tv0,
        )
