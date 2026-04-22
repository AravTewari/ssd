from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import torch
from time import perf_counter
from transformers import AutoTokenizer

from ssd.engine.model_runner import ModelRunner
from ssd.engine.sequence import Sequence
from ssd.engine.scheduler import Scheduler
from ssd.engine.helpers.speculate_types import SpeculatorBase, VerifierBase, VerifyResult
from ssd.utils.misc import decode_tokens


class InferenceStep(ABC):

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    @abstractmethod
    def decode(self, seqs: list[Sequence]) -> int:
        pass

    @abstractmethod
    def prefill(self, seqs: list[Sequence]) -> int:
        pass


class AutoRegressiveStep(InferenceStep):

    def __init__(self, scheduler: Scheduler, model_runner: ModelRunner, tokenizer: AutoTokenizer):
        super().__init__(scheduler)
        self.model_runner = model_runner
        self.tokenizer = tokenizer

    def step(self, seqs: list[Sequence], is_prefill: bool) -> int:
        if __debug__:
            print(f'[auto_regressive_step] is_prefill={is_prefill}', flush=True)

        token_ids = self.model_runner.call("run", seqs, is_prefill)

        if __debug__:
            decoded_tokens = decode_tokens(token_ids, self.tokenizer)
            print(f"[auto_regressive_step] generated tokens: {decoded_tokens}", flush=True)

        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        return len(seqs) if not is_prefill else sum(len(seq) for seq in seqs)

    def prefill(self, seqs: list[Sequence]) -> int:
        return self.step(seqs, is_prefill=True)

    def decode(self, seqs: list[Sequence]) -> int:
        return self.step(seqs, is_prefill=False)


class SpecDecodeStep(InferenceStep):

    def __init__(
        self,
        scheduler: Scheduler,
        speculator: SpeculatorBase,
        verifier: VerifierBase,
        eagle: bool,
        dflash: bool,
        tokenizer: AutoTokenizer,
        async_spec: bool,
        ar_branch_key_mode: str = "normal",
        dflash_branch_key_mode: str = "normal",
        metrics: dict | None = None,
        enable_dflash_diagnostics: bool = False,
        enable_ddtree_diagnostics: bool = False,
    ):
        super().__init__(scheduler)
        self.speculator = speculator
        self.verifier = verifier
        self.eagle = eagle
        self.dflash = dflash
        self.tokenizer = tokenizer
        self.async_spec = async_spec
        self.ar_branch_key_mode = ar_branch_key_mode
        self.dflash_branch_key_mode = dflash_branch_key_mode
        self.metrics = metrics
        self.enable_dflash_diagnostics = enable_dflash_diagnostics
        self.enable_ddtree_diagnostics = enable_ddtree_diagnostics

    def prefill(self, seqs: list[Sequence]) -> int:
        # When doing async speculation and not Eagle, we can do draft and target prefills in parallel.
        if not self.eagle and not self.dflash and self.async_spec:
            empty_verify_result = VerifyResult([], [], None)
            self.speculator.prefill(seqs, empty_verify_result)
            verify_result = self.verifier.prefill(seqs, eagle=False)
        else:
            verify_result = self.verifier.prefill(seqs, eagle=self.eagle)
            self.speculator.prefill(seqs, verify_result)

        for seq in seqs:
            assert seq.recovery_token_id is not None
            seq.num_cached_tokens = seq.num_prompt_tokens
            seq.num_draft_cached_tokens = seq.num_prompt_tokens

        return sum(len(seq) for seq in seqs)

    def decode(self, seqs: list[Sequence]) -> int:
        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        cycle_t0 = perf_counter()
        if _prof:
            torch.cuda.synchronize()
            _t0 = perf_counter()

        # Save lightweight state instead of expensive clone_spec deep copy.
        # speculate() modifies: token_ids (append+extend), num_tokens, last_token, num_draft_cached_tokens
        # verify() modifies: num_cached_tokens (line 77 of verifier.py)
        # postprocess_speculate() needs the ORIGINAL state to apply new suffixes.
        saved = [(len(seq.token_ids), seq.num_tokens, seq.last_token, seq.num_draft_cached_tokens, seq.num_cached_tokens) for seq in seqs]

        eagle_sentinel = True if self.eagle else None
        in_verify_result = VerifyResult(
            new_suffixes=[],
            recovery_tokens=[],
            eagle_acts=eagle_sentinel,
        )
        #### STEP 1: SPECULATE ####
        speculate_t0 = perf_counter()
        speculate_result = self.speculator.speculate(seqs, in_verify_result)
        speculate_t1 = perf_counter()

        if _prof:
            torch.cuda.synchronize()
            _t1 = perf_counter()

        if __debug__:
            if speculate_result.ddtree_entries is not None:
                for i, entry in enumerate(speculate_result.ddtree_entries):
                    print(
                        f"[SpecDecodeStep] ddtree {i}: root={entry.recovery_token} nodes={entry.num_nodes}",
                        flush=True,
                    )
            else:
                speculations = speculate_result.speculations
                print(f"[SpecDecodeStep] speculations: {speculations}", flush=True)
                speculations_list = speculations.tolist()

                for i, speculation in enumerate(speculations_list):
                    decoded_tokens = decode_tokens(speculation, self.tokenizer)
                    print(f"[SpecDecodeStep] speculation {i}: {decoded_tokens}", flush=True)

        #### STEP 2: VERIFY ####
        out_verify_result = self.verifier.verify(seqs, speculate_result, eagle=self.eagle)
        verify_t1 = perf_counter()

        feedback_t0 = perf_counter()
        if self.async_spec and hasattr(self.speculator, "post_verify_feedback"):
            if speculate_result.ddtree_entries is not None:
                self.speculator.post_verify_feedback(seqs, out_verify_result, speculate_result.ddtree_diag)
            elif (
                self.dflash
                and (self.enable_dflash_diagnostics or self.dflash_branch_key_mode == "oracle")
                and speculate_result.dflash_diag is not None
            ):
                self.speculator.post_verify_feedback(seqs, out_verify_result, speculate_result.dflash_diag)
            elif not self.dflash and self.ar_branch_key_mode == "oracle":
                self.speculator.post_verify_feedback(seqs, out_verify_result)
        feedback_t1 = perf_counter()

        if _prof:
            torch.cuda.synchronize()
            _t2 = perf_counter()

        if __debug__:
            recovery_tokens = out_verify_result.recovery_tokens
            new_suffixes = out_verify_result.new_suffixes
            for i, new_suffix in enumerate(new_suffixes):
                decoded_tokens = decode_tokens(new_suffix + [recovery_tokens[i]], self.tokenizer)
                print(f"[SpecDecodeStep] verification {i}: {decoded_tokens}", flush=True)

        # Restore original seq state before postprocess (undo speculate + verify modifications)
        for seq, (orig_len, orig_nt, orig_lt, orig_ndc, orig_nct) in zip(seqs, saved):
            del seq.token_ids[orig_len:]
            seq.num_tokens = orig_nt
            seq.last_token = orig_lt
            seq.num_draft_cached_tokens = orig_ndc
            seq.num_cached_tokens = orig_nct

        #### STEP 3: POSTPROCESS ####
        self.scheduler.postprocess_speculate(
            seqs,
            out_verify_result.new_suffixes,
            out_verify_result.recovery_tokens,
            eagle_acts=out_verify_result.eagle_acts if self.eagle else None,
            dflash_target_features=out_verify_result.dflash_target_features,
        )

        if self.async_spec and not self.dflash and speculate_result.ddtree_entries is None:
            draft_service_s = speculate_t1 - speculate_t0
            post_verify_feedback_s = feedback_t1 - feedback_t0
            if self.metrics is not None:
                self.metrics["ar_draft_service_times"].append(draft_service_s)
                self.metrics["ar_post_verify_feedback_times"].append(post_verify_feedback_s)
                cache_hits = speculate_result.cache_hits.tolist() if speculate_result.cache_hits is not None else [0] * len(seqs)
                target_verify_ms = (out_verify_result.target_verify_s or max(verify_t1 - speculate_t1, 0.0)) * 1000.0
                total_cycle_ms = (perf_counter() - cycle_t0) * 1000.0
                for row_idx, seq in enumerate(seqs):
                    accepted_len = len(out_verify_result.new_suffixes[row_idx])
                    cache_hit = bool(cache_hits[row_idx])
                    self.metrics["ar_cycle_diagnostics"].append(
                        {
                            "seq_id": seq.seq_id,
                            "batch_size": len(seqs),
                            "cycle_idx": seq.dflash_cycle_idx,
                            "accepted_len": accepted_len,
                            "recovery_token": int(out_verify_result.recovery_tokens[row_idx]),
                            "cache_hit": cache_hit,
                            "draft_service_ms": draft_service_s * 1000.0,
                            "post_verify_feedback_ms": post_verify_feedback_s * 1000.0,
                            "target_verify_ms": target_verify_ms,
                            "total_cycle_ms": total_cycle_ms,
                            "tokens_committed_this_cycle": accepted_len,
                        }
                    )
            for seq in seqs:
                seq.dflash_cycle_idx += 1

        if speculate_result.ddtree_entries is not None and speculate_result.ddtree_diag is not None:
            diag = speculate_result.ddtree_diag
            if self.metrics is not None:
                self.metrics["ddtree_draft_step_times"].append(diag.total_dflash_s)
                self.metrics["ddtree_tree_build_times"].append(diag.total_tree_build_s)
            if self.scheduler.draft_backend == "ddtree_ssd" and self.metrics is not None and self.enable_ddtree_diagnostics:
                target_verify_ms = (out_verify_result.target_verify_s or 0.0) * 1000.0
                total_cycle_ms = (perf_counter() - cycle_t0) * 1000.0
                transport_ms = diag.total_transport_s * 1000.0
                cache_hits = speculate_result.cache_hits.tolist() if speculate_result.cache_hits is not None else [0] * len(seqs)
                for row_idx, seq in enumerate(seqs):
                    accepted_len = len(out_verify_result.new_suffixes[row_idx])
                    cache_hit = bool(cache_hits[row_idx])
                    fallback_used = bool(diag.fallback_used[row_idx]) if diag.fallback_used is not None else False
                    verified_node_count = (
                        None
                        if out_verify_result.ddtree_verified_node_counts is None
                        else int(out_verify_result.ddtree_verified_node_counts[row_idx])
                    )
                    tree_node_count = (
                        None
                        if speculate_result.ddtree_entries is None
                        else int(speculate_result.ddtree_entries[row_idx].num_nodes)
                    )
                    self.metrics["ddtree_cycle_diagnostics"].append(
                        {
                            "seq_id": seq.seq_id,
                            "batch_size": len(seqs),
                            "cycle_idx": seq.dflash_cycle_idx,
                            "accepted_len": accepted_len,
                            "recovery_token": int(out_verify_result.recovery_tokens[row_idx]),
                            "cache_hit": cache_hit,
                            "fallback_used": fallback_used,
                            "frontier_candidate_count": (
                                None if diag.frontier_candidate_count is None else int(diag.frontier_candidate_count[row_idx])
                            ),
                            "actual_frontier_rank": (
                                None if diag.actual_frontier_rank is None else diag.actual_frontier_rank[row_idx]
                            ),
                            "committed_tokens_from_cache": accepted_len if cache_hit else 0,
                            "committed_tokens_from_fallback": accepted_len if fallback_used else 0,
                            "ddtree_draft_ms": diag.total_dflash_s * 1000.0,
                            "tree_build_ms": diag.total_tree_build_s * 1000.0,
                            "predictor_ms": diag.total_predictor_s * 1000.0,
                            "target_verify_ms": target_verify_ms,
                            "cache_lookup_ms": diag.cache_lookup_s * 1000.0,
                            "transport_ms": transport_ms,
                            "total_cycle_ms": total_cycle_ms,
                            "tokens_committed_this_cycle": accepted_len,
                            "verified_node_count": verified_node_count,
                            "tree_node_count": tree_node_count,
                        }
                    )
            for seq in seqs:
                seq.dflash_cycle_idx += 1

        if self.dflash and self.async_spec and speculate_result.dflash_diag is not None:
            diag = speculate_result.dflash_diag
            if self.metrics is not None:
                self.metrics["dflash_draft_step_times"].append(diag.total_dflash_s)
                self.metrics["dflash_predictor_times"].append(diag.total_predictor_s)
            if self.enable_dflash_diagnostics and self.metrics is not None:
                cache_hits = speculate_result.cache_hits.tolist() if speculate_result.cache_hits is not None else [0] * len(seqs)
                target_verify_ms = (out_verify_result.target_verify_s or 0.0) * 1000.0
                total_cycle_ms = (perf_counter() - cycle_t0) * 1000.0
                transport_ms = diag.total_transport_s * 1000.0
                for row_idx, seq in enumerate(seqs):
                    accepted_len = len(out_verify_result.new_suffixes[row_idx])
                    cache_hit = bool(cache_hits[row_idx])
                    fallback_used = bool(diag.fallback_used[row_idx]) if diag.fallback_used is not None else False
                    self.metrics["dflash_cycle_diagnostics"].append(
                        {
                            "seq_id": seq.seq_id,
                            "batch_size": len(seqs),
                            "cycle_idx": seq.dflash_cycle_idx,
                            "accepted_len": accepted_len,
                            "recovery_token": int(out_verify_result.recovery_tokens[row_idx]),
                            "cache_hit": cache_hit,
                            "fallback_used": fallback_used,
                            "num_branches_generated": (
                                int(diag.num_branches_generated[row_idx])
                                if diag.num_branches_generated is not None else 0
                            ),
                            "true_branch_rank": (
                                None if diag.true_branch_rank is None else diag.true_branch_rank[row_idx]
                            ),
                            "committed_tokens_from_cache": accepted_len if cache_hit else 0,
                            "committed_tokens_from_fallback": accepted_len if fallback_used else 0,
                            "actual_accept_supported": (
                                None if diag.actual_accept_supported is None else bool(diag.actual_accept_supported[row_idx])
                            ),
                            "actual_recovery_rank_given_accept": (
                                None
                                if diag.actual_recovery_rank_given_accept is None
                                else diag.actual_recovery_rank_given_accept[row_idx]
                            ),
                            "joint_branch_supported": (
                                None if diag.joint_branch_supported is None else bool(diag.joint_branch_supported[row_idx])
                            ),
                            "recovery_entropy_at_actual_accept": (
                                None
                                if diag.recovery_entropy_at_actual_accept is None
                                else diag.recovery_entropy_at_actual_accept[row_idx]
                            ),
                            "recovery_top1_margin_at_actual_accept": (
                                None
                                if diag.recovery_top1_margin_at_actual_accept is None
                                else diag.recovery_top1_margin_at_actual_accept[row_idx]
                            ),
                            "predictor_ms": diag.total_predictor_s * 1000.0,
                            "dflash_ms": diag.total_dflash_s * 1000.0,
                            "target_verify_ms": target_verify_ms,
                            "cache_lookup_ms": diag.cache_lookup_s * 1000.0,
                            "transport_ms": transport_ms,
                            "total_cycle_ms": total_cycle_ms,
                            "tokens_committed_this_cycle": accepted_len,
                        }
                    )
            for seq in seqs:
                seq.dflash_cycle_idx += 1

        if _prof:
            torch.cuda.synchronize()
            _t3 = perf_counter()
            cache_hits = speculate_result.cache_hits
            hits_str = f"hits={cache_hits.sum().item()}/{len(cache_hits)}" if cache_hits is not None else ""
            toks = sum(len(s) for s in out_verify_result.new_suffixes)
            print(f"[PROFILE target] handshake={(_t1-_t0)*1000:.2f}ms verify={(_t2-_t1)*1000:.2f}ms postprocess={(_t3-_t2)*1000:.2f}ms total={(_t3-_t0)*1000:.2f}ms {hits_str} toks={toks}", flush=True)

        return sum(len(s) for s in out_verify_result.new_suffixes)
