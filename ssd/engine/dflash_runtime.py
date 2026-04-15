from dataclasses import dataclass
from time import perf_counter

import torch
from transformers import AutoModel, AutoModelForCausalLM, DynamicCache

from ssd.config import Config
from ssd.engine.dflash_predictor import DFlashFeaturePredictor
from ssd.layers.sampler import Sampler
from ssd.utils.async_helpers.async_spec_helpers import get_forked_recovery_tokens_from_logits


@dataclass
class DFlashBlockOutputs:
    draft_tokens: torch.Tensor
    logits_q: torch.Tensor
    block_hidden: torch.Tensor
    branch_logits: torch.Tensor
    predicted_target_features: torch.Tensor | None


@dataclass
class DFlashCommittedState:
    exact_history: torch.Tensor
    frontier_version: int = 0


@dataclass
class DFlashCacheEntry:
    tokens: torch.Tensor
    logits_q: torch.Tensor
    branch_logits: torch.Tensor
    predicted_target_features: torch.Tensor | None


class DFlashRuntime:
    def __init__(
        self,
        config: Config,
        device: torch.device,
        predictor_path: str | None = None,
        metrics: dict | None = None,
    ):
        self.config = config
        self.device = device
        self.dtype = getattr(config.draft_hf_config, "torch_dtype", None) or torch.bfloat16
        self.block_size = config.dflash_block_size
        self.lookahead = config.speculate_k
        self.mask_token_id = config.dflash_mask_token_id
        self.feature_dim = config.dflash_target_feature_dim
        self.metrics = metrics
        self.base_states: dict[int, DFlashCommittedState] = {}

        self.draft_model = AutoModel.from_pretrained(
            config.draft,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(device).eval()
        self.target_model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=self.dtype,
        ).to(device).eval()
        self.predictor = None
        if predictor_path is not None:
            self.predictor = DFlashFeaturePredictor.from_pretrained(
                predictor_path,
                device=device,
                dtype=self.dtype,
            ).eval()
        self.sampler = Sampler(config.sampler_x, config.async_fan_out).to(device)

    def reset_states(self) -> None:
        self.base_states.clear()

    def prefill_exact_context(
        self,
        seq_ids: list[int],
        prompt_target_features: list[torch.Tensor],
        frontier_version: int = 0,
    ) -> None:
        for seq_id, features in zip(seq_ids, prompt_target_features):
            self.base_states[seq_id] = DFlashCommittedState(
                exact_history=features.to(device=self.device, dtype=self.dtype, non_blocking=False).clone(),
                frontier_version=frontier_version,
            )

    def commit_exact_context(
        self,
        seq_ids: list[int],
        frontier_versions: list[int],
        exact_feature_updates: list[torch.Tensor | None],
    ) -> None:
        for seq_id, frontier_version, update in zip(seq_ids, frontier_versions, exact_feature_updates):
            state = self.base_states.get(seq_id)
            if state is None:
                raise RuntimeError(f"Missing committed DFlash state for seq_id={seq_id}")
            if update is None or update.numel() == 0:
                if state.frontier_version != frontier_version:
                    raise RuntimeError(
                        f"DFlash frontier mismatch for seq_id={seq_id}: "
                        f"state={state.frontier_version}, request={frontier_version}"
                    )
                continue

            if frontier_version != state.frontier_version + 1:
                raise RuntimeError(
                    f"DFlash exact commit out of order for seq_id={seq_id}: "
                    f"state={state.frontier_version}, request={frontier_version}"
                )
            update = update.to(device=self.device, dtype=self.dtype, non_blocking=False)
            state.exact_history = torch.cat([state.exact_history, update], dim=0).contiguous()
            state.frontier_version = frontier_version

    def get_exact_history(self, seq_id: int) -> torch.Tensor:
        state = self.base_states.get(seq_id)
        if state is None:
            raise RuntimeError(f"Missing committed DFlash state for seq_id={seq_id}")
        return state.exact_history

    def generate_block(
        self,
        seq_ids: list[int],
        recovery_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        return_predicted_features: bool = True,
    ) -> DFlashBlockOutputs:
        histories = [self.get_exact_history(seq_id).clone() for seq_id in seq_ids]
        return self.run_block_batch(
            histories,
            recovery_tokens,
            temperatures,
            return_predicted_features=return_predicted_features,
        )

    def populate_branch_cache(
        self,
        seq_ids: list[int],
        frontier_versions: list[int],
        recovery_tokens: list[int],
        temperatures: torch.Tensor,
        cache_hits: torch.Tensor,
        entries: list[DFlashCacheEntry],
    ) -> dict[tuple[int, int, int, int], DFlashCacheEntry]:
        if not entries:
            return {}

        branch_logits = torch.stack([entry.branch_logits for entry in entries], dim=0)
        returned_tokens = torch.stack(
            [
                torch.cat(
                    [
                        torch.tensor([recovery_tokens[i]], dtype=torch.int64, device=self.device),
                        entry.tokens,
                    ],
                    dim=0,
                )
                for i, entry in enumerate(entries)
            ],
            dim=0,
        )
        forked_recovery = get_forked_recovery_tokens_from_logits(
            self.config,
            branch_logits,
            cache_hits,
            returned_tokens,
            tokenizer=None,
        )

        jobs_by_len: dict[int, list[tuple[tuple[int, int, int, int], torch.Tensor, int, float]]] = {}
        for row_idx, entry in enumerate(entries):
            if entry.predicted_target_features is None:
                raise RuntimeError("DFlash SSD branch cache population requires predicted_target_features")
            counts = self.config.fan_out_list if bool(cache_hits[row_idx].item()) else self.config.fan_out_list_miss
            offset = 0
            base_history = self.get_exact_history(seq_ids[row_idx])
            next_frontier_version = frontier_versions[row_idx] + 1
            for accept_idx, count in enumerate(counts):
                child_history = torch.cat(
                    [base_history, entry.predicted_target_features[:accept_idx + 1]],
                    dim=0,
                ).contiguous()
                child_len = int(child_history.shape[0])
                for _ in range(count):
                    recovery_token = int(forked_recovery[row_idx, offset].item())
                    offset += 1
                    cache_key = (seq_ids[row_idx], next_frontier_version, accept_idx, recovery_token)
                    jobs_by_len.setdefault(child_len, []).append(
                        (cache_key, child_history, recovery_token, float(temperatures[row_idx].item()))
                    )

        branch_cache: dict[tuple[int, int, int, int], DFlashCacheEntry] = {}
        for _child_len, jobs in jobs_by_len.items():
            histories = [history for _, history, _, _ in jobs]
            rec_tokens = torch.tensor(
                [recovery for _, _, recovery, _ in jobs],
                dtype=torch.int64,
                device=self.device,
            )
            temps = torch.tensor([temp for _, _, _, temp in jobs], dtype=torch.float32, device=self.device)
            outputs = self.run_block_batch(
                histories,
                rec_tokens,
                temps,
                return_predicted_features=True,
            )
            for row_idx, (cache_key, _history, _recovery, _temp) in enumerate(jobs):
                branch_cache[cache_key] = DFlashCacheEntry(
                    tokens=outputs.draft_tokens[row_idx].clone(),
                    logits_q=outputs.logits_q[row_idx].clone(),
                    branch_logits=outputs.branch_logits[row_idx].clone(),
                    predicted_target_features=(
                        outputs.predicted_target_features[row_idx].clone()
                        if outputs.predicted_target_features is not None else None
                    ),
                )
        return branch_cache

    @torch.inference_mode()
    def run_block_batch(
        self,
        feature_histories: list[torch.Tensor],
        recovery_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        return_predicted_features: bool = True,
    ) -> DFlashBlockOutputs:
        batch_size = len(feature_histories)
        if batch_size == 0:
            raise ValueError("run_block_batch requires at least one feature history")
        if recovery_tokens.shape != (batch_size,):
            raise ValueError(
                f"Expected recovery_tokens with shape ({batch_size},), got {tuple(recovery_tokens.shape)}"
            )
        if temperatures.shape != (batch_size,):
            raise ValueError(
                f"Expected temperatures with shape ({batch_size},), got {tuple(temperatures.shape)}"
            )

        hist_lens = [int(history.shape[0]) for history in feature_histories]
        if any(history.shape[-1] != self.feature_dim for history in feature_histories):
            raise ValueError("All feature histories must match dflash_target_feature_dim")

        vocab_size = self.target_model.lm_head.weight.shape[0]
        hidden_size = getattr(self.draft_model.config, "hidden_size")
        draft_tokens = torch.empty((batch_size, self.lookahead), dtype=torch.int64, device=self.device)
        logits_q = torch.empty((batch_size, self.lookahead, vocab_size), dtype=self.dtype, device=self.device)
        block_hidden = torch.empty((batch_size, self.block_size, hidden_size), dtype=self.dtype, device=self.device)
        branch_logits = torch.empty((batch_size, self.block_size, vocab_size), dtype=self.dtype, device=self.device)
        predicted_target_features = None
        if return_predicted_features and self.predictor is not None:
            predicted_target_features = torch.empty(
                (batch_size, self.block_size, self.feature_dim),
                dtype=self.dtype,
                device=self.device,
            )

        buckets: dict[int, list[int]] = {}
        for idx, hist_len in enumerate(hist_lens):
            buckets.setdefault(hist_len, []).append(idx)

        t0 = perf_counter()
        for hist_len, indices in buckets.items():
            histories = torch.stack(
                [feature_histories[idx].to(device=self.device, dtype=self.dtype, non_blocking=False) for idx in indices],
                dim=0,
            )
            rec_tokens = recovery_tokens[indices]
            temps = temperatures[indices]

            block_output_ids = torch.full(
                (len(indices), self.block_size),
                self.mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
            block_output_ids[:, 0] = rec_tokens
            noise_embedding = self.target_model.model.embed_tokens(block_output_ids)
            position_ids = torch.arange(0, hist_len + self.block_size, device=self.device).unsqueeze(0).expand(len(indices), -1)
            draft_hidden = self.draft_model(
                target_hidden=histories,
                noise_embedding=noise_embedding,
                position_ids=position_ids,
                past_key_values=DynamicCache(),
                use_cache=True,
                is_causal=False,
            )
            bucket_block_hidden = draft_hidden[:, -self.block_size:, :].contiguous()
            bucket_branch_logits = self.target_model.lm_head(bucket_block_hidden).contiguous()
            bucket_logits_q = bucket_branch_logits[:, 1:, :].contiguous()
            flat_temps = temps.repeat_interleave(self.lookahead)
            bucket_tokens = self.sampler(
                bucket_logits_q.view(-1, vocab_size),
                flat_temps,
            ).view(len(indices), self.lookahead)
            bucket_predicted = None
            if predicted_target_features is not None and self.predictor is not None:
                pt0 = perf_counter()
                bucket_predicted = self.predictor(bucket_block_hidden).to(self.dtype)
                if self.metrics is not None:
                    self.metrics["dflash_predictor_times"].append(perf_counter() - pt0)

            for bucket_row, out_idx in enumerate(indices):
                draft_tokens[out_idx] = bucket_tokens[bucket_row]
                logits_q[out_idx] = bucket_logits_q[bucket_row]
                block_hidden[out_idx] = bucket_block_hidden[bucket_row]
                branch_logits[out_idx] = bucket_branch_logits[bucket_row]
                if predicted_target_features is not None and bucket_predicted is not None:
                    predicted_target_features[out_idx] = bucket_predicted[bucket_row]

        if self.metrics is not None:
            self.metrics["dflash_draft_step_times"].append(perf_counter() - t0)
        return DFlashBlockOutputs(
            draft_tokens=draft_tokens,
            logits_q=logits_q,
            block_hidden=block_hidden,
            branch_logits=branch_logits,
            predicted_target_features=predicted_target_features,
        )
