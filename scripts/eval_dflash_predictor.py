import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


BRANCH_TEMPLATES = {
    "baseline48": [3] * 16,
    "front1": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "front2": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "front4": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "front8": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "front12": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    "front16": [1] * 16,
    "top4x2": [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "top4x3": [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "top4x4": [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

MATRIX_A_MODES = [
    "exact_dflash",
    "dflash_ssd_exact_off_normal",
    "dflash_ssd_exact_on_oracle",
    "dflash_ssd_predicted_off_oracle",
    "dflash_ssd_predicted_on_oracle",
    "dflash_ssd_predicted_on_normal",
]

MODE_SPECS = {
    "exact_dflash": {
        "draft_backend": "dflash",
        "draft_async": False,
    },
    "dflash_ssd_exact_off_normal": {
        "draft_backend": "dflash_ssd",
        "draft_async": True,
        "dflash_context_mode": "exact",
        "dflash_branch_cache": "off",
        "dflash_branch_key_mode": "normal",
    },
    "dflash_ssd_exact_on_oracle": {
        "draft_backend": "dflash_ssd",
        "draft_async": True,
        "dflash_context_mode": "exact",
        "dflash_branch_cache": "on",
        "dflash_branch_key_mode": "oracle",
    },
    "dflash_ssd_predicted_off_oracle": {
        "draft_backend": "dflash_ssd",
        "draft_async": True,
        "dflash_context_mode": "predicted",
        "dflash_branch_cache": "off",
        "dflash_branch_key_mode": "oracle",
    },
    "dflash_ssd_predicted_on_normal": {
        "draft_backend": "dflash_ssd",
        "draft_async": True,
        "dflash_context_mode": "predicted",
        "dflash_branch_cache": "on",
        "dflash_branch_key_mode": "normal",
    },
    "dflash_ssd_predicted_on_oracle": {
        "draft_backend": "dflash_ssd",
        "draft_async": True,
        "dflash_context_mode": "predicted",
        "dflash_branch_cache": "on",
        "dflash_branch_key_mode": "oracle",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the DFlash SSD diagnostic matrix and predictor-quality analysis on the held-out split",
    )
    parser.add_argument("--target", type=str, required=True, help="Qwen3-8B snapshot directory")
    parser.add_argument("--draft", type=str, required=True, help="Qwen3-8B-DFlash-b16 draft directory")
    parser.add_argument("--predictor", type=str, required=True, help="Predictor checkpoint directory")
    parser.add_argument("--training-metadata", type=str, required=True, help="training_metadata.json from the predictor run")
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--batch-sizes", type=str, default="1,2,4")
    parser.add_argument("--max-prompts", type=int, default=64, help="Maximum held-out prompts to evaluate")
    parser.add_argument("--topk", type=int, default=10, help="Top-k used for DFlash overlap diagnostics")
    parser.add_argument("--out", type=str, default=None, help="Optional summary JSON path")
    parser.add_argument("--base-dist-port", type=int, default=12470, help="Base SSD_DIST_PORT for subprocess orchestration")
    parser.add_argument(
        "--fanout-template-name",
        type=str,
        default="baseline48",
        choices=list(BRANCH_TEMPLATES.keys()),
        help="Named branch-fanout template used for dflash_ssd runs",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=[*MODE_SPECS.keys(), "quality_metrics"],
        help="Internal single-run mode. When unset, orchestrates the full matrix.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Internal single-run batch size")
    parser.add_argument("--artifact-dir", type=str, default=None, help="Internal artifact directory for a single run")
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_prompt_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _select_prompt_records(args) -> list[dict]:
    metadata = _load_json(Path(args.training_metadata))
    prompt_manifest = _load_prompt_manifest(Path(metadata["prompt_manifest"]))
    test_groups = set(metadata["split_group_keys"]["test"])
    selected = [row for row in prompt_manifest if row["group_key"] in test_groups]
    selected.sort(key=lambda row: (row["dataset_name"], row["prompt_index"]))
    if args.max_prompts > 0:
        selected = selected[:args.max_prompts]
    if not selected:
        raise RuntimeError("No held-out test prompts selected for evaluation")
    return selected


def _safe_mean(values):
    return None if not values else sum(values) / len(values)


def _stderr(values):
    if len(values) <= 1:
        return 0.0 if values else None
    mu = sum(values) / len(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var / len(values))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _enrich_cycle_rows(rows: list[dict], prompt_records: list[dict]) -> list[dict]:
    seq_ids = sorted({int(row["seq_id"]) for row in rows})
    if len(seq_ids) > len(prompt_records):
        raise RuntimeError(
            f"Observed {len(seq_ids)} sequence ids for only {len(prompt_records)} prompt records"
        )
    seq_id_to_prompt = {
        seq_id: prompt_records[idx]
        for idx, seq_id in enumerate(seq_ids)
    }
    enriched = []
    for row in rows:
        seq_id = int(row["seq_id"])
        if seq_id not in seq_id_to_prompt:
            raise RuntimeError(f"Unexpected seq_id={seq_id} for prompt manifest of size {len(prompt_records)}")
        prompt = seq_id_to_prompt[seq_id]
        enriched.append(
            {
                **row,
                "group_key": prompt["group_key"],
                "dataset_name": prompt["dataset_name"],
                "prompt_index": prompt["prompt_index"],
            }
        )
    return enriched


def _summarize_cycle_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "cache_hit_mean": None,
            "miss_rate": None,
            "fallback_rate": None,
            "cache_committed_token_fraction": None,
            "mean_cycle_latency_ms": None,
            "hit_cycle_latency_ms": None,
            "miss_cycle_latency_ms": None,
            "hit_only_accepted_suffix_mean": None,
            "miss_only_accepted_suffix_mean": None,
            "predictor_ms": None,
            "dflash_ms": None,
            "target_verify_ms": None,
            "cache_lookup_ms": None,
            "transport_ms": None,
            "num_branches_generated_mean": None,
            "accepted_length_support_rate": None,
            "recovery_recall_given_supported_accept": None,
            "joint_branch_recall": None,
            "recovery_entropy_at_actual_accept_mean": None,
            "recovery_top1_margin_at_actual_accept_mean": None,
            "true_branch_rank_distribution": {},
            "prompt_group_metrics": {},
            "prompt_group_summary": {},
        }

    cache_hits = [1.0 if row["cache_hit"] else 0.0 for row in rows]
    misses = [1.0 - hit for hit in cache_hits]
    fallback = [1.0 if row["fallback_used"] else 0.0 for row in rows]
    cycle_ms = [row["total_cycle_ms"] for row in rows]
    predictor_ms = [row["predictor_ms"] for row in rows]
    dflash_ms = [row["dflash_ms"] for row in rows]
    verify_ms = [row["target_verify_ms"] for row in rows]
    cache_lookup_ms = [row["cache_lookup_ms"] for row in rows]
    transport_ms = [row["transport_ms"] for row in rows]
    branches = [row["num_branches_generated"] for row in rows]
    committed_from_cache = [row["committed_tokens_from_cache"] for row in rows]
    hit_cycle_ms = [row["total_cycle_ms"] for row in rows if row["cache_hit"]]
    miss_cycle_ms = [row["total_cycle_ms"] for row in rows if not row["cache_hit"]]
    hit_accepted = [row["accepted_len"] for row in rows if row["cache_hit"]]
    miss_accepted = [row["accepted_len"] for row in rows if not row["cache_hit"]]
    true_branch_ranks = [row["true_branch_rank"] for row in rows if row["true_branch_rank"] is not None]
    accept_support = [row["actual_accept_supported"] for row in rows if row["actual_accept_supported"] is not None]
    joint_support = [row["joint_branch_supported"] for row in rows if row["joint_branch_supported"] is not None]
    recovery_recall = [
        row["joint_branch_supported"]
        for row in rows
        if row["actual_accept_supported"] is True and row["joint_branch_supported"] is not None
    ]
    recovery_entropy = [
        row["recovery_entropy_at_actual_accept"]
        for row in rows
        if row["recovery_entropy_at_actual_accept"] is not None
    ]
    recovery_margin = [
        row["recovery_top1_margin_at_actual_accept"]
        for row in rows
        if row["recovery_top1_margin_at_actual_accept"] is not None
    ]
    total_committed = sum(row["tokens_committed_this_cycle"] for row in rows)

    per_group = defaultdict(list)
    for row in rows:
        per_group[row["group_key"]].append(row)

    prompt_group_metrics = {}
    for group_key, group_rows in per_group.items():
        tokens = sum(row["tokens_committed_this_cycle"] for row in group_rows)
        total_s = sum(row["total_cycle_ms"] for row in group_rows) / 1000.0
        prompt_group_metrics[group_key] = {
            "throughput_tok_s": tokens / max(total_s, 1e-6),
            "accepted_suffix_mean": mean([row["accepted_len"] for row in group_rows]),
            "cache_hit_mean": mean([1.0 if row["cache_hit"] else 0.0 for row in group_rows]),
            "miss_rate": mean([0.0 if row["cache_hit"] else 1.0 for row in group_rows]),
            "cache_committed_token_fraction": (
                sum(row["committed_tokens_from_cache"] for row in group_rows)
                / max(sum(row["tokens_committed_this_cycle"] for row in group_rows), 1)
            ),
            "mean_cycle_latency_ms": mean([row["total_cycle_ms"] for row in group_rows]),
            "hit_cycle_latency_ms": _safe_mean([row["total_cycle_ms"] for row in group_rows if row["cache_hit"]]),
            "miss_cycle_latency_ms": _safe_mean([row["total_cycle_ms"] for row in group_rows if not row["cache_hit"]]),
            "hit_only_accepted_suffix_mean": _safe_mean([row["accepted_len"] for row in group_rows if row["cache_hit"]]),
            "miss_only_accepted_suffix_mean": _safe_mean([row["accepted_len"] for row in group_rows if not row["cache_hit"]]),
            "predictor_ms": mean([row["predictor_ms"] for row in group_rows]),
            "dflash_ms": mean([row["dflash_ms"] for row in group_rows]),
            "target_verify_ms": mean([row["target_verify_ms"] for row in group_rows]),
            "accepted_length_support_rate": _safe_mean([
                float(row["actual_accept_supported"]) for row in group_rows
                if row["actual_accept_supported"] is not None
            ]),
            "recovery_recall_given_supported_accept": _safe_mean([
                float(row["joint_branch_supported"]) for row in group_rows
                if row["actual_accept_supported"] is True and row["joint_branch_supported"] is not None
            ]),
            "joint_branch_recall": _safe_mean([
                float(row["joint_branch_supported"]) for row in group_rows
                if row["joint_branch_supported"] is not None
            ]),
        }

    group_metric_names = [
        "throughput_tok_s",
        "accepted_suffix_mean",
        "cache_hit_mean",
        "miss_rate",
        "cache_committed_token_fraction",
        "mean_cycle_latency_ms",
        "hit_only_accepted_suffix_mean",
        "miss_only_accepted_suffix_mean",
        "predictor_ms",
        "dflash_ms",
        "target_verify_ms",
        "accepted_length_support_rate",
        "recovery_recall_given_supported_accept",
        "joint_branch_recall",
    ]
    group_summary = {}
    for metric_name in group_metric_names:
        values = [metrics[metric_name] for metrics in prompt_group_metrics.values() if metrics[metric_name] is not None]
        group_summary[metric_name] = {
            "mean": _safe_mean(values),
            "stderr": _stderr(values),
        }

    return {
        "cache_hit_mean": mean(cache_hits),
        "miss_rate": mean(misses),
        "fallback_rate": mean(fallback),
        "cache_committed_token_fraction": (
            sum(committed_from_cache) / max(total_committed, 1)
        ),
        "mean_cycle_latency_ms": mean(cycle_ms),
        "hit_cycle_latency_ms": _safe_mean(hit_cycle_ms),
        "miss_cycle_latency_ms": _safe_mean(miss_cycle_ms),
        "hit_only_accepted_suffix_mean": _safe_mean(hit_accepted),
        "miss_only_accepted_suffix_mean": _safe_mean(miss_accepted),
        "predictor_ms": mean(predictor_ms),
        "dflash_ms": mean(dflash_ms),
        "target_verify_ms": mean(verify_ms),
        "cache_lookup_ms": mean(cache_lookup_ms),
        "transport_ms": mean(transport_ms),
        "num_branches_generated_mean": mean(branches),
        "accepted_length_support_rate": _safe_mean([float(value) for value in accept_support]),
        "recovery_recall_given_supported_accept": _safe_mean([float(value) for value in recovery_recall]),
        "joint_branch_recall": _safe_mean([float(value) for value in joint_support]),
        "recovery_entropy_at_actual_accept_mean": _safe_mean(recovery_entropy),
        "recovery_top1_margin_at_actual_accept_mean": _safe_mean(recovery_margin),
        "true_branch_rank_distribution": dict(sorted(Counter(true_branch_ranks).items())),
        "prompt_group_metrics": prompt_group_metrics,
        "prompt_group_summary": group_summary,
    }


def _build_llm_kwargs(mode: str, args, batch_size: int) -> dict:
    spec = MODE_SPECS[mode]
    fan_out_list = BRANCH_TEMPLATES[args.fanout_template_name]
    llm_kwargs = {
        "num_gpus": args.gpus,
        "speculate": True,
        "draft": args.draft,
        "draft_backend": spec["draft_backend"],
        "draft_async": spec["draft_async"],
        "max_num_seqs": batch_size,
        "verbose": False,
    }
    if spec["draft_backend"] == "dflash_ssd":
        llm_kwargs.update(
            dflash_predictor=args.predictor,
            dflash_context_mode=spec["dflash_context_mode"],
            dflash_branch_cache=spec["dflash_branch_cache"],
            dflash_branch_key_mode=spec["dflash_branch_key_mode"],
            dflash_enable_diagnostics=True,
        )
        llm_kwargs["fan_out_list"] = fan_out_list
        llm_kwargs["fan_out_list_miss"] = fan_out_list
        llm_kwargs["async_fan_out"] = max(fan_out_list) if fan_out_list else 0
    return llm_kwargs


def _run_online_mode(mode: str, args, prompt_records: list[dict], batch_size: int, artifact_dir: Path | None = None) -> dict:
    from ssd.engine.llm_engine import LLMEngine
    from ssd.sampling_params import SamplingParams

    prompt_token_ids = [row["prompt_token_ids"] for row in prompt_records]
    engine = LLMEngine(args.target, **_build_llm_kwargs(mode, args, batch_size))
    try:
        sampling_params = [
            SamplingParams(
                temperature=0.0,
                draft_temperature=0.0,
                max_new_tokens=args.output_len,
                ignore_eos=True,
            )
            for _ in prompt_token_ids
        ]
        t0 = perf_counter()
        outputs, metrics = engine.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        total_time = perf_counter() - t0
        total_completion_tokens = sum(len(output["token_ids"]) for output in outputs)
        cycle_rows = _enrich_cycle_rows(metrics.get("dflash_cycle_diagnostics", []), prompt_records)
        cycle_rows = [
            {
                **row,
                "fanout_template_name": args.fanout_template_name,
            }
            for row in cycle_rows
        ]
        cycle_summary = _summarize_cycle_rows(cycle_rows)

        result = {
            "mode": mode,
            "batch_size": batch_size,
            "num_prompts": len(prompt_records),
            "fanout_template_name": args.fanout_template_name,
            "throughput_tok_s": total_completion_tokens / max(total_time, 1e-6),
            "accepted_suffix_mean": _safe_mean(metrics.get("accepted_suffix_lens_with_recovery", [])),
            "fraction_accepted": (
                (
                    sum(metrics.get("accepted_suffix_lens_with_recovery", []))
                    - len(metrics.get("accepted_suffix_lens_with_recovery", []))
                ) / (
                    len(metrics.get("accepted_suffix_lens_with_recovery", [])) * engine.config.speculate_k
                )
                if metrics.get("accepted_suffix_lens_with_recovery") else None
            ),
            "target_verify_ms": (
                sum(metrics["target_verify_times"]) * 1000 / len(metrics["target_verify_times"])
                if metrics.get("target_verify_times") else None
            ),
            "dflash_step_ms": (
                sum(metrics["dflash_draft_step_times"]) * 1000 / len(metrics["dflash_draft_step_times"])
                if metrics.get("dflash_draft_step_times") else None
            ),
            "predictor_ms": (
                sum(metrics["dflash_predictor_times"]) * 1000 / len(metrics["dflash_predictor_times"])
                if metrics.get("dflash_predictor_times") else None
            ),
            "cache_hit_mean": cycle_summary["cache_hit_mean"] if cycle_rows else (
                _safe_mean(metrics.get("cache_hits", []))
            ),
            "miss_rate": cycle_summary["miss_rate"],
            "fallback_rate": cycle_summary["fallback_rate"],
            "cache_committed_token_fraction": cycle_summary["cache_committed_token_fraction"],
            "mean_cycle_latency_ms": cycle_summary["mean_cycle_latency_ms"],
            "hit_cycle_latency_ms": cycle_summary["hit_cycle_latency_ms"],
            "miss_cycle_latency_ms": cycle_summary["miss_cycle_latency_ms"],
            "cache_lookup_ms": cycle_summary["cache_lookup_ms"],
            "transport_ms": cycle_summary["transport_ms"],
            "num_branches_generated_mean": cycle_summary["num_branches_generated_mean"],
            "accepted_suffix_mean_on_hit": cycle_summary["hit_only_accepted_suffix_mean"],
            "accepted_suffix_mean_on_miss": cycle_summary["miss_only_accepted_suffix_mean"],
            "hit_only_accepted_suffix_mean": cycle_summary["hit_only_accepted_suffix_mean"],
            "miss_only_accepted_suffix_mean": cycle_summary["miss_only_accepted_suffix_mean"],
            "accepted_length_support_rate": cycle_summary["accepted_length_support_rate"],
            "recovery_recall_given_supported_accept": cycle_summary["recovery_recall_given_supported_accept"],
            "joint_branch_recall": cycle_summary["joint_branch_recall"],
            "recovery_entropy_at_actual_accept_mean": cycle_summary["recovery_entropy_at_actual_accept_mean"],
            "recovery_top1_margin_at_actual_accept_mean": cycle_summary["recovery_top1_margin_at_actual_accept_mean"],
            "true_branch_rank_distribution": cycle_summary["true_branch_rank_distribution"],
            "prompt_group_summary": cycle_summary["prompt_group_summary"],
        }

        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            _write_jsonl(artifact_dir / "cycles.jsonl", cycle_rows)
            _write_json(artifact_dir / "result.json", result)
            _write_json(artifact_dir / "prompt_group_metrics.json", cycle_summary["prompt_group_metrics"])
            result["artifact_dir"] = str(artifact_dir)
        return result
    finally:
        engine.exit(hard=False)


def _concat_selected_hidden(hidden_states: tuple[torch.Tensor, ...], layer_ids: list[int]) -> torch.Tensor:
    selected = [hidden_states[layer_id + 1] for layer_id in layer_ids]
    return torch.cat(selected, dim=-1)


def _kl_divergence(pred_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    pred_log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    ref_log_probs = torch.log_softmax(ref_logits.float(), dim=-1)
    ref_probs = ref_log_probs.exp()
    return torch.sum(ref_probs * (ref_log_probs - pred_log_probs), dim=-1)


def _collect_quality_metrics(args, prompt_records: list[dict]) -> dict:
    from ssd.config import Config
    from ssd.engine.dflash_runtime import DFlashRuntime
    from ssd.utils.verify import verify

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = Config(
        model=args.target,
        draft=args.draft,
        speculate=True,
        draft_backend="dflash_ssd",
        draft_async=True,
        num_gpus=2,
        dflash_predictor=args.predictor,
        max_num_seqs=1,
    )
    runtime = DFlashRuntime(config=config, device=device, predictor_path=args.predictor, metrics=None)
    layer_ids = config.dflash_target_layer_ids
    topk = args.topk

    feature_cos_by_pos = [[] for _ in range(config.speculate_k + 1)]
    feature_norm_err_by_pos = [[] for _ in range(config.speculate_k + 1)]
    token_match_by_pos = [[] for _ in range(config.speculate_k)]
    topk_overlap_by_pos = [[] for _ in range(config.speculate_k)]
    kl_by_pos = [[] for _ in range(config.speculate_k)]

    for prompt in prompt_records:
        prompt_tokens = prompt["prompt_token_ids"]
        prefix_tokens = list(prompt_tokens)
        prompt_ids = torch.tensor(prefix_tokens, dtype=torch.int64, device=device).unsqueeze(0)
        outputs = runtime.target_model(
            input_ids=prompt_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        exact_history = _concat_selected_hidden(outputs.hidden_states, layer_ids)[0].to(runtime.dtype)
        recovery_token = int(outputs.logits[0, -1].argmax().item())
        committed = 0

        while committed < args.output_len:
            current_block = runtime.run_block_batch(
                [exact_history],
                torch.tensor([recovery_token], dtype=torch.int64, device=device),
                torch.tensor([0.0], dtype=torch.float32, device=device),
                return_predicted_features=True,
            )
            speculations = torch.cat(
                [
                    torch.tensor([[recovery_token]], dtype=torch.int64, device=device),
                    current_block.draft_tokens,
                ],
                dim=1,
            )
            verify_tokens = prefix_tokens + speculations[0].tolist()
            verify_ids = torch.tensor(verify_tokens, dtype=torch.int64, device=device).unsqueeze(0)
            verify_outputs = runtime.target_model(
                input_ids=verify_ids,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            logits_p = verify_outputs.logits[:, -(config.speculate_k + 1):, :].contiguous()
            exact_features_full = _concat_selected_hidden(verify_outputs.hidden_states, layer_ids)[:, -(config.speculate_k + 1):, :].contiguous()
            accepted_suffixes, recovery_tokens = verify(
                logits_p=logits_p,
                logits_q=current_block.logits_q,
                speculations=speculations,
                temperatures_target=torch.tensor([0.0], dtype=torch.float32, device=device),
                temperatures_draft=torch.tensor([0.0], dtype=torch.float32, device=device),
                cache_hits=torch.ones(1, dtype=torch.int64, device=device),
            )
            accepted_len = len(accepted_suffixes[0])
            predicted_features = current_block.predicted_target_features[0]
            exact_features = exact_features_full[0]
            cosine = torch.nn.functional.cosine_similarity(predicted_features, exact_features, dim=-1)
            norm_error = (predicted_features.norm(dim=-1) - exact_features.norm(dim=-1)).abs() / exact_features.norm(dim=-1).clamp_min(1e-6)
            for pos in range(config.speculate_k + 1):
                feature_cos_by_pos[pos].append(float(cosine[pos].item()))
                feature_norm_err_by_pos[pos].append(float(norm_error[pos].item()))

            exact_child_history = torch.cat([exact_history, exact_features[:accepted_len]], dim=0).contiguous()
            predicted_child_history = torch.cat([exact_history, predicted_features[:accepted_len]], dim=0).contiguous()
            next_recovery = int(recovery_tokens[0])
            exact_next = runtime.run_block_batch(
                [exact_child_history],
                torch.tensor([next_recovery], dtype=torch.int64, device=device),
                torch.tensor([0.0], dtype=torch.float32, device=device),
                return_predicted_features=False,
            )
            predicted_next = runtime.run_block_batch(
                [predicted_child_history],
                torch.tensor([next_recovery], dtype=torch.int64, device=device),
                torch.tensor([0.0], dtype=torch.float32, device=device),
                return_predicted_features=False,
            )

            for pos in range(config.speculate_k):
                token_match_by_pos[pos].append(
                    float((predicted_next.draft_tokens[0, pos] == exact_next.draft_tokens[0, pos]).item())
                )
                pred_topk = torch.topk(predicted_next.logits_q[0, pos], k=topk).indices.tolist()
                exact_topk = torch.topk(exact_next.logits_q[0, pos], k=topk).indices.tolist()
                overlap = len(set(pred_topk) & set(exact_topk)) / float(topk)
                topk_overlap_by_pos[pos].append(overlap)
                kl_by_pos[pos].append(float(_kl_divergence(predicted_next.logits_q[0, pos], exact_next.logits_q[0, pos]).item()))

            prefix_tokens.extend(accepted_suffixes[0])
            committed += len(accepted_suffixes[0])
            exact_history = exact_child_history
            recovery_token = next_recovery

    return {
        "topk": topk,
        "feature_cosine_by_position": [_safe_mean(values) for values in feature_cos_by_pos],
        "feature_norm_error_by_position": [_safe_mean(values) for values in feature_norm_err_by_pos],
        "dflash_token_match_rate_by_position": [_safe_mean(values) for values in token_match_by_pos],
        "dflash_topk_overlap_by_position": [_safe_mean(values) for values in topk_overlap_by_pos],
        "dflash_logits_kl_by_position": [_safe_mean(values) for values in kl_by_pos],
        "feature_cosine_mean": _safe_mean([value for values in feature_cos_by_pos for value in values]),
        "feature_norm_error_mean": _safe_mean([value for values in feature_norm_err_by_pos for value in values]),
        "dflash_token_match_rate_mean": _safe_mean([value for values in token_match_by_pos for value in values]),
        "dflash_topk_overlap_mean": _safe_mean([value for values in topk_overlap_by_pos for value in values]),
        "dflash_logits_kl_mean": _safe_mean([value for values in kl_by_pos for value in values]),
    }


def _run_single(args):
    prompt_records = _select_prompt_records(args)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    if args.mode == "quality_metrics":
        result = _collect_quality_metrics(args, prompt_records)
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            _write_json(artifact_dir / "result.json", result)
    else:
        if args.batch_size is None:
            raise ValueError("--batch-size is required with --mode for online modes")
        result = _run_online_mode(args.mode, args, prompt_records, args.batch_size, artifact_dir=artifact_dir)
    print(f"RESULT_JSON={json.dumps(result, sort_keys=True)}", flush=True)


def _run_subprocess(mode: str, args, batch_size: int | None, port: int, artifact_dir: Path | None = None) -> dict:
    cmd = [
        sys.executable,
        "-O",
        str(Path(__file__).resolve()),
        "--target", args.target,
        "--draft", args.draft,
        "--predictor", args.predictor,
        "--training-metadata", args.training_metadata,
        "--output-len", str(args.output_len),
        "--gpus", str(args.gpus),
        "--max-prompts", str(args.max_prompts),
        "--topk", str(args.topk),
        "--fanout-template-name", args.fanout_template_name,
        "--mode", mode,
    ]
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if artifact_dir is not None:
        cmd.extend(["--artifact-dir", str(artifact_dir)])
    env = os.environ.copy()
    env["SSD_DIST_PORT"] = str(port)
    completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr, flush=True)
        raise RuntimeError(f"Subprocess failed for mode={mode} batch_size={batch_size}")
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("RESULT_JSON="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError(f"Missing RESULT_JSON for mode={mode} batch_size={batch_size}")


def _clone_args(args, **overrides):
    cloned = argparse.Namespace(**vars(args))
    for key, value in overrides.items():
        setattr(cloned, key, value)
    return cloned


def _best_matrix_b_templates(matrix_b_results: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for batch_size in sorted({row["batch_size"] for row in matrix_b_results}):
        candidates = [row for row in matrix_b_results if row["batch_size"] == batch_size]
        candidates.sort(
            key=lambda row: (
                row["throughput_tok_s"],
                row["cache_committed_token_fraction"] if row["cache_committed_token_fraction"] is not None else -1.0,
                row["joint_branch_recall"] if row["joint_branch_recall"] is not None else -1.0,
            ),
            reverse=True,
        )
        best[str(batch_size)] = {
            "fanout_template_name": candidates[0]["fanout_template_name"],
            "throughput_tok_s": candidates[0]["throughput_tok_s"],
            "cache_committed_token_fraction": candidates[0]["cache_committed_token_fraction"],
            "joint_branch_recall": candidates[0]["joint_branch_recall"],
        }
    return best


def _build_recommendation(matrix_a_results: list[dict], matrix_b_results: list[dict]) -> dict:
    by_mode_batch = {
        (row["mode"], row["batch_size"]): row
        for row in matrix_a_results
    }
    headroom_by_batch = {}
    branch_gap_by_batch = {}
    no_meaningful_headroom = True
    branch_bottleneck = False
    for batch_size in sorted({row["batch_size"] for row in matrix_a_results}):
        exact = by_mode_batch[("exact_dflash", batch_size)]
        exact_oracle = by_mode_batch[("dflash_ssd_exact_on_oracle", batch_size)]
        predicted_normal = by_mode_batch[("dflash_ssd_predicted_on_normal", batch_size)]
        predicted_oracle = by_mode_batch[("dflash_ssd_predicted_on_oracle", batch_size)]
        headroom_ratio = exact_oracle["throughput_tok_s"] / max(exact["throughput_tok_s"], 1e-6)
        within_five_percent = headroom_ratio <= 1.05
        headroom_by_batch[str(batch_size)] = {
            "exact_dflash_tok_s": exact["throughput_tok_s"],
            "exact_on_oracle_tok_s": exact_oracle["throughput_tok_s"],
            "ratio_to_exact_dflash": headroom_ratio,
            "not_more_than_5_percent_faster": within_five_percent,
        }
        no_meaningful_headroom = no_meaningful_headroom and within_five_percent

        normal_vs_oracle_ratio = predicted_normal["throughput_tok_s"] / max(predicted_oracle["throughput_tok_s"], 1e-6)
        branch_gap_by_batch[str(batch_size)] = {
            "predicted_on_normal_tok_s": predicted_normal["throughput_tok_s"],
            "predicted_on_oracle_tok_s": predicted_oracle["throughput_tok_s"],
            "normal_vs_oracle_ratio": normal_vs_oracle_ratio,
            "normal_is_15_percent_slower": normal_vs_oracle_ratio < 0.85,
        }
        if headroom_ratio > 1.05 and normal_vs_oracle_ratio < 0.85:
            branch_bottleneck = True

    best_templates = _best_matrix_b_templates(matrix_b_results)
    recommendation_text = (
        "stop_branch_policy_work"
        if no_meaningful_headroom
        else "prioritize_branch_policy_optimization"
    )
    return {
        "decision_rule_a": {
            "triggered": no_meaningful_headroom,
            "message": (
                "SSD provides no meaningful overlap headroom for DFlash on this setup"
                if no_meaningful_headroom
                else "SSD retains measurable overlap headroom over exact regular DFlash"
            ),
            "by_batch": headroom_by_batch,
        },
        "decision_rule_b": {
            "triggered": branch_bottleneck,
            "message": (
                "Branch selection / branch budget is the dominant remaining bottleneck"
                if branch_bottleneck
                else "Branch selection / branch budget is not yet isolated as the dominant bottleneck"
            ),
            "by_batch": branch_gap_by_batch,
        },
        "decision_rule_c": {
            "best_templates_by_batch": best_templates,
            "message": "Templates ranked by throughput, then cache-committed token fraction, then joint branch recall",
        },
        "recommended_next_step": recommendation_text,
    }


def _build_combined_markdown_table(matrix_a_results: list[dict], matrix_b_results: list[dict]) -> str:
    lines = [
        "| matrix | label | b | tok/s | cache hit | cache-committed frac | joint recall | hit-only accepted | miss-only accepted | dflash ms | predictor ms | branches |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in matrix_a_results:
        lines.append(
            f"| A | {row['mode']} | {row['batch_size']} | {row['throughput_tok_s']:.2f} | "
            f"{row['cache_hit_mean'] if row['cache_hit_mean'] is not None else 'n/a'} | "
            f"{row['cache_committed_token_fraction'] if row['cache_committed_token_fraction'] is not None else 'n/a'} | "
            f"{row['joint_branch_recall'] if row['joint_branch_recall'] is not None else 'n/a'} | "
            f"{row['hit_only_accepted_suffix_mean'] if row['hit_only_accepted_suffix_mean'] is not None else 'n/a'} | "
            f"{row['miss_only_accepted_suffix_mean'] if row['miss_only_accepted_suffix_mean'] is not None else 'n/a'} | "
            f"{row['dflash_step_ms'] if row['dflash_step_ms'] is not None else 'n/a'} | "
            f"{row['predictor_ms'] if row['predictor_ms'] is not None else 'n/a'} | "
            f"{row['num_branches_generated_mean'] if row['num_branches_generated_mean'] is not None else 'n/a'} |"
        )
    for row in matrix_b_results:
        lines.append(
            f"| B | {row['fanout_template_name']} | {row['batch_size']} | {row['throughput_tok_s']:.2f} | "
            f"{row['cache_hit_mean'] if row['cache_hit_mean'] is not None else 'n/a'} | "
            f"{row['cache_committed_token_fraction'] if row['cache_committed_token_fraction'] is not None else 'n/a'} | "
            f"{row['joint_branch_recall'] if row['joint_branch_recall'] is not None else 'n/a'} | "
            f"{row['hit_only_accepted_suffix_mean'] if row['hit_only_accepted_suffix_mean'] is not None else 'n/a'} | "
            f"{row['miss_only_accepted_suffix_mean'] if row['miss_only_accepted_suffix_mean'] is not None else 'n/a'} | "
            f"{row['dflash_step_ms'] if row['dflash_step_ms'] is not None else 'n/a'} | "
            f"{row['predictor_ms'] if row['predictor_ms'] is not None else 'n/a'} | "
            f"{row['num_branches_generated_mean'] if row['num_branches_generated_mean'] is not None else 'n/a'} |"
        )
    return "\n".join(lines)


def _run_all(args):
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    output_path = Path(args.out) if args.out else None
    artifact_root = None
    if output_path is not None:
        artifact_root = output_path.with_suffix("")
        artifact_root = artifact_root.parent / f"{artifact_root.name}_artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)

    matrix_a_results = []
    matrix_b_results = []
    run_idx = 0

    matrix_a_args = _clone_args(args, fanout_template_name="baseline48")
    for batch_size in batch_sizes:
        for mode in MATRIX_A_MODES:
            artifact_dir = (
                artifact_root / "matrix_a" / f"{mode}_b{batch_size}"
                if artifact_root is not None else None
            )
            matrix_a_results.append(
                _run_subprocess(mode, matrix_a_args, batch_size, args.base_dist_port + run_idx, artifact_dir=artifact_dir)
            )
            run_idx += 1

    for template_name in BRANCH_TEMPLATES:
        template_args = _clone_args(args, fanout_template_name=template_name)
        for batch_size in batch_sizes:
            artifact_dir = (
                artifact_root / "matrix_b" / f"{template_name}_b{batch_size}"
                if artifact_root is not None else None
            )
            matrix_b_results.append(
                _run_subprocess(
                    "dflash_ssd_predicted_on_normal",
                    template_args,
                    batch_size,
                    args.base_dist_port + run_idx,
                    artifact_dir=artifact_dir,
                )
            )
            run_idx += 1

    quality_artifact_dir = artifact_root / "quality_metrics" if artifact_root is not None else None
    quality_metrics = _run_subprocess(
        "quality_metrics",
        matrix_a_args,
        batch_size=None,
        port=args.base_dist_port + run_idx,
        artifact_dir=quality_artifact_dir,
    )

    exact_b1 = next(
        row for row in matrix_a_results
        if row["mode"] == "dflash_ssd_exact_off_normal" and row["batch_size"] == 1
    )
    predicted_b1 = next(
        row for row in matrix_a_results
        if row["mode"] == "dflash_ssd_predicted_off_oracle" and row["batch_size"] == 1
    )
    exact_accept = exact_b1["accepted_suffix_mean"]
    predicted_accept = predicted_b1["accepted_suffix_mean"]
    quality_metrics["accepted_suffix_exact_context"] = exact_accept
    quality_metrics["accepted_suffix_predicted_context"] = predicted_accept
    quality_metrics["accepted_suffix_delta"] = (
        None if exact_accept is None or predicted_accept is None else exact_accept - predicted_accept
    )

    recommendation = _build_recommendation(matrix_a_results, matrix_b_results)
    combined_table = _build_combined_markdown_table(matrix_a_results, matrix_b_results)

    print(combined_table)
    print()
    print("Quality metrics:")
    print(json.dumps(quality_metrics, indent=2, sort_keys=True))
    print()
    print("Recommendation:")
    print(json.dumps(recommendation, indent=2, sort_keys=True))

    matrix_a_summary = {
        "results": matrix_a_results,
        "quality_metrics": quality_metrics,
        "recommendation": recommendation,
    }
    matrix_b_summary = {
        "results": matrix_b_results,
        "best_templates_by_batch": recommendation["decision_rule_c"]["best_templates_by_batch"],
    }
    summary = {
        "training_metadata": args.training_metadata,
        "predictor": args.predictor,
        "matrix_a": matrix_a_summary,
        "matrix_b": matrix_b_summary,
        "quality_metrics": quality_metrics,
        "recommendation": recommendation,
        "combined_markdown_table": combined_table,
    }
    if output_path is not None:
        _write_json(artifact_root / "matrix_a_summary.json", matrix_a_summary)
        _write_json(artifact_root / "matrix_b_summary.json", matrix_b_summary)
        _write_json(output_path, summary)


def main():
    args = parse_args()
    if args.mode:
        _run_single(args)
    else:
        _run_all(args)


if __name__ == "__main__":
    main()
