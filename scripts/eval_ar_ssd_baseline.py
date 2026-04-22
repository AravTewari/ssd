import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter

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

MODE_SPECS = {
    "ar_async_exact_off_normal": {
        "ar_branch_cache": "off",
        "ar_branch_key_mode": "normal",
    },
    "ar_async_exact_on_oracle": {
        "ar_branch_cache": "on",
        "ar_branch_key_mode": "oracle",
    },
    "ar_async_normal": {
        "ar_branch_cache": "on",
        "ar_branch_key_mode": "normal",
    },
}


def _fanout_template_for_k(template_name: str, k: int) -> list[int]:
    base = BRANCH_TEMPLATES[template_name]
    needed = k + 1
    if needed > len(base):
        raise ValueError(
            f"fanout template {template_name} only defines {len(base)} positions, but k={k} requires {needed}"
        )
    return list(base[:needed])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the matched AR SSD baseline matrix on the held-out split",
    )
    parser.add_argument("--target", type=str, required=True, help="Qwen3-8B snapshot directory")
    parser.add_argument("--draft", type=str, required=True, help="AR draft snapshot directory")
    parser.add_argument("--training-metadata", type=str, required=True, help="training_metadata.json used for the held-out split")
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--batch-sizes", type=str, default="1,2,4")
    parser.add_argument("--max-prompts", type=int, default=78)
    parser.add_argument("--k", type=int, default=15, help="Speculative lookahead for single-run execution")
    parser.add_argument(
        "--k-values",
        type=str,
        default="15",
        help="Comma-separated speculative lookahead values for full-matrix execution",
    )
    parser.add_argument("--out", type=str, default=None, help="Optional summary JSON path")
    parser.add_argument(
        "--fanout-template-name",
        type=str,
        default="baseline48",
        choices=list(BRANCH_TEMPLATES.keys()),
        help="Branch-fanout template for async AR SSD runs",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=list(MODE_SPECS.keys()),
        help="Internal single-run mode. When unset, runs the full AR matrix.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(MODE_SPECS.keys()),
        help="Comma-separated subset of modes to run in full-matrix execution",
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
    if not rows:
        return []
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
            "cache_committed_token_fraction": None,
            "mean_cycle_latency_ms": None,
            "draft_service_ms": None,
            "post_verify_feedback_ms": None,
            "target_verify_ms": None,
            "hit_only_accepted_suffix_mean": None,
            "miss_only_accepted_suffix_mean": None,
            "prompt_group_metrics": {},
            "prompt_group_summary": {},
        }

    cache_hits = [1.0 if row["cache_hit"] else 0.0 for row in rows]
    cycle_ms = [row["total_cycle_ms"] for row in rows]
    draft_service_ms = [row["draft_service_ms"] for row in rows]
    feedback_ms = [row["post_verify_feedback_ms"] for row in rows]
    verify_ms = [row["target_verify_ms"] for row in rows]
    hit_cycle_ms = [row["total_cycle_ms"] for row in rows if row["cache_hit"]]
    miss_cycle_ms = [row["total_cycle_ms"] for row in rows if not row["cache_hit"]]
    hit_accepted = [row["accepted_len"] for row in rows if row["cache_hit"]]
    miss_accepted = [row["accepted_len"] for row in rows if not row["cache_hit"]]
    total_committed = sum(row["tokens_committed_this_cycle"] for row in rows)

    per_group = defaultdict(list)
    for row in rows:
        per_group[row["group_key"]].append(row)

    prompt_group_metrics = {}
    for group_key, group_rows in per_group.items():
        group_tokens = sum(row["tokens_committed_this_cycle"] for row in group_rows)
        total_s = sum(row["total_cycle_ms"] for row in group_rows) / 1000.0
        prompt_group_metrics[group_key] = {
            "throughput_tok_s": group_tokens / max(total_s, 1e-6),
            "accepted_suffix_mean": mean([row["accepted_len"] for row in group_rows]),
            "cache_hit_mean": mean([1.0 if row["cache_hit"] else 0.0 for row in group_rows]),
            "cache_committed_token_fraction": (
                sum(row["tokens_committed_this_cycle"] for row in group_rows if row["cache_hit"])
                / max(group_tokens, 1)
            ),
            "mean_cycle_latency_ms": mean([row["total_cycle_ms"] for row in group_rows]),
            "draft_service_ms": mean([row["draft_service_ms"] for row in group_rows]),
            "post_verify_feedback_ms": mean([row["post_verify_feedback_ms"] for row in group_rows]),
            "target_verify_ms": mean([row["target_verify_ms"] for row in group_rows]),
        }

    group_summary = {}
    for metric_name in [
        "throughput_tok_s",
        "accepted_suffix_mean",
        "cache_hit_mean",
        "cache_committed_token_fraction",
        "mean_cycle_latency_ms",
        "draft_service_ms",
        "post_verify_feedback_ms",
        "target_verify_ms",
    ]:
        values = [metrics[metric_name] for metrics in prompt_group_metrics.values()]
        group_summary[metric_name] = {
            "mean": _safe_mean(values),
            "stderr": _stderr(values),
        }

    return {
        "cache_hit_mean": mean(cache_hits),
        "miss_rate": 1.0 - mean(cache_hits),
        "cache_committed_token_fraction": (
            sum(row["tokens_committed_this_cycle"] for row in rows if row["cache_hit"])
            / max(total_committed, 1)
        ),
        "mean_cycle_latency_ms": mean(cycle_ms),
        "hit_cycle_latency_ms": _safe_mean(hit_cycle_ms),
        "miss_cycle_latency_ms": _safe_mean(miss_cycle_ms),
        "draft_service_ms": mean(draft_service_ms),
        "post_verify_feedback_ms": mean(feedback_ms),
        "target_verify_ms": mean(verify_ms),
        "hit_only_accepted_suffix_mean": _safe_mean(hit_accepted),
        "miss_only_accepted_suffix_mean": _safe_mean(miss_accepted),
        "prompt_group_metrics": prompt_group_metrics,
        "prompt_group_summary": group_summary,
    }


def _build_llm_kwargs(mode: str, args, batch_size: int, k: int) -> dict:
    spec = MODE_SPECS[mode]
    fan_out_list = _fanout_template_for_k(args.fanout_template_name, k)
    llm_kwargs = {
        "num_gpus": args.gpus,
        "speculate": True,
        "draft": args.draft,
        "draft_backend": "ar",
        "draft_async": True,
        "speculate_k": k,
        "max_num_seqs": batch_size,
        "verbose": False,
        "jit_speculate": True,
        "ar_branch_cache": spec["ar_branch_cache"],
        "ar_branch_key_mode": spec["ar_branch_key_mode"],
    }
    if spec["ar_branch_cache"] == "on":
        llm_kwargs["fan_out_list"] = fan_out_list
        llm_kwargs["fan_out_list_miss"] = fan_out_list
        llm_kwargs["async_fan_out"] = max(fan_out_list) if fan_out_list else 0
    else:
        llm_kwargs["fan_out_list"] = [1] * (k + 1)
        llm_kwargs["fan_out_list_miss"] = [1] * (k + 1)
        llm_kwargs["async_fan_out"] = 1
    return llm_kwargs


def _run_mode(
    mode: str,
    args,
    prompt_records: list[dict],
    batch_size: int,
    k: int,
    artifact_dir: Path | None = None,
) -> dict:
    from ssd.engine.llm_engine import LLMEngine
    from ssd.sampling_params import SamplingParams

    prompt_token_ids = [row["prompt_token_ids"] for row in prompt_records]
    spec = MODE_SPECS[mode]
    engine = LLMEngine(args.target, **_build_llm_kwargs(mode, args, batch_size, k))
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
        cache_hits = metrics.get("cache_hits", [])
        accepted = metrics.get("accepted_suffix_lens_with_recovery", [])
        cycle_rows = _enrich_cycle_rows(metrics.get("ar_cycle_diagnostics", []), prompt_records)
        cycle_rows = [
            {
                **row,
                "fanout_template_name": args.fanout_template_name,
                "mode": mode,
                "k": k,
            }
            for row in cycle_rows
        ]
        cycle_summary = _summarize_cycle_rows(cycle_rows)
        result = {
            "mode": mode,
            "batch_size": batch_size,
            "k": k,
            "num_prompts": len(prompt_records),
            "fanout_template_name": args.fanout_template_name,
            "throughput_tok_s": total_completion_tokens / max(total_time, 1e-6),
            "accepted_suffix_mean": _safe_mean(accepted),
            "fraction_accepted": (
                ((sum(accepted) - len(accepted)) / (len(accepted) * engine.config.speculate_k))
                if accepted else None
            ),
            "target_verify_ms": (
                cycle_summary["target_verify_ms"]
                if cycle_rows else (
                    sum(metrics["target_verify_times"]) * 1000 / len(metrics["target_verify_times"])
                    if metrics.get("target_verify_times") else None
                )
            ),
            "draft_service_ms": cycle_summary["draft_service_ms"],
            "post_verify_feedback_ms": cycle_summary["post_verify_feedback_ms"],
            "mean_cycle_latency_ms": cycle_summary["mean_cycle_latency_ms"],
            "cache_hit_mean": cycle_summary["cache_hit_mean"] if cycle_rows else _safe_mean(cache_hits),
            "miss_rate": cycle_summary["miss_rate"] if cycle_rows else ((1.0 - _safe_mean(cache_hits)) if cache_hits else None),
            "cache_committed_token_fraction": cycle_summary["cache_committed_token_fraction"],
            "accepted_suffix_mean_on_hit": cycle_summary["hit_only_accepted_suffix_mean"],
            "accepted_suffix_mean_on_miss": cycle_summary["miss_only_accepted_suffix_mean"],
            "prompt_group_summary": cycle_summary["prompt_group_summary"],
            "num_branches_generated_mean": (
                0 if spec["ar_branch_cache"] == "off"
                else (1 if spec["ar_branch_key_mode"] == "oracle" else sum(_fanout_template_for_k(args.fanout_template_name, k)))
            ),
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


def _run_single_mode(args) -> dict:
    if args.mode is None or args.batch_size is None:
        raise ValueError("--mode and --batch-size are required for single-mode execution")
    prompt_records = _select_prompt_records(args)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    return _run_mode(args.mode, args, prompt_records, args.batch_size, args.k, artifact_dir=artifact_dir)


def _run_mode_subprocess(
    args,
    mode: str,
    batch_size: int,
    k: int,
    artifact_dir: Path | None,
    port_offset: int,
) -> dict:
    cmd = [
        sys.executable,
        "-O",
        str(Path(__file__).resolve()),
        "--target", args.target,
        "--draft", args.draft,
        "--training-metadata", args.training_metadata,
        "--output-len", str(args.output_len),
        "--gpus", str(args.gpus),
        "--max-prompts", str(args.max_prompts),
        "--fanout-template-name", args.fanout_template_name,
        "--mode", mode,
        "--batch-size", str(batch_size),
        "--k", str(k),
    ]
    if artifact_dir is not None:
        cmd.extend(["--artifact-dir", str(artifact_dir)])
    else:
        raise ValueError("artifact_dir is required for subprocess orchestration")

    env = os.environ.copy()
    base_port = int(env.get("SSD_DIST_PORT", "12355"))
    env["SSD_DIST_PORT"] = str(base_port + port_offset)
    completed = subprocess.run(cmd, text=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            f"AR subprocess failed for mode={mode} batch_size={batch_size} k={k}"
        )
    result_path = artifact_dir / "result.json"
    if not result_path.exists():
        raise RuntimeError(
            f"Missing result.json for mode={mode} batch_size={batch_size} k={k}"
        )
    time.sleep(1.0)
    return _load_json(result_path)


def _run_full_matrix(args) -> dict:
    prompt_records = _select_prompt_records(args)
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item]
    k_values = [int(item) for item in args.k_values.split(",") if item]
    modes = [mode for mode in args.modes.split(",") if mode]
    unknown_modes = [mode for mode in modes if mode not in MODE_SPECS]
    if unknown_modes:
        raise ValueError(f"Unknown AR modes: {unknown_modes}")

    results = []
    out_path = Path(args.out) if args.out else None
    artifact_root = out_path.parent / "summary_artifacts" if out_path else None
    run_idx = 0

    for mode in modes:
        for k in k_values:
            for batch_size in batch_sizes:
                artifact_dir = (
                    artifact_root / mode / f"k{k}_b{batch_size}"
                    if artifact_root else None
                )
                if artifact_dir is not None:
                    result_path = artifact_dir / "result.json"
                    if result_path.exists():
                        result = _load_json(result_path)
                        results.append(result)
                        print(
                            "SKIP_RESULT_JSON "
                            + json.dumps(
                                {
                                    "mode": mode,
                                    "batch_size": batch_size,
                                    "k": k,
                                    "throughput_tok_s": result.get("throughput_tok_s"),
                                    "accepted_suffix_mean": result.get("accepted_suffix_mean"),
                                    "cache_hit_mean": result.get("cache_hit_mean"),
                                    "num_branches_generated_mean": result.get("num_branches_generated_mean"),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        run_idx += 1
                        continue
                result = _run_mode_subprocess(
                    args,
                    mode,
                    batch_size,
                    k,
                    artifact_dir=artifact_dir,
                    port_offset=run_idx,
                )
                results.append(result)
                print(
                    "RESULT_JSON "
                    + json.dumps(
                        {
                            "mode": mode,
                            "batch_size": batch_size,
                            "k": k,
                            "throughput_tok_s": result["throughput_tok_s"],
                            "accepted_suffix_mean": result["accepted_suffix_mean"],
                            "cache_hit_mean": result["cache_hit_mean"],
                            "num_branches_generated_mean": result["num_branches_generated_mean"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                run_idx += 1

    summary = {
        "prompt_count": len(prompt_records),
        "batch_sizes": batch_sizes,
        "k_values": k_values,
        "modes": modes,
        "results": results,
    }
    if out_path is not None:
        _write_json(out_path, summary)
    return summary


def main():
    args = parse_args()
    if args.mode is not None:
        result = _run_single_mode(args)
        print(f"RESULT_JSON={json.dumps(result, sort_keys=True)}")
        return

    summary = _run_full_matrix(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
