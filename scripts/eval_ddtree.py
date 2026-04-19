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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


MODE_SPECS = {
    "ddtree": {
        "draft_backend": "ddtree",
        "draft_async": False,
    },
    "ddtree_ssd_exact_off": {
        "draft_backend": "ddtree_ssd",
        "draft_async": True,
        "ddtree_context_mode": "exact",
        "ddtree_cache": "off",
        "ddtree_frontier_mode": "oracle",
    },
    "ddtree_ssd_exact_on_oracle": {
        "draft_backend": "ddtree_ssd",
        "draft_async": True,
        "ddtree_context_mode": "exact",
        "ddtree_cache": "on",
        "ddtree_frontier_mode": "oracle",
    },
    "ddtree_ssd_predicted_off_oracle": {
        "draft_backend": "ddtree_ssd",
        "draft_async": True,
        "ddtree_context_mode": "predicted",
        "ddtree_cache": "off",
        "ddtree_frontier_mode": "oracle",
    },
    "ddtree_ssd_predicted_on_oracle": {
        "draft_backend": "ddtree_ssd",
        "draft_async": True,
        "ddtree_context_mode": "predicted",
        "ddtree_cache": "on",
        "ddtree_frontier_mode": "oracle",
    },
    "ddtree_ssd_predicted_on_surrogate": {
        "draft_backend": "ddtree_ssd",
        "draft_async": True,
        "ddtree_context_mode": "predicted",
        "ddtree_cache": "on",
        "ddtree_frontier_mode": "surrogate",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the DDTree headroom matrix on the held-out split")
    parser.add_argument("--target", type=str, required=True, help="Qwen3-8B snapshot directory")
    parser.add_argument("--draft", type=str, required=True, help="Qwen3-8B-DFlash-b16 draft directory")
    parser.add_argument("--predictor", type=str, default=None, help="Predictor checkpoint directory for predicted-context DDTree modes")
    parser.add_argument("--training-metadata", type=str, required=True, help="training_metadata.json from the predictor run")
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--batch-sizes", type=str, default="1,2,4")
    parser.add_argument("--tree-budgets", type=str, default="8,16")
    parser.add_argument("--frontier-counts", type=str, default="1,2")
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(MODE_SPECS.keys()),
        help="Comma-separated subset of modes to run",
    )
    parser.add_argument("--max-prompts", type=int, default=78)
    parser.add_argument("--out", type=str, default=None, help="Optional summary JSON path")
    parser.add_argument("--base-dist-port", type=int, default=12650, help="Base SSD_DIST_PORT for subprocess orchestration")
    parser.add_argument("--mode", type=str, default=None, choices=list(MODE_SPECS.keys()), help="Internal single-run mode")
    parser.add_argument("--batch-size", type=int, default=None, help="Internal single-run batch size")
    parser.add_argument("--tree-budget", type=int, default=None, help="Internal single-run DDTree node budget")
    parser.add_argument("--frontier-count", type=int, default=None, help="Internal single-run DDTree frontier count")
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
        raise RuntimeError("No held-out test prompts selected for DDTree evaluation")
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
    seq_id_to_prompt = {seq_id: prompt_records[idx] for idx, seq_id in enumerate(seq_ids)}
    enriched = []
    for row in rows:
        prompt = seq_id_to_prompt[int(row["seq_id"])]
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
            "cache_committed_token_fraction": None,
            "mean_cycle_latency_ms": None,
            "hit_cycle_latency_ms": None,
            "miss_cycle_latency_ms": None,
            "hit_only_accepted_suffix_mean": None,
            "miss_only_accepted_suffix_mean": None,
            "frontier_candidate_count_mean": None,
            "actual_frontier_rank_distribution": {},
            "verified_node_count_mean": None,
            "tree_node_count_mean": None,
            "prompt_group_summary": {},
        }

    cache_hits = [1.0 if row["cache_hit"] else 0.0 for row in rows]
    total_committed = sum(row["tokens_committed_this_cycle"] for row in rows)
    per_group = defaultdict(list)
    for row in rows:
        per_group[row["group_key"]].append(row)

    prompt_group_metrics = {}
    for group_key, group_rows in per_group.items():
        total_s = sum(row["total_cycle_ms"] for row in group_rows) / 1000.0
        tokens = sum(row["tokens_committed_this_cycle"] for row in group_rows)
        prompt_group_metrics[group_key] = {
            "throughput_tok_s": tokens / max(total_s, 1e-6),
            "accepted_suffix_mean": mean([row["accepted_len"] for row in group_rows]),
            "cache_hit_mean": mean([1.0 if row["cache_hit"] else 0.0 for row in group_rows]),
            "cache_committed_token_fraction": (
                sum(row["committed_tokens_from_cache"] for row in group_rows)
                / max(sum(row["tokens_committed_this_cycle"] for row in group_rows), 1)
            ),
        }

    prompt_group_summary = {}
    for metric_name in ["throughput_tok_s", "accepted_suffix_mean", "cache_hit_mean", "cache_committed_token_fraction"]:
        values = [metrics[metric_name] for metrics in prompt_group_metrics.values()]
        prompt_group_summary[metric_name] = {
            "mean": _safe_mean(values),
            "stderr": _stderr(values),
        }

    return {
        "cache_hit_mean": mean(cache_hits),
        "cache_committed_token_fraction": (
            sum(row["committed_tokens_from_cache"] for row in rows) / max(total_committed, 1)
        ),
        "mean_cycle_latency_ms": mean([row["total_cycle_ms"] for row in rows]),
        "hit_cycle_latency_ms": _safe_mean([row["total_cycle_ms"] for row in rows if row["cache_hit"]]),
        "miss_cycle_latency_ms": _safe_mean([row["total_cycle_ms"] for row in rows if not row["cache_hit"]]),
        "hit_only_accepted_suffix_mean": _safe_mean([row["accepted_len"] for row in rows if row["cache_hit"]]),
        "miss_only_accepted_suffix_mean": _safe_mean([row["accepted_len"] for row in rows if not row["cache_hit"]]),
        "frontier_candidate_count_mean": _safe_mean([row["frontier_candidate_count"] for row in rows if row["frontier_candidate_count"] is not None]),
        "actual_frontier_rank_distribution": dict(
            sorted(
                Counter(
                    row["actual_frontier_rank"]
                    for row in rows
                    if row["actual_frontier_rank"] is not None
                ).items()
            )
        ),
        "verified_node_count_mean": _safe_mean([row["verified_node_count"] for row in rows if row["verified_node_count"] is not None]),
        "tree_node_count_mean": _safe_mean([row["tree_node_count"] for row in rows if row["tree_node_count"] is not None]),
        "prompt_group_summary": prompt_group_summary,
        "prompt_group_metrics": prompt_group_metrics,
    }


def _build_llm_kwargs(mode: str, args, batch_size: int, tree_budget: int, frontier_count: int) -> dict:
    spec = MODE_SPECS[mode]
    kwargs = {
        "num_gpus": args.gpus,
        "speculate": True,
        "draft": args.draft,
        "draft_backend": spec["draft_backend"],
        "draft_async": spec["draft_async"],
        "max_num_seqs": batch_size,
        "ddtree_tree_budget": tree_budget,
        "ddtree_frontier_count": frontier_count,
        "verbose": False,
    }
    if spec["draft_backend"] == "ddtree_ssd":
        kwargs.update(
            ddtree_context_mode=spec["ddtree_context_mode"],
            ddtree_cache=spec["ddtree_cache"],
            ddtree_frontier_mode=spec["ddtree_frontier_mode"],
            ddtree_enable_diagnostics=True,
        )
        if spec["ddtree_context_mode"] == "predicted":
            if args.predictor is None:
                raise RuntimeError(f"{mode} requires --predictor")
            kwargs["dflash_predictor"] = args.predictor
    return kwargs


def _run_mode(mode: str, args, prompt_records: list[dict], batch_size: int, tree_budget: int, frontier_count: int, artifact_dir: Path | None) -> dict:
    from ssd.engine.llm_engine import LLMEngine
    from ssd.sampling_params import SamplingParams

    prompt_token_ids = [row["prompt_token_ids"] for row in prompt_records]
    engine = LLMEngine(args.target, **_build_llm_kwargs(mode, args, batch_size, tree_budget, frontier_count))
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

        cycle_rows = _enrich_cycle_rows(metrics.get("ddtree_cycle_diagnostics", []), prompt_records)
        cycle_summary = _summarize_cycle_rows(cycle_rows)
        result = {
            "mode": mode,
            "batch_size": batch_size,
            "tree_budget": tree_budget,
            "frontier_count": frontier_count,
            "num_prompts": len(prompt_records),
            "throughput_tok_s": total_completion_tokens / max(total_time, 1e-6),
            "accepted_suffix_mean": _safe_mean(metrics.get("accepted_suffix_lens_with_recovery", [])),
            "target_verify_ms": _safe_mean([value * 1000.0 for value in metrics.get("target_verify_times", [])]),
            "ddtree_step_ms": _safe_mean([value * 1000.0 for value in metrics.get("ddtree_draft_step_times", [])]),
            "tree_build_ms": _safe_mean([value * 1000.0 for value in metrics.get("ddtree_tree_build_times", [])]),
            "tree_compile_ms": _safe_mean([value * 1000.0 for value in metrics.get("ddtree_tree_compile_times", [])]),
            "verified_node_count_mean": _safe_mean(metrics.get("ddtree_verified_node_counts", [])),
            "tree_node_count_mean": _safe_mean(metrics.get("ddtree_tree_node_counts", [])),
            "cache_hit_mean": cycle_summary["cache_hit_mean"] if cycle_rows else _safe_mean(metrics.get("cache_hits", [])),
            "cache_committed_token_fraction": cycle_summary["cache_committed_token_fraction"],
            "hit_only_accepted_suffix_mean": cycle_summary["hit_only_accepted_suffix_mean"],
            "miss_only_accepted_suffix_mean": cycle_summary["miss_only_accepted_suffix_mean"],
            "mean_cycle_latency_ms": cycle_summary["mean_cycle_latency_ms"],
            "hit_cycle_latency_ms": cycle_summary["hit_cycle_latency_ms"],
            "miss_cycle_latency_ms": cycle_summary["miss_cycle_latency_ms"],
            "frontier_candidate_count_mean": cycle_summary["frontier_candidate_count_mean"],
            "actual_frontier_rank_distribution": cycle_summary["actual_frontier_rank_distribution"],
            "prompt_group_summary": cycle_summary["prompt_group_summary"],
        }
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            _write_json(artifact_dir / "result.json", result)
            if cycle_rows:
                _write_jsonl(artifact_dir / "cycles.jsonl", cycle_rows)
                _write_json(artifact_dir / "prompt_group_metrics.json", cycle_summary["prompt_group_metrics"])
        return result
    finally:
        engine.exit(hard=False)


def _format_markdown_table(rows: list[dict]) -> str:
    lines = [
        "| mode | b | N | F | tok/s | acc suffix | cache hit | cache tok frac | hit acc | miss acc | verify ms | ddtree ms | tree build ms | tree compile ms | frontier count | frontier ranks |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['mode']} | {row['batch_size']} | {row['tree_budget']} | {row['frontier_count']} | "
            f"{row['throughput_tok_s']:.3f} | "
            f"{row['accepted_suffix_mean'] if row['accepted_suffix_mean'] is not None else 'n/a'} | "
            f"{row['cache_hit_mean'] if row['cache_hit_mean'] is not None else 'n/a'} | "
            f"{row['cache_committed_token_fraction'] if row['cache_committed_token_fraction'] is not None else 'n/a'} | "
            f"{row['hit_only_accepted_suffix_mean'] if row['hit_only_accepted_suffix_mean'] is not None else 'n/a'} | "
            f"{row['miss_only_accepted_suffix_mean'] if row['miss_only_accepted_suffix_mean'] is not None else 'n/a'} | "
            f"{row['target_verify_ms'] if row['target_verify_ms'] is not None else 'n/a'} | "
            f"{row['ddtree_step_ms'] if row['ddtree_step_ms'] is not None else 'n/a'} | "
            f"{row['tree_build_ms'] if row['tree_build_ms'] is not None else 'n/a'} | "
            f"{row['tree_compile_ms'] if row['tree_compile_ms'] is not None else 'n/a'} | "
            f"{row['frontier_candidate_count_mean'] if row['frontier_candidate_count_mean'] is not None else 'n/a'} | "
            f"{row['actual_frontier_rank_distribution']} |"
        )
    return "\n".join(lines)


def _build_recommendation(results: list[dict]) -> dict:
    by_setting = defaultdict(dict)
    for row in results:
        key = (row["batch_size"], row["tree_budget"], row["frontier_count"])
        by_setting[key][row["mode"]] = row

    oracle_beats = []
    oracle_deltas = []
    for key, rows in by_setting.items():
        if "ddtree" not in rows or "ddtree_ssd_exact_on_oracle" not in rows:
            continue
        base = rows["ddtree"]["throughput_tok_s"]
        oracle = rows["ddtree_ssd_exact_on_oracle"]["throughput_tok_s"]
        delta = (oracle - base) / max(base, 1e-6)
        oracle_deltas.append({"setting": {"batch_size": key[0], "tree_budget": key[1], "frontier_count": key[2]}, "delta_frac": delta})
        if delta >= 0.05:
            oracle_beats.append((key, delta))

    if not oracle_beats:
        return {
            "decision": "stop",
            "reason": "ddtree_ssd exact+on+oracle did not beat plain ddtree by 5% anywhere in the initial sweep",
            "oracle_deltas": oracle_deltas,
        }

    best_key, best_delta = max(oracle_beats, key=lambda item: item[1])
    return {
        "decision": "continue",
        "reason": "ddtree_ssd exact+on+oracle exceeded the 5% headroom threshold",
        "best_setting": {
            "batch_size": best_key[0],
            "tree_budget": best_key[1],
            "frontier_count": best_key[2],
            "delta_frac": best_delta,
        },
        "oracle_deltas": oracle_deltas,
    }


def _run_single(args):
    prompt_records = _select_prompt_records(args)
    if args.mode is None or args.batch_size is None or args.tree_budget is None or args.frontier_count is None:
        raise ValueError("--mode, --batch-size, --tree-budget, and --frontier-count are required for single-run mode")
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    result = _run_mode(
        mode=args.mode,
        args=args,
        prompt_records=prompt_records,
        batch_size=args.batch_size,
        tree_budget=args.tree_budget,
        frontier_count=args.frontier_count,
        artifact_dir=artifact_dir,
    )
    print(f"RESULT_JSON={json.dumps(result, sort_keys=True)}", flush=True)


def _run_subprocess(mode: str, args, batch_size: int, tree_budget: int, frontier_count: int, port: int, artifact_dir: Path | None = None) -> dict:
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
        "--mode", mode,
        "--batch-size", str(batch_size),
        "--tree-budget", str(tree_budget),
        "--frontier-count", str(frontier_count),
    ]
    if args.predictor is not None:
        cmd.extend(["--predictor", args.predictor])
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
        raise RuntimeError(
            f"Subprocess failed for mode={mode} batch_size={batch_size} tree_budget={tree_budget} frontier_count={frontier_count}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("RESULT_JSON="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError(
        f"Missing RESULT_JSON for mode={mode} batch_size={batch_size} tree_budget={tree_budget} frontier_count={frontier_count}"
    )


def _run_all(args):
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    tree_budgets = [int(x) for x in args.tree_budgets.split(",") if x]
    frontier_counts = [int(x) for x in args.frontier_counts.split(",") if x]
    modes = [mode for mode in args.modes.split(",") if mode]
    unknown_modes = [mode for mode in modes if mode not in MODE_SPECS]
    if unknown_modes:
        raise ValueError(f"Unknown DDTree modes: {unknown_modes}")

    output_path = Path(args.out) if args.out else None
    artifact_root = output_path.with_suffix("") if output_path else Path("artifacts/ddtree_eval")
    if output_path:
        artifact_root = output_path.parent / f"{artifact_root.name}_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    results = []
    run_idx = 0
    for batch_size in batch_sizes:
        for tree_budget in tree_budgets:
            for frontier_count in frontier_counts:
                for mode in modes:
                    artifact_dir = artifact_root / mode / f"b{batch_size}_tb{tree_budget}_fc{frontier_count}"
                    result = _run_subprocess(
                        mode=mode,
                        args=args,
                        batch_size=batch_size,
                        tree_budget=tree_budget,
                        frontier_count=frontier_count,
                        port=args.base_dist_port + run_idx,
                        artifact_dir=artifact_dir,
                    )
                    results.append(result)
                    run_idx += 1

    recommendation = _build_recommendation(results)
    markdown = _format_markdown_table(results)
    summary = {
        "results": results,
        "recommendation": recommendation,
        "markdown_table": markdown,
    }

    output_path = output_path if output_path else artifact_root / "summary.json"
    _write_json(output_path, summary)
    with (artifact_root / "summary.md").open("w", encoding="utf-8") as f:
        f.write(markdown + "\n\n")
        f.write("```json\n")
        f.write(json.dumps(recommendation, indent=2, sort_keys=True))
        f.write("\n```\n")
    print(markdown)
    print(json.dumps(recommendation, indent=2, sort_keys=True))


def main():
    args = parse_args()
    if args.mode is not None:
        _run_single(args)
    else:
        _run_all(args)


if __name__ == "__main__":
    main()
