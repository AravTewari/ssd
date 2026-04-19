import argparse
import json
import os
import sys
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
    parser.add_argument("--k", type=int, default=15, help="Speculative lookahead for the AR draft")
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _build_llm_kwargs(mode: str, args, batch_size: int) -> dict:
    spec = MODE_SPECS[mode]
    fan_out_list = BRANCH_TEMPLATES[args.fanout_template_name]
    llm_kwargs = {
        "num_gpus": args.gpus,
        "speculate": True,
        "draft": args.draft,
        "draft_backend": "ar",
        "draft_async": True,
        "speculate_k": args.k,
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
        llm_kwargs["fan_out_list"] = [1] * (args.k + 1)
        llm_kwargs["fan_out_list_miss"] = [1] * (args.k + 1)
        llm_kwargs["async_fan_out"] = 1
    return llm_kwargs


def _run_mode(mode: str, args, prompt_records: list[dict], batch_size: int, artifact_dir: Path | None = None) -> dict:
    from ssd.engine.llm_engine import LLMEngine
    from ssd.sampling_params import SamplingParams

    prompt_token_ids = [row["prompt_token_ids"] for row in prompt_records]
    spec = MODE_SPECS[mode]
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
        cache_hits = metrics.get("cache_hits", [])
        accepted = metrics.get("accepted_suffix_lens_with_recovery", [])
        result = {
            "mode": mode,
            "batch_size": batch_size,
            "num_prompts": len(prompt_records),
            "fanout_template_name": args.fanout_template_name,
            "throughput_tok_s": total_completion_tokens / max(total_time, 1e-6),
            "accepted_suffix_mean": _safe_mean(accepted),
            "fraction_accepted": (
                ((sum(accepted) - len(accepted)) / (len(accepted) * engine.config.speculate_k))
                if accepted else None
            ),
            "target_verify_ms": (
                sum(metrics["target_verify_times"]) * 1000 / len(metrics["target_verify_times"])
                if metrics.get("target_verify_times") else None
            ),
            "draft_step_ms": (
                sum(metrics["target_step_times"]) * 1000 / len(metrics["target_step_times"])
                if metrics.get("target_step_times") else None
            ),
            "cache_hit_mean": _safe_mean(cache_hits),
            "miss_rate": (1.0 - _safe_mean(cache_hits)) if cache_hits else None,
            "accepted_suffix_mean_on_hit": _safe_mean(metrics.get("accepted_suffix_lens_on_hit", [])),
            "accepted_suffix_mean_on_miss": _safe_mean(metrics.get("accepted_suffix_lens_on_miss", [])),
            "num_branches_generated_mean": (
                0 if spec["ar_branch_cache"] == "off"
                else (1 if spec["ar_branch_key_mode"] == "oracle" else sum(BRANCH_TEMPLATES[args.fanout_template_name]))
            ),
        }
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            _write_json(artifact_dir / "result.json", result)
            result["artifact_dir"] = str(artifact_dir)
        return result
    finally:
        engine.exit(hard=False)


def _run_single_mode(args) -> dict:
    if args.mode is None or args.batch_size is None:
        raise ValueError("--mode and --batch-size are required for single-mode execution")
    prompt_records = _select_prompt_records(args)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    return _run_mode(args.mode, args, prompt_records, args.batch_size, artifact_dir=artifact_dir)


def _run_full_matrix(args) -> dict:
    prompt_records = _select_prompt_records(args)
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item]
    results = []
    out_path = Path(args.out) if args.out else None
    artifact_root = out_path.parent / "summary_artifacts" if out_path else None

    for mode in MODE_SPECS:
        for batch_size in batch_sizes:
            artifact_dir = artifact_root / f"{mode}_b{batch_size}" if artifact_root else None
            result = _run_mode(mode, args, prompt_records, batch_size, artifact_dir=artifact_dir)
            results.append(result)
            print(
                "RESULT_JSON "
                + json.dumps(
                    {
                        "mode": mode,
                        "batch_size": batch_size,
                        "throughput_tok_s": result["throughput_tok_s"],
                        "accepted_suffix_mean": result["accepted_suffix_mean"],
                        "cache_hit_mean": result["cache_hit_mean"],
                        "num_branches_generated_mean": result["num_branches_generated_mean"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = {
        "prompt_count": len(prompt_records),
        "batch_sizes": batch_sizes,
        "results": results,
    }
    if out_path is not None:
        _write_json(out_path, summary)
    return summary


def main():
    args = parse_args()
    if args.mode is not None:
        result = _run_single_mode(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    summary = _run_full_matrix(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
