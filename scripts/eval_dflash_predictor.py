import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DFlash predictor checkpoints online against exact dflash on the held-out test split",
    )
    parser.add_argument("--target", type=str, required=True, help="Qwen3-8B snapshot directory")
    parser.add_argument("--draft", type=str, required=True, help="Qwen3-8B-DFlash-b16 draft directory")
    parser.add_argument("--predictor", type=str, required=True, help="Candidate predictor checkpoint directory")
    parser.add_argument("--baseline-predictor", type=str, default=None, help="Optional baseline predictor checkpoint")
    parser.add_argument("--training-metadata", type=str, required=True, help="training_metadata.json from the predictor run")
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--batch-sizes", type=str, default="1,2,4")
    parser.add_argument("--max-prompts", type=int, default=16, help="Maximum held-out prompts to evaluate")
    parser.add_argument("--out", type=str, default=None, help="Optional output JSON path")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["dflash", "dflash_ssd_candidate", "dflash_ssd_baseline"],
        help="Internal single-run mode. When unset, the script orchestrates all runs in subprocesses.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Internal single-run batch size")
    parser.add_argument("--base-dist-port", type=int, default=12470, help="Base SSD_DIST_PORT for subprocess orchestration")
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


def _select_prompts(args) -> list[list[int]]:
    metadata = _load_json(Path(args.training_metadata))
    prompt_manifest = _load_prompt_manifest(Path(metadata["prompt_manifest"]))
    test_groups = set(metadata["split_group_keys"]["test"])
    selected = [row for row in prompt_manifest if row["group_key"] in test_groups]
    selected.sort(key=lambda row: (row["dataset_name"], row["prompt_index"]))
    if args.max_prompts > 0:
        selected = selected[:args.max_prompts]
    prompt_token_ids = [row["prompt_token_ids"] for row in selected]
    if not prompt_token_ids:
        raise RuntimeError("No held-out test prompts selected for evaluation")
    return prompt_token_ids


def _run_mode(mode: str, args, prompt_token_ids: list[list[int]], batch_size: int):
    from ssd.engine.llm_engine import LLMEngine
    from ssd.sampling_params import SamplingParams

    llm_kwargs = {
        "num_gpus": args.gpus,
        "speculate": True,
        "draft": args.draft,
        "draft_backend": "dflash" if mode == "dflash" else "dflash_ssd",
        "draft_async": mode != "dflash",
        "max_num_seqs": batch_size,
        "verbose": False,
    }
    if mode == "dflash_ssd_candidate":
        llm_kwargs["dflash_predictor"] = args.predictor
    elif mode == "dflash_ssd_baseline":
        llm_kwargs["dflash_predictor"] = args.baseline_predictor

    engine = LLMEngine(args.target, **llm_kwargs)
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
        return {
            "mode": mode,
            "batch_size": batch_size,
            "num_prompts": len(prompt_token_ids),
            "throughput": total_completion_tokens / max(total_time, 1e-6),
            "accepted_suffix_mean": (
                sum(metrics["accepted_suffix_lens_with_recovery"]) / len(metrics["accepted_suffix_lens_with_recovery"])
                if metrics["accepted_suffix_lens_with_recovery"] else None
            ),
            "fraction_accepted": (
                (
                    sum(metrics["accepted_suffix_lens_with_recovery"]) - len(metrics["accepted_suffix_lens_with_recovery"])
                ) / (len(metrics["accepted_suffix_lens_with_recovery"]) * engine.config.speculate_k)
                if metrics["accepted_suffix_lens_with_recovery"] else None
            ),
            "target_verify_ms": (
                sum(metrics["target_verify_times"]) * 1000 / len(metrics["target_verify_times"])
                if metrics["target_verify_times"] else None
            ),
            "dflash_step_ms": (
                sum(metrics["dflash_draft_step_times"]) * 1000 / len(metrics["dflash_draft_step_times"])
                if metrics["dflash_draft_step_times"] else None
            ),
            "predictor_ms": (
                sum(metrics["dflash_predictor_times"]) * 1000 / len(metrics["dflash_predictor_times"])
                if metrics["dflash_predictor_times"] else None
            ),
            "cache_hit_mean": (
                sum(metrics["cache_hits"]) / len(metrics["cache_hits"])
                if metrics["cache_hits"] else None
            ),
            "accepted_suffix_mean_on_hit": (
                sum(metrics["accepted_suffix_lens_on_hit"]) / len(metrics["accepted_suffix_lens_on_hit"])
                if metrics["accepted_suffix_lens_on_hit"] else None
            ),
            "accepted_suffix_mean_on_miss": (
                sum(metrics["accepted_suffix_lens_on_miss"]) / len(metrics["accepted_suffix_lens_on_miss"])
                if metrics["accepted_suffix_lens_on_miss"] else None
            ),
        }
    finally:
        engine.exit(hard=False)


def _run_single(args):
    prompt_token_ids = _select_prompts(args)
    if args.batch_size is None:
        raise ValueError("--batch-size is required with --mode")
    result = _run_mode(args.mode, args, prompt_token_ids, args.batch_size)
    print(f"RESULT_JSON={json.dumps(result, sort_keys=True)}", flush=True)


def _run_subprocess(mode: str, batch_size: int, args, port: int) -> dict:
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
        "--mode", mode,
        "--batch-size", str(batch_size),
    ]
    if args.baseline_predictor:
        cmd.extend(["--baseline-predictor", args.baseline_predictor])
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


def _run_all(args):
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    modes = ["dflash"]
    if args.baseline_predictor:
        modes.append("dflash_ssd_baseline")
    modes.append("dflash_ssd_candidate")

    results = []
    run_idx = 0
    for batch_size in batch_sizes:
        for mode in modes:
            results.append(_run_subprocess(mode, batch_size, args, args.base_dist_port + run_idx))
            run_idx += 1

    print("| mode | b | throughput tok/s | accepted suffix | cache hit | dflash ms | predictor ms |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for row in results:
        print(
            f"| {row['mode']} | {row['batch_size']} | {row['throughput']:.2f} | "
            f"{row['accepted_suffix_mean'] if row['accepted_suffix_mean'] is not None else 'n/a'} | "
            f"{row['cache_hit_mean'] if row['cache_hit_mean'] is not None else 'n/a'} | "
            f"{row['dflash_step_ms'] if row['dflash_step_ms'] is not None else 'n/a'} | "
            f"{row['predictor_ms'] if row['predictor_ms'] is not None else 'n/a'} |"
        )

    if args.out:
        with Path(args.out).open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "training_metadata": args.training_metadata,
                    "predictor": args.predictor,
                    "baseline_predictor": args.baseline_predictor,
                    "results": results,
                },
                f,
                indent=2,
                sort_keys=True,
            )


def main():
    args = parse_args()
    if args.mode:
        _run_single(args)
    else:
        _run_all(args)


if __name__ == "__main__":
    main()
