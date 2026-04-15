import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test for the async approximate dflash_ssd backend")
    parser.add_argument("--target", type=str, required=True, help="Qwen3-8B target snapshot directory")
    parser.add_argument("--draft", type=str, required=True, help="Qwen3-8B-DFlash-b16 draft directory")
    parser.add_argument("--predictor", type=str, required=True, help="Predictor checkpoint directory")
    parser.add_argument("--prompt", type=str, default="Explain speculative decoding in one sentence.")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--f", type=int, default=3, help="Async fan out")
    parser.add_argument("--max-steps", type=int, default=8)
    return parser.parse_args()


def ensure_dir(path: str, label: str) -> None:
    if not Path(path).is_dir():
        raise FileNotFoundError(f"{label} path does not exist or is not a directory: {path}")


def main():
    args = parse_args()
    ensure_dir(args.target, "Target")
    ensure_dir(args.draft, "Draft")
    ensure_dir(args.predictor, "Predictor")

    from ssd.engine.llm_engine import LLMEngine, METRICS
    from ssd.sampling_params import SamplingParams

    engine = LLMEngine(
        args.target,
        num_gpus=args.gpus,
        speculate=True,
        speculate_k=15,
        draft=args.draft,
        draft_backend="dflash_ssd",
        draft_async=True,
        dflash_predictor=args.predictor,
        async_fan_out=args.f,
        max_num_seqs=1,
        max_steps=args.max_steps,
        verbose=False,
    )
    try:
        sampling = SamplingParams(
            temperature=0.0,
            draft_temperature=0.0,
            max_new_tokens=6,
            ignore_eos=True,
        )
        outputs, metrics = engine.generate([args.prompt], [sampling], use_tqdm=False)
        if not outputs:
            raise RuntimeError("dflash_ssd smoke test produced no outputs")
        if not metrics["cache_hits"]:
            raise RuntimeError("dflash_ssd smoke test did not record cache hit metrics")
        print("dflash_ssd smoke test passed")
        print(f"target={args.target}")
        print(f"draft={args.draft}")
        print(f"predictor={args.predictor}")
        print(f"k={engine.config.speculate_k}")
        print(f"generated={outputs[0]['text']!r}")
        print(f"cache_hit_mean={sum(metrics['cache_hits']) / len(metrics['cache_hits']):.3f}")
        if METRICS["dflash_draft_step_times"]:
            avg_ms = sum(METRICS["dflash_draft_step_times"]) * 1000 / len(METRICS["dflash_draft_step_times"])
            print(f"avg_dflash_step_ms={avg_ms:.2f}")
        if METRICS["dflash_predictor_times"]:
            avg_pred_ms = sum(METRICS["dflash_predictor_times"]) * 1000 / len(METRICS["dflash_predictor_times"])
            print(f"avg_predictor_ms={avg_pred_ms:.2f}")
    finally:
        engine.exit(hard=False)


if __name__ == "__main__":
    main()
