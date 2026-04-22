import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke test for the exact sync DFlash draft backend",
    )
    parser.add_argument("--target", type=str, required=True, help="Qwen3 target snapshot directory")
    parser.add_argument("--draft", type=str, required=True, help="DFlash draft model directory")
    parser.add_argument("--prompt", type=str, default="Explain speculative decoding in one sentence.",
                        help="Prompt text to seed the draft path")
    parser.add_argument("--gpus", type=int, default=2, help="Total GPUs to use")
    parser.add_argument("--max-steps", type=int, default=2, help="Maximum generation steps for a full end-to-end smoke")
    return parser.parse_args()


def ensure_model_dir(path: str, label: str):
    if not Path(path).is_dir():
        raise FileNotFoundError(f"{label} path does not exist or is not a directory: {path}")


def main():
    import torch

    from ssd.engine.helpers.speculate_types import VerifyResult
    from ssd.engine.llm_engine import LLMEngine, METRICS
    from ssd.engine.step import SpecDecodeStep
    from ssd.engine.sequence import Sequence
    from ssd.sampling_params import SamplingParams

    args = parse_args()
    ensure_model_dir(args.target, "Target")
    ensure_model_dir(args.draft, "Draft")

    engine = LLMEngine(
        args.target,
        num_gpus=args.gpus,
        speculate=True,
        draft=args.draft,
        draft_backend="dflash",
        max_num_seqs=1,
        max_steps=args.max_steps,
        verbose=False,
    )

    try:
        sampling = SamplingParams(
            temperature=0.0,
            draft_temperature=0.0,
            max_new_tokens=max(engine.config.speculate_k, 8),
            ignore_eos=True,
        )
        prompt_ids = engine.tokenizer.encode(args.prompt, add_special_tokens=False)
        if not prompt_ids:
            raise RuntimeError("Prompt tokenized to an empty sequence")

        seq = Sequence(prompt_ids, sampling)
        engine.scheduler.add(seq)
        seqs, is_prefill = engine.scheduler.schedule()
        assert is_prefill and len(seqs) == 1, "Smoke test expected a single scheduled prefill sequence"

        inference_step = engine.create_inference_step(engine.config)
        if not isinstance(inference_step, SpecDecodeStep):
            raise RuntimeError("DFlash smoke test expected a speculative inference step")

        prefill_result = inference_step.verifier.prefill(seqs, eagle=False)
        assert prefill_result.dflash_target_features is not None, "Missing DFlash prompt target features"
        assert len(prefill_result.dflash_target_features) == 1
        prompt_features = prefill_result.dflash_target_features[0]
        assert tuple(prompt_features.shape) == (
            len(prompt_ids),
            engine.config.dflash_target_feature_dim,
        ), f"Unexpected prompt feature shape: {tuple(prompt_features.shape)}"

        inference_step.speculator.prefill(seqs, prefill_result)
        for s in seqs:
            s.num_cached_tokens = s.num_prompt_tokens
            s.num_draft_cached_tokens = s.num_prompt_tokens

        spec_result = inference_step.speculator.speculate(
            seqs,
            VerifyResult([], []),
        )
        expected_spec_shape = (1, engine.config.speculate_k + 1)
        expected_logits_shape = (1, engine.config.speculate_k, engine.config.hf_config.vocab_size)
        assert tuple(spec_result.speculations.shape) == expected_spec_shape, (
            f"Expected speculations shape {expected_spec_shape}, got {tuple(spec_result.speculations.shape)}"
        )
        assert tuple(spec_result.logits_q.shape) == expected_logits_shape, (
            f"Expected logits_q shape {expected_logits_shape}, got {tuple(spec_result.logits_q.shape)}"
        )
        assert torch.isfinite(spec_result.logits_q.float()).all(), "logits_q contains non-finite values"

        verify_out = inference_step.verifier.verify(seqs, spec_result, eagle=False)
        assert verify_out.dflash_target_features is not None, "Verifier did not return DFlash target features"
        assert len(verify_out.dflash_target_features) == 1
        assert verify_out.dflash_target_features[0].shape[-1] == engine.config.dflash_target_feature_dim

        print("DFlash smoke test passed")
        print(f"target={args.target}")
        print(f"draft={args.draft}")
        print(f"gpus={args.gpus}")
        print(f"k={engine.config.speculate_k} block_size={engine.config.dflash_block_size}")
        print(f"prompt_features_shape={tuple(prompt_features.shape)}")
        print(f"speculations_shape={tuple(spec_result.speculations.shape)}")
        print(f"logits_q_shape={tuple(spec_result.logits_q.shape)}")
        if METRICS["dflash_draft_step_times"]:
            avg_ms = sum(METRICS["dflash_draft_step_times"]) * 1000 / len(METRICS["dflash_draft_step_times"])
            print(f"avg_dflash_step_ms={avg_ms:.2f}")
        print(f"accepted_suffix={verify_out.new_suffixes[0]}")
        print(f"next_recovery_token={verify_out.recovery_tokens[0]}")
    finally:
        engine.exit(hard=False)


if __name__ == "__main__":
    main()
