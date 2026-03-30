import argparse
from pathlib import Path

def parse_args():
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    parser = argparse.ArgumentParser(
        description="Smoke test for the sync LLaDA diffusion draft backend",
    )
    parser.add_argument("--target", type=str, default=None,
                        help="Target model snapshot directory")
    parser.add_argument("--draft", type=str, default=None,
                        help="LLaDA draft model directory")
    parser.add_argument("--prompt", type=str, default="Explain speculative decoding in one sentence.",
                        help="Prompt text to seed the diffusion draft")
    parser.add_argument("--b", type=int, default=1,
                        help="Batch size to exercise")
    parser.add_argument("--k", type=int, default=4,
                        help="Speculative lookahead length")
    parser.add_argument("--dsteps", type=int, default=32,
                        help="Number of diffusion denoising steps")
    parser.add_argument("--device", type=str, default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
                        help="Torch device to use")
    return parser.parse_args()


def ensure_model_dir(path: str, label: str):
    if not Path(path).is_dir():
        raise FileNotFoundError(f"{label} path does not exist or is not a directory: {path}")


def pick_recovery_token(tokenizer: AutoTokenizer) -> int:
    candidates = [
        " the",
        "Speculative",
        ".",
    ]
    for text in candidates:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if token_ids:
            return token_ids[0]
    raise RuntimeError("Unable to find a usable recovery token")


def main():
    args = parse_args()
    import torch
    from transformers import AutoTokenizer

    from ssd.config import Config
    from ssd.engine.diffusion_draft_adapter import LLaDADiffusionAdapter
    from ssd.engine.sequence import Sequence
    from ssd.engine.speculator_sync_diffusion import SpeculatorSyncDiffusion
    from ssd.paths import DEFAULT_LLADA_DRAFT, DEFAULT_TARGET
    from ssd.sampling_params import SamplingParams

    if args.target is None:
        args.target = DEFAULT_TARGET
    if args.draft is None:
        args.draft = DEFAULT_LLADA_DRAFT

    ensure_model_dir(args.target, "Target")
    ensure_model_dir(args.draft, "Draft")

    device = torch.device(args.device)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target, use_fast=True)
    if target_tokenizer.pad_token_id is None and target_tokenizer.eos_token_id is not None:
        target_tokenizer.pad_token = target_tokenizer.eos_token

    config = Config(
        model=args.target,
        draft=args.draft,
        speculate=True,
        draft_backend="llada_diffusion",
        speculate_k=args.k,
        diffusion_steps=args.dsteps,
        max_num_seqs=args.b,
    )

    metrics = {"diffusion_draft_step_times": []}
    adapter = LLaDADiffusionAdapter(
        config=config,
        target_tokenizer=target_tokenizer,
        device=device,
        metrics=metrics,
    )
    speculator = SpeculatorSyncDiffusion(
        lookahead=args.k,
        device=device,
        diffusion_adapter=adapter,
    )

    prompt_ids = target_tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt_ids:
        raise RuntimeError("Prompt tokenized to an empty sequence")

    recovery_token_id = pick_recovery_token(target_tokenizer)
    seqs = []
    for _ in range(args.b):
        seq = Sequence(
            prompt_ids,
            SamplingParams(temperature=0.0, max_new_tokens=max(args.k, 8), ignore_eos=True),
        )
        seq.recovery_token_id = recovery_token_id
        seq.num_draft_cached_tokens = seq.num_tokens
        seqs.append(seq)

    result = speculator.speculate(seqs, verify_result=None)

    expected_spec_shape = (args.b, args.k + 1)
    expected_logits_shape = (args.b, args.k, target_tokenizer.vocab_size)
    assert tuple(result.speculations.shape) == expected_spec_shape, (
        f"Expected speculations shape {expected_spec_shape}, got {tuple(result.speculations.shape)}"
    )
    assert tuple(result.logits_q.shape) == expected_logits_shape, (
        f"Expected logits_q shape {expected_logits_shape}, got {tuple(result.logits_q.shape)}"
    )
    assert torch.isfinite(result.logits_q.float()).all(), "logits_q contains non-finite values"

    decoded = []
    for row in range(args.b):
        decoded.append(target_tokenizer.decode(result.speculations[row, 1:].tolist(), skip_special_tokens=False))

    print("LLaDA diffusion smoke test passed")
    print(f"target={args.target}")
    print(f"draft={args.draft}")
    print(f"device={device}")
    print(f"batch={args.b} k={args.k} dsteps={args.dsteps}")
    print(f"speculations_shape={tuple(result.speculations.shape)}")
    print(f"logits_q_shape={tuple(result.logits_q.shape)}")
    if metrics["diffusion_draft_step_times"]:
        avg_ms = sum(metrics["diffusion_draft_step_times"]) * 1000 / len(metrics["diffusion_draft_step_times"])
        print(f"avg_diffusion_step_ms={avg_ms:.2f}")
    for i, text in enumerate(decoded[: min(3, len(decoded))]):
        print(f"draft[{i}]={text!r}")


if __name__ == "__main__":
    main()
