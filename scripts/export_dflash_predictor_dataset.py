import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export grouped DFlash predictor traces from the exact regular dflash backend",
    )
    parser.add_argument("--target", type=str, required=True, help="Qwen3-8B snapshot directory")
    parser.add_argument("--draft", type=str, required=True, help="Qwen3-8B-DFlash-b16 draft directory")
    parser.add_argument("--out", type=str, required=True, help="Output directory for trace files and manifests")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mixed",
        choices=["mixed", "humaneval", "alpaca", "gsm", "ultrafeedback"],
        help="Prompt source. 'mixed' uses Humaneval+Alpaca+GSM+UltraFeedback.",
    )
    parser.add_argument(
        "--num-prompts-per-dataset",
        type=int,
        default=64,
        help="Number of prompt groups to load per dataset. For --dataset mixed this applies per dataset.",
    )
    parser.add_argument("--input-len", type=int, default=32, help="Minimum prompt token length")
    parser.add_argument("--prompt-offset", type=int, default=0, help="Skip the first N prompts per dataset")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum speculative decode steps per prompt")
    parser.add_argument("--max-traces", type=int, default=0, help="If > 0, stop after exporting this many traces")
    parser.add_argument("--batch-size", type=int, default=4, help="Preferred export batch size")
    parser.add_argument(
        "--fallback-batch-size",
        type=int,
        default=2,
        help="Retry export with this batch size on CUDA OOM. Must be <= batch-size.",
    )
    parser.add_argument("--gpus", type=int, default=2, help="Total GPUs to use")
    return parser.parse_args()


def _group_key(record: dict) -> str:
    return f"{record['dataset_name']}:{record['prompt_index']}"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _save_trace(out_dir: Path, index: int, payload: dict) -> str:
    import torch

    file_name = f"trace_{index:06d}.pt"
    torch.save(payload, out_dir / file_name)
    return file_name


def _load_prompt_records(args) -> list[dict]:
    from bench.bench_helpers import load_all_dataset_prompt_records, load_dataset_prompt_records

    disable_thinking = True
    if args.dataset == "mixed":
        return load_all_dataset_prompt_records(
            model_path=args.target,
            num_prompts_per_dataset=args.num_prompts_per_dataset,
            input_len=args.input_len,
            prompt_offset=args.prompt_offset,
            use_chat_template=False,
            disable_thinking=disable_thinking,
        )
    return load_dataset_prompt_records(
        dataset_name=args.dataset,
        model_path=args.target,
        num_prompts=args.num_prompts_per_dataset,
        input_len=args.input_len,
        prompt_offset=args.prompt_offset,
        use_chat_template=False,
        disable_thinking=disable_thinking,
        strict=True,
    )


def _cleanup_batch_state(engine) -> None:
    if engine.scheduler.running:
        for seq in list(engine.scheduler.running):
            engine.scheduler.block_manager.deallocate(seq)
            engine.scheduler.draft_block_manager.deallocate(seq)
        engine.scheduler.running.clear()
    if engine.scheduler.waiting:
        engine.scheduler.waiting.clear()
    engine.dflash_worker.reset_all()


def export_dataset(args, prompt_records: list[dict], batch_size: int) -> dict:
    from ssd.engine.helpers.speculate_types import VerifyResult
    from ssd.engine.llm_engine import LLMEngine
    from ssd.engine.sequence import Sequence, SequenceStatus
    from ssd.engine.step import SpecDecodeStep
    from ssd.sampling_params import SamplingParams

    out_dir = Path(args.out)
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    prompt_manifest = []
    for rec in prompt_records:
        prompt_manifest.append({
            "dataset_name": rec["dataset_name"],
            "prompt_index": rec["prompt_index"],
            "group_key": _group_key(rec),
            "prompt_token_ids": rec["prompt_token_ids"],
            "text": rec["text"],
        })
    _write_jsonl(out_dir / "prompt_manifest.jsonl", prompt_manifest)

    engine = LLMEngine(
        args.target,
        num_gpus=args.gpus,
        speculate=True,
        draft=args.draft,
        draft_backend="dflash",
        max_num_seqs=batch_size,
        dflash_trace_hidden=True,
        verbose=False,
    )

    trace_index_rows = []
    trace_idx = 0
    skipped_prefill_groups = []
    max_traces = args.max_traces if args.max_traces > 0 else None
    try:
        inference_step = engine.create_inference_step(engine.config)
        if not isinstance(inference_step, SpecDecodeStep):
            raise RuntimeError("Expected a speculative inference step for exact dflash export")

        for batch_start in range(0, len(prompt_records), batch_size):
            if max_traces is not None and trace_idx >= max_traces:
                break
            batch_records = prompt_records[batch_start:batch_start + batch_size]
            seq_to_record = {}
            seq_step_counts = {}
            sampling = SamplingParams(
                temperature=0.0,
                draft_temperature=0.0,
                max_new_tokens=max(engine.config.speculate_k, args.max_steps * engine.config.speculate_k),
                ignore_eos=True,
            )
            for record in batch_records:
                seq = Sequence(record["prompt_token_ids"], sampling)
                seq_to_record[seq.seq_id] = record
                seq_step_counts[seq.seq_id] = 0
                engine.scheduler.add(seq)

            while not engine.is_finished():
                if max_traces is not None and trace_idx >= max_traces:
                    break
                seqs, is_prefill = engine.scheduler.schedule()
                if not seqs:
                    break
                if is_prefill:
                    prefill_result = inference_step.verifier.prefill(seqs, eagle=False)
                    valid_pairs = []
                    for seq, prompt_features in zip(seqs, prefill_result.dflash_target_features or []):
                        if prompt_features.numel() == 0 or prompt_features.shape[0] == 0:
                            record = seq_to_record[seq.seq_id]
                            skipped_prefill_groups.append({
                                "dataset_name": record["dataset_name"],
                                "prompt_index": record["prompt_index"],
                                "group_key": _group_key(record),
                                "reason": "empty_dflash_prefill_features",
                            })
                            seq.status = SequenceStatus.FINISHED
                            engine.scheduler.block_manager.deallocate(seq)
                            engine.scheduler.draft_block_manager.deallocate(seq)
                            if seq in engine.scheduler.running:
                                engine.scheduler.running.remove(seq)
                            continue
                        valid_pairs.append((seq, prompt_features))

                    if not valid_pairs:
                        continue

                    seqs = [seq for seq, _ in valid_pairs]
                    prefill_result = VerifyResult(
                        new_suffixes=[],
                        recovery_tokens=[seq.recovery_token_id for seq in seqs],
                        dflash_target_features=[features for _, features in valid_pairs],
                        dflash_target_features_full=None,
                    )
                    inference_step.speculator.prefill(seqs, prefill_result)
                    for seq in seqs:
                        seq.num_cached_tokens = seq.num_prompt_tokens
                        seq.num_draft_cached_tokens = seq.num_prompt_tokens
                    continue

                saved = [
                    (
                        len(seq.token_ids),
                        seq.num_tokens,
                        seq.last_token,
                        seq.num_draft_cached_tokens,
                        seq.num_cached_tokens,
                    )
                    for seq in seqs
                ]
                spec_result = inference_step.speculator.speculate(seqs, VerifyResult([], []))
                verify_out = inference_step.verifier.verify(seqs, spec_result, eagle=False)

                if spec_result.dflash_block_hidden is None:
                    raise RuntimeError("Expected dflash_block_hidden traces from exact dflash backend")
                if verify_out.dflash_target_features_full is None:
                    raise RuntimeError("Expected full verifier-side DFlash target features")

                for row_idx, seq in enumerate(seqs):
                    if seq_step_counts[seq.seq_id] >= args.max_steps:
                        continue
                    record = seq_to_record[seq.seq_id]
                    payload = {
                        "dataset_name": record["dataset_name"],
                        "prompt_index": record["prompt_index"],
                        "group_key": _group_key(record),
                        "step_index": seq_step_counts[seq.seq_id],
                        "block_hidden": spec_result.dflash_block_hidden[row_idx].cpu(),
                        "target_features_full": verify_out.dflash_target_features_full[row_idx].cpu(),
                        "accepted_len": len(verify_out.new_suffixes[row_idx]),
                        "recovery_token": int(verify_out.recovery_tokens[row_idx]),
                        "speculations": spec_result.speculations[row_idx].cpu(),
                        "logits_q": spec_result.logits_q[row_idx].cpu(),
                    }
                    file_name = _save_trace(trace_dir, trace_idx, payload)
                    trace_index_rows.append({
                        "file_name": file_name,
                        "dataset_name": record["dataset_name"],
                        "prompt_index": record["prompt_index"],
                        "group_key": _group_key(record),
                        "step_index": seq_step_counts[seq.seq_id],
                        "accepted_len": payload["accepted_len"],
                        "recovery_token": payload["recovery_token"],
                    })
                    trace_idx += 1
                    seq_step_counts[seq.seq_id] += 1
                    if max_traces is not None and trace_idx >= max_traces:
                        break

                for seq, (orig_len, orig_nt, orig_lt, orig_ndc, orig_nct) in zip(seqs, saved):
                    del seq.token_ids[orig_len:]
                    seq.num_tokens = orig_nt
                    seq.last_token = orig_lt
                    seq.num_draft_cached_tokens = orig_ndc
                    seq.num_cached_tokens = orig_nct

                engine.scheduler.postprocess_speculate(
                    seqs,
                    verify_out.new_suffixes,
                    verify_out.recovery_tokens,
                    dflash_target_features=verify_out.dflash_target_features,
                )

                for seq in list(seqs):
                    if seq_step_counts[seq.seq_id] >= args.max_steps and not seq.is_finished:
                        seq.status = SequenceStatus.FINISHED
                        engine.scheduler.block_manager.deallocate(seq)
                        engine.scheduler.draft_block_manager.deallocate(seq)
                        if seq in engine.scheduler.running:
                            engine.scheduler.running.remove(seq)

                if max_traces is not None and trace_idx >= max_traces:
                    break

            _cleanup_batch_state(engine)

        _write_jsonl(out_dir / "trace_index.jsonl", trace_index_rows)
        export_metadata = {
            "target": args.target,
            "draft": args.draft,
            "dataset": args.dataset,
            "num_prompts_per_dataset": args.num_prompts_per_dataset,
            "input_len": args.input_len,
            "prompt_offset": args.prompt_offset,
            "max_steps": args.max_steps,
            "max_traces": args.max_traces,
            "effective_batch_size": batch_size,
            "num_prompt_groups": len(prompt_manifest),
            "num_traces": len(trace_index_rows),
            "num_skipped_prefill_groups": len(skipped_prefill_groups),
        }
        with (out_dir / "export_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(export_metadata, f, indent=2, sort_keys=True)
        if skipped_prefill_groups:
            _write_jsonl(out_dir / "skipped_prefill_groups.jsonl", skipped_prefill_groups)
        return export_metadata
    finally:
        engine.exit(hard=False)


def _is_oom(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def main():
    args = parse_args()

    import torch

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fallback_batch_size > args.batch_size:
        raise ValueError("--fallback-batch-size must be <= --batch-size")

    prompt_records = _load_prompt_records(args)
    if not prompt_records:
        raise RuntimeError("No prompt records were loaded for export")

    try:
        metadata = export_dataset(args, prompt_records, batch_size=args.batch_size)
    except Exception as exc:
        if not _is_oom(exc) or args.fallback_batch_size == args.batch_size:
            raise
        print(
            f"Export hit CUDA OOM at batch_size={args.batch_size}; retrying with batch_size={args.fallback_batch_size}",
            flush=True,
        )
        torch.cuda.empty_cache()
        metadata = export_dataset(args, prompt_records, batch_size=args.fallback_batch_size)

    print(
        f"Exported {metadata['num_traces']} traces from {metadata['num_prompt_groups']} prompt groups "
        f"to {out_dir} (batch_size={metadata['effective_batch_size']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
