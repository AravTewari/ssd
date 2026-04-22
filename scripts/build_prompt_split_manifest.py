import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a prompt manifest plus prompt-group train/val/test split metadata",
    )
    parser.add_argument("--target", type=str, required=True, help="Target model directory used for tokenization")
    parser.add_argument("--out", type=str, required=True, help="Output directory for prompt_manifest.jsonl and training_metadata.json")
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
        default=200,
        help="Number of prompts to load per dataset. For --dataset mixed this applies per dataset.",
    )
    parser.add_argument("--input-len", type=int, default=32, help="Minimum prompt token length")
    parser.add_argument("--prompt-offset", type=int, default=0, help="Skip the first N prompts per dataset")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic split seed")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def _group_key(record: dict) -> str:
    return f"{record['dataset_name']}:{record['prompt_index']}"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _load_prompt_records(args) -> list[dict]:
    from bench.bench_helpers import load_all_dataset_prompt_records, load_dataset_prompt_records

    if args.dataset == "mixed":
        return load_all_dataset_prompt_records(
            model_path=args.target,
            num_prompts_per_dataset=args.num_prompts_per_dataset,
            input_len=args.input_len,
            prompt_offset=args.prompt_offset,
            use_chat_template=False,
            disable_thinking=True,
        )
    return load_dataset_prompt_records(
        dataset_name=args.dataset,
        model_path=args.target,
        num_prompts=args.num_prompts_per_dataset,
        input_len=args.input_len,
        prompt_offset=args.prompt_offset,
        use_chat_template=False,
        disable_thinking=True,
        strict=True,
    )


def _build_group_splits(records: list[dict], seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[str]]:
    groups = defaultdict(list)
    for record in records:
        groups[_group_key(record)].append(record)
    group_keys = sorted(groups)
    if len(group_keys) < 3:
        raise RuntimeError(f"Need at least 3 prompt groups for train/val/test, found {len(group_keys)}")
    random.Random(seed).shuffle(group_keys)

    num_groups = len(group_keys)
    train_count = max(1, int(num_groups * train_ratio))
    val_count = max(1, int(num_groups * val_ratio))
    if train_count + val_count >= num_groups:
        val_count = max(1, num_groups - train_count - 1)
    test_count = num_groups - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > val_count:
            train_count -= 1
        else:
            val_count -= 1

    train_keys = group_keys[:train_count]
    val_keys = group_keys[train_count:train_count + val_count]
    test_keys = group_keys[train_count + val_count:]
    if not train_keys or not val_keys or not test_keys:
        raise RuntimeError("Failed to create non-empty train/val/test splits")

    return {
        "train": train_keys,
        "val": val_keys,
        "test": test_keys,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out)
    records = _load_prompt_records(args)
    if not records:
        raise RuntimeError("No prompt records were loaded")

    manifest_rows = [
        {
            "dataset_name": record["dataset_name"],
            "prompt_index": record["prompt_index"],
            "group_key": _group_key(record),
            "prompt_token_ids": record["prompt_token_ids"],
            "text": record["text"],
        }
        for record in records
    ]
    prompt_manifest_path = out_dir / "prompt_manifest.jsonl"
    _write_jsonl(prompt_manifest_path, manifest_rows)

    split_group_keys = _build_group_splits(
        records=records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    metadata = {
        "dataset": args.dataset,
        "num_prompt_records": len(manifest_rows),
        "num_prompt_groups": len({row["group_key"] for row in manifest_rows}),
        "prompt_manifest": str(prompt_manifest_path.resolve()),
        "seed": args.seed,
        "split_group_keys": split_group_keys,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "num_prompts_per_dataset": args.num_prompts_per_dataset,
        "input_len": args.input_len,
        "prompt_offset": args.prompt_offset,
    }
    metadata_path = out_dir / "training_metadata.json"
    _write_json(metadata_path, metadata)
    print(json.dumps({"prompt_manifest": str(prompt_manifest_path), "training_metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
