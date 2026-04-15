from __future__ import annotations

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
        description="Train and evaluate a DFlash SSD predictor from grouped exact-dflash traces",
    )
    parser.add_argument("--trace-dir", type=str, required=True, help="Trace export directory")
    parser.add_argument("--out", type=str, required=True, help="Output directory for checkpoints and metadata")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overfit",
        type=int,
        default=0,
        help="If > 0, restrict training to this many prompt groups for pipeline debugging.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def _load_trace_index(trace_dir: Path) -> list[dict]:
    index_path = trace_dir / "trace_index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing trace index: {index_path}")
    rows = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No trace rows found in {index_path}")
    return rows


class TraceDataset:
    def __init__(self, trace_dir: Path, entries: list[dict]):
        self.trace_dir = trace_dir
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        import torch

        entry = self.entries[index]
        payload = torch.load(self.trace_dir / "traces" / entry["file_name"], map_location="cpu")
        return (
            payload["block_hidden"].float(),
            payload["target_features_full"].float(),
            int(payload["accepted_len"]),
        )


def collate(batch):
    import torch

    block_hidden = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    accepted_lens = torch.tensor([item[2] for item in batch], dtype=torch.int64)
    return block_hidden, targets, accepted_lens


def _compute_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    import torch
    import torch.nn.functional as F

    mse = torch.mean((predicted - target) ** 2)
    cosine_per_pos = F.cosine_similarity(predicted, target, dim=-1)
    cosine_mean = cosine_per_pos.mean()
    cosine_loss = 1.0 - cosine_mean
    loss = 0.5 * mse + 0.5 * cosine_loss
    mse_per_pos = torch.mean((predicted - target) ** 2, dim=-1)
    return {
        "loss": loss,
        "mse": mse,
        "cosine": cosine_mean,
        "cosine_loss": cosine_loss,
        "mse_per_pos": mse_per_pos,
        "cosine_per_pos": cosine_per_pos,
    }


def evaluate(model, loader: DataLoader, device: torch.device) -> dict:
    import torch

    model.eval()
    total = 0
    loss_sum = 0.0
    mse_sum = 0.0
    cosine_sum = 0.0
    mse_by_pos = None
    cosine_by_pos = None
    count_by_pos = None
    accepted_buckets: dict[int, dict[str, float]] = defaultdict(lambda: {"count": 0, "loss": 0.0, "mse": 0.0, "cosine": 0.0})

    with torch.inference_mode():
        for block_hidden, target, accepted_lens in loader:
            block_hidden = block_hidden.to(device)
            target = target.to(device)
            predicted = model(block_hidden)
            metrics = _compute_metrics(predicted, target)
            batch_size = block_hidden.shape[0]

            loss_sum += float(metrics["loss"].item()) * batch_size
            mse_sum += float(metrics["mse"].item()) * batch_size
            cosine_sum += float(metrics["cosine"].item()) * batch_size
            total += batch_size

            if mse_by_pos is None:
                block_size = metrics["mse_per_pos"].shape[1]
                mse_by_pos = torch.zeros(block_size, dtype=torch.float64)
                cosine_by_pos = torch.zeros(block_size, dtype=torch.float64)
                count_by_pos = torch.zeros(block_size, dtype=torch.float64)
            mse_by_pos += metrics["mse_per_pos"].double().sum(dim=0).cpu()
            cosine_by_pos += metrics["cosine_per_pos"].double().sum(dim=0).cpu()
            count_by_pos += batch_size

            for row_idx, accepted_len in enumerate(accepted_lens.tolist()):
                bucket = accepted_buckets[accepted_len]
                row_pred = predicted[row_idx:row_idx + 1]
                row_target = target[row_idx:row_idx + 1]
                row_metrics = _compute_metrics(row_pred, row_target)
                bucket["count"] += 1
                bucket["loss"] += float(row_metrics["loss"].item())
                bucket["mse"] += float(row_metrics["mse"].item())
                bucket["cosine"] += float(row_metrics["cosine"].item())

    overall = {
        "loss": loss_sum / max(total, 1),
        "mse": mse_sum / max(total, 1),
        "cosine": cosine_sum / max(total, 1),
    }
    per_position = {}
    if mse_by_pos is not None and count_by_pos is not None:
        for pos in range(len(mse_by_pos)):
            denom = max(float(count_by_pos[pos].item()), 1.0)
            per_position[str(pos)] = {
                "mse": float(mse_by_pos[pos].item() / denom),
                "cosine": float(cosine_by_pos[pos].item() / denom),
            }
    accepted_length_metrics = {}
    for accepted_len, bucket in sorted(accepted_buckets.items()):
        denom = max(bucket["count"], 1)
        accepted_length_metrics[str(accepted_len)] = {
            "count": bucket["count"],
            "loss": bucket["loss"] / denom,
            "mse": bucket["mse"] / denom,
            "cosine": bucket["cosine"] / denom,
        }
    return {
        "overall": overall,
        "per_position": per_position,
        "accepted_length": accepted_length_metrics,
    }


def _build_group_splits(entries: list[dict], seed: int, train_ratio: float, val_ratio: float, overfit: int) -> tuple[dict[str, list[str]], dict[str, list[dict]]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        groups[entry["group_key"]].append(entry)
    group_keys = sorted(groups)
    random.Random(seed).shuffle(group_keys)
    if overfit > 0:
        group_keys = group_keys[:overfit]

    num_groups = len(group_keys)
    if num_groups < 3:
        raise RuntimeError(f"Need at least 3 prompt groups for train/val/test, found {num_groups}")
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

    split_entries = {
        "train": [entry for key in train_keys for entry in groups[key]],
        "val": [entry for key in val_keys for entry in groups[key]],
        "test": [entry for key in test_keys for entry in groups[key]],
    }
    split_keys = {
        "train": train_keys,
        "val": val_keys,
        "test": test_keys,
    }
    return split_keys, split_entries


def main():
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from ssd.engine.dflash_predictor import DFlashFeaturePredictor, DFlashPredictorConfig

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = _load_trace_index(trace_dir)
    split_keys, split_entries = _build_group_splits(
        entries=entries,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        overfit=args.overfit,
    )

    train_dataset = TraceDataset(trace_dir, split_entries["train"])
    val_dataset = TraceDataset(trace_dir, split_entries["val"])
    test_dataset = TraceDataset(trace_dir, split_entries["test"])

    sample_hidden, sample_target, _ = train_dataset[0]
    config = DFlashPredictorConfig(
        hidden_size=sample_hidden.shape[-1],
        target_feature_dim=sample_target.shape[-1],
        block_size=sample_hidden.shape[0],
        position_dim=sample_hidden.shape[-1],
        mlp_hidden_size=sample_hidden.shape[-1] * 2,
    )
    model = DFlashFeaturePredictor(config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    epoch_summaries = []
    device = torch.device(args.device)
    for epoch in range(args.epochs):
        model.train()
        for block_hidden, target, _accepted_lens in train_loader:
            block_hidden = block_hidden.to(device)
            target = target.to(device)
            predicted = model(block_hidden)
            loss = _compute_metrics(predicted, target)["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_metrics = evaluate(model, val_loader, device)
        overall = val_metrics["overall"]
        epoch_summaries.append({
            "epoch": epoch + 1,
            **overall,
        })
        print(
            f"epoch={epoch + 1} "
            f"val_loss={overall['loss']:.6f} "
            f"val_mse={overall['mse']:.6f} "
            f"val_cosine={overall['cosine']:.6f}",
            flush=True,
        )
        if overall["loss"] < best_val:
            best_val = overall["loss"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    best_dir = out_dir / "best_checkpoint"
    model.load_state_dict(best_state)
    model.save_pretrained(str(best_dir))

    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    metadata = {
        "trace_dir": str(trace_dir),
        "prompt_manifest": str(trace_dir / "prompt_manifest.jsonl"),
        "args": vars(args),
        "model_config": {
            "hidden_size": config.hidden_size,
            "target_feature_dim": config.target_feature_dim,
            "block_size": config.block_size,
            "position_dim": config.position_dim,
            "mlp_hidden_size": config.mlp_hidden_size,
        },
        "best_epoch": best_epoch,
        "split_group_keys": split_keys,
        "split_counts": {
            split: {
                "num_groups": len(split_keys[split]),
                "num_traces": len(split_entries[split]),
            }
            for split in ("train", "val", "test")
        },
        "epoch_summaries": epoch_summaries,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": str(best_dir),
    }
    with (out_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(
        f"saved={best_dir} "
        f"best_epoch={best_epoch} "
        f"val_loss={val_metrics['overall']['loss']:.6f} "
        f"val_mse={val_metrics['overall']['mse']:.6f} "
        f"val_cosine={val_metrics['overall']['cosine']:.6f} "
        f"test_loss={test_metrics['overall']['loss']:.6f} "
        f"test_mse={test_metrics['overall']['mse']:.6f} "
        f"test_cosine={test_metrics['overall']['cosine']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
