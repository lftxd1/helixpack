"""Train small probes on cached frozen Evo2 ETGP features.

Two readouts are compared because they answer different questions:

  mean_linear : mean-pooled window feature -> linear layer.  Identical
                pooling protocol to the fine-tuned external ETGP baselines.
  bins_conv   : per-block mean features -> two 1D convolutions over the block
                axis -> mean pool -> linear.  Keeps coarse position, still
                no per-base output.

The dataset is tiny (2.1k train / 266 valid / 270 test) and heavily imbalanced,
so every result is reported as mean +- std over several head seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = ranks[positives].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    hits = np.cumsum(labels)
    precision = hits / np.arange(1, len(labels) + 1)
    return float((precision * labels).sum() / max(1, labels.sum()))


def load_split(feature_dir: Path, split: str) -> dict[str, torch.Tensor]:
    shards = sorted(feature_dir.glob(f"{split}_shard*.pt"))
    if not shards:
        raise FileNotFoundError(f"no shards for split {split} in {feature_dir}")
    payloads = [torch.load(path, map_location="cpu") for path in shards]
    indices = sum((p["indices"] for p in payloads), [])
    order = np.argsort(indices)
    merged = {
        "labels": torch.cat([p["labels"] for p in payloads])[order],
        "mean": torch.cat([p["mean"] for p in payloads])[order],
        "bins": torch.cat([p["bins"] for p in payloads])[order],
        "context_length": int(payloads[0]["context_length"]),
        "bins_count": int(payloads[0]["bins_count"]),
    }
    return merged


class MeanLinear(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2)

    def forward(self, mean, bins):
        return self.head(self.norm(mean.float()))


class BinsConv(nn.Module):
    def __init__(self, width: int, channels: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Conv1d(width, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.head = nn.Linear(channels, 2)

    def forward(self, mean, bins):
        hidden = self.norm(bins.float()).transpose(1, 2)
        hidden = self.net(hidden).mean(dim=-1)
        return self.head(hidden)


def evaluate(model: nn.Module, data: dict[str, torch.Tensor], device: torch.device) -> dict[str, float]:
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(data["labels"]), 256):
            chunk = slice(start, start + 256)
            logits = model(data["mean"][chunk].to(device), data["bins"][chunk].to(device))
            scores.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu())
    scores = torch.cat(scores).numpy()
    labels = data["labels"].numpy()
    return {"auroc": auroc(labels, scores), "auprc": auprc(labels, scores)}


def train_one(kind: str, width: int, train, valid, test, device, seed: int,
              epochs: int, lr: float, weight_decay: float) -> dict:
    torch.manual_seed(seed)
    model = (MeanLinear(width) if kind == "mean_linear" else BinsConv(width)).to(device)
    positives = float(train["labels"].sum())
    negatives = float(len(train["labels"]) - positives)
    class_weight = torch.tensor([1.0, negatives / max(1.0, positives)], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total = epochs * max(1, len(train["labels"]) // 256 + 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total)

    best = {"auroc": -1.0}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train["labels"]))
        for start in range(0, len(order), 256):
            batch = order[start:start + 256]
            logits = model(train["mean"][batch].to(device), train["bins"][batch].to(device))
            loss = loss_fn(logits, train["labels"][batch].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        valid_metrics = evaluate(model, valid, device)
        if valid_metrics["auroc"] > best["auroc"]:
            best = {"epoch": epoch, **valid_metrics,
                    "test": evaluate(model, test, device)}
    if "test" not in best:  # single-class validation leaves AUROC undefined
        best = {"epoch": epochs, **evaluate(model, valid, device),
                "test": evaluate(model, test, device)}
    return best


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    feature_dir = Path(args.feature_dir)
    splits = {name: load_split(feature_dir, name) for name in ("train", "valid", "test")}
    width = int(splits["train"]["mean"].shape[-1])
    print(json.dumps({
        "event": "loaded",
        "width": width,
        "bins": int(splits["train"]["bins"].shape[1]),
        "n": {k: len(v["labels"]) for k, v in splits.items()},
        "positives": {k: int(v["labels"].sum()) for k, v in splits.items()},
    }), flush=True)

    results = {}
    for kind in ("mean_linear", "bins_conv"):
        runs = []
        for seed in range(args.seeds):
            run = train_one(kind, width, splits["train"], splits["valid"], splits["test"],
                            device, seed=seed, epochs=args.epochs,
                            lr=args.lr, weight_decay=args.weight_decay)
            runs.append(run)
            print(json.dumps({"event": "run", "probe": kind, "seed": seed, **run}), flush=True)
        results[kind] = {
            "valid_auroc": float(np.mean([r["auroc"] for r in runs])),
            "valid_auroc_std": float(np.std([r["auroc"] for r in runs])),
            "test_auroc": float(np.mean([r["test"]["auroc"] for r in runs])),
            "test_auroc_std": float(np.std([r["test"]["auroc"] for r in runs])),
            "test_auprc": float(np.mean([r["test"]["auprc"] for r in runs])),
            "test_auprc_std": float(np.std([r["test"]["auprc"] for r in runs])),
            "seeds": args.seeds,
        }

    summary = {
        "task": "ETGP", "model": "Evo2-1B-base frozen",
        "context_length": splits["train"]["context_length"],
        "feature_dir": str(feature_dir),
        "results": results,
    }
    print(json.dumps({"event": "summary", **summary}, ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
