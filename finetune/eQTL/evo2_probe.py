"""Train per-tissue probes on cached frozen Evo2 eQTL features.

The eQTL benchmark is nine GTEx tissues with their own train/valid/test splits.
Each tissue gets the same two readouts used for ETGP, then the per-tissue
AUROCs are macro-averaged, matching how the fine-tuned baselines are reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from finetune.ETGP.evo2_probe import BinsConv, MeanLinear, evaluate, train_one


def load_split(feature_dir: Path, tissue: str, split: str) -> dict[str, torch.Tensor]:
    shards = sorted(feature_dir.glob(f"{tissue}_{split}_shard*.pt"))
    if not shards:
        raise FileNotFoundError(f"no shards for {tissue}/{split} in {feature_dir}")
    payloads = [torch.load(path, map_location="cpu") for path in shards]
    indices = sum((p["indices"] for p in payloads), [])
    order = np.argsort(indices)
    return {
        "labels": torch.cat([p["labels"] for p in payloads])[order],
        "mean": torch.cat([p["mean"] for p in payloads])[order],
        "bins": torch.cat([p["bins"] for p in payloads])[order],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--tissues", nargs="+", default=None)
    parser.add_argument("--probes", nargs="+", default=["mean_linear", "bins_conv"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    feature_dir = Path(args.feature_dir)
    tissues = args.tissues or sorted({
        path.name.split("_train_shard")[0]
        for path in feature_dir.glob("*_train_shard*.pt")
    })

    summary = {}
    for probe in args.probes:
        per_tissue = {}
        for tissue in tissues:
            splits = {name: load_split(feature_dir, tissue, name)
                      for name in ("train", "valid", "test")}
            width = int(splits["train"]["mean"].shape[-1])
            runs = [
                train_one(probe, width, splits["train"], splits["valid"], splits["test"],
                          device, seed=seed, epochs=args.epochs,
                          lr=args.lr, weight_decay=args.weight_decay)
                for seed in range(args.seeds)
            ]
            per_tissue[tissue] = {
                "runs": runs,
                "valid_auroc": float(np.mean([r["auroc"] for r in runs])),
                "test_auroc": float(np.mean([r["test"]["auroc"] for r in runs])),
                "test_auprc": float(np.mean([r["test"]["auprc"] for r in runs])),
                "n": {k: int(len(v["labels"])) for k, v in splits.items()},
            }
            print(json.dumps({"event": "tissue", "probe": probe, **{tissue: per_tissue[tissue]}}),
                  flush=True)
        summary[probe] = {
            "per_tissue": per_tissue,
            "macro_valid_auroc": float(np.mean([v["valid_auroc"] for v in per_tissue.values()])),
            "macro_test_auroc": float(np.mean([v["test_auroc"] for v in per_tissue.values()])),
            "macro_test_auprc": float(np.mean([v["test_auprc"] for v in per_tissue.values()])),
            "seeds": args.seeds,
        }

    result = {
        "task": "eQTL",
        "model": "Evo2-1B-base frozen",
        "feature_dir": str(feature_dir),
        "results": summary,
    }
    print(json.dumps({"event": "summary", **result}, ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
