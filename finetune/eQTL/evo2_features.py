"""Cache frozen Evo2 features for DNALongBench eQTL windows.

Same contract as ``finetune/ETGP/evo2_features.py``: every window yields a
mean-pooled vector and per-block mean features, so one probe implementation
serves every backbone.  Windows follow the existing eQTL protocol, i.e. the
first 128 kb of each record, with no trailing padding added here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Imported before evo2 on purpose: vericomp pulls in mamba_ssm, and loading that
# extension after flash-attn has registered leaves torch.ops.flash_attn_2_cuda
# without a schema, which turns the first long forward into a hard failure.
from vericomp.eqtl_data import EQTLDataset

TOKEN_BYTE = {"A": 65, "C": 67, "G": 71, "T": 84, "N": 78}
TISSUES = (
    "Adipose_Subcutaneous",
    "Artery_Tibial",
    "Cells_Cultured_fibroblasts",
    "Muscle_Skeletal",
    "Nerve_Tibial",
    "Skin_Not_Sun_Exposed_Suprapubic",
    "Skin_Sun_Exposed_Lower_leg",
    "Thyroid",
    "Whole_Blood",
)


def encode(sequence: str, device: torch.device) -> torch.Tensor:
    codes = [TOKEN_BYTE.get(character.upper(), 78) for character in sequence]
    return torch.tensor(codes, dtype=torch.long, device=device).unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tissues", nargs="+", default=list(TISSUES))
    parser.add_argument("--context-length", type=int, default=128_000)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from evo2.models import Evo2

    device = torch.device(args.device)
    wrapper = Evo2("evo2_1b_base", local_path=args.checkpoint)
    backbone = wrapper.model
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.eval()
    captured: dict[str, torch.Tensor] = {}
    backbone.norm.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("x", out.detach())
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    done = 0

    for tissue in args.tissues:
        for split in ("train", "valid", "test"):
            dataset = EQTLDataset(args.data_root, tissue, split, args.context_length)
            records = dataset.records
            indices = list(range(args.shard, len(records), args.num_shards))
            means, bins, labels = [], [], []
            for index in indices:
                sequence, label = records[index]
                sequence = sequence[: args.context_length]
                if len(sequence) < args.context_length:
                    sequence = sequence.ljust(args.context_length, "N")
                with torch.no_grad():
                    wrapper(encode(sequence, device))
                hidden = captured.pop("x")[0]
                means.append(hidden.float().mean(dim=0))
                bins.append(
                    F.adaptive_avg_pool1d(hidden.t().unsqueeze(0), args.bins)
                    .squeeze(0).t().to(torch.bfloat16)
                )
                labels.append(int(label))
                done += 1
                if done % 100 == 0:
                    elapsed = time.time() - started
                    print(json.dumps({
                        "event": "progress", "done": done, "tissue": tissue,
                        "split": split, "s_per_window": round(elapsed / done, 2),
                    }), flush=True)

            torch.save({
                "tissue": tissue,
                "split": split,
                "indices": indices,
                "labels": torch.tensor(labels, dtype=torch.long),
                "mean": torch.stack(means) if means else torch.zeros(0, 1920),
                "bins": torch.stack(bins) if bins else torch.zeros(0, args.bins, 1920),
                "context_length": args.context_length,
                "bins_count": args.bins,
                "shard": args.shard,
                "num_shards": args.num_shards,
            }, out_dir / f"{tissue}_{split}_shard{args.shard}of{args.num_shards}.pt")

    print(json.dumps({
        "event": "done", "windows": done,
        "minutes": round((time.time() - started) / 60, 1),
    }), flush=True)


if __name__ == "__main__":
    main()
