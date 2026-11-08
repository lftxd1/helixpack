"""Evaluate long-context MLM loss/accuracy separately for every length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pretrain.train import evaluate, make_loader
from vericomp import VeriCompConfig, VeriCompPretrainModel
from vericomp.long_context_data import HybridLongContextDataset


DEFAULT_LENGTHS = "1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576"
DEFAULT_BATCH_SIZES = "1024:128,2048:64,4096:32,8192:32,16384:16,32768:16,65536:8,131072:4,262144:2,524288:2,1048576:2"


def parse_mapping(text):
    result = {}
    for item in text.split(","):
        length, batch = item.split(":", 1)
        result[int(length)] = int(batch)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fasta", default="/gpfs_data/Projects/data/hg38/hg38.ml.fa")
    parser.add_argument("--bed", default="/gpfs_data/Projects/data/hg38/human-sequences.bed")
    parser.add_argument("--lengths", default=DEFAULT_LENGTHS)
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--samples-per-length", type=int, default=28)
    parser.add_argument("--n-max-fraction", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--allocation-mode",
        choices=("dynamic", "uniform", "length_only"),
        default="dynamic",
    )
    parser.add_argument(
        "--strict-uncompressed-short",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=10017)
    args = parser.parse_args()

    lengths = [int(value) for value in args.lengths.split(",") if value.strip()]
    batch_sizes = parse_mapping(args.batch_sizes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("evaluation requires CUDA")

    config = VeriCompConfig(
        num_layers=12,
        allocation_mode=args.allocation_mode,
        strict_uncompressed_short=args.strict_uncompressed_short,
        activation_checkpointing=True,
    )
    model = VeriCompPretrainModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()

    results = []
    per_length_samples = max(1, int(args.samples_per_length))
    for offset, length in enumerate(lengths):
        dataset = HybridLongContextDataset(
            args.bed,
            args.fasta,
            sequence_lengths=[length],
            split="valid",
            max_length=max(lengths),
            seed=args.seed + offset,
            random_window_sampling=False,
            long_samples_per_length=per_length_samples,
            max_samples_per_length=per_length_samples,
            n_max_fraction=args.n_max_fraction,
        )
        loader, sampler = make_loader(
            dataset,
            world_size=1,
            rank=0,
            batch_sizes={length: batch_sizes[length]},
            seed=args.seed + offset,
            shuffle=False,
            drop_last=False,
            workers=args.num_workers,
            equal_length_sampling=False,
        )
        metrics = evaluate(model, loader, sampler, device=device, world_size=1)
        row = {"length": length, **metrics}
        results.append(row)
        print(
            f"L={length:>7} examples={metrics['examples']:>3} "
            f"loss={metrics['loss']:.6f} mlm={metrics['loss_mlm']:.6f} "
            f"mlm_acc={metrics['mlm_acc']:.6f} "
            f"strand={metrics['loss_strand']:.6f} router={metrics['loss_router']:.6f}",
            flush=True,
        )
        del loader, sampler, dataset
        torch.cuda.empty_cache()

    output = Path(args.checkpoint).with_name("per_length_validation.json")
    output.write_text(json.dumps({"checkpoint": args.checkpoint, "results": results}, indent=2) + "\n")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
