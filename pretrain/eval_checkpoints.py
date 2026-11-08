"""Score pretraining checkpoints on one shared validation set.

The August runs and the October grid run each built their own validation
split (300 vs 304 examples, and the BED-derived short buckets have changed
since), so their logged val losses are not comparable.  This script rebuilds
the split once, from the current dataset code, and evaluates every checkpoint
on exactly that split.

Usage:
    python -m pretrain.eval_checkpoints \
        --checkpoint a.pt --checkpoint b.pt [--lengths 1024,...] [--device cuda:0]

Allocation mode and strict_uncompressed_short are taken from each
checkpoint's ``model_config`` when present, otherwise from its pretraining
``args``, otherwise from the long-context defaults.  That reproduces the
forward path each checkpoint was actually trained with.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pretrain.train import masked_accuracy
from vericomp import VeriCompConfig, VeriCompPretrainModel
from vericomp.data import SpanMLMCollator
from vericomp.long_context_data import HybridLongContextDataset


def _parse_lengths(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _checkpoint_config(path: str) -> VeriCompConfig:
    """Recover the forward path a checkpoint was trained with."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    payload = checkpoint.get("model_config")
    if isinstance(payload, dict):
        return VeriCompConfig(**payload)
    args = checkpoint.get("args") or {}
    return VeriCompConfig(
        num_layers=args.get("num_layers", 12),
        allocation_mode=args.get("allocation_mode", "dynamic"),
        grid_stride=args.get("grid_stride", 32),
        strict_uncompressed_short=args.get("strict_uncompressed_short", True),
        duplex_mode=args.get("duplex_mode", "dual"),
        activation_checkpointing=False,
    )


def _load_model(path: str, device: torch.device) -> VeriCompPretrainModel:
    config = _checkpoint_config(path)
    config.force_flash_attention = True
    model = VeriCompPretrainModel(config).to(device)
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{path}: state dict mismatch, missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model


def _build_valid(args) -> HybridLongContextDataset:
    lengths = _parse_lengths(args.lengths)
    per_length = max(1, (args.validation_samples + len(lengths) - 1) // len(lengths))
    return HybridLongContextDataset(
        args.bed,
        args.fasta,
        sequence_lengths=lengths,
        split="valid",
        max_length=max(lengths),
        seed=args.seed + 10_000,
        random_window_sampling=False,
        long_samples_per_length=per_length,
        max_samples_per_length=per_length,
        n_max_fraction=0.01,
        max_window_attempts=32,
        boundary_margin=0,
    )


@torch.no_grad()
def _evaluate(model, dataset, lengths, args, device) -> dict:
    """Evaluate per length so compression-vs-no-compression is visible."""
    rows = {}
    for length in lengths:
        indices = [i for i, (_, _, sample_length) in enumerate(dataset.samples)
                   if sample_length == length]
        if not indices:
            continue
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(
            subset, batch_size=args.batch_size, shuffle=False,
            collate_fn=SpanMLMCollator(), num_workers=args.num_workers,
        )
        total_loss = total_mlm = total_correct = total_masked = 0.0
        count = 0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                out = model(**batch)
            n = batch["input_ids_pos"].shape[0]
            total_loss += float(out["loss"].float()) * n
            total_mlm += float(out["loss_mlm"].float()) * n
            correct, masked = masked_accuracy(out["logits"], batch["labels"])
            total_correct += correct
            total_masked += masked
            count += n
        rows[length] = {
            "examples": count,
            "loss": total_loss / count,
            "mlm": total_mlm / count,
            "mlm_acc": total_correct / max(1.0, total_masked),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--bed", default="/gpfs_data/Projects/data/hg38/human-sequences.bed")
    parser.add_argument("--fasta", default="/gpfs_data/Projects/data/hg38/hg38.ml.fa")
    parser.add_argument("--lengths", default="1024,2048,4096,8192,16384,32768,65536,131072")
    parser.add_argument("--validation-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    lengths = _parse_lengths(args.lengths)
    dataset = _build_valid(args)
    print(f"validation examples={len(dataset)} lengths={lengths}", flush=True)

    results = {}
    for path in args.checkpoint:
        model = _load_model(path, device)
        config = model.config
        tag = f"{Path(path).parent.name}/{Path(path).name}"
        rows = _evaluate(model, dataset, lengths, args, device)
        results[tag] = {
            "allocation_mode": config.allocation_mode,
            "grid_stride": config.grid_stride if config.allocation_mode == "grid" else None,
            "strict_uncompressed_short": config.strict_uncompressed_short,
            "per_length": rows,
        }
        total = sum(r["examples"] for r in rows.values())
        pooled_loss = sum(r["loss"] * r["examples"] for r in rows.values()) / total
        pooled_mlm = sum(r["mlm"] * r["examples"] for r in rows.values()) / total
        results[tag]["pooled"] = {
            "examples": total, "loss": pooled_loss, "mlm": pooled_mlm,
        }
        print(f"{tag}: mode={config.allocation_mode} "
              f"strict_short={config.strict_uncompressed_short} "
              f"loss={pooled_loss:.5f} mlm={pooled_mlm:.5f}", flush=True)
        for length, row in rows.items():
            print("    L=%7d n=%3d loss=%.5f mlm=%.5f acc=%.4f" % (
                length, row["examples"], row["loss"], row["mlm"], row["mlm_acc"]),
                flush=True)
        del model
        torch.cuda.empty_cache()

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
