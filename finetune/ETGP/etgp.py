"""VeriComp downstream ETGP binary classification.

The encoder is fully fine-tuned by default.  The classifier consumes the
mean-pooled Transformer/MoE bottleneck, matching the VeriComp eQTL protocol.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from finetune.eQTL.eqtl import (
    VeriCompEQTLClassifier,
    auprc_binary,
    auroc_binary,
    load_pretrained_backbone,
    make_loader,
    run_epoch,
)
from vericomp.etgp_data import ETGPDataset
from vericomp import VeriCompConfig
from vericomp.model_presets import available_model_presets, model_config_from_checkpoint
from finetune.run_config import (print_resolved_config, resolve_model_path,
                                  write_finetune_config)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-size", choices=("auto", *available_model_presets()),
                        default="auto",
                        help="architecture preset; auto reads new checkpoint metadata")
    parser.add_argument("--model", default="vericomp",
                        help="model .py path, /path/model.py:ClassName, or built-in alias")
    parser.add_argument("--feature", choices=("backbone", "decoder", "base"), default="backbone")
    parser.add_argument("--data-root", default="./data/dnalongbench")
    parser.add_argument("--cell-type", default="CRISPRi_EPI_K562_hg19")
    parser.add_argument("--data-max-length", type=int, default=12_800)
    parser.add_argument("--pooling-method", choices=("mean",), default="mean")
    parser.add_argument(
        "--dual-strand", action=argparse.BooleanOptionalAction, default=False,
        help="use the reverse-complement path during ETGP backbone inference",
    )
    parser.add_argument("--allocation-mode",
                        choices=("dynamic", "uniform", "length_only"),
                        default=None,
                        help="override allocation mode; pre-model_config "
                             "checkpoints default to dynamic")
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-optimizer-steps", type=int, default=None,
                        help="optional cap for a short speed or smoke run")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--test-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-valid-records", type=int, default=None)
    parser.add_argument("--max-test-records", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=str(Path("weights/finetune/ETGP") / time.strftime("%Y%m%d")),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


class LimitedBatchLoader:
    """Expose only the first N batches while preserving the loader interface."""

    def __init__(self, loader, max_batches):
        self.loader = loader
        self.max_batches = min(int(max_batches), len(loader))
        self.batch_size = loader.batch_size

    def __len__(self):
        return self.max_batches

    def __iter__(self):
        yield from itertools.islice(iter(self.loader), self.max_batches)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("ETGP downstream evaluation requires CUDA")
    if args.batch_size < 1 or args.grad_accumulation < 1:
        raise ValueError("batch size and gradient accumulation must be positive")
    if args.max_optimizer_steps is not None and args.max_optimizer_steps < 1:
        raise ValueError("max optimizer steps must be positive when provided")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    datasets = {
        subset: ETGPDataset(
            args.data_root,
            args.cell_type,
            subset,
            args.data_max_length,
            max_records=limit,
        )
        for subset, limit in (
            ("train", args.max_train_records),
            ("valid", args.max_valid_records),
            ("test", args.max_test_records),
        )
    }
    loaders = {
        "train": make_loader(datasets["train"], args.batch_size, args.num_workers, True),
        "valid": make_loader(datasets["valid"], args.batch_size, args.num_workers, False),
        "test": make_loader(datasets["test"], args.batch_size, args.num_workers, False),
    }

    config_overrides = {"force_flash_attention": True,
                        "activation_checkpointing": False}
    if getattr(args, "allocation_mode", None) is not None:
        config_overrides["allocation_mode"] = args.allocation_mode
    config = model_config_from_checkpoint(
        args.checkpoint, args.model_size, **config_overrides,
    )
    model = VeriCompEQTLClassifier(
        config,
        freeze_encoder=args.freeze_encoder,
        model_name=args.model,
        feature=args.feature,
        dual_strand=args.dual_strand,
    ).to(device)
    loaded, skipped = load_pretrained_backbone(model, args.checkpoint)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=max(1, args.epochs * max(1, len(loaders["train"]) // 10)),
    )

    output_dir = Path(args.output_dir) / args.cell_type
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_path(args.model)
    write_finetune_config(output_dir, task="ETGP", subtask=args.cell_type,
                          args=args, model_path=model_path, model_config=config,
                          extra={"data_root": str(Path(args.data_root).resolve())})
    print_resolved_config(
        task="ETGP", subtask=args.cell_type, args=args, model_path=model_path,
        model_config=config,
        metadata={
            "checkpoint": {
                "path": str(Path(args.checkpoint).expanduser().resolve()),
                "loaded_backbone_tensors": loaded,
                "skipped_backbone_tensors": skipped,
            },
            "data": {
                "train_examples": len(datasets["train"]),
                "valid_examples": len(datasets["valid"]),
                "test_examples": len(datasets["test"]),
                "max_length": args.data_max_length,
                "bottleneck_pooling": "mean",
                "bottleneck_tokens": min(args.data_max_length, config.max_tokens),
                "num_workers": args.num_workers,
            },
            "optimization": {
                "optimizer": "AdamW",
                "scheduler": "LinearLR",
                "total_scheduler_iters": max(1, args.epochs * max(1, len(loaders["train"]) // 10)),
                "train_batch_size": args.batch_size,
                "effective_batch_size": args.batch_size * args.grad_accumulation,
            },
            "parameters": {
                "total": sum(p.numel() for p in model.parameters()),
                "trainable": sum(p.numel() for p in trainable),
            },
            "runtime": {
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "autocast": "bfloat16" if device.type == "cuda" else "disabled",
            },
        },
    )

    best_auprc = -math.inf
    history = []
    step_state = {"batch_step": 0, "optimizer_step": 0, "test_history": [],
                  "next_test_batch": args.test_every}
    for epoch in range(1, args.epochs + 1):
        # ``run_epoch`` is shared with eQTL and performs the optimizer loop;
        # ETGP evaluates validation after each epoch and test once at the end.
        # Keep the task-specific ``test_every`` argument for CLI compatibility,
        # but do not pass unsupported callback arguments into the shared loop.
        train_loader = loaders["train"]
        if args.max_optimizer_steps is not None:
            remaining_steps = args.max_optimizer_steps - step_state["optimizer_step"]
            if remaining_steps <= 0:
                break
            train_loader = LimitedBatchLoader(
                train_loader, remaining_steps * args.grad_accumulation
            )
        steps_before_epoch = step_state["optimizer_step"]
        train_started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            grad_accumulation=args.grad_accumulation,
            max_grad_norm=args.max_grad_norm,
            step_state=step_state,
        )
        train_elapsed = time.perf_counter() - train_started
        updates_in_epoch = step_state["optimizer_step"] - steps_before_epoch
        print(json.dumps({
            "event": "train_epoch_timing",
            "epoch": epoch,
            "optimizer_steps": updates_in_epoch,
            "elapsed_seconds": train_elapsed,
            "seconds_per_optimizer_step": train_elapsed / max(updates_in_epoch, 1),
        }), flush=True)
        scheduler.step()
        with torch.inference_mode():
            valid_metrics = run_epoch(model, loaders["valid"], device)
        record = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        if np.isfinite(valid_metrics["auprc"]) and valid_metrics["auprc"] > best_auprc:
            best_auprc = valid_metrics["auprc"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "valid": valid_metrics, "args": vars(args)}, output_dir / "best.pt")
        if (args.max_optimizer_steps is not None
                and step_state["optimizer_step"] >= args.max_optimizer_steps):
            break

    best_path = output_dir / "best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
    with torch.inference_mode():
        test_metrics = run_epoch(model, loaders["test"], device)
    result = {
        "task": "ETGP",
        "cell_type": args.cell_type,
        "valid_best_auprc": best_auprc,
        "valid_best_auroc": max(
            (record["valid"]["auroc"] for record in history
             if np.isfinite(record["valid"]["auroc"])),
            default=-math.inf,
        ),
        "test": test_metrics,
        "test_history": step_state["test_history"],
        "history": history,
    }
    torch.save({"model": model.state_dict(), "result": result, "args": vars(args)}, output_dir / "last.pt")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"test": test_metrics, "output_dir": str(output_dir)}), flush=True)


if __name__ == "__main__":
    main()
