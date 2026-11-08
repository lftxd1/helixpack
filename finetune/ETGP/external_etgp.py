"""External-backbone ETGP fine-tuning at long context.

This runner adapts the DNALongBench ETGP records to external DNA backbones
supported by ``finetune.eQTL.external_eqtl.Backbone``. The ETGP classifier
uses mean-pooled per-base hidden states and a two-class head.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from finetune.eQTL.eqtl import auprc_binary, auroc_binary
from finetune.eQTL.external_eqtl import Backbone
from vericomp.etgp_data import ETGPDataset


_ID_TO_BASE = ("A", "C", "G", "T", "N", "N")


class _SequenceETGPDataset(Dataset):
    """ETGP records exposed as strings for external tokenizers."""

    def __init__(self, root: str, cell_type: str, subset: str, context_length: int):
        source = ETGPDataset(root, cell_type, subset, context_length)
        self.records = source.records
        self.context_length = int(context_length)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        sequence, label = self.records[int(index)]
        sequence = sequence[: self.context_length].ljust(self.context_length, "N")
        return {"sequence": sequence, "labels": int(label)}


def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "sequence": [str(item["sequence"]) for item in batch],
        "labels": torch.tensor([int(item["labels"]) for item in batch], dtype=torch.long),
    }


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    labels_np = np.asarray(labels, dtype=np.int64)
    scores_np = np.asarray(scores, dtype=np.float64)
    result = {
        "auroc": auroc_binary(labels_np, scores_np),
        "auprc": auprc_binary(labels_np, scores_np),
    }
    return result


def _evaluate(backbone: Backbone, head: nn.Module, loader: DataLoader, device: torch.device):
    model_training, head_training = backbone.model.training, head.training
    backbone.model.eval(); head.eval()
    labels, scores = [], []
    with torch.no_grad():
        for batch in loader:
            features = backbone.features(batch["sequence"], grad=False)
            probabilities = torch.softmax(head(features), dim=-1)[:, 1]
            labels.extend(batch["labels"].tolist())
            scores.extend(probabilities.float().cpu().tolist())
    if model_training:
        backbone.model.train()
    if head_training:
        head.train()
    return _metrics(labels, scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("hyenadna", "caduceus_ph", "caduceus_ps", "ntv3", "janusdna"),
        required=True,
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cell-type", default="CRISPRi_EPI_K562_hg19")
    parser.add_argument("--context-length", type=int, default=128000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--eval-batch-size", type=int, default=None,
        help="Validation/test batch size; defaults to --batch-size.",
    )
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.context_length != 128000:
        raise ValueError("external ETGP runner currently supports 128k context only")
    if args.batch_size * args.grad_accumulation < 1:
        raise ValueError("effective batch size must be positive")
    eval_batch_size = args.eval_batch_size or args.batch_size
    if eval_batch_size < 1:
        raise ValueError("evaluation batch size must be positive")

    _seed(args.seed)
    if args.model == "janusdna":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = Backbone(
        args.model, args.model_path, device,
        janusdna_bf16_autocast=(args.model == "janusdna"),
    )
    datasets = {
        split: _SequenceETGPDataset(args.data_root, args.cell_type, split, args.context_length)
        for split in ("train", "valid", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size if split == "train" else eval_batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers, pin_memory=True, collate_fn=_collate,
        )
        for split, dataset in datasets.items()
    }
    if min(len(dataset) for dataset in datasets.values()) == 0:
        raise RuntimeError({split: len(dataset) for split, dataset in datasets.items()})

    with torch.no_grad():
        width = int(backbone.features([datasets["train"][0]["sequence"]], grad=False).shape[-1])
    head = nn.Linear(width, 2, bias=True).to(device, dtype=torch.float32)
    for parameter in backbone.model.parameters():
        parameter.requires_grad_(True)
    backbone.model.train(); head.train()
    trainable = [p for p in backbone.model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    batches_per_epoch = len(loaders["train"])
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accumulation)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=max(1, warmup_steps)
    ) if warmup_steps else None
    loss_fn = nn.CrossEntropyLoss()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "ETGP", "model": args.model, "model_path": str(Path(args.model_path).resolve()),
        "cell_type": args.cell_type, "context_length": args.context_length,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "eval_batch_size": eval_batch_size,
        "grad_accumulation": args.grad_accumulation,
        "effective_batch_size": args.batch_size * args.grad_accumulation,
        "steps_per_epoch": steps_per_epoch, "total_optimizer_steps": total_steps,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio, "seed": args.seed,
        "n": {split: len(dataset) for split, dataset in datasets.items()},
        "dtype": "bfloat16_backbone_autocast_float32_head_loss",
        "tf32_enabled": bool(args.model == "janusdna"),
        "router_aux_loss_coef": 0.02 if args.model == "janusdna" else None,
        "training_objective": (
            "cross_entropy_plus_model_moe_auxiliary_loss"
            if args.model == "janusdna" else "cross_entropy"
        ),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"event": "resolved_config", **config}, ensure_ascii=False), flush=True)

    best_auroc = -float("inf")
    best_state = None
    history = []
    global_step = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        backbone.model.train(); head.train(); optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_main_loss = 0.0
        train_moe_loss = 0.0
        for batch_index, batch in enumerate(loaders["train"], start=1):
            output_router_logits = args.model == "janusdna"
            extracted = backbone.features(
                batch["sequence"], grad=True,
                output_router_logits=output_router_logits,
            )
            if output_router_logits:
                features, moe_loss = extracted
                if moe_loss is None:
                    raise RuntimeError("JanusDNA did not return its MoE auxiliary loss")
            else:
                features, moe_loss = extracted, None
            labels = batch["labels"].to(device, non_blocking=True)
            logits = head(features)
            main_loss = loss_fn(logits.float(), labels)
            total_loss = main_loss + moe_loss if moe_loss is not None else main_loss
            (total_loss / args.grad_accumulation).backward()
            train_loss += float(total_loss.detach().cpu())
            train_main_loss += float(main_loss.detach().cpu())
            if moe_loss is not None:
                train_moe_loss += float(moe_loss.detach().cpu())
            if batch_index % args.grad_accumulation == 0 or batch_index == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if scheduler is not None and global_step <= warmup_steps:
                    scheduler.step()
                if global_step == 1 or global_step % 50 == 0:
                    print(json.dumps({"epoch": epoch, "step": global_step,
                                      "train_loss": float(total_loss.detach().cpu()),
                                      "main_loss": float(main_loss.detach().cpu()),
                                      "moe_loss": (
                                          float(moe_loss.detach().cpu())
                                          if moe_loss is not None else None
                                      ),
                                      "elapsed_seconds": round(time.time() - started, 1)}), flush=True)
        valid = _evaluate(backbone, head, loaders["valid"], device)
        test = _evaluate(backbone, head, loaders["test"], device)
        record = {"epoch": epoch, "step": global_step,
                  "train_loss": train_loss / max(1, len(loaders["train"])),
                  "train_main_loss": train_main_loss / max(1, len(loaders["train"])),
                  "train_moe_loss": train_moe_loss / max(1, len(loaders["train"])),
                  "valid": valid, "test": test}
        history.append(record)
        with (output_dir / "epoch_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if np.isfinite(valid["auroc"]) and valid["auroc"] > best_auroc:
            best_auroc = valid["auroc"]
            best_state = {
                "head": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
                "backbone": {key: value.detach().cpu().clone() for key, value in backbone.model.state_dict().items()},
                "epoch": epoch, "step": global_step, "valid": valid,
            }
            torch.save(best_state, output_dir / "best.pt")
    result = {"task": "ETGP", "model": args.model, "cell_type": args.cell_type,
              "validation_best_auroc": best_auroc, "history": history}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "completed", **result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
