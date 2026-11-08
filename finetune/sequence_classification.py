"""Shared VeriComp runner for JanusDNA NT and GenomicBenchmarks tasks."""

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
from torch.utils.data import DataLoader

from finetune.eQTL.eqtl import load_pretrained_backbone
from vericomp import VeriCompConfig
from vericomp.backbone import build_backbone, extract_feature
from vericomp.model_presets import available_model_presets, model_config_from_checkpoint
from finetune.run_config import (print_resolved_config, resolve_model_path,
                                  write_finetune_config)


def multiclass_mcc(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.size == 0:
        return float("nan")
    classes = np.unique(np.concatenate((labels, predictions)))
    matrix = np.zeros((len(classes), len(classes)), dtype=np.float64)
    lookup = {int(value): index for index, value in enumerate(classes)}
    for truth, pred in zip(labels, predictions):
        matrix[lookup[int(truth)], lookup[int(pred)]] += 1.0
    t_sum = matrix.sum(axis=1)
    p_sum = matrix.sum(axis=0)
    n = matrix.sum()
    numerator = n * np.trace(matrix) - np.dot(t_sum, p_sum)
    denominator = np.sqrt((n * n - np.dot(p_sum, p_sum)) * (n * n - np.dot(t_sum, t_sum)))
    return float(numerator / denominator) if denominator > 0 else 0.0


def classification_metrics(labels: np.ndarray, logits: np.ndarray,
                          threshold: float | None = None) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    if threshold is not None and logits.shape[-1] == 2:
        scores = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[:, 1]
        predictions = (scores >= float(threshold)).astype(np.int64)
    else:
        predictions = logits.argmax(axis=-1)
    accuracy = float(np.mean(predictions == labels)) if labels.size else float("nan")
    mcc = multiclass_mcc(labels, predictions)
    f1 = float("nan")
    if logits.shape[-1] == 2:
        truth = labels == 1
        pred = predictions == 1
        tp = float(np.sum(truth & pred)); fp = float(np.sum(~truth & pred)); fn = float(np.sum(truth & ~pred))
        f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    return {"accuracy": accuracy, "mcc": mcc, "f1_binary": f1, "examples": int(labels.size)}


def best_binary_threshold(labels: np.ndarray, logits: np.ndarray) -> tuple[float, float]:
    """Choose a binary decision threshold using only a validation split."""
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape[-1] != 2 or labels.size == 0:
        return 0.5, float("nan")
    scores = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[:, 1]
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_threshold, best_mcc = 0.5, -math.inf
    for threshold in candidates:
        value = classification_metrics(labels, logits, float(threshold))["mcc"]
        if np.isfinite(value) and value > best_mcc:
            best_threshold, best_mcc = float(threshold), float(value)
    return best_threshold, best_mcc


class VeriCompSequenceClassifier(nn.Module):
    def __init__(self, config: VeriCompConfig, num_classes: int, *, freeze_encoder: bool,
                 model_name: str = "vericomp", feature: str = "backbone",
                 pooling: str = "mean", head: str = "linear",
                 head_hidden_dim: int = 256, head_dropout: float = 0.1,
                 dual_strand: bool = False, mask_padding: bool = True,
                 pad_token_id: int = 5):
        super().__init__()
        self.model_name = model_name
        self.mask_padding = bool(mask_padding)
        self.pad_token_id = int(pad_token_id)
        self.feature = feature
        self.pooling = str(pooling).lower()
        self.head_type = str(head).lower()
        self.dual_strand = bool(dual_strand)
        self.backbone = build_backbone(model_name, config)
        if self.pooling not in {"mean", "meanmax", "attention", "convmax"}:
            raise ValueError("pooling must be mean, meanmax, attention, or convmax")
        if self.head_type not in {"linear", "mlp"}:
            raise ValueError("head must be linear or mlp")
        feature_dim = config.d_model
        if self.pooling == "meanmax":
            feature_dim *= 2
        if self.pooling == "attention":
            self.pool_score = nn.Linear(config.d_model, 1)
        if self.pooling == "convmax":
            self.token_pool = nn.Sequential(
                nn.Conv1d(config.d_model, int(head_hidden_dim), kernel_size=7, padding=3),
                nn.GELU(),
                nn.Conv1d(int(head_hidden_dim), int(head_hidden_dim), kernel_size=1),
                nn.GELU(),
            )
            feature_dim = int(head_hidden_dim)
        if self.head_type == "linear":
            self.classifier = nn.Linear(feature_dim, int(num_classes))
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, int(head_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(head_dropout)),
                nn.Linear(int(head_hidden_dim), int(num_classes)),
            )
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids, labels=None):
        if self.freeze_encoder:
            self.backbone.eval()
            with torch.no_grad():
                features, output = extract_feature(
                    self.backbone, input_ids, self.feature,
                    dual_strand=self.dual_strand)
        else:
            features, output = extract_feature(
                self.backbone, input_ids, self.feature,
                dual_strand=self.dual_strand)
        # Sequences are right-aligned with N padding, so a plain mean over
        # positions averages in padding rows that the encoder never saw as real
        # input. On tasks whose true length is well below max_length this
        # dominates the pooled vector, so the mask is applied by default.
        keep = None
        if self.mask_padding:
            keep = (input_ids != self.pad_token_id).unsqueeze(-1).to(features.dtype)
            if keep.shape[1] != features.shape[1]:
                keep = None
        if self.pooling == "mean":
            if keep is None:
                pooled = features.mean(dim=1)
            else:
                pooled = (features * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        elif self.pooling == "meanmax":
            if keep is None:
                pooled = torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=-1)
            else:
                mean_part = (features * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
                neg_inf = torch.finfo(features.dtype).min
                max_part = (features * keep + (keep - 1.0) * neg_inf).amax(dim=1)
                pooled = torch.cat((mean_part, max_part), dim=-1)
        elif self.pooling == "attention":
            scores = self.pool_score(features).squeeze(-1).float()
            weights = torch.softmax(scores, dim=1).to(features.dtype)
            pooled = torch.sum(features * weights.unsqueeze(-1), dim=1)
        else:
            pooled = self.token_pool(features.transpose(1, 2)).amax(dim=-1)
        logits = self.classifier(pooled)
        result = {"logits": logits, "router": output["loss_router"]}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result


def evaluate(model, loader, device, *, threshold: float | None = None,
             return_outputs: bool = False):
    model.eval()
    total_loss = 0.0
    labels_all, logits_all = [], []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(input_ids, labels)
            total_loss += float(output["loss"].float().item()) * labels.numel()
            labels_all.append(labels.cpu().numpy())
            logits_all.append(output["logits"].float().cpu().numpy())
    labels_np = np.concatenate(labels_all) if labels_all else np.empty(0, dtype=np.int64)
    logits_np = np.concatenate(logits_all) if logits_all else np.empty((0, 1), dtype=np.float32)
    metrics = classification_metrics(labels_np, logits_np, threshold)
    metrics["loss"] = total_loss / max(1, labels_np.size)
    if return_outputs:
        return metrics, labels_np, logits_np
    return metrics


def train_task(args, *, task_family: str, dataset_loader):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{task_family} downstream evaluation requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_ds, valid_ds, test_ds, spec = dataset_loader(
        args.data_root, args.dataset, args.fold,
        args.max_train_records, args.max_valid_records, args.max_test_records,
        args.num_folds,
    )
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True),
        "valid": DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True),
    }
    config_overrides = {"force_flash_attention": True,
                        "activation_checkpointing": False}
    if args.allocation_mode is not None:
        config_overrides["allocation_mode"] = args.allocation_mode
    config = model_config_from_checkpoint(
        args.checkpoint, args.model_size, **config_overrides,
    )
    model = VeriCompSequenceClassifier(config, spec["num_classes"],
                                      freeze_encoder=args.freeze_encoder,
                                      model_name=args.model, feature=args.feature,
                                      pooling=args.pooling, head=args.head,
                                      head_hidden_dim=args.head_hidden_dim,
                                      head_dropout=args.head_dropout,
                                      dual_strand=args.dual_strand,
                                      mask_padding=args.mask_padding,
                                      pad_token_id=config.pad_token_id).to(device)
    loaded, skipped = load_pretrained_backbone(model, args.checkpoint)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )
    total_updates = args.epochs * max(1, len(loaders["train"]))
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.plateau_factor,
            patience=args.plateau_patience)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_updates))
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=max(1, total_updates // 10))
    output_dir = Path(args.output_dir) / args.dataset / f"fold-{args.fold}-seed-{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_path(args.model)
    write_finetune_config(output_dir, task=task_family, subtask=args.dataset,
                          args=args, model_path=model_path, model_config=config,
                          extra={"data_root": str(Path(args.data_root).resolve()),
               "fold": args.fold, "num_folds": args.num_folds})
    print_resolved_config(
        task=task_family, subtask=args.dataset, args=args, model_path=model_path,
        model_config=config,
        metadata={
            "checkpoint": {
                "path": str(Path(args.checkpoint).expanduser().resolve()),
                "loaded_backbone_tensors": loaded,
                "skipped_backbone_tensors": skipped,
            },
            "data": {
                "train_examples": len(train_ds),
                "valid_examples": len(valid_ds),
                "test_examples": len(test_ds),
                "max_length": spec["max_length"],
                "num_classes": spec["num_classes"],
                "metric": spec["metric"],
                "num_workers": args.num_workers,
            },
            "head": {
                "pooling": args.pooling,
                "head": args.head,
                "head_hidden_dim": args.head_hidden_dim,
                "head_dropout": args.head_dropout,
            },
            "optimization": {
                "optimizer": "AdamW",
                "scheduler": args.scheduler,
                "train_batch_size": args.batch_size,
                "effective_batch_size": args.batch_size,
                "total_updates": total_updates,
            },
            "parameters": {
                "total": sum(p.numel() for p in model.parameters()),
                "trainable": sum(p.numel() for p in params),
            },
            "runtime": {
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "autocast": "bfloat16" if device.type == "cuda" else "disabled",
            },
        },
    )

    best_value = -math.inf
    best_threshold = 0.5
    history = []
    test_history = []
    global_step = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(True)
        if model.freeze_encoder:
            model.backbone.eval()
        running_loss = 0.0; seen = 0
        for batch_index, batch in enumerate(loaders["train"], start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(input_ids, labels)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if args.scheduler != "plateau":
                scheduler.step()
            global_step += 1
            running_loss += float(output["loss"].detach().float()) * labels.numel(); seen += labels.numel()
            if args.test_every > 0 and global_step % args.test_every == 0:
                test_metrics = evaluate(model, loaders["test"], device)
                test_record = {"epoch": epoch, "step": global_step, "test": test_metrics}
                test_history.append(test_record); print(json.dumps(test_record), flush=True)
        valid_raw, valid_labels, valid_logits = evaluate(
            model, loaders["valid"], device, return_outputs=True
        )
        threshold = 0.5
        valid = valid_raw
        if args.tune_threshold and spec["num_classes"] == 2:
            threshold, _ = best_binary_threshold(valid_labels, valid_logits)
            valid = classification_metrics(valid_labels, valid_logits, threshold)
            valid["loss"] = valid_raw["loss"]
            valid["threshold"] = threshold
        train_record = {"epoch": epoch, "step": global_step,
                        "train": {"loss": running_loss / max(seen, 1), "examples": seen},
                        "valid": valid}
        if args.tune_threshold and spec["num_classes"] == 2:
            test_tuned = evaluate(model, loaders["test"], device, threshold=threshold)
            train_record["test_tuned"] = test_tuned
            print(json.dumps({"epoch": epoch, "step": global_step,
                              "test_tuned": test_tuned,
                              "threshold": threshold}), flush=True)
        if args.test_every_epoch:
            epoch_test = evaluate(model, loaders["test"], device)
            epoch_record = {"epoch": epoch, "step": global_step, "test": epoch_test}
            test_history.append(epoch_record)
            print(json.dumps(epoch_record), flush=True)
        if args.scheduler == "plateau":
            scheduler.step(float(valid_raw["loss"]))
        train_record["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        history.append(train_record); print(json.dumps(train_record), flush=True)
        value = float(valid.get(spec["metric"], valid["mcc"]))
        if np.isfinite(value) and value > best_value:
            best_value = value
            best_threshold = threshold
            torch.save({"model": model.state_dict(), "epoch": epoch, "valid": valid,
                        "threshold": best_threshold, "args": vars(args)}, output_dir / "best.pt")

    best_path = output_dir / "best.pt"
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
        best_threshold = float(best_checkpoint.get("threshold", best_threshold))
    final_test = evaluate(model, loaders["test"], device,
                          threshold=best_threshold if args.tune_threshold else None)
    result = {"task": task_family, "dataset": args.dataset, "metric": spec["metric"],
              "valid_best": best_value, "test": final_test,
              "threshold": best_threshold if args.tune_threshold else None,
              "history": history,
              "test_history": test_history, "elapsed_s": time.perf_counter() - started}
    torch.save({"model": model.state_dict(), "result": result, "args": vars(args)}, output_dir / "last.pt")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"test": final_test, "output_dir": str(output_dir)}), flush=True)


def build_parser(description):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-size", choices=("auto", *available_model_presets()),
                        default="auto",
                        help="architecture preset; auto reads new checkpoint metadata")
    parser.add_argument("--model", default="vericomp",
                        help="model .py path, /path/model.py:ClassName, or built-in alias")
    parser.add_argument("--feature", choices=("backbone", "decoder", "base"), default="backbone")
    parser.add_argument("--pooling", choices=("mean", "meanmax", "attention", "convmax"), default="mean")
    parser.add_argument("--head", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--dual-strand", action="store_true")
    parser.add_argument("--mask-padding", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="exclude N-padding positions from pooling; GB tasks whose "
                             "true length is far below max_length are sensitive to this")
    parser.add_argument("--allocation-mode", choices=("dynamic", "uniform", "length_only", "grid"),
                        default=None,
                        help="override the allocation mode read from the checkpoint; "
                             "checkpoints written before model_config existed default "
                             "to dynamic, so ablation runs must pass this explicitly")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", type=int, default=0,
                        help="0-based cross-validation fold used as the validation split")
    parser.add_argument("--num-folds", type=int, default=5,
                        help="total number of cross-validation folds")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler", choices=("linear", "plateau", "cosine"), default="linear")
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--test-every", type=int, default=100,
                        help="run a full test sweep every N optimizer steps; 0 disables it")
    parser.add_argument("--test-every-epoch", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="run one full test sweep at the end of each epoch; use this "
                             "instead of a small --test-every on large GB test splits, "
                             "where periodic sweeps dominate wall-clock time")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-valid-records", type=int, default=None)
    parser.add_argument("--max-test-records", type=int, default=None)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    return parser
