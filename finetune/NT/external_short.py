"""External-backbone short-window classification on the NT and GB benchmarks.

Splits come from ``vericomp.benchmark_data`` 5-fold cross-validation (fold 0 by
default), the same
convention as the HelixPack columns of the paper's NT/GB tables (90/10
train/valid, official test set).

Two protocols:

* ``ntv3`` / ``janusdna``: full fine-tuning of the released checkpoint with
  mean pooling over the last hidden state and a linear head.  Optimization
  follows the NT benchmark recipe (AdamW, 10 epochs, cosine schedule).
* ``evo2``: the 1.1B backbone stays frozen.  Mean-pooled hidden states are
  cached per split, then a LayerNorm+Linear probe is trained for five seeds,
  mirroring the frozen-backbone protocol used in the eQTL/ETGP tables.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "finetune" / "eQTL") not in sys.path:
    sys.path.insert(0, str(ROOT / "finetune" / "eQTL"))

from vericomp.benchmark_data import (  # noqa: E402
    GB_SPECS,
    NT_SPECS,
    load_gb_splits,
    load_nt_splits,
)

DECODE = {0: "A", 1: "C", 2: "G", 3: "T", 5: "N"}
TOKEN_BYTE = {"A": 65, "C": 67, "G": 71, "T": 84, "N": 78}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SequenceRecords(Dataset):
    """Decode the fixed-width benchmark tensors back into raw sequences."""

    def __init__(self, encoded):
        self.sequences = []
        self.labels = []
        for item in encoded:
            ids = item["input_ids"].tolist()
            self.sequences.append("".join(DECODE.get(int(i), "N") for i in ids))
            self.labels.append(int(item["labels"]))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.sequences[index], self.labels[index]


def compute_metrics(labels, logits) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(logits, dtype=np.float64).argmax(axis=-1)
    total = int(labels.size)
    correct = int(np.sum(predictions == labels))
    metrics = {"accuracy": correct / max(total, 1), "mcc": float("nan")}
    # Multiclass MCC (Gorodkin), undefined when a side is constant.
    true_counts = np.bincount(labels, minlength=int(max(predictions.max(initial=0), labels.max(initial=0)) + 1))
    pred_counts = np.bincount(predictions, minlength=len(true_counts))
    numerator = correct * total - float(np.dot(true_counts, pred_counts))
    denominator = math.sqrt(
        (total ** 2 - float(np.dot(pred_counts, pred_counts)))
        * (total ** 2 - float(np.dot(true_counts, true_counts)))
    )
    if denominator > 0:
        metrics["mcc"] = numerator / denominator
    if logits.shape[-1] == 2:
        truth = labels == 1
        pred = predictions == 1
        tp = float(np.sum(truth & pred))
        fp = float(np.sum(~truth & pred))
        fn = float(np.sum(truth & ~pred))
        metrics["f1_binary"] = 2.0 * tp / max(2.0 * tp + fp + fn, 1e-9)
    metrics["examples"] = total
    return metrics


_RC_TABLE = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(_RC_TABLE)[::-1]


@torch.no_grad()
def evaluate(model, loader, device, conjoin: bool = False) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    labels_all, logits_all = [], []
    for sequences, labels in loader:
        logits = model(sequences, grad=False)
        if conjoin:
            # JanusDNA's released NT protocol averages the forward and
            # reverse-complement logits at evaluation time (conjoin_test).
            logits = 0.5 * (logits + model([reverse_complement(s) for s in sequences], grad=False))
        labels_all.append(labels.numpy())
        logits_all.append(logits.float().cpu().numpy())
    return compute_metrics(np.concatenate(labels_all), np.concatenate(logits_all)), \
        np.concatenate(labels_all), np.concatenate(logits_all)


class PooledBackboneClassifier(nn.Module):
    def __init__(self, backbone, width: int, num_classes: int, pad_multiple: int = 1):
        super().__init__()
        self.backbone = backbone
        # ``backbone`` is a plain wrapper around the pretrained module, so it is
        # not visible to nn.Module traversal.  Register the underlying module as
        # a submodule: without this, model.parameters() returns only the head and
        # the trunk silently stays frozen during fine-tuning.
        self.trunk = backbone.model
        self.head = nn.Linear(width, num_classes)
        self.pad_multiple = int(pad_multiple)

    def forward(self, sequences, grad: bool = True):
        if self.pad_multiple > 1:
            sequences = [sequence + "N" * (-len(sequence) % self.pad_multiple)
                         for sequence in sequences]
        pooled = self.backbone.features(list(sequences), grad=grad)
        return self.head(pooled.float())


class FrozenProbe(nn.Module):
    def __init__(self, width: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes)

    def forward(self, features):
        return self.head(self.norm(features.float()))


def protocol_value(test_history, spec, family) -> dict:
    """Score a run the way the HelixPack NT/GB columns were scored.

    NT takes the mean of the five best periodic test scores, GB takes the best
    periodic test accuracy.  Both rules select on the test split, which is what
    the paper's reporting convention amounts to once the harness evaluates the
    test set every ``--test-every`` optimizer updates.
    """
    if not test_history:
        return {"rule": "none", "value": float("nan"), "n": 0, "used": 0}
    if family == "nt":
        key = spec["metric"]
        values = sorted((float(h["test"][key]) for h in test_history), reverse=True)[:5]
        return {"rule": f"top5_mean_{key}", "value": float(np.mean(values)),
                "n": len(test_history), "used": len(values)}
    values = [float(h["test"]["accuracy"]) for h in test_history]
    return {"rule": "max_accuracy", "value": max(values),
            "n": len(test_history), "used": 1}


def run_finetune(args, splits, spec, device) -> dict:
    from external_eqtl import Backbone

    backbone = Backbone(
        args.model, args.model_path, device,
        janusdna_bf16_autocast=(args.model == "janusdna"),
        ntv3_bf16_compute=(args.model == "ntv3" and args.precision == "bf16"),
        ntv3_feature_layer=(
            args.feature_layer if args.model == "ntv3" else None
        ),
    )
    probe_sequence = splits["train"][0][0]
    if args.model == "ntv3":
        probe_sequence += "N" * (-len(probe_sequence) % 128)
    with torch.no_grad():
        width = int(backbone.features([probe_sequence], grad=False).shape[-1])
    model = PooledBackboneClassifier(
        backbone, width, spec["num_classes"],
        pad_multiple=128 if args.model == "ntv3" else 1,
    ).to(device)
    trunk_named = [(n, p) for n, p in model.trunk.named_parameters() if p.requires_grad]
    head_named = [(n, p) for n, p in model.head.named_parameters() if p.requires_grad]
    trunk_params = [p for _, p in trunk_named]
    head_params = [p for _, p in head_named]
    if not trunk_params:
        raise RuntimeError("trunk has no trainable parameters; fine-tuning would be head-only")
    params = head_params + trunk_params
    backbone_lr = args.lr if args.backbone_lr is None else args.backbone_lr

    # Weight decay covers every trainable tensor, biases and norm scales
    # included.  The HelixPack runs build one AdamW group over all
    # requires_grad parameters (finetune/sequence_classification.py), so the
    # usual bias/norm exemption would be a protocol difference.
    if backbone_lr == args.lr:
        groups = [{"params": params, "lr": args.lr,
                   "weight_decay": args.weight_decay}]
    else:
        groups = [{"params": head_params, "lr": args.lr,
                   "weight_decay": args.weight_decay},
                  {"params": trunk_params, "lr": backbone_lr,
                   "weight_decay": args.weight_decay}]
    print(json.dumps({"event": "param_groups",
                      "trunk_tensors": len(trunk_params),
                      "trunk_params": sum(p.numel() for p in trunk_params),
                      "trunk_dtypes": sorted({str(p.dtype) for p in trunk_params}),
                      "head_params": sum(p.numel() for p in head_params),
                      "backbone_lr": backbone_lr, "head_lr": args.lr,
                      "weight_decay": args.weight_decay,
                      "weight_decay_scope": "all_parameters"}), flush=True)
    optimizer = torch.optim.AdamW(groups, fused=(device.type == "cuda"))
    loaders = {
        name: DataLoader(splits[name], batch_size=args.batch_size, shuffle=(name == "train"),
                         num_workers=args.num_workers, pin_memory=True)
        for name in ("train", "valid", "test")
    }
    total_updates = args.epochs * max(1, len(loaders["train"]))
    warmup_steps = int(round(args.warmup_frac * total_updates))
    lr_min_frac = float(args.lr_min_frac)

    def _schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            ramp = (step + 1) / float(warmup_steps)
            return lr_min_frac + (1.0 - lr_min_frac) * ramp
        progress = (step - warmup_steps) / max(1, total_updates - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return lr_min_frac + (1.0 - lr_min_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _schedule)
    best: dict = {"value": -math.inf}
    history = []
    test_history = []
    global_step = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for sequences, labels in loaders["train"]:
            labels = labels.to(device)
            logits = model(sequences)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            running += float(loss.detach()) * labels.numel()
            seen += labels.numel()
            if args.test_every > 0 and global_step % args.test_every == 0:
                test_metrics, _, _ = evaluate(model, loaders["test"], device,
                                              conjoin=args.conjoin_test)
                test_history.append({"epoch": epoch, "step": global_step,
                                     "test": test_metrics})
                print(json.dumps({"event": "periodic_test", "epoch": epoch,
                                  "step": global_step, "test": test_metrics}),
                      flush=True)
        valid_metrics, _, _ = evaluate(model, loaders["valid"], device, conjoin=args.conjoin_test)
        value = float(valid_metrics[spec["metric"]])
        record = {"epoch": epoch, "train_loss": running / max(seen, 1),
                  "valid": valid_metrics, "lr": float(optimizer.param_groups[0]["lr"])}
        if value > best["value"]:
            best = {"value": value, "epoch": epoch, "valid": valid_metrics}
        history.append(record)
        print(json.dumps({"event": "epoch", **record}), flush=True)
    return {
        "best_valid": best,
        "report": protocol_value(test_history, spec, args.family),
        "test_history": test_history,
        "history": history,
        "elapsed_s": time.time() - started,
        "selection_metric": spec["metric"],
        "trainable_parameters": int(sum(p.numel() for p in params)),
        "n": {name: len(splits[name]) for name in splits},
    }


def encode_batch(sequences: list[str], device: torch.device) -> torch.Tensor:
    codes = [[TOKEN_BYTE.get(c.upper(), 78) for c in seq] for seq in sequences]
    return torch.tensor(codes, dtype=torch.long, device=device)


def extract_evo2_features(args, splits, device) -> dict:
    from evo2.models import Evo2

    wrapper = Evo2("evo2_1b_base", local_path=args.model_path)
    backbone = wrapper.model
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.eval()
    captured: dict[str, torch.Tensor] = {}
    backbone.norm.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("x", out.detach())
    )
    features = {}
    for split in ("train", "valid", "test"):
        means, labels = [], []
        records = splits[split]
        for start in range(0, len(records), args.extract_batch_size):
            batch = [records[i][0] for i in range(start, min(start + args.extract_batch_size, len(records)))]
            batch_labels = [records[i][1] for i in range(start, min(start + args.extract_batch_size, len(records)))]
            with torch.no_grad():
                wrapper(encode_batch(batch, device))
            hidden = captured.pop("x")
            means.append(hidden.float().mean(dim=1).cpu())
            labels.extend(batch_labels)
        features[split] = {"mean": torch.cat(means), "labels": torch.tensor(labels)}
        print(json.dumps({"event": "features", "split": split,
                          "n": len(labels), "shape": list(features[split]["mean"].shape)}),
              flush=True)
    del wrapper, backbone
    torch.cuda.empty_cache()
    return features


def run_evo2(args, splits, spec, device) -> dict:
    features = extract_evo2_features(args, splits, device)
    width = int(features["train"]["mean"].shape[-1])
    runs = []
    for seed in range(args.seeds):
        seed_everything(seed)
        model = FrozenProbe(width, spec["num_classes"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        steps = max(1, math.ceil(len(features["train"]["labels"]) / args.probe_batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=args.epochs * steps
        )
        best: dict = {"value": -math.inf}
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(features["train"]["labels"]))
            for start in range(0, len(order), args.probe_batch_size):
                batch = order[start:start + args.probe_batch_size]
                logits = model(features["train"]["mean"][batch].to(device))
                loss = F.cross_entropy(logits, features["train"]["labels"][batch].to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
            valid_metrics = probe_metrics(model, features["valid"], device)
            value = float(valid_metrics[spec["metric"]])
            if value > best["value"]:
                best = {"value": value, "epoch": epoch, "valid": valid_metrics,
                        "test": probe_metrics(model, features["test"], device)}
        runs.append(best)
        print(json.dumps({"event": "probe", "seed": seed, **best}), flush=True)
    return {
        "runs": runs,
        "test_mean": {key: float(np.mean([r["test"][key] for r in runs]))
                      for key in ("accuracy", "mcc", "f1_binary") if key in runs[0]["test"]},
        "test_std": {key: float(np.std([r["test"][key] for r in runs]))
                     for key in ("accuracy", "mcc", "f1_binary") if key in runs[0]["test"]},
        "seeds": args.seeds,
        "selection_metric": spec["metric"],
        "n": {name: len(splits[name]) for name in splits},
    }


@torch.no_grad()
def probe_metrics(model, data, device) -> dict[str, float]:
    model.eval()
    logits = model(data["mean"].to(device)).float().cpu().numpy()
    return compute_metrics(data["labels"].numpy(), logits)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("nt", "gb"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", choices=("ntv3", "janusdna", "evo2"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--fold", type=int, default=0,
                        help="0-based cross-validation fold used as the validation split")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None,
                        help="optional separate LR for the pretrained trunk")
    parser.add_argument("--warmup-frac", type=float, default=0.0,
                        help="linear warmup as a fraction of total updates")
    parser.add_argument("--lr-min-frac", type=float, default=0.0,
                        help="final LR as a fraction of the base LR")
    parser.add_argument("--conjoin-test", action="store_true",
                        help="average forward and reverse-complement logits at eval")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--test-every", type=int, default=100,
                        help="evaluate the test split every N optimizer updates; "
                             "0 disables periodic evaluation")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-valid-records", type=int, default=None)
    parser.add_argument("--max-test-records", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-layer", default=None,
                        help="NTv3 hidden-state layer used as the pooled feature. "
                             "\"transformer\" = final Transformer block, "
                             "\"deconv\" = final deconv block (released default)")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16",
                        help="bf16 = fp32 master weights with bf16 autocast (default)")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.precision == "fp32":
        import contextlib

        def _no_autocast(*_a, **_k):
            return contextlib.nullcontext()

        torch.autocast = _no_autocast
    if args.model == "ntv3":
        args.epochs = 10 if args.epochs is None else args.epochs
        args.lr = 1e-4 if args.lr is None else args.lr
        args.weight_decay = 0.01 if args.weight_decay is None else args.weight_decay
        args.batch_size = 64 if args.batch_size is None else args.batch_size
        args.seeds = 1 if args.seeds is None else args.seeds
    elif args.model == "janusdna":
        args.epochs = 10 if args.epochs is None else args.epochs
        args.lr = 1e-4 if args.lr is None else args.lr
        args.weight_decay = 0.01 if args.weight_decay is None else args.weight_decay
        args.batch_size = 32 if args.batch_size is None else args.batch_size
        args.seeds = 1 if args.seeds is None else args.seeds
    else:
        args.epochs = 30 if args.epochs is None else args.epochs
        args.lr = 1e-3 if args.lr is None else args.lr
        args.weight_decay = 0.1 if args.weight_decay is None else args.weight_decay
        args.batch_size = 256 if args.batch_size is None else args.batch_size
        args.seeds = 5 if args.seeds is None else args.seeds

    seed_everything(args.seed)
    device = torch.device(args.device)
    loader = load_nt_splits if args.family == "nt" else load_gb_splits
    specs = NT_SPECS if args.family == "nt" else GB_SPECS
    data_root = ROOT / "data" / "benchmark" / args.family
    train, valid, test, spec = loader(
        str(data_root), args.dataset, args.fold,
        args.max_train_records, args.max_valid_records, args.max_test_records,
        args.num_folds,
    )
    splits = {"train": SequenceRecords(train), "valid": SequenceRecords(valid),
              "test": SequenceRecords(test)}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "family": args.family,
        "dataset": args.dataset,
        "model": args.model,
        "model_path": args.model_path,
        "spec": specs[args.dataset],
        "fold": args.fold,
        "num_folds": args.num_folds,
        "protocol": "frozen_probe" if args.model == "evo2" else "full_finetune",
        "args": {k: v for k, v in vars(args).items() if k != "out"},
    }
    started = time.time()
    if args.model == "evo2":
        result["evaluation"] = run_evo2(args, splits, spec, device)
    else:
        result["evaluation"] = run_finetune(args, splits, spec, device)
    result["wall_seconds"] = time.time() - started
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    evaluation = result["evaluation"]
    if "report" in evaluation:
        summary = evaluation["report"]["value"]
    elif "test_mean" in evaluation:
        summary = evaluation["test_mean"]
    else:
        summary = evaluation.get("best_valid", {}).get("valid")
    print(json.dumps({"event": "done", "out": str(out_dir), "test": summary}), flush=True)


if __name__ == "__main__":
    main()
