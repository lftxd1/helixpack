"""VeriComp DNALongBench TISP sequence-regression task."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from finetune.eQTL.eqtl import load_pretrained_backbone
from finetune.TISP.chromosome_eval import ChromosomeTISP, EVAL_STRIDE, evaluate_chromosome_level
from vericomp import VeriCompConfig
from vericomp.model_presets import available_model_presets, model_config_from_checkpoint
from vericomp.backbone import build_backbone, extract_feature
from vericomp.tisp_data import TISP_SEQUENCE_LENGTH, TISP_WINDOW_STEP, TISPRandomDataset
from finetune.run_config import resolve_model_path, write_finetune_config


TRACK_NAMES = ("cage_plus", "encodecage_plus", "encoderampage_plus", "grocap_plus",
               "procap_plus", "procap_minus", "grocap_minus", "encoderampage_minus",
               "encodecage_minus", "cage_minus")

# Official DNALongBench scoring: one PCC per assay after merging its plus and
# minus strand tracks, followed by the mean of the five assay-level PCCs.
# The third field is the transform used by GeneZip before calculating PCC.
TISP_ASSAY_GROUPS = {
    "fc": (0, 9, "cage_like"),  # FANTOM CAGE
    "ec": (1, 8, "cage_like"),  # ENCODE CAGE
    "er": (2, 7, "cage_like"),  # ENCODE RAMPAGE
    "gc": (3, 6, "natlog_like"),  # GRO-cap
    "pc": (4, 5, "natlog_like"),  # PRO-cap
}
TISP_EVAL_TRIM_BP = (TISP_SEQUENCE_LENGTH - TISP_WINDOW_STEP) // 2
TISP_VALID_MASK_FLANK_BP = 500



def _pcc(pred: np.ndarray, target: np.ndarray) -> float:
    valid = np.isfinite(pred) & np.isfinite(target)
    if valid.sum() < 2:
        return float("nan")
    x, y = pred[valid].astype(np.float64), target[valid].astype(np.float64)
    x -= x.mean(); y -= y.mean()
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denom) if denom > 0 else float("nan")


def _assay_pcc(pred: np.ndarray, target: np.ndarray,
               plus_index: int, minus_index: int) -> float:
    """Compute PCC after concatenating one assay's plus/minus tracks."""
    merged_pred = np.concatenate((
        pred[..., plus_index].reshape(-1),
        pred[..., minus_index].reshape(-1),
    ))
    merged_target = np.concatenate((
        target[..., plus_index].reshape(-1),
        target[..., minus_index].reshape(-1),
    ))
    return _pcc(merged_pred, merged_target)


class _PearsonAccumulator:
    """Streaming Pearson statistics, avoiding a full test-set tensor copy."""

    def __init__(self):
        self.count = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, x: np.ndarray, y: np.ndarray):
        valid = np.isfinite(x) & np.isfinite(y)
        if not bool(valid.any()):
            return
        x = x[valid].astype(np.float64, copy=False)
        y = y[valid].astype(np.float64, copy=False)
        self.count += int(x.size)
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_x2 += float(np.dot(x, x))
        self.sum_y2 += float(np.dot(y, y))
        self.sum_xy += float(np.dot(x, y))

    def compute(self) -> float:
        if self.count <= 1:
            return float("nan")
        n = float(self.count)
        cov = self.sum_xy - self.sum_x * self.sum_y / n
        var_x = self.sum_x2 - self.sum_x * self.sum_x / n
        var_y = self.sum_y2 - self.sum_y * self.sum_y / n
        return float(cov / math.sqrt(var_x * var_y)) if var_x > 0 and var_y > 0 else float("nan")


def _transform_tisp_values(values: torch.Tensor, transform_name: str) -> torch.Tensor:
    if transform_name == "cage_like":
        return torch.log10(torch.pow(10.0, values) - 1.0 + 0.1)
    if transform_name == "natlog_like":
        return values / math.log(10.0)
    raise ValueError(f"unknown TISP transform {transform_name!r}")


def _pseudo_poisson_kl(
    log_rates: torch.Tensor, target: torch.Tensor, *, all_bf16: bool = False
) -> torch.Tensor:
    if all_bf16:
        rates = torch.exp(log_rates)
        target = target.to(dtype=log_rates.dtype)
        eps = torch.tensor(1e-5, dtype=log_rates.dtype, device=log_rates.device)
    else:
        rates = torch.exp(log_rates.float())
        target = target.float()
        eps = 1e-5
    return (target * torch.log((target + eps) / (rates + eps)) + rates - target).mean()


class VeriCompTISPModel(nn.Module):
    def __init__(self, config: VeriCompConfig, *, freeze_encoder: bool,
                 model_name: str = "vericomp", feature: str = "backbone",
                 router_loss_coef: float = 0.0, dual_strand: bool = False,
                 all_bf16: bool = False, fp32_head: bool = False):
        super().__init__()
        self.model_name = model_name
        self.feature = feature
        self.router_loss_coef = float(router_loss_coef)
        self.dual_strand = bool(dual_strand)
        self.all_bf16 = bool(all_bf16)
        self.fp32_head = bool(fp32_head)
        self.backbone = build_backbone(model_name, config)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, len(TRACK_NAMES))
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, target=None):
        if self.freeze_encoder:
            self.backbone.eval()
            with torch.no_grad():
                features, output = extract_feature(
                    self.backbone, input_ids, self.feature,
                    dual_strand=self.dual_strand,
                )
        else:
            features, output = extract_feature(
                self.backbone, input_ids, self.feature,
                dual_strand=self.dual_strand,
            )
        if features.shape[1] != TISP_SEQUENCE_LENGTH:
            features = F.interpolate(
                features.transpose(1, 2), size=TISP_SEQUENCE_LENGTH,
                mode="linear", align_corners=False,
            ).transpose(1, 2)
        if self.fp32_head:
            # Keep the readout out of the ambient BF16 autocast so its small
            # FP32 parameters receive FP32 matmuls.
            with torch.autocast(device_type="cuda", enabled=False):
                log_rates = self.head(self.norm(features.float()))
        else:
            log_rates = self.head(self.norm(features))
        pred = torch.exp(log_rates if self.all_bf16 else log_rates.float())
        router = output["loss_router"]
        result = {"log_rates": log_rates, "pred": pred, "router": router}
        if target is not None:
            valid = torch.isfinite(target) & torch.isfinite(log_rates)
            task_loss = (
                _pseudo_poisson_kl(log_rates[valid], target[valid], all_bf16=self.all_bf16)
                if bool(valid.any()) else log_rates.sum() * 0.0
            )
            router_term = router.to(task_loss.dtype) * self.router_loss_coef
            result["loss_task"] = task_loss
            result["loss_router_term"] = router_term
            result["loss"] = task_loss + router_term
        return result


def evaluate(model, loader, device):
    model.eval()
    losses = []
    examples = 0
    accumulators = {name: _PearsonAccumulator() for name in TISP_ASSAY_GROUPS}
    trim = TISP_EVAL_TRIM_BP
    center_stop = TISP_SEQUENCE_LENGTH - trim
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(input_ids, target)
            losses.append(float(output["loss"].float().item()))
            examples += int(input_ids.shape[0])
            pred = output["pred"].float()[:, trim:center_stop]
            true = target.float()[:, trim:center_stop]

            # Match GeneZip's N masking: exclude each N and a +/-500 bp flank.
            invalid = input_ids.eq(5).float().unsqueeze(1)
            invalid = F.max_pool1d(invalid, kernel_size=2 * TISP_VALID_MASK_FLANK_BP + 1,
                                   stride=1, padding=TISP_VALID_MASK_FLANK_BP).squeeze(1).bool()
            valid = ~invalid[:, trim:center_stop]
            for name, (plus_index, minus_index, transform_name) in TISP_ASSAY_GROUPS.items():
                pred_plus = _transform_tisp_values(pred[..., plus_index], transform_name)
                pred_minus = _transform_tisp_values(pred[..., minus_index], transform_name)
                true_plus = _transform_tisp_values(true[..., plus_index], transform_name)
                true_minus = _transform_tisp_values(true[..., minus_index], transform_name)
                mask = valid
                accumulators[name].update(
                    torch.cat((pred_plus[mask], pred_minus[mask])).cpu().numpy(),
                    torch.cat((true_plus[mask], true_minus[mask])).cpu().numpy(),
                )
    pcc = {f"{name}_pcc": accumulator.compute() for name, accumulator in accumulators.items()}
    finite = [value for value in pcc.values() if np.isfinite(value)]
    pcc["avg_pcc"] = float(np.mean(finite)) if finite else float("nan")
    return {"loss": float(np.mean(losses)) if losses else float("nan"), **pcc,
            "examples": examples}


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-size", choices=("auto", *available_model_presets()),
                        default="auto",
                        help="architecture preset; auto reads new checkpoint metadata")
    parser.add_argument("--model", default="vericomp",
                        help="model .py path, /path/model.py:ClassName, or built-in alias")
    parser.add_argument("--feature", choices=("backbone", "decoder", "base"), default="decoder")
    parser.add_argument("--param-dtype", choices=("fp32", "bf16"), default="fp32",
                        help="storage dtype for backbone weights; the LayerNorm and head stay FP32")
    parser.add_argument(
        "--dual-strand", action=argparse.BooleanOptionalAction, default=False,
        help="use the reverse-complement path during TISP backbone inference",
    )
    parser.add_argument("--data-root", default="./data/dnalongbench")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32,
                        help="separate validation/test batch size; inference uses no gradients")
    parser.add_argument("--eval-num-workers", type=int, default=4,
                        help="BigWig-prefetch workers used only by validation/test")
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=25000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--num-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("warmup", "linear"), default="warmup")
    parser.add_argument("--scheduler-start-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--router-loss-coef", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--allocation-mode",
                        choices=("dynamic", "uniform", "length_only", "grid"),
                        default=None,
                        help="override allocation mode; pre-model_config "
                             "checkpoints default to dynamic")
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default=str(Path("weights/finetune/TISP") / time.strftime("%Y%m%d")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--init-head-bias", choices=("none", "mean"), default="none")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("TISP downstream evaluation requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_ds = TISPRandomDataset(args.data_root, args.num_train_samples, seed=args.seed)
    valid_ds = ChromosomeTISP(args.data_root, ["chr10", "10"])
    test_ds = ChromosomeTISP(args.data_root, ["chr8", "8", "chr9", "9"])
    if args.max_valid_samples:
        valid_ds.windows = valid_ds.windows[: int(args.max_valid_samples)]
    if args.max_test_samples:
        test_ds.windows = test_ds.windows[: int(args.max_test_samples)]
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, num_workers=0),
        "valid": DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.eval_num_workers, pin_memory=True,
                            persistent_workers=args.eval_num_workers > 0),
        "test": DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.eval_num_workers, pin_memory=True,
                            persistent_workers=args.eval_num_workers > 0),
    }
    config_overrides = {"force_flash_attention": True,
                        "activation_checkpointing": False}
    if getattr(args, "allocation_mode", None) is not None:
        config_overrides["allocation_mode"] = args.allocation_mode
    config = model_config_from_checkpoint(
        args.checkpoint,
        args.model_size,
        **config_overrides,
    )
    model = VeriCompTISPModel(config, freeze_encoder=args.freeze_encoder,
                             model_name=args.model, feature=args.feature,
                             router_loss_coef=args.router_loss_coef,
                             dual_strand=args.dual_strand).to(device)
    model.use_bf16 = device.type == "cuda"
    loaded, skipped = load_pretrained_backbone(model, args.checkpoint)
    if args.param_dtype == "bf16":
        model.backbone.to(dtype=torch.bfloat16)
        model.fp32_head = True
    if args.init_head_bias == "mean":
        totals = None
        seen = 0
        for index, batch in enumerate(loaders["train"]):
            if index >= 32:
                break
            target = batch["target"].to(device).float()
            batch_sum = target.sum(dim=(0, 1))
            totals = batch_sum if totals is None else totals + batch_sum
            seen += target.shape[0] * target.shape[1]
        per_track_mean = (totals / max(seen, 1)).clamp_min(1e-8)
        with torch.no_grad():
            model.head.bias.copy_(torch.log(per_track_mean).to(model.head.bias.dtype))
        print(json.dumps({
            "event": "head_bias_init",
            "per_track_mean": [float(value) for value in per_track_mean.cpu()],
        }), flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )
    warmup_steps = int(args.train_steps * args.warmup_ratio)
    if args.scheduler == "linear":
        def lr_scale(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return args.scheduler_start_factor + (1.0 - args.scheduler_start_factor) * (step + 1) / warmup_steps
            decay_steps = max(1, args.train_steps - warmup_steps)
            return max(0.0, 1.0 - (step - warmup_steps) / decay_steps)
    elif warmup_steps:
        def lr_scale(step: int) -> float:
            if step < warmup_steps:
                return args.scheduler_start_factor + (1.0 - args.scheduler_start_factor) * (step + 1) / warmup_steps
            return 1.0
    else:
        def lr_scale(step: int) -> float:
            return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_path(args.model)
    write_finetune_config(output_dir, task="TISP", subtask="TISP",
                          args=args, model_path=model_path, model_config=config,
                          extra={"data_root": str(Path(args.data_root).resolve())})
    trainable_parameters = sum(p.numel() for p in params)
    total_parameters = sum(p.numel() for p in model.parameters())
    resolved_config = {
        "event": "resolved_config_before_training",
        "task": "TISP",
        "subtask": "TISP",
        "arguments": vars(args),
        "model": {
            "model": args.model,
            "model_path": model_path,
            "feature": args.feature,
            "config": dataclasses.asdict(config),
            "dual_strand": args.dual_strand,
            "activation_checkpointing": config.activation_checkpointing,
            "freeze_encoder": args.freeze_encoder,
        },
        "checkpoint": {
            "path": str(Path(args.checkpoint).expanduser().resolve()),
            "loaded_backbone_tensors": loaded,
            "skipped_backbone_tensors": skipped,
        },
        "data": {
            "root": str(Path(args.data_root).expanduser().resolve()),
            "train_examples": len(train_ds),
            "valid_examples": len(valid_ds),
            "test_examples": len(test_ds),
            "sequence_length": TISP_SEQUENCE_LENGTH,
            "window_step": EVAL_STRIDE,
            "train_batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "num_workers": args.eval_num_workers,
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "router_loss_coef": args.router_loss_coef,
            "max_grad_norm": args.max_grad_norm,
            "grad_accumulation": args.grad_accumulation,
            "effective_batch_size": args.batch_size * args.grad_accumulation,
            "train_steps": args.train_steps,
            "eval_every": args.eval_every,
        },
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "autocast": "bfloat16" if device.type == "cuda" else "disabled",
            "evaluator": "chromosome_weighted_overlap_stitching_paired_assay_pcc",
        },
    }
    print(json.dumps(resolved_config, indent=2, ensure_ascii=False), flush=True)

    best_pcc = -math.inf
    history = []
    iterator = iter(loaders["train"])
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    for step in range(1, args.train_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loaders["train"]); batch = next(iterator)
        model.train(not args.freeze_encoder)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(input_ids, target)
            loss = output["loss"] / args.grad_accumulation
        loss.backward()
        if step % args.grad_accumulation == 0 or step == args.train_steps:
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 50 == 0 or step == args.train_steps:
            elapsed = max(time.perf_counter() - start, 1e-6)
            print(json.dumps({"step": step, "train_loss": float(output["loss"].detach().float()),
                              "router": float(output["router"].detach().float()),
                              "step_s": elapsed / step}), flush=True)
        if step % args.eval_every == 0 or step == args.train_steps:
            valid = evaluate_chromosome_level(model, loaders["valid"], device, valid_ds)
            record = {"step": step, "valid": valid}
            history.append(record); print(json.dumps(record), flush=True)
            if np.isfinite(valid["avg_pcc"]) and valid["avg_pcc"] > best_pcc:
                best_pcc = valid["avg_pcc"]
                torch.save({"model": model.state_dict(), "step": step, "valid": valid,
                            "args": vars(args)}, output_dir / "best.pt")

    final_test = evaluate_chromosome_level(model, loaders["test"], device, test_ds)
    torch.save({"model": model.state_dict(), "step": args.train_steps,
                "test_final": final_test, "args": vars(args)}, output_dir / "last.pt")
    best_test = None
    best_checkpoint = output_dir / "best.pt"
    if best_checkpoint.exists():
        payload = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(payload["model"])
        best_test = evaluate_chromosome_level(model, loaders["test"], device, test_ds)
    result = {"task": "TISP", "valid_best_avg_pcc": best_pcc,
              "test_final": final_test, "test_validation_best": best_test,
              "history": history}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "valid_best_avg_pcc": best_pcc}), flush=True)


if __name__ == "__main__":
    main()
