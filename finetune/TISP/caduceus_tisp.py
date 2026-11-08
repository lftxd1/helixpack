"""All-BF16 Caduceus TISP fine-tuning with chromosome-level evaluation."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from finetune.TISP.external_tisp import evaluate_tisp_chromosome
from finetune.TISP.tisp import _pseudo_poisson_kl
from finetune.eQTL.external_eqtl import Backbone
from vericomp.tisp_data import TISP_SEQUENCE_LENGTH, TISP_WINDOW_STEP, TISPRandomDataset, TISPWindowDataset


_ID_TO_BASE = ("A", "C", "G", "T", "N", "N")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CaduceusTISPModel(nn.Module):
    """Caduceus final token features followed by a per-base TISP decoder."""

    def __init__(self, model_name: str, model_path: str, device: torch.device, precision: str = "bf16"):
        super().__init__()
        loaded = Backbone(model_name, model_path, device)
        self.backbone = loaded.model
        self.tokenizer = loaded.tokenizer
        self.device = device
        self.precision = precision
        param_dtype = torch.bfloat16 if precision in ("bf16", "mixed") else torch.float32
        head_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.backbone.to(device=device, dtype=param_dtype)
        # The MLM projection is not part of TISP. Caduceus' final hidden state
        # is read directly from its shared Caduceus module.
        self.backbone.lm_head.requires_grad_(False)
        width = int(self.backbone.config.d_model) * (2 if self.backbone.config.rcps else 1)
        self.norm = nn.LayerNorm(width, device=device, dtype=head_dtype)
        self.head = nn.Linear(width, 10, device=device, dtype=head_dtype)
        self.sep_token_id = int(self.tokenizer.sep_token_id)

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        sequences = ["".join(_ID_TO_BASE[int(token)] for token in row) for row in input_ids.cpu().tolist()]
        encoded = self.tokenizer(
            sequences, return_tensors="pt", padding=False, truncation=False
        )["input_ids"].to(self.device)
        # Caduceus adds one trailing [SEP] token. Remove it so each output
        # position remains aligned with its corresponding TISP target base.
        if encoded.shape[1] == TISP_SEQUENCE_LENGTH + 1:
            if not bool(torch.all(encoded[:, -1] == self.sep_token_id)):
                raise RuntimeError("Caduceus tokenizer did not append the expected terminal [SEP]")
            encoded = encoded[:, :TISP_SEQUENCE_LENGTH]
        if encoded.shape[1] != TISP_SEQUENCE_LENGTH:
            raise RuntimeError(
                f"Caduceus produced {encoded.shape[1]} tokens for a {TISP_SEQUENCE_LENGTH}-bp TISP window"
            )
        return encoded

    def forward(self, input_ids: torch.Tensor, target: torch.Tensor | None = None):
        encoded = self._encode(input_ids)
        outputs = self.backbone.caduceus(
            input_ids=encoded,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        if hidden.shape[1] != TISP_SEQUENCE_LENGTH:
            raise RuntimeError(f"Caduceus hidden length mismatch: {hidden.shape[1]}")
        if self.precision == "mixed":
            # Keep the tiny readout out of the ambient BF16 autocast so the
            # head bias can represent sub-ulp Adam updates.
            with torch.autocast(device_type="cuda", enabled=False):
                log_rates = self.head(self.norm(hidden.float()))
        else:
            head_input = hidden.to(torch.bfloat16) if self.precision == "bf16" else hidden.float()
            log_rates = self.head(self.norm(head_input))
        pred = torch.exp(log_rates)
        result = {"log_rates": log_rates, "pred": pred, "hidden": hidden}
        if target is not None:
            valid = torch.isfinite(target) & torch.isfinite(log_rates)
            target_bf16 = target.to(dtype=torch.bfloat16)
            loss = (
                _pseudo_poisson_kl(
                    log_rates[valid],
                    target_bf16[valid] if self.precision == "bf16" else target[valid],
                    all_bf16=self.precision == "bf16",
                )
                if bool(valid.any()) else log_rates.sum() * 0.0
            )
            result["loss"] = loss
            result["loss_task"] = loss
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("caduceus_ph", "caduceus_ps"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--precision", choices=("bf16", "fp32", "mixed"), default="bf16")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--init-head-bias", choices=("none", "mean"), default="none")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accumulation", type=int, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("warmup", "linear"), default="warmup",
                        help="warmup ramps for warmup_ratio then holds; linear decays to zero afterwards")
    parser.add_argument("--scheduler-start-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    _seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Caduceus TISP training requires a CUDA device")
    if args.batch_size * args.grad_accumulation != 8:
        raise ValueError("These paired Caduceus runs use effective batch size 8")

    data_root = str(Path(args.data_root).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = TISPRandomDataset(data_root, 0, seed=3)
    valid = TISPWindowDataset(data_root, ["10"])
    test = TISPWindowDataset(data_root, ["8", "9"])
    loaders = {
        "train": DataLoader(train, batch_size=args.batch_size, num_workers=0),
        "valid": DataLoader(valid, batch_size=args.eval_batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(test, batch_size=args.eval_batch_size, shuffle=False, num_workers=0),
    }
    if not len(valid) or not len(test):
        raise RuntimeError(f"empty validation or test set: valid={len(valid)}, test={len(test)}")

    model = CaduceusTISPModel(
        args.model, args.model_path, device, precision=args.precision
    ).to(device=device)
    if args.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    if args.init_head_bias == "mean":
        total = None
        seen = 0
        for index, batch in enumerate(loaders["train"]):
            if index >= 32:
                break
            target = batch["target"].to(device).float()
            batch_sum = target.sum(dim=(0, 1))
            total = batch_sum if total is None else total + batch_sum
            seen += target.shape[0] * target.shape[1]
        per_track_mean = (total / max(seen, 1)).clamp_min(1e-8)
        with torch.no_grad():
            model.head.bias.copy_(torch.log(per_track_mean).to(model.head.bias.dtype))
        print(json.dumps({
            "event": "head_bias_init",
            "per_track_mean": [float(value) for value in per_track_mean.cpu()],
        }), flush=True)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
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

    config = {
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "task": "TISP",
        "model": args.model,
        "model_path": str(Path(args.model_path).resolve()),
        "data_root": data_root,
        "sequence_length": TISP_SEQUENCE_LENGTH,
        "center_scored_length": TISP_WINDOW_STEP,
        "train_chromosomes": "all except 8, 9, 10",
        "validation": "chromosome 10",
        "test": ["chromosome 8", "chromosome 9"],
        "train_sampling": "TISPRandomDataset random windows, seed 3, N fraction <= 0.30",
        "precision": {
            "bf16": "BF16 all model modules, TISP adapter, loss, and optimizer states",
            "mixed": "BF16 backbone parameters; FP32 LayerNorm, TISP head, loss, and head optimizer states",
            "fp32": "FP32 parameters, adapter, loss, and optimizer states; BF16 autocast eligible ops",
        }[args.precision],
        "feature": "Caduceus final last_hidden_state; remove terminal tokenizer SEP for base alignment",
        "decoder": (
            "BF16 LayerNorm + Linear(hidden, 10)"
            if args.precision == "bf16"
            else "FP32 LayerNorm + Linear(hidden, 10)"
        ),
        "loss": (
            "pseudo_poisson_kl computed in BF16"
            if args.precision == "bf16"
            else "pseudo_poisson_kl computed in FP32"
        ),
        "evaluation": "GeneZip-compatible chromosome-level 50-kb centre de-duplication and five strand-paired PCCs; metrics accumulated in FP32",
        "validation_best": "validation average PCC",
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accumulation,
        "effective_batch_size": args.batch_size * args.grad_accumulation,
        "eval_batch_size": args.eval_batch_size,
        "train_steps": args.train_steps,
        "validation_every_steps": args.eval_every,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "scheduler_start_factor": args.scheduler_start_factor,
        "head_bias_init": args.init_head_bias,
        "optimizer": "AdamW",
        "weight_decay": args.weight_decay,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "gradient_clip": args.max_grad_norm,
        "scheduler": args.scheduler,
        "seed": args.seed,
        "dataset_sizes": {"train_nominal": len(train), "valid_windows": len(valid), "test_windows": len(test)},
        "arguments": vars(args),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "resolved_config", **config}, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps({
        "event": "parameter_summary",
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in params),
        "trainable_dtypes": sorted({str(parameter.dtype) for parameter in params}),
        "device": str(device),
    }), flush=True)

    best_pcc = -math.inf
    history: list[dict[str, object]] = []
    iterator = iter(loaders["train"])
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for step in range(1, args.train_steps + 1):
        for _ in range(args.grad_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loaders["train"])
                batch = next(iterator)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids, target)
                loss = output["loss"] / args.grad_accumulation
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite BF16 TISP loss at optimizer step {step}")
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % args.log_every == 0:
            print(json.dumps({
                "step": step,
                "train_loss": float(output["loss"].detach().float().item()),
                "grad_norm": float(torch.as_tensor(grad_norm).detach().float().item()),
                "lr": optimizer.param_groups[0]["lr"],
                "step_s": (time.perf_counter() - started) / step,
                "pred_mean": float(output["pred"].detach().float().mean()),
                "pred_pos_std": float(output["pred"].detach().float().std(dim=1).mean()),
                "hidden_pos_std": float(output["hidden"].detach().float().std(dim=1).mean()),
            }), flush=True)

        if step % args.eval_every == 0 or step == args.train_steps:
            valid_metrics = evaluate_tisp_chromosome(model, loaders["valid"], device)
            record = {"step": step, "valid": valid_metrics}
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if np.isfinite(valid_metrics["avg_pcc"]) and valid_metrics["avg_pcc"] > best_pcc:
                best_pcc = float(valid_metrics["avg_pcc"])
                torch.save({"model": model.state_dict(), "step": step, "valid": valid_metrics, "config": config}, output_dir / "best.pt")

        if step % 1000 == 0 and step % args.eval_every != 0:
            torch.save({"model": model.state_dict(), "step": step, "config": config}, output_dir / f"step_{step:05d}.pt")

    test_final = evaluate_tisp_chromosome(model, loaders["test"], device)
    torch.save({"model": model.state_dict(), "step": args.train_steps, "test_final": test_final, "config": config}, output_dir / "last.pt")
    best_path = output_dir / "best.pt"
    test_best = None
    best_step = None
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
        best_step = int(best_checkpoint["step"])
        test_best = evaluate_tisp_chromosome(model, loaders["test"], device)
    metrics = {
        "task": "TISP",
        "model": args.model,
        "valid_best_avg_pcc": best_pcc,
        "validation_best_step": best_step,
        "test_final": test_final,
        "test_validation_best": test_best,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "completed", "output_dir": str(output_dir), **metrics}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
