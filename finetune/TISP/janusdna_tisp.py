"""JanusDNA TISP fine-tuning with BF16 backbone autocast and FP32 loss."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from finetune.TISP.external_tisp import evaluate_tisp_chromosome
from vericomp.tisp_data import TISPWindowDataset, TISPRandomDataset


JANUSDNA_SOURCE = Path("/gpfs_data/Projects/GenomeModelProject/sources/JanusDNA")
JANUSDNA_MODEL_ROOT = Path("/gpfs_data/Projects/GenomeModelProject/models/janusdna")
JANUS_INPUT_IDS = (7, 8, 9, 10, 6, 6)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Match JanusDNA's official training entry point while retaining FP32
    # parameter, activation, and loss dtypes.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _load_janusdna(model_root: Path, device: torch.device, router_coef: float):
    sys.path.insert(0, str(JANUSDNA_SOURCE))
    sys.path.insert(0, str(JANUSDNA_SOURCE / "src"))
    from janusdna.configuration_janusdna import JanusDNAConfig
    from janusdna.modeling_janusdna import (
        JanusDNAForCausalLM,
        JanusDNAModel,
        JanusDNAPreTrainedModel,
    )

    # JanusDNA wraps FlexAttention in torch.compile.  Its custom architecture is
    # not advertised through the generic Transformers capability check.
    # Transformers performs its backend-capability check when the nested
    # JanusDNAModel is instantiated, so mark both the base class and model.
    JanusDNAPreTrainedModel._supports_flex_attn = True
    JanusDNAModel._supports_flex_attn = True
    JanusDNAForCausalLM._supports_flex_attn = True
    config_file = next(model_root.glob("*/model_config.json"), None)
    checkpoint_file = next(model_root.glob("*/checkpoints/last.ckpt"), None)
    if config_file is None or checkpoint_file is None:
        raise FileNotFoundError(f"JanusDNA checkpoint layout not found under {model_root}")

    raw = json.loads(config_file.read_text(encoding="utf-8"))
    config_values = dict(raw["config"])
    config_values.pop("_target_", None)
    config_values.update(
        output_router_logits=True,
        router_aux_loss_coef=router_coef,
        gradient_checkpointing=False,
        use_cache=False,
        attn_implementation="flex_attention",
        final_attention_class="flex_attention",
    )
    config = JanusDNAConfig(**config_values)
    print(
        "[janusdna] FP32 with TF32 enabled; custom FlexAttention path enabled; "
        "gradient checkpointing disabled; use_cache disabled",
        flush=True,
    )

    model = JanusDNAForCausalLM(config)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key[len("model."):]: value for key, value in state.items()
             if key.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected JanusDNA checkpoint keys: {unexpected[:10]}")
    print(
        f"[janusdna] loaded checkpoint={checkpoint_file}; "
        f"missing_keys={len(missing)}; unexpected_keys={len(unexpected)}",
        flush=True,
    )
    del checkpoint, state
    model.to(device=device, dtype=torch.float32)
    model.lm_head.requires_grad_(False)
    return model, config


class JanusDNATISPModel(nn.Module):
    """Base-resolution TISP adapter over JanusDNA's final fused hidden state."""

    def __init__(self, backbone: nn.Module, hidden_size: int, router_coef: float):
        super().__init__()
        self.backbone = backbone
        self.router_coef = float(router_coef)
        self.norm = nn.LayerNorm(hidden_size, dtype=torch.float32)
        self.head = nn.Linear(hidden_size, 10, dtype=torch.float32)
        self.id_map = torch.tensor(JANUS_INPUT_IDS, dtype=torch.long)

    def forward(self, input_ids: torch.Tensor, target: torch.Tensor | None = None):
        # TISP datasets use VeriComp's compact A/C/G/T/N ids.  JanusDNA's
        # character vocabulary uses A=7, C=8, G=9, T=10, N/other=6.
        mapped_ids = self.id_map.to(input_ids.device)[input_ids.long()]
        # Keep FP32 parameters and optimizer state, while using BF16 for
        # autocast-eligible backbone operations during training. Evaluation
        # stays in FP32 so validation-best selection uses the reference path.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.training,
        ):
            outputs = self.backbone.model(
                input_ids=mapped_ids,
                output_router_logits=self.training,
                use_cache=False,
                return_dict=True,
            )
        hidden = outputs.last_hidden_state
        if hidden.shape[1] != input_ids.shape[1]:
            raise RuntimeError(
                f"JanusDNA hidden length {hidden.shape[1]} != input length {input_ids.shape[1]}"
            )
        log_rates = self.head(self.norm(hidden.float()))
        predictions = torch.exp(log_rates.float())
        result = {"pred": predictions, "log_rates": log_rates}

        if target is not None:
            target = target.float()
            valid = torch.isfinite(target) & torch.isfinite(log_rates)
            if bool(valid.any()):
                rates = torch.exp(log_rates[valid].float())
                labels = target[valid]
                task_loss = (
                    labels * torch.log((labels + 1e-5) / (rates + 1e-5))
                    + rates - labels
                ).mean()
            else:
                task_loss = log_rates.sum() * 0.0

            if self.training:
                # JanusDNAModel returns its already-coefficient-weighted MoE
                # load-balancing loss in the router_logits output field.
                moe_loss = outputs.router_logits.float()
                if not torch.is_tensor(moe_loss) or moe_loss.ndim != 0:
                    raise RuntimeError("JanusDNA did not return a scalar MoE auxiliary loss")
            else:
                moe_loss = task_loss.detach() * 0.0
            result.update(
                loss_task=task_loss,
                moe_loss=moe_loss,
                loss=task_loss + moe_loss,
            )
        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default=str(JANUSDNA_MODEL_ROOT))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--param-dtype", choices=("fp32", "bf16"), default="fp32",
                        help="storage dtype for backbone weights; adapter and loss stay FP32")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("warmup", "linear"), default="warmup")
    parser.add_argument("--scheduler-start-factor", type=float, default=0.1)
    parser.add_argument("--router-aux-loss-coef", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This JanusDNA FlexAttention run requires a CUDA GPU")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = str(Path(args.data_root).resolve())

    backbone, config = _load_janusdna(Path(args.model_root), device, args.router_aux_loss_coef)
    model = JanusDNATISPModel(
        backbone, int(config.hidden_size), args.router_aux_loss_coef
    ).to(device=device, dtype=torch.float32)
    if args.param_dtype == "bf16":
        # BF16 backbone weights; the adapter keeps its FP32 parameters and
        # the Poisson loss is evaluated in FP32.
        model.backbone.to(torch.bfloat16)

    # Match the comparison runs' TISP splits and random-window distribution.
    train_dataset = TISPRandomDataset(data_root, 0, seed=3)
    valid_dataset = TISPWindowDataset(data_root, ["10"])
    test_dataset = TISPWindowDataset(data_root, ["8", "9"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
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
    run_config = {
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "task": "TISP", "model": "JanusDNA",
        "protocol": "VeriComp_TISP_GeneZip_chromosome_eval",
        "checkpoint": str(next(Path(args.model_root).glob("*/checkpoints/last.ckpt"))),
        "precision": "FP32 model parameters and optimizer state; BF16 autocast for eligible training-backbone ops; FP32 adapter, TISP loss, MoE loss accumulation, and evaluation; TF32 enabled for remaining eligible CUDA matmuls/cuDNN",
        "training_autocast": "torch.autocast(cuda, bfloat16) around backbone only; disabled for evaluation",
        "sequence_length": 100000, "center_scored_length": 50000,
        "train_chromosomes": "all except 8, 9, 10",
        "validation": "chromosome 10", "test": ["chromosome 8", "chromosome 9"],
        "train_sampling": "TISPRandomDataset random intervals; seed 3; N fraction <= 0.30",
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accumulation,
        "effective_batch_size": args.batch_size * args.grad_accumulation,
        "train_steps": args.train_steps, "validation_every_steps": args.eval_every,
        "learning_rate": args.lr, "scheduler": args.scheduler,
        "warmup_ratio": args.warmup_ratio,
        "param_dtype": args.param_dtype,
        "optimizer": "AdamW", "weight_decay": args.weight_decay,
        "betas": [0.9, 0.999], "eps": 1e-8,
        "gradient_clip": args.max_grad_norm,
        "router_aux_loss_coef": args.router_aux_loss_coef,
        "router_auxiliary": "model-returned coefficient-weighted JanusDNA MoE loss, added separately to TISP loss during training",
        "feature": "JanusDNA last_hidden_state after final_layernorm + final_mlp + residual",
        "decoder": "FP32 LayerNorm + Linear(hidden_size, 10)",
        "input_id_mapping": {"A": 7, "C": 8, "G": 9, "T": 10, "N/other": 6},
        "attention": "FlexAttention; torch.compile path from JanusDNA implementation",
        "gradient_checkpointing": False,
        "evaluation": "GeneZip-compatible chromosome-level 50-kb center de-duplication and five strand-paired PCCs; no autocast",
        "validation_best": "validation average PCC",
        "dataset_sizes": {"train_samples_per_iterator": len(train_dataset),
                           "valid_windows": len(valid_dataset), "test_windows": len(test_dataset)},
        "arguments": vars(args),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    print(json.dumps(run_config, indent=2), flush=True)

    best_pcc = -math.inf
    history: list[dict] = []
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()

    for step in range(1, args.train_steps + 1):
        last_output = None
        for _micro_step in range(args.grad_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            output = model(batch["input_ids"].to(device), batch["target"].to(device))
            (output["loss"] / args.grad_accumulation).backward()
            last_output = output

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % args.log_every == 0:
            print(json.dumps({
                "task": "TISP", "model": "JanusDNA", "step": step,
                "train_loss": float(last_output["loss"].detach().float()),
                "tisp_loss": float(last_output["loss_task"].detach().float()),
                "moe_loss": float(last_output["moe_loss"].detach().float()),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm.detach().float()),
                "elapsed_s": time.perf_counter() - started,
            }), flush=True)

        if step % args.eval_every == 0 or step == args.train_steps:
            eval_started = time.perf_counter()
            print(json.dumps({"task": "TISP", "phase": "validation_start", "step": step,
                              "windows": len(valid_dataset)}), flush=True)
            valid_metrics = evaluate_tisp_chromosome(
                model, valid_loader, device, use_bf16_autocast=False
            )
            record = {"step": step, "valid": valid_metrics}
            history.append(record)
            print(json.dumps(record), flush=True)
            if np.isfinite(valid_metrics["avg_pcc"]) and valid_metrics["avg_pcc"] > best_pcc:
                best_pcc = valid_metrics["avg_pcc"]
                torch.save({
                    "backbone": model.backbone.model.state_dict(),
                    "norm": model.norm.state_dict(), "head": model.head.state_dict(),
                    "step": step, "valid": valid_metrics, "config": run_config,
                }, output_dir / "best.pt")
            print(json.dumps({"task": "TISP", "phase": "validation_complete", "step": step,
                              "elapsed_s": time.perf_counter() - eval_started}), flush=True)

        if step % 300 == 0:
            torch.save({
                "backbone": model.backbone.model.state_dict(),
                "norm": model.norm.state_dict(), "head": model.head.state_dict(),
                "step": step, "config": run_config,
            }, output_dir / f"step_{step:05d}.pt")

    print(json.dumps({"task": "TISP", "phase": "test_final_start", "step": args.train_steps,
                      "windows": len(test_dataset)}), flush=True)
    final_test = evaluate_tisp_chromosome(
        model, test_loader, device, use_bf16_autocast=False
    )
    torch.save({
        "backbone": model.backbone.model.state_dict(),
        "norm": model.norm.state_dict(), "head": model.head.state_dict(),
        "step": args.train_steps, "test_final": final_test, "config": run_config,
    }, output_dir / "last.pt")

    best_test = None
    best_path = output_dir / "best.pt"
    if best_path.exists():
        best_state = torch.load(best_path, map_location=device, weights_only=False)
        model.backbone.model.load_state_dict(best_state["backbone"])
        model.norm.load_state_dict(best_state["norm"])
        model.head.load_state_dict(best_state["head"])
        best_test = evaluate_tisp_chromosome(
            model, test_loader, device, use_bf16_autocast=False
        )

    result = {
        "task": "TISP", "model": "JanusDNA",
        "valid_best_avg_pcc": best_pcc,
        "test_final": final_test,
        "test_validation_best": best_test,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
