"""VeriComp TISP run under the DNALongBench fine-tuning schedule.

The backbone/head remain VeriComp's native TISP implementation.  Dataset
sampling, effective batch, optimizer, schedule, validation rule, and final
evaluation are shared with ``ntv3_dnalongbench.py`` so the comparison uses one
training protocol.  BF16 is used for CUDA forward/backward autocast while the
loss statistics are accumulated in FP32.
"""

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
from torch.utils.data import DataLoader

from finetune.TISP.external_tisp import TISP_ASSAY_GROUPS, evaluate_tisp_chromosome
from finetune.TISP.chromosome_eval import (
    ChromosomeTISP,
    EVAL_STRIDE,
    evaluate_chromosome_level,
)
from finetune.TISP.ntv3_dnalongbench import _cosine_lr
from finetune.TISP.tisp import VeriCompTISPModel
from finetune.eQTL.eqtl import load_pretrained_backbone
from vericomp.model_presets import model_config_from_checkpoint
from vericomp.tisp_data import TISPWindowDataset, TISPRandomDataset


def _seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-num-workers", type=int, default=4)
    p.add_argument("--grad-accumulation", type=int, default=4)
    p.add_argument("--train-steps", type=int, default=25000)
    p.add_argument("--eval-every", type=int, default=500,
                   help="full-chromosome validation interval in optimizer steps")
    p.add_argument("--all-bf16", action="store_true",
                   help="store model parameters and compute, including loss, in BF16")
    p.add_argument("--fp32-master-optimizer", action="store_true",
                   help=("keep FP32 master parameter copies and AdamW states while "
                         "the model parameters, forward/backward compute, and loss stay BF16"))
    p.add_argument("--dual-strand", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--evaluation", choices=("genezip", "weighted"), default="genezip",
                   help="primary evaluator; genezip is the historical chromosome-level protocol")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine",
                   help="learning-rate schedule; constant disables warmup and decay")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--router-loss-coef", type=float, default=0.02)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--allocation-mode",
                   choices=("dynamic", "uniform", "length_only", "grid"),
                   default=None,
                   help="override allocation mode; pre-model_config checkpoints "
                        "default to dynamic")
    p.add_argument("--skip-test", action="store_true",
                   help="do not evaluate the held-out test chromosomes; use for validation-only tuning")
    return p.parse_args()


def main():
    args = parse_args(); _seed(args.seed)
    if args.fp32_master_optimizer and not args.all_bf16:
        raise ValueError("--fp32-master-optimizer requires --all-bf16")
    device = torch.device(args.device)
    root = str(Path(args.data_root).resolve())
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)

    # Match the historical VeriComp runner's random-window distribution and
    # seed.  The NTv3 DNALongBench sampler uses a separate fixed seed and is
    # not interchangeable with this VeriComp experiment.
    # ``build_tisp_datasets`` used by the historical VeriComp runner keeps
    # the dataset sampler seed at its default 3; the command-line seed only
    # controls model/optimizer initialization.
    train = TISPRandomDataset(root, 0)
    if args.evaluation == "genezip":
        valid = TISPWindowDataset(root, ["10"])
        test = TISPWindowDataset(root, ["8", "9"])
    else:
        valid = ChromosomeTISP(root, ["10"])
        test = ChromosomeTISP(root, ["8", "9"])
    train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=0)
    valid_loader = DataLoader(
        valid, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.eval_num_workers, pin_memory=True,
        persistent_workers=args.eval_num_workers > 0,
    )
    test_loader = DataLoader(
        test, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.eval_num_workers, pin_memory=True,
        persistent_workers=args.eval_num_workers > 0,
    )

    config_overrides = {"force_flash_attention": True,
                        "activation_checkpointing": False}
    if getattr(args, "allocation_mode", None) is not None:
        config_overrides["allocation_mode"] = args.allocation_mode
    config = model_config_from_checkpoint(
        args.checkpoint, "auto", **config_overrides,
    )
    model = VeriCompTISPModel(
        config, freeze_encoder=False, model_name="vericomp", feature="decoder",
        router_loss_coef=args.router_loss_coef, dual_strand=args.dual_strand,
        all_bf16=args.all_bf16,
    ).to(device)
    if args.all_bf16:
        model = model.to(dtype=torch.bfloat16)

    def evaluate_split(loader, dataset):
        if args.evaluation == "genezip":
            return evaluate_tisp_chromosome(model, loader, device)
        return evaluate_chromosome_level(model, loader, device, dataset)
    loaded, skipped = load_pretrained_backbone(model, args.checkpoint)
    # The shared evaluation helpers use this flag to apply the same BF16
    # autocast policy as training.
    model.use_bf16 = device.type == "cuda"
    params = [p for p in model.parameters() if p.requires_grad]
    master_params = (
        [torch.nn.Parameter(p.detach().float().clone()) for p in params]
        if args.fp32_master_optimizer else None
    )
    optimizer_params = master_params if master_params is not None else params
    optimizer = torch.optim.AdamW(
        optimizer_params, lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )

    run_config = {
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "task": "TISP", "model": "VeriComp",
        "protocol": "DNALongBench_original_schedule",
        "source_repository": "https://github.com/wenduocheng/DNALongBench",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "sequence_length": 100000, "center_scored_length": 50000,
        "train_chromosomes": "all except 8, 9, 10",
        "validation": "chromosome 10", "test": ["chromosome 8", "chromosome 9"],
        "train_sampling": "TISPRandomDataset random intervals, seed 3, N fraction <= 0.30",
        "batch_size": args.batch_size, "gradient_accumulation": args.grad_accumulation,
        "effective_batch_size": args.batch_size * args.grad_accumulation,
        "train_steps": args.train_steps, "validation_every_steps": args.eval_every,
        "validation_limit_batches": None,
        "learning_rate": args.lr, "optimizer": "AdamW",
        "weight_decay": args.weight_decay, "betas": [0.9, 0.999], "eps": 1e-8,
        "gradient_clip": args.max_grad_norm,
        "scheduler": "constant" if args.lr_schedule == "constant" else "cosine warmup, 1% warmup, lr_min=0.1*lr",
        "precision": (
            "BF16 model parameters, forward/backward compute, and loss; "
            "FP32 master optimizer parameters and AdamW states"
            if args.fp32_master_optimizer else
            "BF16 all modules and loss" if args.all_bf16 else
            "BF16 autocast; loss and metrics FP32"
        ),
        "optimizer_parameter_storage": (
            "FP32 master copies; gradients copied from BF16 model parameters; "
            "updated values cast back to BF16 after each step"
            if args.fp32_master_optimizer else
            "model parameter dtype"
        ),
        "vericomp_feature": "decoder", "vericomp_head": "LayerNorm + Linear(10)",
        "dual_strand": args.dual_strand,
        "loss": "pseudo_poisson_kl + router_aux_loss_coef * router_loss",
        "router_aux_loss_coef": args.router_loss_coef,
        "evaluation_batch_size": args.eval_batch_size,
        "evaluation": (
            "GeneZip-compatible chromosome-level 50-kb centre de-duplication "
            "and five strand-paired PCCs"
            if args.evaluation == "genezip" else
            "full-chromosome 50-kb scored centers, 25-kb stride, "
            "centre-weighted overlap stitching, N/+/-500-bp masking, "
            "five strand-paired PCCs"
        ),
        "validation_best": "validation average PCC",
        "test_evaluation_skipped": args.skip_test,
        "loaded_backbone_tensors": loaded, "skipped_backbone_tensors": skipped,
        "dataset_sizes": {"valid_windows": len(valid), "test_windows": len(test)},
        "arguments": vars(args),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    print(json.dumps(run_config, indent=2), flush=True)

    best = -math.inf; history = []; iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True); started = time.perf_counter()
    for step in range(1, args.train_steps + 1):
        for _ in range(args.grad_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                model_output = model(batch["input_ids"].to(device, non_blocking=True),
                                     batch["target"].to(device, non_blocking=True))
                loss = model_output["loss"] / args.grad_accumulation
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        lr = args.lr if args.lr_schedule == "constant" else _cosine_lr(step - 1, args.train_steps, args.lr)
        for group in optimizer.param_groups: group["lr"] = lr
        if master_params is not None:
            for model_param, master_param in zip(params, master_params):
                master_param.grad = (
                    None if model_param.grad is None
                    else model_param.grad.detach().float()
                )
        optimizer.step()
        if master_params is not None:
            with torch.no_grad():
                for model_param, master_param in zip(params, master_params):
                    model_param.copy_(master_param)
            for model_param in params:
                model_param.grad = None
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 50 == 0:
            print(json.dumps({"step": step,
                              "train_loss": float(model_output["loss"].detach().float()),
                              "router": float(model_output["router"].detach().float()),
                              "lr": lr,
                              "step_s": (time.perf_counter() - started) / step}), flush=True)
        if step % args.eval_every == 0 or step == args.train_steps:
            valid_metrics = evaluate_split(valid_loader, valid)
            record = {"step": step, "valid": valid_metrics}; history.append(record)
            print(json.dumps(record), flush=True)
            if np.isfinite(valid_metrics["avg_pcc"]) and valid_metrics["avg_pcc"] > best:
                best = valid_metrics["avg_pcc"]
                torch.save({"model": model.state_dict(), "step": step,
                            "valid": valid_metrics, "config": run_config}, output_dir / "best.pt")
        if step % 300 == 0:
            torch.save({"model": model.state_dict(), "step": step, "config": run_config},
                       output_dir / f"step_{step:05d}.pt")

    final_test = None
    if not args.skip_test:
        final_test = evaluate_split(test_loader, test)
    # Preserve the final model before loading validation-best for its test run.
    torch.save({"model": model.state_dict(), "step": args.train_steps,
                "test_final": final_test, "config": run_config}, output_dir / "last.pt")
    best_test = None
    best_checkpoint = output_dir / "best.pt"
    if not args.skip_test and best_checkpoint.exists():
        payload = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(payload["model"])
        best_test = evaluate_split(test_loader, test)
    result = {"task": "TISP", "model": "VeriComp", "valid_best_avg_pcc": best,
              "test_final": final_test, "test_validation_best": best_test,
              "history": history}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
