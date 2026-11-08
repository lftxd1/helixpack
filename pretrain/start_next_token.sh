#!/usr/bin/env bash
set -euo pipefail

# Detached four-GPU Next-Token training from an existing VeriComp checkpoint.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-fa}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/weights/pretrain/20260814/vericomp-residual-skip-30k/checkpoint-010000.pt}"
LENGTHS="${LENGTHS:-1024,2048,4096,8192,16384,32768,65536,131072}"
BATCH_SIZES="${BATCH_SIZES:-1024:128,2048:64,4096:32,8192:32,16384:16,32768:16,65536:8,131072:8}"
MAX_LENGTH="${MAX_LENGTH:-131072}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-1}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-2.0}"
TOTAL_UPDATES="${TOTAL_UPDATES:-30600}"
WARMUP_UPDATES="${WARMUP_UPDATES:-1000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-300}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
CAUSAL_LOCAL_SKIP="${CAUSAL_LOCAL_SKIP:-0}"
LOCAL_SKIP_GATE_BIAS="${LOCAL_SKIP_GATE_BIAS:--2.0}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/weights/pretrain/$(date +%Y%m%d)/vericomp-next-token}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/runs/logs/pretrain/$(date +%Y%m%d)}"
DRY_RUN="${DRY_RUN:-0}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate is not installed in conda environment: ${CONDA_ENV}" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
log_file="${LOG_DIR}/next-token-${timestamp}.log"
cmd=(
  "$(command -v accelerate)" launch
  --num_machines 1 --num_processes 4 --mixed_precision bf16
  --dynamo_backend no -m pretrain.next_token
  --init-from "$CHECKPOINT"
  --lengths "$LENGTHS" --batch-sizes "$BATCH_SIZES"
  --max-length "$MAX_LENGTH" --grad-accumulation "$GRAD_ACCUMULATION"
  --lr "$LEARNING_RATE" --weight-decay "$WEIGHT_DECAY"
  --max-grad-norm "$MAX_GRAD_NORM" --total-updates "$TOTAL_UPDATES"
  --warmup-updates "$WARMUP_UPDATES" --eval-every "$EVAL_EVERY"
  --save-every "$SAVE_EVERY" --validation-samples "$VALIDATION_SAMPLES"
  --num-workers "$NUM_WORKERS" --max-tokens "$MAX_TOKENS"
  --output-dir "$OUTPUT_DIR"
)
if [[ "$CAUSAL_LOCAL_SKIP" == "1" ]]; then
  cmd+=(--causal-local-skip --local-skip-gate-bias "$LOCAL_SKIP_GATE_BIAS")
fi

printf 'Running command:'
printf ' %q' "${cmd[@]}"
printf '\nLog: %s\n' "$log_file"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export PYTHONUNBUFFERED=1
nohup setsid "${cmd[@]}" >"$log_file" 2>&1 < /dev/null &
pid=$!
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  sed -n '1,200p' "$log_file" >&2 || true
  exit 1
fi
echo "Started Next-Token training PID ${pid}"
