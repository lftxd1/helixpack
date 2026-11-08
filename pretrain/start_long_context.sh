#!/usr/bin/env bash
set -euo pipefail

# Independent launcher for long-context MLM.  The existing short-context
# launcher is pretrain/start_train.sh and is intentionally not modified.
#
# Example overrides:
#   CUDA_DEVICES=0,1,2,3 TOTAL_UPDATES=1000 ./pretrain/start_long_context.sh
#   LENGTHS=131072,262144,524288,1048576 BATCH_SIZES=131072:4,262144:2,524288:2,1048576:2 ...

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-fa}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"

LENGTHS="${LENGTHS:-1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576}"
BATCH_SIZES="${BATCH_SIZES:-1024:128,2048:64,4096:32,8192:32,16384:16,32768:16,65536:8,131072:4,262144:2,524288:2,1048576:2}"
LENGTH_WEIGHTS="${LENGTH_WEIGHTS:-1024:1,2048:1,4096:1,8192:1,16384:1,32768:1,65536:1,131072:1,262144:1.5,524288:2.25,1048576:3}"
MAX_LENGTH="${MAX_LENGTH:-1048576}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-1}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-2.0}"
TOTAL_UPDATES="${TOTAL_UPDATES:-25000}"
WARMUP_UPDATES="${WARMUP_UPDATES:-1000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-300}"
LONG_SAMPLES_PER_LENGTH="${LONG_SAMPLES_PER_LENGTH:-}"
N_MAX_FRACTION="${N_MAX_FRACTION:-0.01}"
MAX_WINDOW_ATTEMPTS="${MAX_WINDOW_ATTEMPTS:-32}"
BOUNDARY_MARGIN="${BOUNDARY_MARGIN:-0}"
SEED="${SEED:-17}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/weights/pretrain/$(date +%Y%m%d)/vericomp-long-context-weighted-1m}"
DEFAULT_RESUME="${PROJECT_DIR}/weights/pretrain/20260817/vericomp-long-context-1m/latest.pt"
if [[ ! -f "$DEFAULT_RESUME" ]]; then
  DEFAULT_RESUME="${PROJECT_DIR}/weights/pretrain/20260814/vericomp-residual-skip-30k/checkpoint-010000.pt"
fi
RESUME="${RESUME:-$DEFAULT_RESUME}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/runs/logs/pretrain/$(date +%Y%m%d)/long-context-weighted}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
timestamp="$(date +%Y%m%d-%H%M%S)"
log_file="${LOG_DIR}/train-${timestamp}.log"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "conda initialization script not found: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate is not installed in conda environment: ${CONDA_ENV}" >&2
  exit 1
fi
ACCELERATE_BIN="$(command -v accelerate)"

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

cmd=(
  "$ACCELERATE_BIN" launch
  --num_machines 1
  --num_processes 4
  --mixed_precision bf16
  --dynamo_backend no
  -m
  pretrain.train_long_context
  --lengths "$LENGTHS"
  --batch-sizes "$BATCH_SIZES"
  --length-weights "$LENGTH_WEIGHTS"
  --max-length "$MAX_LENGTH"
  --grad-accumulation "$GRAD_ACCUMULATION"
  --lr "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --max-grad-norm "$MAX_GRAD_NORM"
  --total-updates "$TOTAL_UPDATES"
  --warmup-updates "$WARMUP_UPDATES"
  --eval-every "$EVAL_EVERY"
  --save-every "$SAVE_EVERY"
  --validation-samples "$VALIDATION_SAMPLES"
  --n-max-fraction "$N_MAX_FRACTION"
  --max-window-attempts "$MAX_WINDOW_ATTEMPTS"
  --boundary-margin "$BOUNDARY_MARGIN"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$LONG_SAMPLES_PER_LENGTH" ]]; then
  cmd+=(--long-samples-per-length "$LONG_SAMPLES_PER_LENGTH")
fi
if [[ -n "$RESUME" ]]; then
  cmd+=(--resume "$RESUME")
fi

echo "Starting VeriComp long-context MLM training"
echo "  environment: ${CONDA_ENV}"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "  lengths: ${LENGTHS}"
echo "  per-rank batch sizes: ${BATCH_SIZES}"
echo "  length weights: ${LENGTH_WEIGHTS}"
echo "  total updates: ${TOTAL_UPDATES}"
echo "  long-window N threshold: ${N_MAX_FRACTION}"
echo "  workers: ${NUM_WORKERS}"
echo "  activation checkpointing: enabled"
echo "  resume: ${RESUME:-none}"
echo "  log: ${log_file}"
printf '  command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run; training was not started."
  exit 0
fi

# Keep the detached launch behavior consistent with the project's production
# jobs.  No .pid file is written; the launcher PID is printed only here.
nohup setsid "${cmd[@]}" >"$log_file" 2>&1 < /dev/null &
training_pid=$!

sleep 2
if ! kill -0 "$training_pid" 2>/dev/null; then
  echo "Training exited during startup; log follows:" >&2
  sed -n '1,240p' "$log_file" >&2 || true
  exit 1
fi

echo "Started PID ${training_pid}"
echo "Follow log: tail -f ${log_file}"
