#!/usr/bin/env bash
set -euo pipefail

# Isolated 1B pretraining launcher.  The existing start_train.sh remains the
# unchanged 100M entry point; this script only selects the 1B architecture and
# a conservative memory recipe.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-fa}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"

LENGTHS="${LENGTHS:-1024,2048,4096,8192,16384,32768,65536,131072}"
# These values are intentionally conservative for a replicated 1B model.
# Override them after a memory smoke test on the target GPUs.
BATCH_SIZES="${BATCH_SIZES:-1024:8,2048:4,4096:2,8192:1,16384:1,32768:1,65536:1,131072:1}"
MAX_LENGTH="${MAX_LENGTH:-131072}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-2.0}"
TOTAL_UPDATES="${TOTAL_UPDATES:-24414}"
WARMUP_UPDATES="${WARMUP_UPDATES:-1000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-300}"
SEED="${SEED:-17}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/weights/pretrain/$(date +%Y%m%d)/vericomp-1b}"
RESUME="${RESUME:-}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/runs/logs/pretrain/$(date +%Y%m%d)/vericomp-1b}"
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

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

cmd=(
  "$(command -v accelerate)" launch
  --num_machines 1
  --num_processes 4
  --mixed_precision bf16
  --dynamo_backend no
  -m pretrain.train
  --model-size 1b
  --lengths "$LENGTHS"
  --batch-sizes "$BATCH_SIZES"
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
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$RESUME" ]]; then
  cmd+=(--resume "$RESUME")
fi

echo "Starting VeriComp 1B training"
echo "  environment: ${CONDA_ENV}"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "  mixed lengths: ${LENGTHS}"
echo "  per-rank batch sizes: ${BATCH_SIZES}"
echo "  gradient accumulation: ${GRAD_ACCUMULATION}"
echo "  activation checkpointing: enabled"
echo "  output: ${OUTPUT_DIR}"
echo "  log: ${log_file}"
printf '  command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run; training was not started."
  exit 0
fi

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
