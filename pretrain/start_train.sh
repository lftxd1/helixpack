#!/usr/bin/env bash
set -euo pipefail

# Start the production VeriComp mixed-length training job in the fa conda
# environment.  Each entry in LENGTHS is a separate exact-length bucket.
# DistributedLengthBatchSampler forms a complete same-length global batch
# inside one bucket, using the configured per-length local batch size, then
# splits that global batch across ranks.
#
# Override any variable when needed, for example:
#   LEARNING_RATE=5e-4 TOTAL_UPDATES=1000 ./pretrain/start_train.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-fa}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"

LENGTHS="${LENGTHS:-1024,2048,4096,8192,16384,32768,65536,131072}"
BATCH_SIZES="${BATCH_SIZES:-1024:128,2048:64,4096:32,8192:32,16384:16,32768:16,65536:8,131072:8}"
MAX_LENGTH="${MAX_LENGTH:-131072}"
MODEL_SIZE="${MODEL_SIZE:-100m}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-1}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-2.0}"
TOTAL_UPDATES="${TOTAL_UPDATES:-24414}"
WARMUP_UPDATES="${WARMUP_UPDATES:-1000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-300}"
SEED="${SEED:-17}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RESUME="${RESUME:-}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/runs/logs/pretrain/$(date +%Y%m%d)}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "$OUTPUT_DIR" ]]; then
  # Keep the historical 100M path unchanged, but never let a new model
  # preset accidentally write into an existing model's checkpoint directory.
  if [[ "$MODEL_SIZE" == "100m" ]]; then
    output_name="vericomp-128k"
  else
    output_name="vericomp-${MODEL_SIZE}-128k"
  fi
  OUTPUT_DIR="${PROJECT_DIR}/weights/pretrain/$(date +%Y%m%d)/${output_name}"
fi

# Preserve the historical 100M launch behavior.  Larger presets default to
# checkpointing because their activation footprint is substantially higher;
# callers can still override either choice with ACTIVATION_CHECKPOINTING=0/1.
if [[ -z "${ACTIVATION_CHECKPOINTING+x}" ]]; then
  if [[ "$MODEL_SIZE" == "100m" ]]; then
    ACTIVATION_CHECKPOINTING=0
  else
    ACTIVATION_CHECKPOINTING=1
  fi
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
timestamp="$(date +%Y%m%d-%H%M%S)"
log_file="${LOG_DIR}/train-${timestamp}.log"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "conda initialization script not found: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi

# Load conda into this non-interactive shell, then activate the requested env.
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate is not installed in conda environment: ${CONDA_ENV}" >&2
  echo "Install it with: conda run -n ${CONDA_ENV} pip install accelerate" >&2
  exit 1
fi
ACCELERATE_BIN="$(command -v accelerate)"

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

case "$ACTIVATION_CHECKPOINTING" in
  0|1) ;;
  *) echo "ACTIVATION_CHECKPOINTING must be 0 or 1" >&2; exit 1 ;;
esac

cmd=(
  "$ACCELERATE_BIN" launch
  --num_machines 1
  --num_processes 4
  --mixed_precision bf16
  --dynamo_backend no
  -m
  pretrain.train
  --lengths "$LENGTHS"
  --batch-sizes "$BATCH_SIZES"
  --max-length "$MAX_LENGTH"
  --model-size "$MODEL_SIZE"
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
if [[ "$ACTIVATION_CHECKPOINTING" == "0" ]]; then
  cmd+=(--no-activation-checkpointing)
fi
if [[ -n "$RESUME" ]]; then
  cmd+=(--resume "$RESUME")
fi

write_launch_manifest() {
  {
    echo "launch_manifest_begin"
    echo "launch_time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname)"
    echo "project_dir=${PROJECT_DIR}"
    echo "working_dir=$(pwd)"
    echo "git_revision=$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo unavailable)"
    echo "conda_env=${CONDA_ENV}"
    echo "python=$(command -v python)"
    echo "python_version=$(python --version 2>&1)"
    echo "torch_version=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unavailable)"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "num_processes=4"
    echo "model_size=${MODEL_SIZE}"
    echo "lengths=${LENGTHS}"
    echo "max_length=${MAX_LENGTH}"
    echo "per_gpu_batch_sizes=${BATCH_SIZES}"
    echo "gradient_accumulation=${GRAD_ACCUMULATION}"
    echo "learning_rate=${LEARNING_RATE}"
    echo "weight_decay=${WEIGHT_DECAY}"
    echo "max_grad_norm=${MAX_GRAD_NORM}"
    echo "total_updates=${TOTAL_UPDATES}"
    echo "warmup_updates=${WARMUP_UPDATES}"
    echo "eval_every=${EVAL_EVERY}"
    echo "save_every=${SAVE_EVERY}"
    echo "validation_samples=${VALIDATION_SAMPLES}"
    echo "num_workers=${NUM_WORKERS}"
    echo "seed=${SEED}"
    echo "activation_checkpointing=$([[ "$ACTIVATION_CHECKPOINTING" == "1" ]] && echo enabled || echo disabled)"
    echo "output_dir=${OUTPUT_DIR}"
    echo "log_file=${log_file}"
    echo "resume=${RESUME:-none}"
    echo "same_length_global_batch=true"
    echo "resolved_model_config=printed_by_pretrain.train_after_launch"
    echo "gpu_inventory_begin"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 || echo unavailable
    echo "gpu_inventory_end"
    printf 'command='
    printf '%q ' "${cmd[@]}"
    printf '\nlaunch_manifest_end\n'
  } | tee "$log_file"
}

write_launch_manifest

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run; training was not started."
  exit 0
fi

# `setsid` makes the Accelerate launcher and its worker processes independent
# of the shell that started this script.  `nohup` keeps them alive across a
# terminal/session disconnect.  Use the absolute binary path captured after
# conda activation so the detached process cannot lose the fa environment.
nohup setsid "${cmd[@]}" >>"$log_file" 2>&1 < /dev/null &
training_pid=$!

sleep 2
if ! kill -0 "$training_pid" 2>/dev/null; then
  echo "Training exited during startup; log follows:" | tee -a "$log_file" >&2
  sed -n '1,240p' "$log_file" >&2 || true
  exit 1
fi

echo "Started PID ${training_pid}" | tee -a "$log_file"
echo "Follow log: tail -f ${log_file}" | tee -a "$log_file"
echo "Check process: ps -fp ${training_pid}" | tee -a "$log_file"
