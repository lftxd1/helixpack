#!/usr/bin/env bash
set -euo pipefail

# VeriComp equivalent of GeneZip's shell/dnalongbench/eqtl.sh.
# It trains/evaluates a binary eQTL head using mean-pooled bottleneck tokens.
# Repeated runs: 3 seeds (override with SEEDS="5 6 7").

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-fa}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/weights/pretrain/20260814/vericomp-residual-skip-30k/latest.pt}"
MODEL_NAME="${MODEL_NAME:-vericomp}"
FEATURE="${FEATURE:-backbone}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data/dnalongbench}"
CELL_TYPE="${CELL_TYPE:-Artery_Tibial}"
SEEDS="${SEEDS:-1 2 3}"
DATA_MAX_LENGTH="${DATA_MAX_LENGTH:-12800}"
POOLING_METHOD="${POOLING_METHOD:-mean}"
FREEZE_ENCODER="${FREEZE_ENCODER:-false}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-1}"
EPOCHS="${EPOCHS:-30}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
STRICT_UNCOMPRESSED_SHORT="${STRICT_UNCOMPRESSED_SHORT:-false}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/weights/finetune/eQTL/$(date +%Y%m%d)}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/runs/logs/finetune/eQTL/$(date +%Y%m%d)}"
LOG_FILE="${LOG_FILE:-}"
DETACH="${DETACH:-0}"
MAX_TRAIN_RECORDS="${MAX_TRAIN_RECORDS:-}"
MAX_VALID_RECORDS="${MAX_VALID_RECORDS:-}"
MAX_TEST_RECORDS="${MAX_TEST_RECORDS:-}"
DRY_RUN="${DRY_RUN:-0}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

pids=()
for SEED in $SEEDS; do
  cmd=(python -m finetune.eQTL.eqtl
    --checkpoint "$CHECKPOINT"
    --model "$MODEL_NAME"
    --feature "$FEATURE"
    --data-root "$DATA_ROOT"
    --cell-type "$CELL_TYPE"
    --data-max-length "$DATA_MAX_LENGTH"
    --pooling-method "$POOLING_METHOD"
    --batch-size "$BATCH_SIZE"
    --grad-accumulation "$GRAD_ACCUMULATION"
    --epochs "$EPOCHS"
    --lr "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --max-grad-norm "$MAX_GRAD_NORM"
    --num-workers "$NUM_WORKERS"
    --seed "$SEED"
    --output-dir "${OUTPUT_DIR}/seed-${SEED}")
  if [[ "$FREEZE_ENCODER" == "true" ]]; then
    cmd+=(--freeze-encoder)
  else
    cmd+=(--no-freeze-encoder)
  fi
  if [[ "$STRICT_UNCOMPRESSED_SHORT" == "true" ]]; then
    cmd+=(--strict-uncompressed-short)
  else
    cmd+=(--no-strict-uncompressed-short)
  fi
  [[ -n "$MAX_TRAIN_RECORDS" ]] && cmd+=(--max-train-records "$MAX_TRAIN_RECORDS")
  [[ -n "$MAX_VALID_RECORDS" ]] && cmd+=(--max-valid-records "$MAX_VALID_RECORDS")
  [[ -n "$MAX_TEST_RECORDS" ]] && cmd+=(--max-test-records "$MAX_TEST_RECORDS")

  printf 'Running command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  [[ "$DRY_RUN" == "1" ]] && continue

  if [[ "$DETACH" == "1" ]]; then
    mkdir -p "$LOG_DIR"
    if [[ -z "$LOG_FILE" ]]; then
      log_file="${LOG_DIR}/eqtl-seed-${SEED}-$(date +%Y%m%d-%H%M%S).log"
    else
      log_file="${LOG_FILE%.log}-seed-${SEED}.log"
    fi
    export PYTHONUNBUFFERED=1
    nohup setsid "${cmd[@]}" >"$log_file" 2>&1 < /dev/null &
    pid=$!
    sleep 2
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "eQTL process (seed ${SEED}) exited during startup; log follows:" >&2
      sed -n '1,160p' "$log_file" >&2 || true
      exit 1
    fi
    echo "Started eQTL seed ${SEED} PID ${pid}"
    echo "Log: ${log_file}"
    pids+=("$pid")
  else
    "${cmd[@]}"
  fi
done

[[ "$DRY_RUN" == "1" || "$DETACH" != "1" ]] && exit 0
echo "All eQTL seed runs started (${pids[*]})"
