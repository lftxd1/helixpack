#!/usr/bin/env bash
set -euo pipefail

# VeriComp DNALongBench ETGP launcher.  Logs are unbuffered and persistent when
# DETACH=1; no .pid sidecar is created.
# Repeated runs: 3 seeds (override with SEEDS="5 6 7").

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-fa}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/weights/pretrain/20260814/vericomp-residual-skip-30k/latest.pt}"
MODEL_NAME="${MODEL_NAME:-vericomp}"
FEATURE="${FEATURE:-backbone}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data/dnalongbench}"
CELL_TYPE="${CELL_TYPE:-CRISPRi_EPI_K562_hg19}"
SEEDS="${SEEDS:-1 2 3}"
DATA_MAX_LENGTH="${DATA_MAX_LENGTH:-12800}"
POOLING_METHOD="${POOLING_METHOD:-mean}"
DUAL_STRAND="${DUAL_STRAND:-false}"
FREEZE_ENCODER="${FREEZE_ENCODER:-false}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-1}"
EPOCHS="${EPOCHS:-30}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
TEST_EVERY="${TEST_EVERY:-100}"
NUM_WORKERS="${NUM_WORKERS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/weights/finetune/ETGP/$(date +%Y%m%d)}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/runs/logs/finetune/ETGP/$(date +%Y%m%d)}"
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
  cmd=(python -m finetune.ETGP.etgp
    --checkpoint "$CHECKPOINT" --model "$MODEL_NAME" --feature "$FEATURE"
    --data-root "$DATA_ROOT" --cell-type "$CELL_TYPE"
    --data-max-length "$DATA_MAX_LENGTH" --pooling-method "$POOLING_METHOD"
    --batch-size "$BATCH_SIZE" --grad-accumulation "$GRAD_ACCUMULATION"
    --epochs "$EPOCHS" --lr "$LEARNING_RATE" --weight-decay "$WEIGHT_DECAY"
    --max-grad-norm "$MAX_GRAD_NORM" --test-every "$TEST_EVERY"
    --num-workers "$NUM_WORKERS" --output-dir "${OUTPUT_DIR}/seed-${SEED}")
  cmd+=(--seed "$SEED")
  if [[ "$DUAL_STRAND" == "true" ]]; then cmd+=(--dual-strand); else cmd+=(--no-dual-strand); fi
  if [[ "$FREEZE_ENCODER" == "true" ]]; then cmd+=(--freeze-encoder); else cmd+=(--no-freeze-encoder); fi
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
      log_file="${LOG_DIR}/etgp-seed-${SEED}-$(date +%Y%m%d-%H%M%S).log"
    else
      log_file="${LOG_FILE%.log}-seed-${SEED}.log"
    fi
    export PYTHONUNBUFFERED=1
    nohup setsid "${cmd[@]}" >"$log_file" 2>&1 < /dev/null &
    pid=$!
    sleep 2
    kill -0 "$pid" 2>/dev/null || { sed -n '1,160p' "$log_file" >&2 || true; exit 1; }
    echo "Started ETGP seed ${SEED} PID ${pid}"
    echo "Log: ${log_file}"
    pids+=("$pid")
  else
    "${cmd[@]}"
  fi
done

[[ "$DRY_RUN" == "1" || "$DETACH" != "1" ]] && exit 0
echo "All ETGP seed runs started (${pids[*]})"
