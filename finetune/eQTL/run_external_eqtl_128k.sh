#!/usr/bin/env bash
set -euo pipefail

# Five 128k experiments, excluding Evo2-1B as requested. Four GPUs are used in
# two waves so that each large backbone has a dedicated 80 GB card.
PROJECT_ROOT="/gpfs_data/Projects/HelixPack"
MODEL_ROOT="/gpfs_data/Projects/GenomeModelProject/models"
RUN_ROOT="${PROJECT_ROOT}/runs/supplementary_experiments/20260917/eqtl-128000-external-batch8-5epoch"
OUTPUT_ROOT="${PROJECT_ROOT}/weights/finetune/eQTL/20260917/external-128000-batch8-5epoch"
SCRIPT="${PROJECT_ROOT}/finetune/eQTL/external_eqtl.py"
CONFIG="${PROJECT_ROOT}/config/external_eqtl_128k.json"
PYTHON_BIN="/root/miniconda3/envs/janusdna/bin/python"
mkdir -p "${RUN_ROOT}/logs"
LAUNCH_START_UNIX="$(date +%s)"
LAUNCH_START="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "experiment_start=${LAUNCH_START} context=128000 models=hyenadna,caduceus_ph,caduceus_ps,janusdna,ntv3" \
  | tee "${RUN_ROOT}/launcher.log"

batch_config() {
  "${PYTHON_BIN}" - "${CONFIG}" "$1" <<'PY'
import json
import sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
item = cfg["model_batch_sizes"][sys.argv[2]]
print(item["batch_size_per_device"], item["gradient_accumulation_steps"])
PY
}

read -r HYENA_BATCH HYENA_ACCUM <<<"$(batch_config hyenadna)"
read -r CPH_BATCH CPH_ACCUM <<<"$(batch_config caduceus_ph)"
read -r CPS_BATCH CPS_ACCUM <<<"$(batch_config caduceus_ps)"
read -r JANUS_BATCH JANUS_ACCUM <<<"$(batch_config janusdna)"
read -r NTV3_BATCH NTV3_ACCUM <<<"$(batch_config ntv3)"

run_one() {
  local gpu="$1" model="$2" path="$3" batch_size="$4" grad_accumulation="$5"
  local start_epoch start_unix end_epoch end_unix
  start_epoch="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  start_unix="$(date +%s)"
  echo "task_start=${start_epoch} model=${model} gpu=${gpu} batch_size=${batch_size} grad_accumulation=${grad_accumulation} effective_batch_size=8" \
    | tee "${RUN_ROOT}/logs/${model}.log"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT}" \
    --model "${model}" --model-path "${path}" \
    --config "${CONFIG}" \
    --output-root "${OUTPUT_ROOT}" --cell-type all \
    --context-length 128000 --epochs 5 --batch-size "${batch_size}" \
    --grad-accumulation "${grad_accumulation}" --learning-rate 2e-4 --weight-decay 0.1 \
    --warmup-ratio 0.1 --max-grad-norm 1.0 --seed 17 \
    --log-interval-steps 1 --eval-interval-epochs 1 \
    >> "${RUN_ROOT}/logs/${model}.log" 2>&1
  local status=$?
  set -e
  end_epoch="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  end_unix="$(date +%s)"
  echo "task_end=${end_epoch} model=${model} status=${status} elapsed_seconds=$((end_unix-start_unix))" \
    | tee -a "${RUN_ROOT}/logs/${model}.log"
  return "$status"
}

run_one 0 hyenadna "${MODEL_ROOT}/hyenadna-large-1m" "${HYENA_BATCH}" "${HYENA_ACCUM}" & p0=$!
run_one 1 caduceus_ph "${MODEL_ROOT}/caduceus-ph-131k" "${CPH_BATCH}" "${CPH_ACCUM}" & p1=$!
run_one 2 caduceus_ps "${MODEL_ROOT}/caduceus-ps-131k" "${CPS_BATCH}" "${CPS_ACCUM}" & p2=$!
run_one 3 janusdna "${MODEL_ROOT}/janusdna" "${JANUS_BATCH}" "${JANUS_ACCUM}" & p3=$!
set +e
wait "$p0"; s0=$?
wait "$p1"; s1=$?
wait "$p2"; s2=$?
wait "$p3"; s3=$?
set -e
echo "First wave exit codes: Hyena=${s0}, Caduceus-Ph=${s1}, Caduceus-PS=${s2}, JanusDNA=${s3}" | tee -a "${RUN_ROOT}/logs/launcher-status"

# NTv3 is run after the first wave, while retaining four-card scheduling and
# leaving room for its convolutional 128k activation footprint.
run_one 0 ntv3 "${MODEL_ROOT}/ntv3-100m-pre" "${NTV3_BATCH}" "${NTV3_ACCUM}"

LAUNCH_END_UNIX="$(date +%s)"
LAUNCH_END="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "experiment_end=${LAUNCH_END} elapsed_seconds=$((LAUNCH_END_UNIX-LAUNCH_START_UNIX))" \
  | tee "${RUN_ROOT}/logs/complete"
