#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# TokenGT LoRA comparison automation
# Runs:
#   1) sparse-lap-type with nn.Linear
#   2) sparse-lap-type with LoRA rank 4
#   3) sparse-lap-type with LoRA rank 16
#   4) dense-lap-type with nn.Linear
#   5) dense-lap-type with LoRA rank 4
#   6) dense-lap-type with LoRA rank 16
#
# For each variant:
#   train script -> test script -> save logs
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p comparison_logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "============================================================"
echo "Running TokenGT LoRA comparison"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

check_gpu() {
    echo "================ GPU CHECK ================"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi
    else
        echo "nvidia-smi not found. If you are on Mac, CUDA GPU will not be available."
    fi

    python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current cuda device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(0))
PY
    echo "==========================================="
}

run_one() {
    local input_name="$1"      # sparse or dense
    local linear_impl="$2"     # linear or lora
    local rank="$3"            # 0, 4, 16

    local train_script="${input_name}-lap-type.sh"
    local test_script="${input_name}-lap-type-test.sh"

    local tag="${input_name}_lap_type_${linear_impl}_rank${rank}"
    local log_dir="comparison_logs/${tag}"

    mkdir -p "$log_dir"

    echo ""
    echo "============================================================"
    echo "RUN: ${tag}"
    echo "Train script: ${train_script}"
    echo "Test script : ${test_script}"
    echo "============================================================"

    if [[ ! -f "$train_script" ]]; then
        echo "ERROR: Missing train script: $train_script"
        exit 1
    fi

    if [[ ! -f "$test_script" ]]; then
        echo "ERROR: Missing test script: $test_script"
        exit 1
    fi

    export LINEAR_IMPL="$linear_impl"
    export LORA_RANK="$rank"
    export EXP_SUFFIX="$tag"

    echo "LINEAR_IMPL=${LINEAR_IMPL}"
    echo "LORA_RANK=${LORA_RANK}"
    echo "EXP_SUFFIX=${EXP_SUFFIX}"

    echo ""
    echo "---------------- TRAINING: ${tag} ----------------"
    bash "$train_script" 2>&1 | tee "${log_dir}/train.log"

    echo ""
    echo "---------------- TESTING: ${tag} ----------------"
    bash "$test_script" 2>&1 | tee "${log_dir}/test.log"

    echo ""
    echo "Finished: ${tag}"
}

check_gpu

# Sparse experiments
run_one "sparse" "linear" "0"
run_one "sparse" "lora" "4"
run_one "sparse" "lora" "16"

# Dense experiments
run_one "dense" "linear" "0"
run_one "dense" "lora" "4"
run_one "dense" "lora" "16"

echo ""
echo "============================================================"
echo "All experiments completed."
echo "Logs saved in: comparison_logs/"
echo "============================================================"