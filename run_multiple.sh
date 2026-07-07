#!/bin/bash
# Run both benchmarks across a grid of (tp × model × batch_size × sweep_mode).
#
# Loop order (outer → inner):
#   tp  →  model  →  batch_size  →  sweep_mode
#
# NOTE: run_mini_verl_benchmark.py does not accept --tensor-parallel-size, so
# it is launched with identical args for every tp value.  This is intentional:
# each (model, bs, sweep) triple is still cleanly bracketed by compilation
# cleanups even though mini_verl's tp is not configurable here.
#
# Each inner iteration:
#   1. cleanup compilations
#   2. run_mini_verl_benchmark.py
#   3. cleanup compilations
#   4. run_benchmark.py
#   5. cleanup compilations

set -uo pipefail

# ─── SWEEP PARAMETERS ─────────────────────────────────────────────────────────
TP_SIZES=(1 2)
MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
)
BATCH_SIZES=(1 4)
SWEEP_MODES=("sequential" "random" "alternating")

# ─── PYTHON INTERPRETERS (one per venv) ───────────────────────────────────────
PYTHON_MINI_VERL="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python3"
PYTHON_VLLM="/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python3"

# ─── FIXED PARAMETERS ─────────────────────────────────────────────────────────
DTYPE="bfloat16"
INPUT_LENS="64 127 128 256 257 512"
OUTPUT_LENS="64 127 128 256 257 512"
MAX_CACHE_LEN=1024        # --max-cache-len  (mini_verl)
MAX_MODEL_LEN=1024        # --max-model-len  (vllm)
DECODE_INPUT_LEN=128

# ─── HELPERS ──────────────────────────────────────────────────────────────────
cleanup() {
    echo "--- cleanup: removing compilation artifacts ---"
    pkill -f neuron-monitor 2>/dev/null || true
    rm -rf /tmp/nxd_model/ /tmp/neuronxcc-*/
    rm -rf ~/.cache/neuron/ /var/tmp/neuron-compile-cache/
}

# ─── MAIN GRID ────────────────────────────────────────────────────────────────
for tp in "${TP_SIZES[@]}"; do
    for model in "${MODELS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            for sweep in "${SWEEP_MODES[@]}"; do

                echo "========================================================================="
                echo "  TP=$tp | Model=$model | BS=$bs | Sweep=$sweep"
                echo "========================================================================="

                # 1. cleanup before mini_verl run
                cleanup

                # 2. mini_verl benchmark (no tp arg)
                echo "--- run_mini_verl_benchmark ---"
                "$PYTHON_MINI_VERL" run_mini_verl_benchmark.py \
                    --model          "$model" \
                    --dtype          "$DTYPE" \
                    --batch-size     "$bs" \
                    --max-cache-len  "$MAX_CACHE_LEN" \
                    --input-lens     $INPUT_LENS \
                    --output-lens    $OUTPUT_LENS \
                    --decode-input-len "$DECODE_INPUT_LEN" \
                    --test           both \
                    --sweep-mode     "$sweep"

                # 3. cleanup between the two benchmarks
                cleanup

                # 4. vllm benchmark (passes tp)
                echo "--- run_benchmark ---"
                "$PYTHON_VLLM" run_benchmark.py \
                    --model                "$model" \
                    --dtype                "$DTYPE" \
                    --tensor-parallel-size "$tp" \
                    --max-num-seqs         "$bs" \
                    --max-model-len        "$MAX_MODEL_LEN" \
                    --input-lens           $INPUT_LENS \
                    --output-lens          $OUTPUT_LENS \
                    --decode-input-len     "$DECODE_INPUT_LEN" \
                    --test                 both \
                    --sweep-mode           "$sweep"

                # 5. cleanup after
                cleanup

            done  # sweep
        done  # batch_size
    done  # model
done  # tp
