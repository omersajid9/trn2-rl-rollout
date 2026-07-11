#!/bin/bash
# Run the mini_verl ROLLOUT benchmark across a grid of (tp × model × batch_size × prefill_chunk_size).
#
# Key differences from run_multiple.sh:
#   - --tp-size is actually passed to run_mini_verl_rollout.py (the critical fix)
#   - --do-sample: stochastic generation matching real RL rollout (not greedy)
#   - --n-samples: GRPO-style N completions per prompt (effective batch = bs × n_samples)
#   - vllm section dropped — rollout-only focus
#
# Loop order (outer → inner):
#   tp  →  model  →  batch_size  →  n_samples  →  prefill_chunk_size
#
# Prefill chunk sizes tested:
#   0   – no chunking (one NEFF per unique input_len; baseline)
#   64  – fine-grained; all shapes are multiples of 64
#   128 – balanced; aligns with Neuron 128-bucket convention

set -uo pipefail

# ─── SWEEP PARAMETERS ─────────────────────────────────────────────────────────
TP_SIZES=(1 2 4)
MODELS=(
    "Qwen/Qwen3-8B"
)
BATCH_SIZES=(1 4 8 16)
N_SAMPLES=(1 4 8)          # extend to (1 4 8) to sweep GRPO group sizes

# ─── PYTHON INTERPRETER ───────────────────────────────────────────────────────
PYTHON_MINI_VERL="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python3"

# ─── FIXED PARAMETERS ─────────────────────────────────────────────────────────
DTYPE="bfloat16"
INPUT_LENS="64 127 128 256 257 512"
OUTPUT_LENS="64 127 128 256 257 512"
MAX_CACHE_LEN=1024
DECODE_INPUT_LEN=128
TEMPERATURE=1.0

PREFILL_CHUNK_SIZES=(64 127 128 256 257 512)

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
        for ns in "${N_SAMPLES[@]}"; do
            for bs in "${BATCH_SIZES[@]}"; do

                echo "========================================================================="
                echo "  TP=$tp | Model=$model | BS=$bs | N_SAMPLES=$ns"
                echo "  Effective engine batch: $((bs * ns))"
                echo "========================================================================="

                for chunk in "${PREFILL_CHUNK_SIZES[@]}"; do

                    echo "--- run_mini_verl_rollout (tp=$tp, chunk=$chunk, n_samples=$ns) ---"

                    cleanup

                    "$PYTHON_MINI_VERL" run_mini_verl_rollout.py \
                        --model               "$model"           \
                        --tp-size             "$tp"              \
                        --dtype               "$DTYPE"           \
                        --batch-size          "$bs"              \
                        --n-samples           "$ns"              \
                        --max-cache-len       "$MAX_CACHE_LEN"   \
                        --input-lens          $INPUT_LENS        \
                        --output-lens         $OUTPUT_LENS       \
                        --decode-input-len    "$DECODE_INPUT_LEN" \
                        --prefill-chunk-size  "$chunk"           \
                        --test                both               \
                        --do-sample                              \
                        --temperature         "$TEMPERATURE"

                    cleanup

                done  # prefill_chunk_size
            done  # n_samples
        done  # batch_size
    done  # model
done  # tp
