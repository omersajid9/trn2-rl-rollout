#!/bin/bash
# Run both benchmarks across a grid of (tp × model × batch_size × prefill_chunk_size).
#
# Loop order (outer → inner):
#   tp  →  model  →  batch_size  →  prefill_chunk_size
#
# NOTE: run_mini_verl_benchmark.py does not accept --tensor-parallel-size, so
# it is launched with identical args for every tp value.  This is intentional:
# each (model, bs, chunk) triple is still cleanly bracketed by compilation
# cleanups even though mini_verl's tp is not configurable here.
#
# Prefill chunk sizes tested:
#   0   – no chunking (one NEFF per unique input_len; baseline)
#   64  – fine-grained; all shapes are multiples of 64; least padding waste
#   128 – balanced; aligns with Neuron 128-bucket convention; 4 unique shapes
#   256 – coarsest; only 2 unique shapes (256, 512); most padding waste
#
# Each inner iteration:
#   1. cleanup compilations
#   2. run_mini_verl_benchmark.py  (once per prefill_chunk_size)
#   3. cleanup compilations
#   4. run_benchmark.py            (once, after all chunk-size runs)
#   5. cleanup compilations

set -uo pipefail

# ─── SWEEP PARAMETERS ─────────────────────────────────────────────────────────
TP_SIZES=(1 2 4)
MODELS=(
    "Qwen/Qwen3-8B"
)
BATCH_SIZES=(1 4 8 16)

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

PREFILL_CHUNK_SIZES=(0 64 100 128 200 256)

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

            echo "========================================================================="
            echo "  TP=$tp | Model=$model | BS=$bs"
            echo "========================================================================="

            # 2. mini_verl benchmark — one run per prefill chunk size
            for chunk in "${PREFILL_CHUNK_SIZES[@]}"; do

                echo "--- run_mini_verl_benchmark (prefill_chunk_size=$chunk) ---"

                # 1. cleanup before each mini_verl run
                cleanup

                "$PYTHON_MINI_VERL" run_mini_verl_benchmark.py \
                    --model               "$model" \
                    --dtype               "$DTYPE" \
                    --batch-size          "$bs" \
                    --max-cache-len       "$MAX_CACHE_LEN" \
                    --input-lens          $INPUT_LENS \
                    --output-lens         $OUTPUT_LENS \
                    --decode-input-len    "$DECODE_INPUT_LEN" \
                    --prefill-chunk-size  "$chunk" \
                    --test                both \

            done  # prefill_chunk_size

            # 3. cleanup between mini_verl and vllm
            cleanup

            # 4. vllm benchmark (passes tp; no prefill-chunk-size concept)
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

            # 5. cleanup after
            cleanup

        done  # batch_size
    done  # model
done  # tp
