#!/bin/bash
set -e

MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
)
DTYPE="bfloat16"
BATCH_SIZES=(1 4)
OUTPUT_LENS=(128 256 257 512)
TP_SIZES=(2 4)
INPUT_BUDGET=512

for model in "${MODELS[@]}"; do
    for tp in "${TP_SIZES[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            for ol in "${OUTPUT_LENS[@]}"; do
                mml=$((INPUT_BUDGET + ol))
                echo "=========================================="
                echo "Cleaning up compilation artifacts and cache"
                echo "=========================================="
                rm -rf /tmp/nxd_model/ /tmp/neuronxcc-*/ ~/.cache/neuron/
                rm -rf /var/tmp/neuron-compile-cache/
                echo "=========================================="
                echo "Model: $model | TP: $tp | BS: $bs | OL: $ol | MML: $mml"
                echo "=========================================="
                python run_benchmark.py \
                    --model "$model" \
                    --dtype "$DTYPE" \
                    --tensor-parallel-size "$tp" \
                    --max-num-seqs "$bs" \
                    --output-len "$ol" \
                    --max-model-len "$mml"
                echo "=========================================="
                echo "Cleaning up compilation artifacts and cache"
                echo "=========================================="
                rm -rf /tmp/nxd_model/ /tmp/neuronxcc-*/ ~/.cache/neuron/
                rm -rf /var/tmp/neuron-compile-cache/
            done
        done
    done
done
