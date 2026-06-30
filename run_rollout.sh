#!/bin/bash

MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
)
DTYPE="bfloat16"
BATCH_SIZES=(1 3 4 5 12)
OUTPUT_LENS=(1 2 127 128 129 255 256 257 512)
TP_SIZES=(1 2)
INPUT_LENS=(64 100 127 128 129 200 255 256 257 512)
MAX_INPUT_LEN=512  # max of INPUT_LENS above

for model in "${MODELS[@]}"; do
    for tp in "${TP_SIZES[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            for ol in "${OUTPUT_LENS[@]}"; do
                mml=$((MAX_INPUT_LEN + ol))

                echo "=========================================="
                echo "Cleaning up compilation artifacts and cache"
                echo "=========================================="
                rm -rf /tmp/nxd_model/ /tmp/neuronxcc-*/ ~/.cache/neuron/
                rm -rf /var/tmp/neuron-compile-cache/

                echo "=========================================="
                echo "Model: $model | TP: $tp | BS: $bs | OL: $ol | MML: $mml"
                echo "Mode: uniform (--same-input-len)"
                echo "Input lens: ${INPUT_LENS[*]}"
                echo "=========================================="
                python rollout_benchmark.py \
                    --model "$model" \
                    --dtype "$DTYPE" \
                    --tensor-parallel-size "$tp" \
                    --max-num-seqs "$bs" \
                    --output-len "$ol" \
                    --max-model-len "$mml" \
                    --input-lens "${INPUT_LENS[@]}" \
                    --same-input-len

                echo "=========================================="
                echo "Model: $model | TP: $tp | BS: $bs | OL: $ol | MML: $mml"
                echo "Mode: mixed (heterogeneous batch)"
                echo "Input lens: ${INPUT_LENS[*]}"
                echo "=========================================="
                python rollout_benchmark.py \
                    --model "$model" \
                    --dtype "$DTYPE" \
                    --tensor-parallel-size "$tp" \
                    --max-num-seqs "$bs" \
                    --output-len "$ol" \
                    --max-model-len "$mml" \
                    --input-lens "${INPUT_LENS[@]}"

            done
        done
    done
done

echo "=========================================="
echo "Cleaning up compilation artifacts and cache"
echo "=========================================="
rm -rf /tmp/nxd_model/ /tmp/neuronxcc-*/ ~/.cache/neuron/
rm -rf /var/tmp/neuron-compile-cache/

echo "=========================================="
echo "All runs complete"
echo "=========================================="
