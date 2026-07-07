#!/bin/bash
# Sweep colocate GRPO training cells across a grid of (model × tp × batch_size ×
# prefill_chunk × ppo_micro_batch_size).
#
# Loop order (outer → inner):
#   model  →  tp  →  batch_size  →  prefill_chunk  →  ppo_micro_batch_size
#
# Each inner iteration:
#   1. cleanup (kill stale neuron-monitor + Ray workers, drop compile cache)
#   2. run_colocate_benchmark.py  (writes logs/<cell>/result.json + run.log + out.log)
#   3. cleanup
#
# A cell that OOMs or crashes is recorded in result.json (succeeded=false) and
# the sweep continues — no cell failure can abort the grid.
#
# To run a subset of the grid, override the arrays inline:
#   MODELS="Qwen/Qwen2.5-0.5B-Instruct" TP_SIZES="1" BATCH_SIZES="8" \
#       bash run_colocate_multiple.sh

set -uo pipefail

# ─── SWEEP PARAMETERS ─────────────────────────────────────────────────────────
MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
)
# trn2.3xlarge has 4 NeuronCores (LNC-pair IDs 0-3, 96 GB HBM total).
# world_size = procs_per_node must be a power-of-two topology-aligned value.
# With 4 cores: TP=1 → fsdp=4 ranks; TP=2 → fsdp=2 ranks (2×2 grid).
TP_SIZES=(1 2)
# OOM sweep: increase BS until we hit HBM limit.  With 0.5B fsdp=4 each shard
# is ~250 MB; with 1.5B fsdp=4 each shard is ~750 MB.  KV-cache + optimizer
# offloading during training means small batches can still OOM at high N.
BATCH_SIZES=(4 8 16)
PREFILL_CHUNKS=(-1 128)   # -1 = no chunk; 128 must divide MAX_PROMPT_LEN
MBS_SIZES=(1 2)           # actor.ppo_micro_batch_size

# ─── PYTHON INTERPRETER ───────────────────────────────────────────────────────
PYTHON="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python3"

# ─── FIXED PARAMETERS ─────────────────────────────────────────────────────────
DTYPE="bfloat16"
CORES=4                   # NeuronCores: trn2.3xlarge has 4 (IDs 0-3, 96 GB HBM)
MAX_PROMPT_LEN=128        # keep prompt + response <= 512 (neuronx-cc compile limit)
MAX_RESP_LEN=128
ROLLOUT_N=4               # GRPO group size
TOTAL_STEPS=8             # > perf_warmup_steps (3) for a valid avg breakdown

# NEFF cache — override if you want a persistent shared cache across cells.
export TORCH_NEURONX_NEFF_CACHE_DIR="${TORCH_NEURONX_NEFF_CACHE_DIR:-/tmp/neff_cache}"
export TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR="${TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR:-/tmp/neff_local_cache}"

# ─── HELPERS ──────────────────────────────────────────────────────────────────
cleanup() {
    echo "--- cleanup: killing stale monitors and workers ---"
    pkill -f neuron-monitor 2>/dev/null || true
    # Graceful Ray shutdown first, then force-kill lingering workers.
    "$PYTHON" -c "import ray; ray.init(ignore_reinit_error=True); ray.shutdown()" \
        2>/dev/null || true
    # Send SIGTERM first, give workers 10 s to flush, then SIGKILL.
    pkill -f "ray::" 2>/dev/null || true
    sleep 10
    pkill -9 -f "ray::" 2>/dev/null || true
    # Also clear the local NRT lock dir so the next run doesn't hit stale locks.
    echo "--- cleanup: dropping compile artifacts and locks ---"
    rm -rf /tmp/nxd_model/ /tmp/neuronxcc-*/
    rm -rf /tmp/local_cache/ /tmp/ray/
    rm -rf ~/.cache/neuron/ /var/tmp/neuron-compile-cache/
    sleep 10
}

# ─── MAIN GRID ────────────────────────────────────────────────────────────────
for model in "${MODELS[@]}"; do
    for tp in "${TP_SIZES[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            for chunk in "${PREFILL_CHUNKS[@]}"; do
                for mbs in "${MBS_SIZES[@]}"; do

                    echo "========================================================================="
                    echo "  Model=$model | TP=$tp | BS=$bs | Chunk=$chunk | MBS=$mbs"
                    echo "========================================================================="

                    # 1. cleanup before run
                    cleanup

                    # 2. run one colocate cell
                    echo "--- run_colocate_benchmark ---"
                    "$PYTHON" run_colocate_benchmark.py \
                        --model               "$model" \
                        --dtype               "$DTYPE" \
                        --cores               "$CORES" \
                        --tp-size             "$tp" \
                        --train-batch-size    "$bs" \
                        --ppo-micro-batch-size "$mbs" \
                        --rollout-n           "$ROLLOUT_N" \
                        --max-prompt-length   "$MAX_PROMPT_LEN" \
                        --max-response-length "$MAX_RESP_LEN" \
                        --prefill-chunk-size  "$chunk" \
                        --total-steps         "$TOTAL_STEPS" \
                    || echo "--- cell exited non-zero (OOM or crash); continuing ---"

                    # 3. cleanup after run
                    cleanup

                done  # mbs
            done  # chunk
        done  # bs
    done  # tp
done  # model
