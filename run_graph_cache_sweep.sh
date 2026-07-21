#!/bin/bash
# Experiment 2: Graph-cache memory scaling (Q2/Q3/Q4 from the mini-verl vs
# vLLM HBM thread).
#
# Forces --prefill-chunk-size 0 so every distinct input_len is a genuinely
# distinct NEFF (no chunk-boundary reuse collapsing shapes onto one graph).
# Sweeps a wide range of input_lens in ONE long-lived process and, after each
# shape's COLD+WARM pair, takes a synchronous neuron-monitor snapshot to
# record `model_code` (NEFF HBM) plus the full per-category hbm_breakdown —
# this is the "marginal HBM per cached graph" series.
#
# A REVISIT pass then re-runs every input_len once more, in the same order.
#   - recompiled_graphs == 0 for every revisit -> all N NEFFs coexisted, no
#     eviction observed.
#   - recompiled_graphs > 0 on an early shape's revisit -> that NEFF was
#     evicted under memory pressure from the later shapes and had to
#     recompile. This is the direct falsification test for "does the Neuron
#     runtime evict old NEFFs once too many accumulate".
#
# Output: logs_graphcache/<run_dir>/graph_cache_sweep.json has one record per
# (input_len, phase) with model_code_total_gb + hbm_breakdown_gb, ready to
# plot vs. graphs_generated_cumulative.

set -uo pipefail

# ─── NEFF MEMORY LOGGING ──────────────────────────────────────────────────────
# Per-NEFF memory breakdown on every model load (TDRV:dml_log_dev_neff_mem):
# model code / constants / scratchpad / runtime / dma-rings, plus the full
# per-NEFF OOM table + NEFF-id->name mapping on OOM. Captured into each run's
# out.log (stderr is redirected there). run_mini_verl_rollout.py also sets this
# via setdefault; exporting here makes it explicit and overridable.
#   https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/explore/device-memory.html
export NEURON_RT_LOG_LEVEL_TDRV="${NEURON_RT_LOG_LEVEL_TDRV:-info}"

# ─── SWEEP PARAMETERS ─────────────────────────────────────────────────────────
TP_SIZES=(1)              # extend to (1 2 4) to see if HBM headroom differs by TP
MODEL="Qwen/Qwen3-8B"
BATCH_SIZE=1

# Wide input_len sweep: COUNT distinct shapes, STEP tokens apart, starting at
# START -> COUNT distinct NEFFs (chunk=0). Defaults to 200 shapes from 64 up
# to 2054 in steps of 10 — deliberately far more graphs than any reasonable
# runtime cache is likely to hold, to find the eviction wall (Q3/Q4).
START=64
STEP=10
COUNT=200

INPUT_LENS_ARR=()
for ((i = 0; i < COUNT; i++)); do
    INPUT_LENS_ARR+=($((START + i * STEP)))
done
INPUT_LENS="${INPUT_LENS_ARR[*]}"
MAX_INPUT_LEN=${INPUT_LENS_ARR[-1]}
MAX_CACHE_LEN=$((MAX_INPUT_LEN + 64))   # headroom above the largest prefill shape

# Unused by graphcache mode itself, but the shared max_cache_len guard in
# mini_verl_run() folds these into its sizing check — keep them small so the
# guard doesn't inflate max_cache_len unnecessarily.
OUTPUT_LENS=1
DECODE_INPUT_LEN=64

# ─── PYTHON INTERPRETER ───────────────────────────────────────────────────────
PYTHON_MINI_VERL="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python3"

# ─── HELPERS ──────────────────────────────────────────────────────────────────
cleanup() {
    echo "--- cleanup: removing compilation artifacts ---"
    pkill -f neuron-monitor 2>/dev/null || true
    rm -rf /tmp/nxd_model/ /tmp/neuronxcc-*/
    rm -rf ~/.cache/neuron/ /var/tmp/neuron-compile-cache/
}

# ─── MAIN GRID ────────────────────────────────────────────────────────────────
for tp in "${TP_SIZES[@]}"; do

    echo "========================================================================="
    echo "  GRAPH-CACHE SWEEP | Model=$MODEL | TP=$tp | BS=$BATCH_SIZE"
    echo "  input_lens: $COUNT shapes, $START..$MAX_INPUT_LEN step $STEP"
    echo "  (3 passes/shape: COLD+WARM forward, then REVISIT -> ~$((COUNT * 3)) rollouts)"
    echo "========================================================================="

    # Cold environment: no stale NEFFs / HLO cache from a prior run leaking
    # into this process's "distinct graph" count.
    cleanup

    "$PYTHON_MINI_VERL" run_mini_verl_rollout.py \
        --model               "$MODEL"            \
        --tp-size             "$tp"                \
        --batch-size          "$BATCH_SIZE"        \
        --max-cache-len       "$MAX_CACHE_LEN"     \
        --input-lens          $INPUT_LENS          \
        --output-lens         "$OUTPUT_LENS"       \
        --decode-input-len    "$DECODE_INPUT_LEN"  \
        --prefill-chunk-size  0                    \
        --test                graphcache           \
        --no-do-sample

    cleanup

done  # tp
