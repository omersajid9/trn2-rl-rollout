#!/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python3
"""
Smoke test for the colocate GRPO benchmark.

Runs the smallest possible colocate cell to verify the full stack before
launching run_colocate_multiple.sh:

  - imports (verl, ray, omegaconf, tensordict, mini_verl) all resolve
  - Ray initialises and allocates NeuronCores
  - the colocate FSDP actor + rollout model loads and compiles
  - 2 training steps complete (too few for a perf avg, but enough to confirm
    the loop runs: rollout → log_prob → reward → advantage → update_actor)

Expected wall time: 20–40 min on first run (NEFF compilation dominates).
Subsequent runs reuse the NEFF cache and finish in a few minutes.

Run from trn2-rl-rollout/:
    python run_colocate_smoke.py
"""

import sys

# Minimal args — override anything here before running the full grid.
sys.argv = [
    "run_colocate_smoke.py",
    "--model",                 "Qwen/Qwen2.5-0.5B-Instruct",
    "--cores",                 "4",
    "--tp-size",               "1",
    "--train-batch-size",      "4",
    "--ppo-micro-batch-size",  "1",
    "--rollout-n",             "2",
    "--max-prompt-length",     "64",
    "--max-response-length",   "64",
    "--prefill-chunk-size",    "-1",
    "--total-steps",           "2",
]

from run_colocate_benchmark import main  # noqa: E402

main()
