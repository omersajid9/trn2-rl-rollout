#!/bin/bash

# Change into the repo directory — cron runs with a minimal environment,
# so we must cd explicitly rather than relying on a working directory.
cd /home/ubuntu/trn2-rl-rollout || exit 1

# Set git identity for automated commits.
git config user.name "log-sync"
git config user.email "log-sync@trn2"

# Snapshot the important, ephemeral /tmp artifacts into logs/ so they get
# committed. The Neuron Runtime writes an untruncated per-NEFF memory table to
# /tmp/neuron_mem_table_device_*_hbm_*.log on OOM; that lives in /tmp and would
# be lost when the instance goes away. Copy any that currently exist into a
# timestamped dir so an overnight OOM crash is still visible tomorrow.
# (Compile caches like /tmp/nxd_model and /tmp/neuronxcc-* are huge and
# regenerable, so we deliberately do NOT copy those.)
snap_dir="logs/_tmp_artifacts/$(date -u '+%Y%m%dT%H%M%SZ')"
if ls /tmp/neuron_mem_table_device_*_hbm_*.log >/dev/null 2>&1; then
    mkdir -p "$snap_dir"
    cp -f /tmp/neuron_mem_table_device_*_hbm_*.log "$snap_dir"/ 2>/dev/null || true
fi

# Stage logs and supporting scripts.
git add logs* run* push_logs.sh mem_check.py memory_config.conf make_plots.py prompts.py rollout_benchmark.py

# Commit the staged changes.
# '|| true' prevents the script from failing if there is nothing new to commit
# (git exits with code 1 when there are no changes).
git commit -m "auto-save: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" || true

# Push to the master branch on the 'origin' remote.
# '-u' sets the upstream tracking reference so future bare 'git push'
# commands know where to send commits.
git push -u origin master
