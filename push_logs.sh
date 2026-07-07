#!/bin/bash

# Change into the repo directory — cron runs with a minimal environment,
# so we must cd explicitly rather than relying on a working directory.
cd /home/ubuntu/trn2-rl-rollout || exit 1

# Stage everything inside the logs/ folder.
# 'git add <path>' only stages changes under that path.
git add logs run* push_logs.sh mem_check.py memory_config.conf make_plots.py prompts.py rollout_benchmark.py

# Commit the staged changes.
# '|| true' prevents the script from failing if there is nothing new to commit
# (git exits with code 1 when there are no changes).
git commit -m "auto-save: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" || true

# Push to the master branch on the 'origin' remote.
# '-u' sets the upstream tracking reference so future bare 'git push'
# commands know where to send commits.
git push -u origin master
