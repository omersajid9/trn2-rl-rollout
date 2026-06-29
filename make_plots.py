#!/usr/bin/env python3
"""
make_plots.py

Scans logs/ for subdirectories that don't already have a plots/ subfolder.
For each qualifying run, reads run.log and generates 4 PNG files into
logs/<run-id>/plots/:

    memory.png   – host RAM + swap over time
    cpu.png      – CPU user % + system % over time
    hbm.png      – HBM (per-device) over time
    unified.png  – all three stacked in one figure

Run from /home/ubuntu/inf2/:
    python make_plots.py
"""

import re
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── constants ──────────────────────────────────────────────────────────────────

LOGS_DIR = Path("logs")

C_HOST     = "#2196F3"
C_SWAP     = "#90CAF9"
C_USER     = "#4CAF50"
C_SYS      = "#81C784"
HBM_COLORS = ["#E53935", "#FB8C00", "#8E24AA", "#00897B",
               "#3949AB", "#F4511E", "#039BE5", "#43A047"]

# (color, linestyle) per phase label — solid=START, dashed=DONE, same hue per event
PHASE_STYLES = {
    "PROGRAM STARTED":   ("#1A237E", "-"),
    "PROGRAM ENDED":     ("#3949AB", "--"),
    "CACHE_DROP START":  ("#BF360C", "-"),
    "CACHE_DROP DONE":   ("#E64A19", "--"),
    "VLLM_IMPORT START": ("#1B5E20", "-"),
    "VLLM_IMPORT DONE":  ("#388E3C", "--"),
    "COMPILATION START": ("#E65100", "-"),
    "COMPILATION DONE":  ("#FFA726", "--"),
    "GENERATION START":  ("#4A148C", "-"),
    "GENERATION DONE":   ("#AB47BC", "--"),
}
_DEFAULT_PHASE = ("#607D8B", ":")


# ── parser ─────────────────────────────────────────────────────────────────────

def parse_log(log_path: Path):
    """
    Read run.log and return a data dict, or None if the file has no MEM records.

    Reads the entire file (not gated by PROGRAM STARTED/ENDED) so the
    pre-run baseline is visible in the plots.
    """
    mem_re  = re.compile(r">>> \[MEM\] (\S+) (\{.*\}) <<<")
    post_re = re.compile(r">>> \[POST\] (\S+) (.+?) <<<")

    mem_records, post_events = [], []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            m = mem_re.match(line)
            if m:
                mem_records.append((
                    datetime.fromisoformat(m.group(1)),
                    json.loads(m.group(2)),
                ))
                continue
            m = post_re.match(line)
            if m:
                post_events.append((
                    datetime.fromisoformat(m.group(1)),
                    m.group(2).strip(),
                ))

    if not mem_records:
        return None

    t0     = mem_records[0][0]
    to_min = lambda dt: (dt - t0).total_seconds() / 60.0

    times_min = [to_min(dt) for dt, _ in mem_records]

    # HBM: only devices that appear at least once
    hbm_devices = sorted(
        {k for _, r in mem_records for k in r.get("hbm_gb", {})}, key=int
    )
    hbm_times  = {d: [] for d in hbm_devices}
    hbm_series = {d: [] for d in hbm_devices}
    for i, (_, r) in enumerate(mem_records):
        for dev in hbm_devices:
            val = r.get("hbm_gb", {}).get(dev)
            if val is not None:
                hbm_times[dev].append(times_min[i])
                hbm_series[dev].append(val)

    return {
        "times_min":    times_min,
        "host_mem":     [r["host_mem_used_gb"] for _, r in mem_records],
        "swap_mem":     [r["swap_used_gb"]      for _, r in mem_records],
        "cpu_user":     [r["cpu_user_pct"]      for _, r in mem_records],
        "cpu_sys":      [r["cpu_system_pct"]    for _, r in mem_records],
        "hbm_devices":  hbm_devices,
        "hbm_times":    hbm_times,
        "hbm_series":   hbm_series,
        "phase_events": [(to_min(dt), lbl) for dt, lbl in post_events],
    }


# ── subplot content ────────────────────────────────────────────────────────────

def draw_memory(ax, data):
    t = data["times_min"]
    ax.fill_between(t, data["host_mem"], alpha=0.28, color=C_HOST)
    ax.plot(t, data["host_mem"], lw=1.2, color=C_HOST, label="Host RAM (GB)")
    ax.fill_between(t, data["swap_mem"], alpha=0.55, color=C_SWAP)
    ax.plot(t, data["swap_mem"], lw=0.8, color=C_SWAP,  label="Swap (GB)")
    ax.set_ylabel("Memory (GB)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(axis="y", ls=":", alpha=0.4)


def draw_cpu(ax, data):
    t = data["times_min"]
    ax.fill_between(t, data["cpu_user"], alpha=0.28, color=C_USER)
    ax.plot(t, data["cpu_user"], lw=1.2, color=C_USER, label="CPU user %")
    ax.fill_between(t, data["cpu_sys"],  alpha=0.45, color=C_SYS)
    ax.plot(t, data["cpu_sys"],  lw=0.8, color=C_SYS,  label="CPU system %")
    ax.set_ylabel("CPU Usage (%)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(axis="y", ls=":", alpha=0.4)


def draw_hbm(ax, data):
    if data["hbm_devices"]:
        for j, dev in enumerate(data["hbm_devices"]):
            col = HBM_COLORS[j % len(HBM_COLORS)]
            ax.plot(data["hbm_times"][dev], data["hbm_series"][dev],
                    lw=1.4, color=col, label=f"Device {dev}")
    else:
        ax.text(0.5, 0.5, "No HBM data in log", transform=ax.transAxes,
                ha="center", va="center", color="grey")
    ax.set_ylabel("HBM (GB)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", ls=":", alpha=0.4)


# ── phase markers ──────────────────────────────────────────────────────────────

def draw_phase_markers(axes, phase_events):
    """
    Draw one vertical line per event on every axis in `axes`.
    Returns {label: handle} — one handle per unique label — for use in legends.
    Labels are NOT set on the axvline itself to keep ax.get_legend_handles_labels()
    clean; we attach them manually when building the legend.
    """
    seen = {}
    for t, label in phase_events:
        color, ls = PHASE_STYLES.get(label, _DEFAULT_PHASE)
        for i, ax in enumerate(axes):
            line = ax.axvline(t, color=color, lw=1.1, ls=ls, alpha=0.85, zorder=3)
            if i == 0 and label not in seen:
                seen[label] = line
    return seen


# ── shared helpers ─────────────────────────────────────────────────────────────

def format_xaxis(ax):
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.set_xlabel("Time (min)")


def add_top_legend(fig, data_handles, data_labels, phase_handles, run_id):
    """Place a single legend panel at the top of the figure."""
    handles = data_handles + list(phase_handles.values())
    labels  = data_labels  + list(phase_handles.keys())
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(handles), 6),
        fontsize=7.5,
        framealpha=0.93,
        edgecolor="#aaaaaa",
        title=f"Run: {run_id}",
        title_fontsize=8,
        handlelength=1.8,
    )


# ── plot builders ──────────────────────────────────────────────────────────────

def _save_single(draw_fn, data, path, run_id):
    """Generic builder for a single-subplot figure."""
    fig, ax = plt.subplots(figsize=(12, 4))
    draw_fn(ax, data)
    phase_handles = draw_phase_markers([ax], data["phase_events"])
    format_xaxis(ax)
    dh, dl = ax.get_legend_handles_labels()
    add_top_legend(fig, dh, dl, phase_handles, run_id)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_unified(data, path, run_id):
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"hspace": 0.06, "height_ratios": [1, 1, 1]},
    )
    ax_mem, ax_cpu, ax_hbm = axes

    draw_memory(ax_mem, data)
    draw_cpu(ax_cpu, data)
    draw_hbm(ax_hbm, data)
    phase_handles = draw_phase_markers(list(axes), data["phase_events"])
    format_xaxis(ax_hbm)

    # Deduplicate data handles across all three axes
    seen_labels, all_dh, all_dl = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen_labels:
                all_dh.append(h); all_dl.append(l)
                seen_labels.add(l)

    add_top_legend(fig, all_dh, all_dl, phase_handles, run_id)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── folder processing ──────────────────────────────────────────────────────────

def process_log_folder(log_dir: Path):
    plots_dir = log_dir / "plots"
    if plots_dir.exists():
        print(f"  [skip] {log_dir.name}  (plots/ already exists)")
        return

    log_file = log_dir / "run.log"
    if not log_file.exists():
        print(f"  [skip] {log_dir.name}  (no run.log)")
        return

    data = parse_log(log_file)
    if data is None:
        print(f"  [skip] {log_dir.name}  (no MEM records in run.log)")
        return

    plots_dir.mkdir()
    run_id = log_dir.name
    print(f"  → {run_id}  ({len(data['times_min'])} samples, "
          f"{len(data['phase_events'])} phase events)")

    _save_single(draw_memory, data, plots_dir / "memory.png",  run_id)
    _save_single(draw_cpu,    data, plots_dir / "cpu.png",     run_id)
    _save_single(draw_hbm,    data, plots_dir / "hbm.png",     run_id)
    _save_unified(data, plots_dir / "unified.png", run_id)

    print(f"    ✓ 4 plots saved → {plots_dir}")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    if not LOGS_DIR.exists():
        print(f"Error: '{LOGS_DIR}/' not found. Run this script from inf2/")
        return

    log_dirs = sorted(d for d in LOGS_DIR.iterdir() if d.is_dir())
    if not log_dirs:
        print("No subdirectories found in logs/")
        return

    print(f"Scanning {len(log_dirs)} log folder(s) in {LOGS_DIR}/\n")
    for log_dir in log_dirs:
        process_log_folder(log_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
