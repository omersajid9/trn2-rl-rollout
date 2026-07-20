#!/usr/bin/env python3
"""
make_plots.py

Scans logs/ for subdirectories that don't already have a plots/ subfolder.
For each qualifying run, reads run.log and generates PNG files into
logs/<run-id>/plots/:

    memory.png        – host RAM + swap over time
    cpu.png           – CPU user % + system % over time
    hbm.png           – HBM (per-device) over time
    neuroncore.png    – NeuronCore utilization (per-core) over time
    hbm_breakdown.png – HBM breakdown (tensors/constants/model_code/
                        model_shared_scratchpad/runtime_memory), one
                        subplot per category, one line per device
    unified.png       – memory/cpu/hbm/neuroncore stacked in one figure
    latency.png       – per-size PREFILL/DECODE latency (COLD vs WARM)

Run from /home/ubuntu/inf2/:
    python make_plots.py
"""

import re
import json
from datetime import datetime
from pathlib import Path
import shutil

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
NC_COLORS  = ["#FF6F00", "#FF8F00", "#FFA000", "#FFB300",
              "#FFC107", "#FFD54F", "#FFE082", "#FFECB3"]

# (color, linestyle) per phase label — solid=START, dashed=DONE, same hue per event
PHASE_STYLES = {
    # ── program lifecycle ──────────────────────────── near-black, lw=2.5
    "PROGRAM START":       ("#212121", "-",  2.5),
    "PROGRAM DONE":        ("#212121", "--", 2.5),
    "PROGRAM STARTED":     ("#212121", "-",  2.5),
    "PROGRAM ENDED":       ("#212121", "--", 2.5),

    # ── cache drops ────────────────────────────────── bold red, lw=2.0
    "CACHE_DROP START":    ("#D32F2F", "-",  2.0),
    "CACHE_DROP DONE":     ("#D32F2F", "--", 2.0),

    # ── setup / loading ────────────────────────────── steel-blue family, lw=1.6
    "TOKENIZER START":     ("#1565C0", "-",  1.6),
    "TOKENIZER DONE":      ("#1565C0", "--", 1.6),
    "VLLM_IMPORT START":   ("#1976D2", "-",  1.6),
    "VLLM_IMPORT DONE":    ("#1976D2", "--", 1.6),
    "COMPILATION START":   ("#1E88E5", "-",  1.6),
    "COMPILATION DONE":    ("#1E88E5", "--", 1.6),
    "WARMUP START":        ("#90A4AE", "-",  1.2),
    "WARMUP DONE":         ("#90A4AE", "--", 1.2),
    "GENERATION START":    ("#00838F", "-",  1.4),
    "GENERATION DONE":     ("#00838F", "--", 1.4),
    "MODEL_LOAD START":    ("#0277BD", "-",  1.6),
    "MODEL_LOAD DONE":     ("#0277BD", "--", 1.6),
    "MODEL_DEVICE START":  ("#0288D1", "-",  1.4),
    "MODEL_DEVICE DONE":   ("#0288D1", "--", 1.4),
    "TORCH_COMPILE START": ("#039BE5", "-",  1.4),
    "TORCH_COMPILE DONE":  ("#039BE5", "--", 1.4),

    # ── NxDI backend ───────────────────────────────── orange/amber family, lw=1.8
    "NXDI_COMPILE START":  ("#E65100", "-",  1.8),
    "NXDI_COMPILE DONE":   ("#E65100", "--", 1.8),
    "NXDI_LOAD START":     ("#EF6C00", "-",  1.6),  # no DONE — followed by NXDI_READY
    "NXDI_READY DONE":     ("#FB8C00", "--", 1.6),

    # ── Trace backend ──────────────────────────────── pink/rose family, lw=1.6
    "TRACE_COMPILE START": ("#AD1457", "-",  1.6),
    "TRACE_COMPILE DONE":  ("#AD1457", "--", 1.6),
    "TRACE_LOAD START":    ("#C2185B", "-",  1.4),
    "TRACE_LOAD DONE":     ("#C2185B", "--", 1.4),
    "TRACE_READY DONE":    ("#E91E63", "--", 1.4),

    # ── PREFILL family ─────────────────────────────────── green, lw=2.5/1.4/1.0
    "PREFILL START":       ("#1B5E20", "-",  2.5),   # outer PREFILL_TEST wrapper
    "PREFILL DONE":        ("#1B5E20", "--", 2.5),
    "PREFILL_COLD START":  ("#388E3C", "-",  1.4),   # nxdi-style: each cold run
    "PREFILL_COLD DONE":   ("#388E3C", "--", 1.4),
    "PREFILL_WARM START":  ("#81C784", "-",  1.0),   # nxdi-style: each warm run
    "PREFILL_WARM DONE":   ("#81C784", "--", 1.0),
    "PREFILL_RUN START":   ("#66BB6A", "-",  1.4),   # torch-style: single run per size
    "PREFILL_RUN DONE":    ("#66BB6A", "--", 1.4),

    # ── DECODE family ──────────────────────────────────── purple, lw=2.5/1.4/1.0
    "DECODE START":        ("#311B92", "-",  2.5),   # outer DECODE_TEST wrapper
    "DECODE DONE":         ("#311B92", "--", 2.5),
    "DECODE_COLD START":   ("#4527A0", "-",  1.4),   # nxdi-style: each cold run
    "DECODE_COLD DONE":    ("#4527A0", "--", 1.4),
    "DECODE_WARM START":   ("#9575CD", "-",  1.0),   # nxdi-style: each warm run
    "DECODE_WARM DONE":    ("#9575CD", "--", 1.0),
    "DECODE_RUN START":    ("#7E57C2", "-",  1.4),   # torch-style: single run per size
    "DECODE_RUN DONE":     ("#7E57C2", "--", 1.4),
}
_DEFAULT_PHASE = ("#607D8B", ":", 0.9)

# Regex used by both the parser (latency extraction) and draw_phase_markers
_PER_SIZE_RE = re.compile(
    r"^(PREFILL|DECODE)_(\d+)_(COLD|WARM)\s+(START|DONE)$"
)


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
    _lat_raw = []   # (datetime, (phase, size_str, temp, action)) — raw before normalization

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
                dt    = datetime.fromisoformat(m.group(1))
                raw   = m.group(2).strip()

                # Capture raw label for latency extraction before normalization
                lat_m = _PER_SIZE_RE.match(raw)
                if lat_m:
                    _lat_raw.append((dt, lat_m.groups()))

                # Normalize label for the timeline markers:
                label = raw
                # PREFILL_TEST / DECODE_TEST → PREFILL / DECODE  (outer wrapper)
                label = re.sub(r"(PREFILL|DECODE)_TEST\s+(START|DONE)", r"\1 \2", label)
                # PREFILL_64_COLD START → PREFILL_COLD START  (nxdi-style: has _COLD/_WARM)
                # DECODE_64_COLD START  → DECODE_COLD START
                label = re.sub(r"(PREFILL|DECODE)_\d+_(COLD|WARM)\s+(START|DONE)", r"\1_\2 \3", label)
                # DECODE_128 START → DECODE_RUN START  (torch-style: size only, no _COLD/_WARM)
                label = re.sub(r"(PREFILL|DECODE)_\d+\s+(START|DONE)", r"\1_RUN \2", label)
                # TRACE_LOAD START b128 → TRACE_LOAD START
                label = re.sub(r"(TRACE_(?:LOAD|COMPILE)\s+(?:START|DONE))\s+b\d+", r"\1", label)
                post_events.append((dt, label))

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

    # HBM breakdown: per-device sub-metrics (tensors, constants, model_code,
    # model_shared_scratchpad, runtime_memory, ...) — only categories/devices
    # that appear at least once
    hbm_breakdown_devices = sorted(
        {k for _, r in mem_records for k in r.get("hbm_breakdown", {})}, key=int
    )
    hbm_breakdown_categories = sorted({
        cat
        for _, r in mem_records
        for dev_data in r.get("hbm_breakdown", {}).values()
        for cat in dev_data
    })
    hbm_breakdown_times  = {cat: {d: [] for d in hbm_breakdown_devices}
                            for cat in hbm_breakdown_categories}
    hbm_breakdown_series = {cat: {d: [] for d in hbm_breakdown_devices}
                            for cat in hbm_breakdown_categories}
    for i, (_, r) in enumerate(mem_records):
        breakdown = r.get("hbm_breakdown", {})
        for dev in hbm_breakdown_devices:
            dev_data = breakdown.get(dev)
            if not dev_data:
                continue
            for cat in hbm_breakdown_categories:
                val = dev_data.get(cat)
                if val is not None:
                    hbm_breakdown_times[cat][dev].append(times_min[i])
                    hbm_breakdown_series[cat][dev].append(val)

    # NeuronCore utilization: only cores that appear at least once
    nc_devices = sorted(
        {k for _, r in mem_records for k in r.get("neuroncore_util_pct", {})}, key=int
    )
    nc_times  = {d: [] for d in nc_devices}
    nc_series = {d: [] for d in nc_devices}
    for i, (_, r) in enumerate(mem_records):
        for dev in nc_devices:
            val = r.get("neuroncore_util_pct", {}).get(dev)
            if val is not None:
                nc_times[dev].append(times_min[i])
                nc_series[dev].append(val)

    # Extract per-size latencies from raw START/DONE pairs (before normalization)
    _lat_pending = {}
    latencies = {}   # (phase, size, temp) -> latency_ms
    for dt, (phase, size_str, temp, action) in _lat_raw:
        key = (phase, int(size_str), temp)
        if action == "START":
            _lat_pending[key] = dt
        elif action == "DONE" and key in _lat_pending:
            ms = (dt - _lat_pending.pop(key)).total_seconds() * 1000
            latencies[key] = round(ms, 3)

    return {
        "times_min":    times_min,
        "host_mem":     [r["host_mem_used_gb"] for _, r in mem_records],
        "swap_mem":     [r["swap_used_gb"]      for _, r in mem_records],
        "cpu_user":     [r["cpu_user_pct"]      for _, r in mem_records],
        "cpu_sys":      [r["cpu_system_pct"]    for _, r in mem_records],
        "hbm_devices":  hbm_devices,
        "hbm_times":    hbm_times,
        "hbm_series":   hbm_series,
        "hbm_breakdown_devices":    hbm_breakdown_devices,
        "hbm_breakdown_categories": hbm_breakdown_categories,
        "hbm_breakdown_times":      hbm_breakdown_times,
        "hbm_breakdown_series":     hbm_breakdown_series,
        "nc_devices":   nc_devices,
        "nc_times":     nc_times,
        "nc_series":    nc_series,
        "phase_events": [(to_min(dt), lbl) for dt, lbl in post_events],
        "latencies":    latencies,
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


def draw_hbm_breakdown_category(ax, data, category):
    """Draw one HBM-breakdown category (e.g. 'tensors') — one line per device."""
    devices = data["hbm_breakdown_devices"]
    if devices:
        for j, dev in enumerate(devices):
            col = HBM_COLORS[j % len(HBM_COLORS)]
            ax.plot(data["hbm_breakdown_times"][category][dev],
                    data["hbm_breakdown_series"][category][dev],
                    lw=1.4, color=col, label=f"Device {dev}")
    else:
        ax.text(0.5, 0.5, "No HBM breakdown data in log", transform=ax.transAxes,
                ha="center", va="center", color="grey")
    ax.set_ylabel(category.replace("_", " ").title(), fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", ls=":", alpha=0.4)


def draw_neuroncore(ax, data):
    if data["nc_devices"]:
        for j, dev in enumerate(data["nc_devices"]):
            col = NC_COLORS[j % len(NC_COLORS)]
            ax.plot(data["nc_times"][dev], data["nc_series"][dev],
                    lw=1.4, color=col, label=f"Core {dev}")
    else:
        ax.text(0.5, 0.5, "No NeuronCore data in log", transform=ax.transAxes,
                ha="center", va="center", color="grey")
    ax.set_ylabel("NeuronCore Util (%)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.set_ylim(bottom=0)
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
        color, ls, lw = PHASE_STYLES.get(label, _DEFAULT_PHASE)
        for i, ax in enumerate(axes):
            line = ax.axvline(t, color=color, lw=lw, ls=ls, alpha=0.85, zorder=3)
            if i == 0 and label not in seen:
                seen[label] = line
    return seen


# ── shared helpers ─────────────────────────────────────────────────────────────

def format_xaxis(ax):
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.set_xlabel("Time (min)")


def add_top_legend(fig, data_handles, data_labels, phase_handles, run_id):
    """
    Place a single legend panel at the top of the figure.

    The legend can grow to many rows (one per unique phase-marker label),
    so instead of shrinking/overlapping the axes we grow the figure itself
    by exactly the legend's rendered height, guaranteeing no overlap
    regardless of how many entries end up in it.
    """
    handles = data_handles + list(phase_handles.values())
    labels  = data_labels  + list(phase_handles.keys())
    legend = fig.legend(
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

    # Measure the legend's actual rendered height, then grow the figure by
    # that much (rather than shrinking the axes) and push the axes' top
    # boundary down below the legend. Fractions used with bbox_to_anchor
    # and subplots_adjust are resolution-independent, so this keeps the
    # legend pinned to the new top edge without touching axes sizing.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_height_in = legend.get_window_extent(renderer).height / fig.dpi
    pad_in = 0.15
    width_in, height_in = fig.get_size_inches()
    new_height_in = height_in + legend_height_in + pad_in
    fig.set_size_inches(width_in, new_height_in)
    new_top = 1.0 - (legend_height_in + pad_in) / new_height_in
    fig.subplots_adjust(top=new_top)


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
        4, 1, figsize=(14, 13), sharex=True,
        gridspec_kw={"hspace": 0.06, "height_ratios": [1, 1, 1, 1]},
    )
    ax_mem, ax_cpu, ax_hbm, ax_nc = axes

    draw_memory(ax_mem, data)
    draw_cpu(ax_cpu, data)
    draw_hbm(ax_hbm, data)
    draw_neuroncore(ax_nc, data)
    phase_handles = draw_phase_markers(list(axes), data["phase_events"])
    format_xaxis(ax_nc)

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


def _save_hbm_breakdown(data, path, run_id):
    """
    One subplot per HBM-breakdown category (tensors, constants, model_code,
    model_shared_scratchpad, runtime_memory, ...), each showing one line per
    device, so that a category driving an HBM increase is easy to spot.
    """
    categories = data.get("hbm_breakdown_categories", [])
    if not categories:
        return

    n = len(categories)
    fig, axes = plt.subplots(
        n, 1, figsize=(14, max(2.6, 13 / 4) * n), sharex=True,
        gridspec_kw={"hspace": 0.08, "height_ratios": [1] * n},
    )
    axes = [axes] if n == 1 else list(axes)

    for ax, cat in zip(axes, categories):
        draw_hbm_breakdown_category(ax, data, cat)

    phase_handles = draw_phase_markers(axes, data["phase_events"])
    format_xaxis(axes[-1])

    # Deduplicate data handles (device labels) across all category axes
    seen_labels, all_dh, all_dl = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen_labels:
                all_dh.append(h); all_dl.append(l)
                seen_labels.add(l)

    add_top_legend(fig, all_dh, all_dl, phase_handles, run_id)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── latency chart ──────────────────────────────────────────────────────────────

def _save_latency(data, path, run_id):
    """
    2-panel line chart: PREFILL latency vs input length | DECODE latency vs output length.
    Each panel shows COLD (solid) and WARM (dashed) lines.
    """
    lats = data.get("latencies", {})
    prefill_sizes = sorted({s for (p, s, _) in lats if p == "PREFILL"})
    decode_sizes  = sorted({s for (p, s, _) in lats if p == "DECODE"})

    has_prefill = bool(prefill_sizes)
    has_decode  = bool(decode_sizes)
    if not has_prefill and not has_decode:
        return

    ncols = (1 if has_prefill else 0) + (1 if has_decode else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4.5),
                             squeeze=False)
    fig.suptitle(f"Per-size benchmark latency — {run_id}", fontsize=9, y=1.01)

    def _panel(ax, phase, sizes, xlabel):
        cold = [lats.get((phase, s, "COLD")) for s in sizes]
        warm = [lats.get((phase, s, "WARM")) for s in sizes]
        x    = list(range(len(sizes)))
        c_col = "#1B5E20" if phase == "PREFILL" else "#311B92"
        w_col = "#66BB6A" if phase == "PREFILL" else "#9575CD"

        def _plot_line(vals, color, ls, ms_size, label):
            xi = [x[i] for i, v in enumerate(vals) if v is not None]
            yi = [v    for v in vals if v is not None]
            if xi:
                ax.plot(xi, yi, ls, color=color, lw=2, markersize=ms_size,
                        marker="o", label=label)
                for xi_, yi_ in zip(xi, yi):
                    ax.annotate(f"{yi_:.1f}", (xi_, yi_),
                                textcoords="offset points", xytext=(0, 6),
                                fontsize=7, ha="center", color=color)

        _plot_line(cold, c_col, "-",  6, "COLD")
        _plot_line(warm, w_col, "--", 5, "WARM")

        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sizes], rotation=45, ha="right")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Latency (ms)", fontsize=9)
        ax.set_title(f"{phase} latency  (COLD vs WARM)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    col = 0
    if has_prefill:
        _panel(axes[0][col], "PREFILL", prefill_sizes, "Input length (tokens)")
        col += 1
    if has_decode:
        _panel(axes[0][col], "DECODE",  decode_sizes,  "Output length (tokens)")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── folder processing ──────────────────────────────────────────────────────────

def process_log_folder(log_dir: Path):
    plots_dir = log_dir / "plots"
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
        # print(f"  [skip] {log_dir.name}  (plots/ already exists)")
        # return

    log_file = log_dir / "run.log"
    if not log_file.exists():
        print(f"  [skip] {log_dir.name}  (no run.log)")
        return
    # metric_file = log_dir / "metrics.json"
    # if not metric_file.exists():
    #     print(f"  [skip] {log_dir.name}  (no metrics.json)")
    #     return

    data = parse_log(log_file)
    if data is None:
        print(f"  [skip] {log_dir.name}  (no MEM records in run.log)")
        return

    plots_dir.mkdir()
    run_id = log_dir.name
    print(f"  → {run_id}  ({len(data['times_min'])} samples, "
          f"{len(data['phase_events'])} phase events)")

    _save_single(draw_memory,     data, plots_dir / "memory.png",      run_id)
    _save_single(draw_cpu,        data, plots_dir / "cpu.png",         run_id)
    _save_single(draw_hbm,        data, plots_dir / "hbm.png",         run_id)
    _save_single(draw_neuroncore, data, plots_dir / "neuroncore.png",  run_id)
    _save_unified(data, plots_dir / "unified.png", run_id)
    _save_hbm_breakdown(data, plots_dir / "hbm_breakdown.png", run_id)
    _save_latency(data, plots_dir / "latency.png", run_id)

    print(f"    ✓ plots saved → {plots_dir}")


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
