#!/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python3
"""
Colocated GRPO training benchmark for mini-verl.

Runs ONE mini-verl GRPO training cell in colocate mode (actor + rollout on the
same NeuronCores, sharing a single FSDP model with no weight-sync transfer).
Mirrors the structure of run_mini_verl_benchmark.py:

  * Tee + log() + setup_run_dir() + set_output_file() — same conventions.
  * Background neuron-monitor thread writing [MEM] lines to run.log — no PID
    filter, because Ray worker subprocesses own the NeuronCores, not the driver
    process.  All runtimes are aggregated so the whole-device colocate HBM
    footprint is visible in every [MEM] record.  NeuronCore indices are
    globally unique on Trainium so hbm_gb / neuroncore_util_pct remain
    non-colliding.  make_plots.py parses the format unchanged.
  * result.json — cell config + succeeded/failed + OOM signature + perf
    breakdown extracted from the trainer's avg perf line.
  * out.log — full stdout+stderr of the training run (Tee'd to the terminal).
  * config.json — snapshot of CellConfig for post-hoc reference.

Companion sweep: run_colocate_multiple.sh iterates the grid and calls cleanup()
(including `ray stop --force`) between cells.

Colocate constraints enforced by mini-verl:
  * actor.parallel.procs_per_node == rollout.parallel.procs_per_node  (same pool)
  * actor.parallel.tp_size == rollout.parallel.tp_size                 (matched TP)
  * both must be powers of two; tp_size must divide procs_per_node
  * total seq (max_prompt_length + max_response_length) must stay <= ~512
    or the trainer's eager forward fails to compile (neuronx-cc rotate_half limit)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# ─── Neuron / Ray env ─────────────────────────────────────────────────────────
# Must be set before importing mini_verl (ray_trainer imports ray + omegaconf
# at module level; MINI_VERL_DEVICE is resolved inside main()).
os.environ.setdefault("NEURON_CC_FLAGS",                "--model-type transformer")
os.environ.setdefault("ACCELERATE_TORCH_DEVICE",        "neuron")
os.environ.setdefault("RAY_DEDUP_LOGS",                 "0")
os.environ.setdefault("PYTHONUNBUFFERED",               "1")
os.environ.setdefault("TORCH_NEURONX_ENABLE_LAZY_ALLOC","1")
os.environ.setdefault("HF_HUB_OFFLINE",                 "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE",            "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM",          "false")

from mini_verl.config import Config                 # noqa: E402
from mini_verl.ray_trainer import main as run_grpo  # noqa: E402


# ─── DEFAULTS ─────────────────────────────────────────────────────────────────

DEFAULTS: dict = {
    "model":                 "Qwen/Qwen2.5-0.5B-Instruct",
    "dtype":                 "bfloat16",
    "cores":                 32,
    "tp_size":               1,
    "train_batch_size":      16,
    "ppo_micro_batch_size":  1,
    "rollout_n":             4,
    "max_prompt_length":     128,
    "max_response_length":   128,
    "prefill_chunk_size":    -1,
    "total_steps":           8,
    "train_files":           os.path.expanduser("~/data/gsm8k/train.parquet"),
}

# Mirrors mini-verl/exp/parse_perf.py OOM signatures.
_OOM_SIGNATURES: tuple[str, ...] = (
    "out of memory",
    "OOM",
    "RESOURCE_EXHAUSTED",
    "Allocation ",
    "failed to allocate",
    "NRT_INSUFFICIENT",
    "bad_alloc",
)

# Category order matches neuron-monitor JSON and mem_check.py.
_NC_USAGE_KEYS: tuple[str, ...] = (
    "tensors", "constants", "model_code", "model_shared_scratchpad", "runtime_memory"
)


# ─── DATACLASSES ──────────────────────────────────────────────────────────────

@dataclass
class CellConfig:
    model:                str = DEFAULTS["model"]
    dtype:                str = DEFAULTS["dtype"]
    # colocate: ONE pool for both actor and rollout — cores is shared.
    cores:                int = DEFAULTS["cores"]
    # actor tp_size == rollout tp_size (enforced by mini-verl).
    tp_size:              int = DEFAULTS["tp_size"]
    train_batch_size:     int = DEFAULTS["train_batch_size"]
    # Micro-batch for actor forward/backward (gradient accumulation).
    ppo_micro_batch_size: int = DEFAULTS["ppo_micro_batch_size"]
    # GRPO group size: scales KV cache and rollout batch linearly.
    rollout_n:            int = DEFAULTS["rollout_n"]
    max_prompt_length:    int = DEFAULTS["max_prompt_length"]
    max_response_length:  int = DEFAULTS["max_response_length"]
    # -1 = no chunking (one prefill forward per full prompt).
    # >0 = chunked prefill; value must divide max_prompt_length.
    prefill_chunk_size:   int = DEFAULTS["prefill_chunk_size"]
    # Must exceed perf_warmup_steps (=3) for a valid avg breakdown line.
    total_steps:          int = DEFAULTS["total_steps"]
    train_files:          str = DEFAULTS["train_files"]


# ─── UTILITIES ────────────────────────────────────────────────────────────────

class Tee:
    """Fan output to multiple file objects simultaneously."""
    def __init__(self, *files):
        self._files = files
    def write(self, data):
        for f in self._files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self._files:
            f.flush()
    def isatty(self):
        return False
    def fileno(self):
        # Ray and some stdlib code call fileno() on sys.stdout/stderr.
        # Delegate to the first underlying file that supports it.
        import io
        for f in self._files:
            try:
                return f.fileno()
            except (AttributeError, io.UnsupportedOperation):
                continue
        raise io.UnsupportedOperation("fileno")


def log(log_file: Path, text: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n>>> [POST] {ts} {text} <<<\n")


def setup_run_dir(cell: CellConfig) -> tuple[Path, Path, Path]:
    model_short = cell.model.split("/")[-1]
    chunk_tag   = f"chunk{cell.prefill_chunk_size}" if cell.prefill_chunk_size > 0 else "nochunk"
    base = (
        f"{model_short}_{cell.dtype[:5]}"
        f"_cores{cell.cores}"
        f"_tp{cell.tp_size}"
        f"_bs{cell.train_batch_size}"
        f"_mbs{cell.ppo_micro_batch_size}"
        f"_n{cell.rollout_n}"
        f"_p{cell.max_prompt_length}"
        f"_r{cell.max_response_length}"
        f"_{chunk_tag}"
        f"_steps{cell.total_steps}"
    )
    logs_dir  = Path("logs")
    candidate = logs_dir / base
    counter   = 1
    while candidate.exists():
        candidate = logs_dir / f"{base} ({counter})"
        counter  += 1
    candidate.mkdir(parents=True)
    return candidate, candidate / "run.log", candidate / "out.log"


def set_output_file(out_log: Path) -> None:
    f = open(out_log, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
    # Redirect the OS-level fd so subprocess output (e.g. Ray worker logs) also
    # goes to the file.  Guarded: in some environments (IDE wrappers, test
    # harnesses) sys.__stdout__ is not a real file and fileno() will raise.
    try:
        import io
        os.dup2(f.fileno(), sys.__stdout__.fileno())
        os.dup2(f.fileno(), sys.__stderr__.fileno())
    except (AttributeError, io.UnsupportedOperation, OSError):
        pass


# ─── MEMORY LOGGER ────────────────────────────────────────────────────────────
#
# Unlike run_mini_verl_benchmark.py (one process owns all NeuronCores),
# GRPO training spawns Ray worker subprocesses that own the NeuronCores.
# The driver process has no Neuron runtime entry, so PID-filtering would yield
# empty HBM dicts.  We drop the PID filter and aggregate ALL runtimes from
# every neuron-monitor report.  NeuronCore indices are globally assigned on
# Trainium and non-overlapping across workers, so the hbm_gb dict keys remain
# unique.  The [MEM] line format is otherwise identical to run_mini_verl_benchmark.py.

def _parse_neuron_monitor_line(data: dict) -> dict:
    """Convert one neuron-monitor JSON line into a run.log [MEM] record.

    Aggregates ALL neuron_runtime_data entries (no PID filter) so the full
    colocate HBM footprint across every Ray worker is captured.
    """
    gb = lambda b: round(int(b) / 1_073_741_824, 2)

    mem  = data.get("system_data", {}).get("memory_info", {})
    vcpu = data.get("system_data", {}).get("vcpu_usage", {}).get("average_usage", {})
    record: dict = {
        "host_mem_used_gb": gb(mem.get("memory_used_bytes", 0)),
        "swap_used_gb":     gb(mem.get("swap_used_bytes",   0)),
        "cpu_user_pct":     vcpu.get("user",   0),
        "cpu_system_pct":   vcpu.get("system", 0),
    }

    hbm: dict           = {}
    hbm_breakdown: dict = {}
    util: dict          = {}

    for rt in data.get("neuron_runtime_data", []) or []:
        report = rt.get("report", {})

        nc_usage = (
            report
            .get("memory_used", {})
            .get("neuron_runtime_used_bytes", {})
            .get("usage_breakdown", {})
            .get("neuroncore_memory_usage", {})
        )
        for nc_idx, cats in nc_usage.items():
            total = sum(int(cats.get(k, 0)) for k in cats)
            hbm[nc_idx]           = gb(total)
            hbm_breakdown[nc_idx] = {k: gb(cats.get(k, 0)) for k in _NC_USAGE_KEYS}

        for nc_idx, v in (
            report
            .get("neuroncore_counters", {})
            .get("neuroncores_in_use", {})
            .items()
        ):
            util[nc_idx] = round(v.get("neuroncore_utilization", 0), 1)

    record["hbm_gb"]              = hbm
    record["hbm_breakdown"]       = hbm_breakdown
    record["neuroncore_util_pct"] = util
    return record


def _write_mem_record(log_file: Path, record: dict) -> None:
    """Append one [MEM] entry to run.log (same format as mem_check.py)."""
    ts   = datetime.now(timezone.utc).isoformat()
    line = f"\n>>> [MEM] {ts} {json.dumps(record)} <<<\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)


def start_mem_logger(log_file: Path) -> tuple:
    """Start the background neuron-monitor logger.

    Returns (monitor_proc, thread, stop_event).
    """
    monitor = subprocess.Popen(
        ["neuron-monitor", "-c", "memory_config.conf"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    stop = threading.Event()

    def _reader():
        for raw_line in monitor.stdout:
            if stop.is_set():
                break
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                data   = json.loads(raw_line)
                record = _parse_neuron_monitor_line(data)
                _write_mem_record(log_file, record)
            except Exception:
                pass

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return monitor, thread, stop


def stop_mem_logger(monitor, thread, stop_event) -> None:
    stop_event.set()
    monitor.terminate()
    monitor.wait(timeout=5)
    thread.join(timeout=5)


# ─── OOM DETECTION ────────────────────────────────────────────────────────────

def scan_for_oom(out_log: Path) -> str | None:
    """Scan out.log for known OOM / failure signatures; return first match or None."""
    try:
        text = out_log.read_text(errors="replace")
    except OSError:
        return None
    lower = text.lower()
    for sig in _OOM_SIGNATURES:
        if sig.lower() in lower:
            return sig
    return None


def _extract_perf_breakdown(out_log: Path) -> dict | None:
    """Parse the avg perf breakdown line from the trainer log (last match wins).

    Mirrors the field extraction in mini-verl/exp/parse_perf.py.
    Returns None if no valid breakdown line was found (e.g. too few steps,
    or the run crashed before the end-of-fit print).
    """
    _FIELDS = {
        "rollout":     "rollout",
        "fwd":         "forward",
        "bwd":         "backward",
        "actor":       "actor",
        "weight_sync": "weight_sync",
        "step":        "step",
    }
    try:
        text = out_log.read_text(errors="replace")
    except OSError:
        return None
    breakdown = None
    for line in text.splitlines():
        if "avg perf breakdown" not in line or "no steps to average" in line:
            continue
        vals = {}
        for tok, key in _FIELDS.items():
            m = re.search(rf"\b{re.escape(tok)}=([0-9]*\.?[0-9]+)s", line)
            if m:
                vals[key] = float(m.group(1))
        if vals:
            breakdown = vals
    return breakdown


def save_cell_result(
    result_path: Path,
    cell: CellConfig,
    status: str,
    oom: str | None,
    breakdown: dict | None,
    elapsed_s: float,
) -> None:
    record = {
        **dataclasses.asdict(cell),
        "status":      status,
        "succeeded":   status == "ok" and oom is None,
        "oom_signature": oom,
        "breakdown_s": breakdown,
        "wall_clock_s": round(elapsed_s, 1),
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")


# ─── CONFIG BUILDER ───────────────────────────────────────────────────────────

def build_mini_verl_config(cell: CellConfig) -> Config:
    """Build a mini-verl Config from a CellConfig via dotted-override strings.

    Uses Config.apply_overrides() so mini-verl's own type coercion and key
    validation run — a misspelled key raises immediately.

    weight_sync.channel is intentionally omitted: mini-verl ignores it in
    colocate (the rollout generates on the actor's shared FSDP model).
    """
    overrides = [
        "device=auto",
        "colocate=true",
        f"model.path={cell.model}",
        f"model.dtype={cell.dtype}",
        f"data.train_files={cell.train_files}",
        f"data.train_batch_size={cell.train_batch_size}",
        f"data.max_prompt_length={cell.max_prompt_length}",
        f"data.max_response_length={cell.max_response_length}",
        f"rollout.n={cell.rollout_n}",
        f"rollout.prefill_chunk_size={cell.prefill_chunk_size}",
        f"rollout.parallel.procs_per_node={cell.cores}",
        f"rollout.parallel.tp_size={cell.tp_size}",
        "rollout.sample_on_device=true",
        "rollout.mem_snapshot=false",
        f"actor.parallel.procs_per_node={cell.cores}",
        f"actor.parallel.tp_size={cell.tp_size}",
        f"actor.ppo_micro_batch_size={cell.ppo_micro_batch_size}",
        "actor.use_kl_loss=true",
        "actor.mem_snapshot=false",
        f"trainer.total_training_steps={cell.total_steps}",
        "trainer.logger=[console]",
        "trainer.print_rollouts=1",
    ]
    return Config().apply_overrides(overrides)


# ─── TOP-LEVEL RUN ────────────────────────────────────────────────────────────

def run_one_cell(cell: CellConfig, log_file: Path, run_dir: Path) -> str:
    """Run one colocate GRPO training cell.

    Wraps run_grpo() in try/except so an OOM or other exception is recorded in
    result.json rather than aborting the caller's grid loop.  Returns the
    status string ("ok" or "failed:<ExceptionType>").
    """
    log(log_file, "PROGRAM STARTED")

    cfg    = build_mini_verl_config(cell)
    status = "ok"
    t0     = time.time()
    try:
        run_grpo(cfg)
    except Exception as e:
        status = f"failed:{type(e).__name__}"
        log(log_file, f"RUN EXCEPTION {e!r}")

    elapsed   = time.time() - t0
    out_log   = run_dir / "out.log"
    oom       = scan_for_oom(out_log)
    breakdown = _extract_perf_breakdown(out_log)

    save_cell_result(run_dir / "result.json", cell, status, oom, breakdown, elapsed)

    if oom:
        log(log_file, f"OOM DETECTED: {oom}")
    log(log_file, f"PROGRAM ENDED status={status} elapsed={elapsed:.1f}s")
    return status


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="mini-verl colocate GRPO training benchmark (one cell).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",                 default=DEFAULTS["model"])
    p.add_argument("--dtype",                 default=DEFAULTS["dtype"])
    p.add_argument("--cores",       type=int, default=DEFAULTS["cores"],
                   help="NeuronCore count (actor == rollout in colocate). Power of two.")
    p.add_argument("--tp-size",     type=int, default=DEFAULTS["tp_size"],
                   help="Tensor-parallel degree. Power of two, must divide --cores.")
    p.add_argument("--train-batch-size",     type=int, default=DEFAULTS["train_batch_size"],
                   help="Distinct prompts per training step.")
    p.add_argument("--ppo-micro-batch-size", type=int, default=DEFAULTS["ppo_micro_batch_size"],
                   help="Actor forward/backward micro-batch (gradient accumulation).")
    p.add_argument("--rollout-n",   type=int, default=DEFAULTS["rollout_n"],
                   help="GRPO group size (samples per prompt). Scales KV cache linearly.")
    p.add_argument("--max-prompt-length",    type=int, default=DEFAULTS["max_prompt_length"],
                   help="Prompt token budget. Keep prompt+response <= 512 (compiler limit).")
    p.add_argument("--max-response-length",  type=int, default=DEFAULTS["max_response_length"],
                   help="Response token budget. Keep prompt+response <= 512.")
    p.add_argument("--prefill-chunk-size",   type=int, default=DEFAULTS["prefill_chunk_size"],
                   help="-1 = no chunk. >0 = chunked prefill; must divide max-prompt-length.")
    p.add_argument("--total-steps", type=int, default=DEFAULTS["total_steps"],
                   help="Training steps. Needs > 3 (perf_warmup_steps) for a valid breakdown.")
    p.add_argument("--train-files",           default=DEFAULTS["train_files"],
                   help="Path to GSM8K train.parquet.")
    return p.parse_args()


def main():
    args = parse_args()
    cell = CellConfig(
        model=args.model,
        dtype=args.dtype,
        cores=args.cores,
        tp_size=args.tp_size,
        train_batch_size=args.train_batch_size,
        ppo_micro_batch_size=args.ppo_micro_batch_size,
        rollout_n=args.rollout_n,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        prefill_chunk_size=args.prefill_chunk_size,
        total_steps=args.total_steps,
        train_files=args.train_files,
    )

    run_dir, log_file, out_log = setup_run_dir(cell)
    set_output_file(out_log)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cell), f, indent=2)

    print(f"Run directory : {run_dir}")
    print(f"Log file      : {log_file}")
    print(f"Config        : {run_dir / 'config.json'}")

    monitor, thread, stop = start_mem_logger(log_file)
    time.sleep(5)

    run_one_cell(cell, log_file, run_dir)

    time.sleep(5)
    stop_mem_logger(monitor, thread, stop)


if __name__ == "__main__":
    main()
