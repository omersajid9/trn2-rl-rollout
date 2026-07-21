#!/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import types

import torch
import torch._logging

torch._dynamo.config.recompile_limit = 10000

# ─── RECOMPILE OBSERVABILITY ──────────────────────────────────────────────────
# One-line signal, native to torch (no vLLM involved): fires the moment a guard
# on the compiled forward fails and Dynamo has to retrace, naming the guard that
# broke, e.g. "tensor 'input_ids' stride mismatch at index 0. expected 8, actual
# 16" — silent otherwise. Pairs with compiled_graph_count() below, which tells
# you whether that retrace actually reached neuronx-cc and built a new NEFF.
torch._logging.set_logs(recompiles=True)


def compiled_graph_count() -> int:
    """Total NEFFs the neuron torch.compile backend has built so far in this process.

    Backed by torch_neuronx.neuron_dynamo_backend.metrics.record_compilation(), which
    bumps counters["neuron"]["compiled_graphs"] exactly once per graph that reaches
    neuronx-cc — i.e. once per Dynamo recompile NOT served by an already-compiled
    graph. Always 0 on CPU (the backend module is never even loaded). Diffing this
    before/after a call is a cheap way to prove a "WARM" run stayed warm.
    """
    return torch._dynamo.utils.counters["neuron"]["compiled_graphs"]


# mini_verl.platform.__init__ imports ray_resources → verl, which is not
# installed in every venv.  Stub it out before the package __init__ runs so
# the submodules we actually use (mem_snapshot, device, generation) load fine.
# for _mod in ("verl", "verl.single_controller", "verl.single_controller.ray"):
#     if _mod not in sys.modules:
#         sys.modules[_mod] = types.ModuleType(_mod)
# sys.modules["verl.single_controller.ray"].RayResourcePool = object  # type: ignore[attr-defined]

from mini_verl.workers.generation import GenEngine, GenerationParams
from mini_verl.platform.mem_snapshot import log_memory_snapshot
from mini_verl.platform.device import device_time, is_neuron_active

sys.path.insert(0, str(Path(__file__).parent))
from prompts import prompts


DEFAULTS = {
    "model": "Qwen/Qwen3-8B",
    "dtype": "bfloat16",
    "batch_size": 1,
    "max_cache_len": 1024,
    "prefill_chunk_size": 64,
    "input_lens": [64, 127, 128, 129, 255, 256, 257, 512],
    "output_lens": [0],
    # "output_lens": [64, 127, 128, 129, 255, 256, 257, 512],
    "decode_input_len": 130,
    "test": "prefill",
    "tp_size": 1,
}

# Prefill sweep generates exactly one token so total time ≈ TTFT.
_PREFILL_OUTPUT_LEN = 1

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}
_DTYPE_ABBREV = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}

# Category keys as ordered in the neuron-monitor JSON.
_NC_USAGE_KEYS = (
    "tensors", "constants", "model_code", "model_shared_scratchpad", "runtime_memory"
)


@dataclass
class RunMetrics:
    batch_size: int = 0
    input_len: int = 0
    prompt_len: int = 0
    output_len: int = 0
    run_idx: int = 0
    total_time_ms: float = 0.0
    tps: float = 0.0
    tpt_ms: float = 0.0
    recompiled_graphs: int = 0  # new NEFFs built by engine.generate() during this run; see compiled_graph_count()


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


def log(log_file: Path, text: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n>>> [POST] {ts} {text} <<<\n")


def save_metrics(path: Path, metrics: list[RunMetrics]) -> None:
    payload = {
        "batch_size": metrics[0].batch_size if metrics else 0,
        "runs": [
            {
                "run_idx": m.run_idx,
                "tag": "COLD" if m.run_idx == 1 else "WARM",
                "input_len": m.input_len,
                "prompt_len": m.prompt_len,
                "output_len": m.output_len,
                "total_time_ms": round(m.total_time_ms, 2),
                "tps": round(m.tps, 2),
                "tpt_ms": round(m.tpt_ms, 3),
                "recompiled_graphs": m.recompiled_graphs,
            }
            for m in metrics
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def drop_caches(log_file: Path) -> None:
    log(log_file, "CACHE_DROP START")
    subprocess.run(["sync"], check=True)
    subprocess.run(
        ["sudo", "tee", "/proc/sys/vm/drop_caches"],
        input=b"3\n", check=True, capture_output=True,
    )
    log(log_file, "CACHE_DROP DONE")


def setup_run_dir(cfg) -> tuple[Path, Path, Path]:
    model_short = cfg.model.split("/")[-1]
    chunk_tag = f"chunk{cfg.prefill_chunk_size}" if cfg.prefill_chunk_size > 0 else "nochunk"
    il_vals = "-".join(str(x) for x in cfg.input_lens)
    ol_vals = "-".join(str(x) for x in cfg.output_lens)
    base = (
        f"{model_short}_{_DTYPE_ABBREV.get(cfg.dtype, cfg.dtype)}"
        f"_bs{cfg.batch_size}"
        f"_tp{cfg.tp_size}"
        f"_cache{cfg.max_cache_len}"
        f"_{chunk_tag}"
        f"_test-{cfg.test}"
        f"_il[{il_vals}]_ol[{ol_vals}]"
    )
    logs_dir  = Path("logs").resolve()
    candidate = logs_dir / base
    counter   = 1
    while candidate.exists():
        candidate = logs_dir / f"{base} ({counter})"
        counter  += 1
    candidate.mkdir(parents=True)
    return candidate, candidate / "run.log", candidate / "out.log"


def set_output_file(out_log: Path) -> None:
    f = open(out_log, "w", encoding="utf-8")

    tty_stdout = os.fdopen(os.dup(sys.__stdout__.fileno()), "w", closefd=True)
    tty_stderr = os.fdopen(os.dup(sys.__stderr__.fileno()), "w", closefd=True)
    os.dup2(f.fileno(), sys.__stdout__.fileno())
    os.dup2(f.fileno(), sys.__stderr__.fileno())

    sys.stdout = Tee(tty_stdout, f)
    sys.stderr = Tee(tty_stderr, f)


# ─── TOKEN CORPUS ─────────────────────────────────────────────────────────────

def build_token_corpus(model_name: str) -> list[int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return tok.encode(" ".join(prompts), add_special_tokens=False)


def _slice_corpus(corpus: list[int], n: int) -> list[int]:
    tiled = corpus * ((n // len(corpus)) + 1)
    return tiled[:n]


def make_input_tensors(
    corpus:             list[int],
    input_len:          int,
    batch_size:         int,
    prefill_chunk_size: int,
    pad_token_id:       int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build ``(input_ids, attention_mask, prompt_len)``.

    ``prompt_len`` equals ``input_len`` rounded up to the next chunk boundary
    when ``prefill_chunk_size > 0`` and the length is not already a multiple.
    Padding is left-aligned (zeros in the mask, ``pad_token_id`` in the ids),
    matching the verl rollout convention.
    """
    token_ids = _slice_corpus(corpus, input_len)

    if prefill_chunk_size > 0 and input_len % prefill_chunk_size != 0:
        pad_len    = prefill_chunk_size - (input_len % prefill_chunk_size)
        prompt_len = input_len + pad_len
        ids_row    = [pad_token_id] * pad_len + token_ids
        mask_row   = [0] * pad_len + [1] * input_len
    else:
        prompt_len = input_len
        ids_row    = token_ids
        mask_row   = [1] * input_len

    input_ids      = torch.tensor([ids_row]  * batch_size, dtype=torch.long)
    attention_mask = torch.tensor([mask_row] * batch_size, dtype=torch.long)
    return input_ids, attention_mask, prompt_len


# ─── MEMORY LOGGER ────────────────────────────────────────────────────────────

def _parse_neuron_monitor_line(data: dict, pid: int) -> dict:
    """Convert one neuron-monitor JSON line into a run.log [MEM] record.

    System-wide fields (host RAM, swap, CPU) match mem_check.py exactly.
    HBM and NeuronCore utilization are PID-filtered: only the runtime entry
    for this process is used, so workers on other cores are excluded.

    Extra ``hbm_breakdown`` field carries the per-category split that
    mem_check.py discards.  make_plots.py ignores unknown keys, so this does
    not break any existing tooling.
    """
    gb = lambda b: round(int(b) / 1_073_741_824, 2)

    # System-wide host RAM + CPU (mirrors mem_check.py parse_snapshot exactly).
    mem  = data.get("system_data", {}).get("memory_info", {})
    vcpu = data.get("system_data", {}).get("vcpu_usage", {}).get("average_usage", {})
    record: dict = {
        "host_mem_used_gb": gb(mem.get("memory_used_bytes", 0)),
        "swap_used_gb":     gb(mem.get("swap_used_bytes",   0)),
        "cpu_user_pct":     vcpu.get("user",   0),
        "cpu_system_pct":   vcpu.get("system", 0),
    }

    hbm: dict       = {}
    hbm_breakdown: dict = {}
    util: dict      = {}

    for rt in data.get("neuron_runtime_data", []) or []:
        if rt.get("pid") != pid:
            continue
        report = rt.get("report", {})

        # Per-NeuronCore HBM: sum all categories for the hbm_gb value that
        # make_plots.py plots; keep the per-category split in hbm_breakdown.
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

        # Per-NeuronCore utilization.
        for nc_idx, v in (
            report
            .get("neuroncore_counters", {})
            .get("neuroncores_in_use", {})
            .items()
        ):
            util[nc_idx] = round(v.get("neuroncore_utilization", 0), 1)

    record["hbm_gb"]           = hbm
    record["hbm_breakdown"]    = hbm_breakdown
    record["neuroncore_util_pct"] = util
    return record


def _write_mem_record(log_file: Path, record: dict) -> None:
    """Append one ``[MEM]`` entry to run.log (same format as mem_check.py)."""
    ts   = datetime.now(timezone.utc).isoformat()
    line = f"\n>>> [MEM] {ts} {json.dumps(record)} <<<\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)


def start_mem_logger(log_file: Path) -> tuple:
    """Start the background memory logger.

    Runs ``neuron-monitor -c memory_config.conf`` as a subprocess and
    processes each JSON line in a daemon thread, writing ``[MEM]`` entries to
    run.log.  This replaces the ``start_memory_profiler`` + ``mem_check.py``
    pair from run_benchmark.py with an equivalent single-process solution that
    is PID-filtered and carries the richer ``hbm_breakdown`` field.

    Returns ``(monitor_proc, thread, stop_event)``.
    """
    pid     = os.getpid()
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
                record = _parse_neuron_monitor_line(data, pid)
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


# ─── ENGINE ───────────────────────────────────────────────────────────────────

def build_engine(cfg, log_file: Path) -> tuple[GenEngine, int, int]:
    """Load model + tokenizer, create GenEngine.

    Returns ``(engine, eos_token_id, pad_token_id)``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(log_file, "TOKENIZER START")
    tok = AutoTokenizer.from_pretrained(cfg.model)
    eos_token_id = tok.eos_token_id
    pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else eos_token_id
    log(log_file, f"TOKENIZER DONE")

    log(log_file, "MODEL_LOAD START")
    dtype = _DTYPE_MAP.get(cfg.dtype, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype)
    log(log_file, "MODEL_LOAD DONE")

    device = torch.device("neuron") if is_neuron_active() else torch.device("cpu")

    log(log_file, "MODEL_DEVICE START")
    model = model.to(device)
    log(log_file, "MODEL_DEVICE DONE")

    log(log_file, "ENGINE_CREATE START")
    engine = GenEngine(
        model,
        device=device,
        max_cache_len=cfg.max_cache_len,
        compile_forward=is_neuron_active(),
        inference_mode=True,
        tp_size=cfg.tp_size,
    )
    log(log_file, "ENGINE_CREATE DONE")
    return engine, eos_token_id, pad_token_id


def do_warmup(
    engine:       GenEngine,
    cfg,
    corpus:       list[int],
    log_file:     Path,
    eos_token_id: int,
    pad_token_id: int,
) -> None:
    """Single warm-up call at a small fixed shape so the decode graph compiles once."""
    log(log_file, "WARMUP START")
    warmup_len = min(64, cfg.input_lens[0]) if cfg.input_lens else 64
    if cfg.prefill_chunk_size > 0 and warmup_len % cfg.prefill_chunk_size != 0:
        warmup_len = cfg.prefill_chunk_size

    input_ids, attention_mask, _ = make_input_tensors(
        corpus, warmup_len, cfg.batch_size, cfg.prefill_chunk_size, pad_token_id,
    )
    params = GenerationParams(
        max_new_tokens=32,
        prefill_chunk_size=cfg.prefill_chunk_size,
        do_sample=False,
        ignore_eos=True,
    )
    engine.generate(
        input_ids, attention_mask, params,
        eos_token_id=eos_token_id, pad_token_id=pad_token_id,
    )
    log(log_file, "WARMUP DONE")
    print(f"  graphs_generated={compiled_graph_count()}")


# ─── SINGLE TIMED RUN ─────────────────────────────────────────────────────────

def _run_one(
    engine:       GenEngine,
    cfg,
    corpus:       list[int],
    input_len:    int,
    output_len:   int,
    run_idx:      int,
    eos_token_id: int,
    pad_token_id: int,
) -> RunMetrics:
    """One timed engine.generate() call.

    Uses device_time() which drains the async Neuron kernel queue before
    reading the clock, so both endpoints reflect completed work.
    """
    input_ids, attention_mask, prompt_len = make_input_tensors(
        corpus, input_len, cfg.batch_size, cfg.prefill_chunk_size, pad_token_id,
    )
    params = GenerationParams(
        max_new_tokens=output_len,
        prefill_chunk_size=cfg.prefill_chunk_size,
        do_sample=False,    # greedy — deterministic and compilation-friendly
        ignore_eos=True,    # always run full max_new_tokens for fair comparisons
    )

    graphs_before = compiled_graph_count()
    t0 = device_time()
    engine.generate(
        input_ids, attention_mask, params,
        eos_token_id=eos_token_id, pad_token_id=pad_token_id,
    )
    t1 = device_time()
    recompiled_graphs = compiled_graph_count() - graphs_before

    total_ms = (t1 - t0) * 1_000.0
    tokens   = cfg.batch_size * output_len
    tps      = tokens / (total_ms / 1_000.0) if total_ms > 0 else 0.0

    return RunMetrics(
        batch_size=cfg.batch_size,
        input_len=input_len,
        prompt_len=prompt_len,
        output_len=output_len,
        run_idx=run_idx,
        total_time_ms=total_ms,
        tps=tps,
        recompiled_graphs=recompiled_graphs,
    )


# ─── PREFILL SWEEP ────────────────────────────────────────────────────────────
# Vary input_len, fix output_len=_PREFILL_OUTPUT_LEN (=1). total_time ≈ TTFT.
#
# Label format ``PREFILL_{input_len}_{COLD|WARM} START/DONE`` is understood by
# make_plots.py's latency extractor (_PER_SIZE_RE) and phase-marker normalizer.

def prefill_sweep(
    engine:       GenEngine,
    cfg,
    corpus:       list[int],
    log_file:     Path,
    run_dir:      Path,
    eos_token_id: int,
    pad_token_id: int,
) -> dict[int, float]:
    """Run the prefill sweep. Returns {input_len: warm_total_ms} for decode TPT."""
    log(log_file, "PREFILL_TEST START")

    warm_prefill_ms: dict[int, float] = {}

    for input_len in cfg.input_lens:
        batch: list[RunMetrics] = []
        for run_idx in (1, 2):
            tag   = "COLD" if run_idx == 1 else "WARM"
            label = f"PREFILL_{input_len}_{tag}"
            log(log_file, f"{label} START")
            m = _run_one(
                engine, cfg, corpus,
                input_len, _PREFILL_OUTPUT_LEN, run_idx,
                eos_token_id, pad_token_id,
            )
            log(log_file, f"{label} DONE")
            recompile_flag = f"  [RECOMPILE x{m.recompiled_graphs}]" if m.recompiled_graphs else ""
            print(f"  {label}  total_ms={m.total_time_ms:.1f}  tps={m.tps:.1f}{recompile_flag}")
            batch.append(m)

        warm_prefill_ms[input_len] = batch[1].total_time_ms
        save_metrics(run_dir / f"metrics_prefill_il{input_len}.json", batch)
        time.sleep(2)

    log(log_file, "PREFILL_TEST DONE")
    return warm_prefill_ms


# ─── DECODE SWEEP ─────────────────────────────────────────────────────────────
# Vary output_len, fix input_len=cfg.decode_input_len.
#
# tpt_ms is estimated as (total_ms - warm_prefill_baseline) / (output_len - 1),
# removing the prefill overhead so only the decode forward passes are timed.
# GenEngine runs (output_len - 1) decode forward passes: the first token comes
# from the last prefill chunk logits at no additional cost.

def decode_sweep(
    engine:          GenEngine,
    cfg,
    corpus:          list[int],
    log_file:        Path,
    run_dir:         Path,
    eos_token_id:    int,
    pad_token_id:    int,
    warm_prefill_ms: dict[int, float],
) -> None:
    """Run the decode sweep."""
    log(log_file, "DECODE_TEST START")
    prefill_baseline = warm_prefill_ms.get(cfg.decode_input_len, 0.0)
    dil = cfg.decode_input_len

    for output_len in cfg.output_lens:
        batch: list[RunMetrics] = []
        for run_idx in (1, 2):
            tag   = "COLD" if run_idx == 1 else "WARM"
            label = f"DECODE_{output_len}_{tag}"
            log(log_file, f"{label} START")
            m = _run_one(
                engine, cfg, corpus,
                dil, output_len, run_idx,
                eos_token_id, pad_token_id,
            )
            if output_len > 1 and prefill_baseline > 0:
                decode_only_ms = max(0.0, m.total_time_ms - prefill_baseline)
                m.tpt_ms = decode_only_ms / (output_len - 1)
            log(log_file, f"{label} DONE")
            recompile_flag = f"  [RECOMPILE x{m.recompiled_graphs}]" if m.recompiled_graphs else ""
            print(
                f"  {label}  total_ms={m.total_time_ms:.1f}"
                f"  tps={m.tps:.1f}  tpt_ms={m.tpt_ms:.3f}{recompile_flag}"
            )
            batch.append(m)

        save_metrics(run_dir / f"metrics_decode_il{dil}_ol{output_len}.json", batch)
        time.sleep(2)

    log(log_file, "DECODE_TEST DONE")


# ─── TOP-LEVEL RUN ────────────────────────────────────────────────────────────

def mini_verl_run(cfg, log_file: Path, run_dir: Path) -> None:
    drop_caches(log_file)
    time.sleep(10)

    log(log_file, "PROGRAM STARTED")

    log(log_file, "CORPUS_BUILD START")
    corpus = build_token_corpus(cfg.model)
    log(log_file, f"CORPUS_BUILD DONE corpus_len={len(corpus)}")

    # Guard: max_cache_len must fit the largest (input_len, output_len) combination.
    max_needed = max(cfg.input_lens + [cfg.decode_input_len]) + max(cfg.output_lens)
    if cfg.max_cache_len < max_needed:
        log(log_file, (
            f"WARNING max_cache_len={cfg.max_cache_len} < required {max_needed}. "
            "Increasing automatically."
        ))
        cfg.max_cache_len = max_needed

    engine, eos_token_id, pad_token_id = build_engine(cfg, log_file)
    time.sleep(10)

    # Rich point-in-time snapshot: GC tensor walk + PyTorch allocator.
    # neuron_monitor=False avoids a competing neuron-monitor subprocess while
    # the background logger is already running.
    log_memory_snapshot("after_engine_create", neuron_monitor=False)

    do_warmup(engine, cfg, corpus, log_file, eos_token_id, pad_token_id)
    time.sleep(10)

    log_memory_snapshot("after_warmup", neuron_monitor=False)

    warm_prefill_ms: dict[int, float] = {}

    if cfg.test in ("prefill", "both"):
        warm_prefill_ms = prefill_sweep(
            engine, cfg, corpus, log_file, run_dir, eos_token_id, pad_token_id,
        )
        time.sleep(10)
        log_memory_snapshot("after_prefill_sweep", neuron_monitor=False)

    if cfg.test in ("decode", "both"):
        # If the prefill sweep was skipped, measure a quick baseline for
        # decode_input_len so tpt_ms can still be estimated.
        if cfg.decode_input_len not in warm_prefill_ms:
            log(log_file, "PREFILL_BASELINE START")
            _run_one(engine, cfg, corpus, cfg.decode_input_len, 1, 1, eos_token_id, pad_token_id)
            m_warm = _run_one(engine, cfg, corpus, cfg.decode_input_len, 1, 2, eos_token_id, pad_token_id)
            warm_prefill_ms[cfg.decode_input_len] = m_warm.total_time_ms
            log(log_file, f"PREFILL_BASELINE DONE")

        decode_sweep(
            engine, cfg, corpus, log_file, run_dir,
            eos_token_id, pad_token_id, warm_prefill_ms,
        )
        time.sleep(10)
        log_memory_snapshot("after_decode_sweep", neuron_monitor=False)

    drop_caches(log_file)
    time.sleep(10)

    log(log_file, "PROGRAM ENDED")
    print(f"DONE graphs_generated={compiled_graph_count()}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="mini_verl GenEngine benchmark (dynamic input/output shape sweep)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",              default=DEFAULTS["model"])
    p.add_argument("--dtype",              default=DEFAULTS["dtype"],
                   choices=list(_DTYPE_MAP))
    p.add_argument("--batch-size",         type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--max-cache-len",      type=int, default=DEFAULTS["max_cache_len"],
                   help="StaticCache length. Auto-increased if smaller than "
                        "max(input_lens) + max(output_lens).")
    p.add_argument("--prefill-chunk-size", type=int, default=DEFAULTS["prefill_chunk_size"],
                   help="0 = one forward per full prompt (each unique input_len compiles a "
                        "new NEFF). >0 = chunked prefill; non-multiples are left-padded.")
    p.add_argument("--input-lens",  nargs="+", type=int, default=DEFAULTS["input_lens"],
                   help="Input lengths for the prefill sweep.")
    p.add_argument("--output-lens", nargs="+", type=int, default=DEFAULTS["output_lens"],
                   help="Output lengths for the decode sweep.")
    p.add_argument("--decode-input-len", type=int, default=DEFAULTS["decode_input_len"],
                   help="Fixed input length used throughout the decode sweep.")
    p.add_argument("--test", choices=["prefill", "decode", "both"], default=DEFAULTS["test"])
    p.add_argument("--tp-size", type=int, default=DEFAULTS["tp_size"],
                   help="Tensor-parallel degree passed to GenEngine (tp_size=1 = single replica).")
    return p.parse_args()


def main():
    cfg = parse_args()
    run_dir, log_file, out_file = setup_run_dir(cfg)

    # Ask torch_neuronx to dump a full per-graph compile ledger (graph_id/cache_key,
    # graph_name, node count, timestamp, phase timings) into this run's directory at
    # exit — one row per NEFF actually built, i.e. one row per recompile that wasn't
    # served by an existing compiled graph. setdefault so an explicit env var (or a
    # different metrics dir) from the caller always wins. No-op on CPU.
    os.environ.setdefault("TORCH_NEURONX_ENABLED_METRIC_TABLES", "graph_stats")
    os.environ.setdefault("TORCH_NEURONX_METRICS_DIR", str(run_dir))

    set_output_file(out_file)

    config_snapshot = {
        "model":              cfg.model,
        "dtype":              cfg.dtype,
        "batch_size":         cfg.batch_size,
        "max_cache_len":      cfg.max_cache_len,
        "prefill_chunk_size": cfg.prefill_chunk_size,
        "input_lens":         cfg.input_lens,
        "output_lens":        cfg.output_lens,
        "decode_input_len":   cfg.decode_input_len,
        "prefill_output_len": _PREFILL_OUTPUT_LEN,
        "tp_size":            cfg.tp_size,
        "test":               cfg.test,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)

    print(f"Run directory : {run_dir}")
    print(f"Log file      : {log_file}")
    print(f"Config        : {run_dir / 'config.json'}")

    monitor, thread, stop = start_mem_logger(log_file)
    time.sleep(10)

    mini_verl_run(cfg, log_file, run_dir)

    time.sleep(10)
    stop_mem_logger(monitor, thread, stop)


if __name__ == "__main__":
    main()
