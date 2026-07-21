#!/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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


from mini_verl.workers.generation import GenEngine, GenerationParams
from mini_verl.platform.mem_snapshot import log_memory_snapshot
from mini_verl.platform.device import device_time, is_neuron_active

sys.path.insert(0, str(Path(__file__).parent))
from prompts import prompts


DEFAULTS = {
    "model":              "Qwen/Qwen3-8B",
    "dtype":              "bfloat16",
    "batch_size":         1,
    "max_cache_len":      1024,
    "prefill_chunk_size": 0,
    "input_lens":         [64, 127, 128, 129, 255, 256, 257, 512],
    "output_lens":        [64, 127, 128, 129, 255, 256, 257, 512],
    "decode_input_len":   130,
    "test":               "both",
    "tp_size":            1,
    # Rollout-specific: match the real RL training rollout config
    "do_sample":          True,   # stochastic, as in actual GRPO/PPO rollout
    "temperature":        1.0,
    "top_p":              1.0,
    "top_k":              0,
    "n_samples":          1,      # GRPO samples per prompt; effective batch = batch_size × n_samples
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
    batch_size:      int   = 0
    n_samples:       int   = 1
    input_len:       int   = 0
    prompt_len:      int   = 0
    output_len:      int   = 0
    run_idx:         int   = 0
    total_time_ms:   float = 0.0
    tps:             float = 0.0
    tpt_ms:          float = 0.0
    response_tokens: float = 0.0  # mean effective tokens per sample (non-pad after first EOS)
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
        "batch_size":  metrics[0].batch_size if metrics else 0,
        "n_samples":   metrics[0].n_samples  if metrics else 1,
        "runs": [
            {
                "run_idx":         m.run_idx,
                "tag":             "COLD" if m.run_idx == 1 else "WARM",
                "input_len":       m.input_len,
                "prompt_len":      m.prompt_len,
                "output_len":      m.output_len,
                "total_time_ms":   round(m.total_time_ms, 2),
                "tps":             round(m.tps, 2),
                "tpt_ms":          round(m.tpt_ms, 3),
                "response_tokens": round(m.response_tokens, 2),
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
    chunk_tag   = f"chunk{cfg.prefill_chunk_size}" if cfg.prefill_chunk_size > 0 else "nochunk"
    sample_tag  = f"sample_t{cfg.temperature}" if cfg.do_sample else "greedy"
    ns_tag      = f"_ns{cfg.n_samples}" if cfg.n_samples > 1 else ""
    il_vals     = "-".join(str(x) for x in cfg.input_lens)
    ol_vals     = "-".join(str(x) for x in cfg.output_lens)
    base = (
        f"{model_short}_{_DTYPE_ABBREV.get(cfg.dtype, cfg.dtype)}"
        f"_bs{cfg.batch_size}{ns_tag}"
        f"_tp{cfg.tp_size}"
        f"_cache{cfg.max_cache_len}"
        f"_{chunk_tag}"
        f"_{sample_tag}"
        f"_test-{cfg.test}"
        f"_il[{il_vals}]_ol[{ol_vals}]"
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build ``(input_ids, attention_mask, position_ids, prompt_len)``.

    ``prompt_len`` equals ``input_len`` rounded up to the next chunk boundary
    when ``prefill_chunk_size > 0`` and the length is not already a multiple.
    Padding is left-aligned (zeros in the mask, ``pad_token_id`` in the ids),
    matching the verl rollout convention.
    position_ids mirror ``_rollout_gen.py``: ``(attention_mask.cumsum(-1) - 1).clamp(0)``.
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
    position_ids   = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    return input_ids, attention_mask, position_ids, prompt_len


# ─── RESPONSE MASK ────────────────────────────────────────────────────────────

def compute_response_mask(
    response:     torch.Tensor,
    eos_token_id: int | list[int],
    dtype:        torch.dtype = torch.long,
) -> torch.Tensor:
    """Mask that is 1 from the first token up to and including the first EOS.

    Mirrors verl's ``get_response_mask``: tokens after the first EOS are 0.
    Rows with no EOS have an all-1 mask (the full response is valid).
    """
    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)

    B, L = response.shape
    is_eos = torch.zeros(B, L, dtype=torch.bool)
    for eid in eos_ids:
        is_eos |= (response == eid)

    has_eos   = is_eos.any(dim=1)
    first_eos = is_eos.long().argmax(dim=1)  # 0 when has_eos is False
    first_eos[~has_eos] = L                  # push past-end when no EOS found

    arange = torch.arange(L).unsqueeze(0).expand(B, -1)
    return (arange <= first_eos.unsqueeze(1)).to(dtype)


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

    mem  = data.get("system_data", {}).get("memory_info", {})
    vcpu = data.get("system_data", {}).get("vcpu_usage", {}).get("average_usage", {})
    record: dict = {
        "host_mem_used_gb": gb(mem.get("memory_used_bytes", 0)),
        "swap_used_gb":     gb(mem.get("swap_used_bytes",   0)),
        "cpu_user_pct":     vcpu.get("user",   0),
        "cpu_system_pct":   vcpu.get("system", 0),
    }

    hbm: dict        = {}
    hbm_breakdown: dict = {}
    util: dict       = {}

    for rt in data.get("neuron_runtime_data", []) or []:
        if rt.get("pid") != pid:
            continue
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
    log(log_file, "TOKENIZER DONE")

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
    """Single warm-up call at a small fixed shape so the decode graph compiles once.

    Uses greedy + ignore_eos so the warmup shape is deterministic and matches the
    generation benchmark's warmup — only the timed runs use sampling.
    """
    log(log_file, "WARMUP START")
    warmup_len = min(64, cfg.input_lens[0]) if cfg.input_lens else 64
    if cfg.prefill_chunk_size > 0 and warmup_len % cfg.prefill_chunk_size != 0:
        warmup_len = cfg.prefill_chunk_size

    input_ids, attention_mask, _, _ = make_input_tensors(
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


# ─── SINGLE TIMED ROLLOUT ─────────────────────────────────────────────────────

def _run_rollout(
    engine:       GenEngine,
    cfg,
    corpus:       list[int],
    input_len:    int,
    output_len:   int,
    run_idx:      int,
    eos_token_id: int,
    pad_token_id: int,
) -> RunMetrics:
    """One timed rollout: prefill + sampled decode + post-generation assembly.

    Mirrors the full pipeline executed by ``_rollout_gen.py::_generate_core``:
      1. engine.generate() with stochastic sampling and real EOS stopping
      2. compute_response_mask() — tokens valid up to and including first EOS
      3. Extend attention_mask (prompt_mask ‖ response_mask)
      4. Extend position_ids over the response tokens

    n_samples tiles the prompt batch so the engine sees batch_size × n_samples rows,
    matching GRPO's pattern of generating N completions per prompt in one call.

    device_time() drains the async Neuron kernel queue before reading the clock,
    so both t0 and t1 reflect completed work.
    """
    effective_batch = cfg.batch_size * cfg.n_samples

    input_ids, attention_mask, position_ids, prompt_len = make_input_tensors(
        corpus, input_len, effective_batch, cfg.prefill_chunk_size, pad_token_id,
    )
    params = GenerationParams(
        max_new_tokens=output_len,
        prefill_chunk_size=cfg.prefill_chunk_size,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        ignore_eos=False,       # real rollout stops at EOS (unlike generation benchmark)
        cpu_multinomial=False,
    )

    graphs_before = compiled_graph_count()
    t0 = device_time()

    # 1. Generation (prefill + decode loop on Neuron)
    response = engine.generate(
        input_ids, attention_mask, params,
        eos_token_id=eos_token_id, pad_token_id=pad_token_id,
    )  # [effective_batch, output_len] on CPU

    # 2. Post-generation assembly — mirrors _rollout_gen.py::_generate_core on CPU.
    #    Included in the timed region because this is part of the rollout pipeline cost.
    response       = response.cpu()
    input_ids      = input_ids.cpu()
    attention_mask = attention_mask.cpu()
    position_ids   = position_ids.cpu()

    # response_mask: 1 up to and including first EOS (verl rollout convention)
    response_mask  = compute_response_mask(response, eos_token_id, dtype=attention_mask.dtype)
    attention_mask = torch.cat([attention_mask, response_mask], dim=-1)

    # Extend position_ids over the response
    response_len = response.size(1)
    delta        = torch.arange(1, response_len + 1).unsqueeze(0).expand(effective_batch, -1)
    position_ids = torch.cat([position_ids, position_ids[:, -1:] + delta], dim=-1)

    # Full sequence tensor (prompt ‖ response) — assembled but not measured separately
    _seq = torch.cat([input_ids, response], dim=-1)

    t1 = device_time()
    recompiled_graphs = compiled_graph_count() - graphs_before

    total_ms         = (t1 - t0) * 1_000.0
    effective_tokens = float(response_mask.sum().item())          # non-pad tokens across batch
    mean_resp_tokens = effective_tokens / effective_batch          # mean per sample
    tps              = effective_tokens / (total_ms / 1_000.0) if total_ms > 0 else 0.0

    return RunMetrics(
        batch_size=cfg.batch_size,
        n_samples=cfg.n_samples,
        input_len=input_len,
        prompt_len=prompt_len,
        output_len=output_len,
        run_idx=run_idx,
        total_time_ms=total_ms,
        tps=tps,
        response_tokens=mean_resp_tokens,
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
            m = _run_rollout(
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


# ─── ROLLOUT DECODE SWEEP ─────────────────────────────────────────────────────
# Vary output_len, fix input_len=cfg.decode_input_len.
#
# tpt_ms = (total_ms - warm_prefill_baseline) / (mean_response_tokens - 1).
# Dividing by mean_response_tokens (not output_len) reflects actual decode steps
# taken under real EOS stopping, giving a fair per-step latency even when EOS
# fires before max_new_tokens.

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
    """Run the rollout decode sweep (sampled generation + EOS stopping + post-assembly)."""
    log(log_file, "DECODE_TEST START")
    prefill_baseline = warm_prefill_ms.get(cfg.decode_input_len, 0.0)
    dil = cfg.decode_input_len

    for output_len in cfg.output_lens:
        batch: list[RunMetrics] = []
        for run_idx in (1, 2):
            tag   = "COLD" if run_idx == 1 else "WARM"
            label = f"DECODE_{output_len}_{tag}"
            log(log_file, f"{label} START")
            m = _run_rollout(
                engine, cfg, corpus,
                dil, output_len, run_idx,
                eos_token_id, pad_token_id,
            )
            # tpt_ms: subtract prefill baseline, divide by actual decode steps.
            # First token comes from the last prefill chunk logits at no extra cost,
            # so decode steps = mean_response_tokens - 1.
            if output_len > 1 and prefill_baseline > 0 and m.response_tokens > 1:
                decode_only_ms = max(0.0, m.total_time_ms - prefill_baseline)
                m.tpt_ms = decode_only_ms / (m.response_tokens - 1)
            log(log_file, f"{label} DONE")
            recompile_flag = f"  [RECOMPILE x{m.recompiled_graphs}]" if m.recompiled_graphs else ""
            print(
                f"  {label}  total_ms={m.total_time_ms:.1f}"
                f"  tps={m.tps:.1f}  tpt_ms={m.tpt_ms:.3f}"
                f"  resp_tokens={m.response_tokens:.1f}/{output_len}{recompile_flag}"
            )
            batch.append(m)

        save_metrics(run_dir / f"metrics_decode_il{dil}_ol{output_len}.json", batch)
        time.sleep(2)

    log(log_file, "DECODE_TEST DONE")


# ─── GRAPH-CACHE MEMORY SWEEP ─────────────────────────────────────────────────
# Experiment 2: with --prefill-chunk-size 0, every distinct input_len is a
# genuinely distinct NEFF shape (no chunk-boundary reuse). Sweeping a wide
# range of input_lens in ONE long-lived process therefore accumulates N
# distinct compiled graphs, and a synchronous neuron-monitor snapshot taken
# right after each shape's WARM run gives the marginal HBM cost of caching
# that Nth graph (mainly the ``model_code`` category) — this is the direct
# data for "how many graphs fit before HBM pressure".
#
# A REVISIT pass then re-runs every input_len from the forward pass ONE more
# time, in the same order. If the runtime keeps all N NEFFs loaded,
# ``recompiled_graphs`` stays 0 for every revisit. If it evicts under memory
# pressure, an early shape will show ``recompiled_graphs=1`` again on revisit —
# a real recompile forced by that NEFF having been unloaded. That is the
# direct, falsifiable answer to "does it evict old NEFFs once too many
# accumulate", rather than a guess about undocumented runtime internals.
#
# Label format ``GRAPHCACHE_{input_len}_{COLD|WARM}`` / ``GRAPHCACHE_REVISIT_{input_len}``.

def _bytes_to_gb(n: float) -> float:
    return round(n / 1_073_741_824, 3)


def _graphcache_snapshot(
    run_dir:   Path,
    records:   list[dict],
    input_len: int,
    phase:     str,
    m:         RunMetrics,
) -> dict:
    """Take one synchronous neuron-monitor snapshot, append + persist a record.

    Uses ``log_memory_snapshot`` directly (not the 1s-period background
    logger) so the reading is taken exactly when this shape's run just
    finished — a deterministic point in time, not raced against a poll tick.
    """
    snap = log_memory_snapshot(f"graphcache_{phase}_il{input_len}", neuron_monitor=True)
    nm = snap.neuron_monitor

    if not nm or "error" in nm:
        hbm_by_nc: dict            = {}
        model_code_total_gb: float | None = None
        device_used_gb:      float | None = None
    else:
        nc_data = nm.get("neuroncores", {})
        hbm_by_nc = {
            str(nc_idx): {k: _bytes_to_gb(v) for k, v in cats.items()}
            for nc_idx, cats in nc_data.items()
        }
        model_code_total_gb = _bytes_to_gb(sum(cats.get("model_code", 0) for cats in nc_data.values()))
        device_used_gb       = _bytes_to_gb(nm.get("neuron_device_used_bytes", 0))

    record = {
        "phase":                       phase,  # "cold" | "warm" | "revisit"
        "input_len":                   input_len,
        "total_time_ms":               round(m.total_time_ms, 2),
        "recompiled_graphs":           m.recompiled_graphs,
        "graphs_generated_cumulative": compiled_graph_count(),
        "device_used_gb":              device_used_gb,
        "model_code_total_gb":         model_code_total_gb,
        "hbm_breakdown_gb":            hbm_by_nc,
    }
    records.append(record)
    with open(run_dir / "graph_cache_sweep.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return record


def graph_cache_sweep(
    engine:       GenEngine,
    cfg,
    corpus:       list[int],
    log_file:     Path,
    run_dir:      Path,
    eos_token_id: int,
    pad_token_id: int,
) -> None:
    """Accumulate one distinct NEFF per input_len; track HBM growth + eviction.

    Writes every record to ``run_dir / "graph_cache_sweep.json"`` (overwritten
    after each shape, so a partial sweep is still usable) with one entry per
    (input_len, phase) — the series to plot is ``model_code_total_gb`` vs
    ``graphs_generated_cumulative``.
    """
    if cfg.prefill_chunk_size != 0:
        log(log_file, (
            f"WARNING GRAPHCACHE expects --prefill-chunk-size 0 "
            f"(got {cfg.prefill_chunk_size}); shapes may collapse onto shared chunk graphs."
        ))

    log(log_file, "GRAPHCACHE_TEST START")
    records: list[dict] = []

    # Pass 1 — forward accumulation: visit each shape once, COLD (compiles if
    # new) + WARM (should show recompiled_graphs=0 — the shape it JUST built).
    for input_len in cfg.input_lens:
        for run_idx, tag in ((1, "COLD"), (2, "WARM")):
            label = f"GRAPHCACHE_{input_len}_{tag}"
            log(log_file, f"{label} START")
            m = _run_rollout(
                engine, cfg, corpus, input_len, _PREFILL_OUTPUT_LEN, run_idx,
                eos_token_id, pad_token_id,
            )
            log(log_file, f"{label} DONE")
            rec = _graphcache_snapshot(run_dir, records, input_len, tag.lower(), m)
            recompile_flag = f"  [RECOMPILE x{m.recompiled_graphs}]" if m.recompiled_graphs else ""
            mc = rec["model_code_total_gb"]
            mc_str = "n/a" if mc is None else f"{mc:.3f}GiB"
            print(
                f"  {label}  total_ms={m.total_time_ms:.1f}"
                f"  model_code_total={mc_str}"
                f"  graphs={rec['graphs_generated_cumulative']}{recompile_flag}"
            )
        time.sleep(1)

    # Pass 2 — revisit every shape once more, now that N-1 OTHER shapes have
    # been compiled in between. recompiled_graphs>0 here means eviction.
    log(log_file, "GRAPHCACHE_REVISIT START")
    for input_len in cfg.input_lens:
        label = f"GRAPHCACHE_REVISIT_{input_len}"
        log(log_file, f"{label} START")
        m = _run_rollout(
            engine, cfg, corpus, input_len, _PREFILL_OUTPUT_LEN, 3,
            eos_token_id, pad_token_id,
        )
        log(log_file, f"{label} DONE")
        rec = _graphcache_snapshot(run_dir, records, input_len, "revisit", m)
        verdict = "EVICTED, recompiled" if m.recompiled_graphs else "still cached"
        print(f"  {label}  total_ms={m.total_time_ms:.1f}  {verdict}")
        time.sleep(1)
    log(log_file, "GRAPHCACHE_REVISIT DONE")

    log(log_file, "GRAPHCACHE_TEST DONE")


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

    log_memory_snapshot("after_engine_create", neuron_monitor=True)

    do_warmup(engine, cfg, corpus, log_file, eos_token_id, pad_token_id)
    time.sleep(10)

    log_memory_snapshot("after_warmup", neuron_monitor=True)

    warm_prefill_ms: dict[int, float] = {}

    if cfg.test in ("prefill", "both"):
        warm_prefill_ms = prefill_sweep(
            engine, cfg, corpus, log_file, run_dir, eos_token_id, pad_token_id,
        )
        time.sleep(10)
        log_memory_snapshot("after_prefill_sweep", neuron_monitor=True)

    if cfg.test in ("decode", "both"):
        # If the prefill sweep was skipped, measure a quick baseline for
        # decode_input_len so tpt_ms can still be estimated.
        if cfg.decode_input_len not in warm_prefill_ms:
            log(log_file, "PREFILL_BASELINE START")
            _run_rollout(engine, cfg, corpus, cfg.decode_input_len, 1, 1, eos_token_id, pad_token_id)
            m_warm = _run_rollout(engine, cfg, corpus, cfg.decode_input_len, 1, 2, eos_token_id, pad_token_id)
            warm_prefill_ms[cfg.decode_input_len] = m_warm.total_time_ms
            log(log_file, "PREFILL_BASELINE DONE")

        decode_sweep(
            engine, cfg, corpus, log_file, run_dir,
            eos_token_id, pad_token_id, warm_prefill_ms,
        )
        time.sleep(10)
        log_memory_snapshot("after_decode_sweep", neuron_monitor=True)

    if cfg.test == "graphcache":
        graph_cache_sweep(
            engine, cfg, corpus, log_file, run_dir, eos_token_id, pad_token_id,
        )
        time.sleep(10)
        log_memory_snapshot("after_graphcache_sweep", neuron_monitor=True)

    drop_caches(log_file)
    time.sleep(10)

    log(log_file, "PROGRAM ENDED")
    print(f"DONE graphs_generated={compiled_graph_count()}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "mini_verl rollout benchmark — full rollout pipeline: "
            "sampled generation + response-mask assembly, swept across TP sizes. "
            "Mirrors _rollout_gen.py::_generate_core (the actual RL training rollout path)."
        ),
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
                   help="0 = one forward per full prompt. >0 = chunked prefill; "
                        "non-multiples are left-padded.")
    p.add_argument("--input-lens",  nargs="+", type=int, default=DEFAULTS["input_lens"],
                   help="Input lengths for the prefill sweep.")
    p.add_argument("--output-lens", nargs="+", type=int, default=DEFAULTS["output_lens"],
                   help="Output lengths (max_new_tokens) for the decode sweep.")
    p.add_argument("--decode-input-len", type=int, default=DEFAULTS["decode_input_len"],
                   help="Fixed input length used throughout the decode sweep.")
    p.add_argument("--test", choices=["prefill", "decode", "both", "graphcache"],
                   default=DEFAULTS["test"],
                   help="'graphcache' runs the graph-cache HBM-scaling sweep "
                        "(see graph_cache_sweep()) instead of the prefill/decode benchmarks.")
    p.add_argument("--tp-size", type=int, default=DEFAULTS["tp_size"],
                   help="Tensor-parallel degree passed to GenEngine (1 = single replica).")
    # Rollout-specific parameters
    p.add_argument("--do-sample",    action="store_true",  default=DEFAULTS["do_sample"],
                   help="Stochastic sampling (default on). Use --no-do-sample for greedy.")
    p.add_argument("--no-do-sample", action="store_false", dest="do_sample",
                   help="Greedy decoding (overrides --do-sample).")
    p.add_argument("--temperature",  type=float, default=DEFAULTS["temperature"],
                   help="Sampling temperature.")
    p.add_argument("--top-p",        type=float, default=DEFAULTS["top_p"],
                   help="Nucleus (top-p) sampling threshold.")
    p.add_argument("--top-k",        type=int,   default=DEFAULTS["top_k"],
                   help="Top-k sampling (0 = disabled).")
    p.add_argument("--n-samples",    type=int,   default=DEFAULTS["n_samples"],
                   help="GRPO samples per prompt. Effective engine batch = batch_size × n_samples.")
    return p.parse_args()


def main():
    cfg = parse_args()
    run_dir, log_file, out_file = setup_run_dir(cfg)

    os.environ.setdefault("TORCH_NEURONX_ENABLED_METRIC_TABLES", "graph_stats")
    os.environ.setdefault("TORCH_NEURONX_METRICS_DIR", str(run_dir))

    set_output_file(out_file)

    sample_mode = (
        f"do_sample=True temperature={cfg.temperature} top_p={cfg.top_p} top_k={cfg.top_k}"
        if cfg.do_sample else "do_sample=False (greedy)"
    )
    config_snapshot = {
        "model":              cfg.model,
        "dtype":              cfg.dtype,
        "batch_size":         cfg.batch_size,
        "n_samples":          cfg.n_samples,
        "effective_batch":    cfg.batch_size * cfg.n_samples,
        "max_cache_len":      cfg.max_cache_len,
        "prefill_chunk_size": cfg.prefill_chunk_size,
        "input_lens":         cfg.input_lens,
        "output_lens":        cfg.output_lens,
        "decode_input_len":   cfg.decode_input_len,
        "prefill_output_len": _PREFILL_OUTPUT_LEN,
        "tp_size":            cfg.tp_size,
        "test":               cfg.test,
        "do_sample":          cfg.do_sample,
        "temperature":        cfg.temperature,
        "top_p":              cfg.top_p,
        "top_k":              cfg.top_k,
        "sampling_mode":      sample_mode,
        "ignore_eos":         False,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)

    print(f"Run directory : {run_dir}")
    print(f"Log file      : {log_file}")
    print(f"Config        : {run_dir / 'config.json'}")
    print(f"Sampling      : {sample_mode}")
    print(f"TP size       : {cfg.tp_size}")
    print(f"n_samples     : {cfg.n_samples}  (effective batch: {cfg.batch_size * cfg.n_samples})")

    monitor, thread, stop = start_mem_logger(log_file)
    time.sleep(10)

    mini_verl_run(cfg, log_file, run_dir)

    time.sleep(10)
    stop_mem_logger(monitor, thread, stop)


if __name__ == "__main__":
    main()
