import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
from dataclasses import dataclass, field

import torch
from prompts import prompts


DEFAULTS = {
    "model":           "Qwen/Qwen2.5-1.5B-Instruct",
    "dtype":           "bfloat16",
    "compile_backend": "openxla",       # Neuron's torch.compile backend
    "output_lens":     [255, 256, 257], # decode step counts for the decode test
    "input_lens":      [64, 127, 128, 129, 256, 257, 512],
    "test":            "both",          # "prefill" | "decode" | "both"
}

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}

_DTYPE_ABBREV = {
    "bfloat16": "bf16",
    "float16":  "fp16",
    "float32":  "fp32",
}


# ─── SCAFFOLDING (reused from rollout_benchmark) ──────────────────────────────

class Tee:
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


@dataclass
class RunMetrics:
    """Metrics for one test point (prefill call or full decode generation)."""
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0                          # prefill latency
    tpt_ms: float = 0.0                           # mean decode step latency
    total_time_ms: float = 0.0
    per_token_latencies_ms: list[float] = field(default_factory=list)
    per_step_compile_deltas: list[int] = field(default_factory=list)
    total_compile_delta: int = 0                  # total new Neuron graphs compiled


def parse_args():
    p = argparse.ArgumentParser(description="torch.compile Neuron recompilation benchmark")
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--dtype", default=DEFAULTS["dtype"])
    p.add_argument("--compile-backend", default=DEFAULTS["compile_backend"],
                   help="torch.compile backend (default: openxla for Neuron)")
    p.add_argument("--output-lens", nargs="+", type=int, default=DEFAULTS["output_lens"],
                   help="Decode step counts to sweep in the decode test (default: 255 256 257)")
    p.add_argument("--input-lens", nargs="+", type=int, default=DEFAULTS["input_lens"],
                   help="Prefill lengths to sweep in the prefill test")
    p.add_argument("--test", choices=["prefill", "decode", "both"], default=DEFAULTS["test"],
                   help="Which test(s) to run")
    return p.parse_args()


def dtype_abbrev(dtype: str) -> str:
    return _DTYPE_ABBREV.get(dtype.lower(), dtype)


def setup_run_dir(cfg, rerun=True):
    model_short = cfg.model.split("/")[-1]
    il_vals = "-".join(str(x) for x in cfg.input_lens)
    ol_vals = "-".join(str(x) for x in cfg.output_lens)
    base_name = (
        f"torch_{model_short}_{dtype_abbrev(cfg.dtype)}"
        f"_{cfg.compile_backend}"
        f"_test-{cfg.test}"
        f"_ol[{ol_vals}]"
        f"_il[{il_vals}]"
    )
    logs_dir = Path("logs")
    candidate = logs_dir / base_name
    counter = 1
    while candidate.exists():
        candidate = logs_dir / f"{base_name} ({counter})"
        counter += 1
    if rerun or counter == 1:
        candidate.mkdir(parents=True)
    return candidate, candidate / "run.log", candidate / "out.log", counter


def get_post(label):
    ts = datetime.now(timezone.utc).isoformat()
    return f"\n>>> [POST] {ts} {label} <<<\n"


def log(log_file, text):
    with open(log_file, "a", encoding="UTF-8") as f:
        f.write(get_post(text))


def save_metrics(path: Path, metrics: RunMetrics):
    payload = {
        "input_tokens":            metrics.input_tokens,
        "output_tokens":           metrics.output_tokens,
        "ttft_ms":                 round(metrics.ttft_ms, 2),
        "tpt_ms":                  round(metrics.tpt_ms, 2),
        "total_time_ms":           round(metrics.total_time_ms, 2),
        "total_compile_delta":     metrics.total_compile_delta,
        "per_token_latencies_ms":  metrics.per_token_latencies_ms,
        "per_step_compile_deltas": metrics.per_step_compile_deltas,
    }
    with open(path, "w", encoding="UTF-8") as f:
        json.dump(payload, f, indent=2)


def drop_caches(log_file):
    log(log_file, "CACHE_DROP START")
    subprocess.run(["sync"], check=True)
    subprocess.run(
        ["sudo", "tee", "/proc/sys/vm/drop_caches"],
        input=b"3\n", check=True, capture_output=True,
    )
    log(log_file, "CACHE_DROP DONE")


def start_memory_profiler(log_file):
    monitor = subprocess.Popen(
        ["neuron-monitor", "-c", "memory_config.conf"],
        stdout=subprocess.PIPE,
    )
    checker = subprocess.Popen(
        ["python3", "mem_check.py", str(log_file)],
        stdin=monitor.stdout,
    )
    monitor.stdout.close()
    return monitor, checker


def stop_memory_profiler(monitor, checker):
    monitor.terminate()
    checker.wait()


def set_output_file(out_log):
    f = open(out_log, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
    os.dup2(f.fileno(), sys.__stdout__.fileno())
    os.dup2(f.fileno(), sys.__stderr__.fileno())


# ─── NEURON COMPILE-CACHE COUNTING ────────────────────────────────────────────
#
# Each new tensor shape that torch.compile hasn't seen → a new Neuron graph
# compilation.  Compiled graphs are persisted as .neff files.  Counting them
# before/after a call gives a direct, unambiguous recompile signal.

_NEURON_CACHE_DIRS = [
    Path.home() / ".cache" / "neuron",
    Path("/var/tmp/neuron-compile-cache"),
]

def _count_neuron_neff_files() -> int:
    count = 0
    for base in _NEURON_CACHE_DIRS:
        if base.exists():
            count += sum(1 for _ in base.rglob("*.neff"))
    # /tmp/neuronxcc-* dirs are created per compilation job
    count += sum(1 for p in Path("/tmp").glob("neuronxcc-*") if p.is_dir())
    return count


def cache_delta(before: int, after: int) -> int:
    return max(0, after - before)


# ─── MODEL LOADING ────────────────────────────────────────────────────────────

def load_model(cfg, log_file):
    from transformers import AutoModelForCausalLM
    torch_dtype = _DTYPE_MAP.get(cfg.dtype, torch.bfloat16)

    log(log_file, "MODEL_LOAD START")
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch_dtype)
    model.eval()
    log(log_file, "MODEL_LOAD DONE")

    # torch.compile is lazy on Neuron/XLA: the actual compilation happens on
    # the first forward call with a given set of tensor shapes.
    log(log_file, f"TORCH_COMPILE START backend={cfg.compile_backend}")
    model = torch.compile(model, backend=cfg.compile_backend)
    log(log_file, "TORCH_COMPILE DONE (trace deferred to first forward call)")
    return model


# ─── INPUT HELPERS ────────────────────────────────────────────────────────────

def build_token_corpus(model_name: str) -> list[int]:
    """Tokenize all prompts into one long sequence for deterministic slicing."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return tok.encode(" ".join(prompts), add_special_tokens=False)


def make_input_ids(token_corpus: list[int], input_len: int) -> torch.Tensor:
    """Return a [1, input_len] int64 tensor, tiling corpus if needed."""
    tiled = token_corpus * ((input_len // len(token_corpus)) + 1)
    return torch.tensor([tiled[:input_len]], dtype=torch.long)


# ─── TEST 1: PREFILL LENGTH SWEEP ─────────────────────────────────────────────
#
# Question: does changing the prefill (prompt) length trigger recompilation?
#
# Method: for each input_len, call model() twice.
#   Run 1 (cold)   → first encounter of this shape → expect compile + high latency
#   Run 2 (repeat) → same shape again              → expect cache hit + low latency
#
# A retracing system shows: run1 slow, run2 fast, compile_delta=1 on run1.
# A static-shape system shows: both slow on first length, flat thereafter.

def _prefill_call(model, input_ids: torch.Tensor) -> tuple[float, int]:
    """Single prefill forward pass. Returns (latency_ms, compile_delta)."""
    before = _count_neuron_neff_files()
    t0 = time.monotonic()
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    elapsed_ms = (time.monotonic() - t0) * 1000
    return elapsed_ms, cache_delta(before, _count_neuron_neff_files())


def torch_prefill_test(model, token_corpus, cfg, log_file, run_dir):
    log(log_file, "PREFILL_TEST START")
    for input_len in cfg.input_lens:
        input_ids = make_input_ids(token_corpus, input_len)
        for run_idx in (1, 2):
            tag = "cold" if run_idx == 1 else "repeat"
            log(log_file, f"PREFILL il={input_len} run={run_idx} ({tag}) START")
            latency_ms, compile_delta = _prefill_call(model, input_ids)
            log(log_file, (
                f"PREFILL il={input_len} run={run_idx} ({tag}) DONE "
                f"latency={latency_ms:.1f}ms compile_delta={compile_delta}"
            ))
            metrics = RunMetrics(
                input_tokens=input_len,
                ttft_ms=latency_ms,
                total_time_ms=latency_ms,
                per_token_latencies_ms=[round(latency_ms, 3)],
                per_step_compile_deltas=[compile_delta],
                total_compile_delta=compile_delta,
            )
            save_metrics(
                run_dir / f"metrics_prefill_il{input_len}_run{run_idx}.json",
                metrics,
            )
    log(log_file, "PREFILL_TEST DONE")


# ─── TEST 2: DECODE LOOP (KV CACHE GROWTH) ────────────────────────────────────
#
# Question: does the autoregressive decode loop trigger recompilation every step?
#
# Method: prefill once at a fixed length, then decode for output_len steps.
#   At each decode step, past_key_values grows by 1 token → new tensor shape.
#   Under torch.compile this means a new graph trace every step → 10–20x slowdown.
#
# Diagnostic signals:
#   per_token_latencies_ms — a retracing system shows high latency every step
#                            or at least on first-encounter steps; static = flat
#   per_step_compile_deltas — direct count of new Neuron graphs compiled per step

def run_decode_loop(model, input_ids: torch.Tensor, max_new_tokens: int) -> RunMetrics:
    """
    Naive autoregressive loop: past_key_values grows by 1 every decode step.
    This is the failure mode AWS reported — every new KV length is a new shape.
    """
    m = RunMetrics(input_tokens=input_ids.shape[1])

    # ── prefill: full input sequence, no past ──
    before = _count_neuron_neff_files()
    t0 = time.monotonic()
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=None, use_cache=True)
    prefill_ms = (time.monotonic() - t0) * 1000
    m.ttft_ms = prefill_ms
    m.per_token_latencies_ms.append(round(prefill_ms, 3))
    m.per_step_compile_deltas.append(cache_delta(before, _count_neuron_neff_files()))

    past = out.past_key_values                         # shape: [layers, 2, 1, heads, input_len, d]
    next_tok = out.logits[:, -1].argmax(-1, keepdim=True)

    # ── decode: one new token per step, past grows ──
    for _ in range(max_new_tokens):
        before = _count_neuron_neff_files()
        t_step = time.monotonic()
        with torch.no_grad():
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
        step_ms = (time.monotonic() - t_step) * 1000

        m.per_token_latencies_ms.append(round(step_ms, 3))
        m.per_step_compile_deltas.append(cache_delta(before, _count_neuron_neff_files()))

        # This grows the KV cache seq_len by 1 → new shape every step
        past = out.past_key_values
        next_tok = out.logits[:, -1].argmax(-1, keepdim=True)

    m.output_tokens = max_new_tokens
    m.total_time_ms = sum(m.per_token_latencies_ms)
    decode_steps = m.per_token_latencies_ms[1:]   # exclude prefill
    m.tpt_ms = sum(decode_steps) / len(decode_steps) if decode_steps else 0.0
    m.total_compile_delta = sum(m.per_step_compile_deltas)
    return m


def torch_decode_test(model, token_corpus, cfg, log_file, run_dir):
    """
    Run the naive decode loop for each output_len in cfg.output_lens sequentially.
    Running 255 → 256 → 257 in one session lets us see whether the compile cache
    carries over between lengths (different total KV extents = different shapes).
    """
    input_len = cfg.input_lens[0]
    input_ids = make_input_ids(token_corpus, input_len)

    log(log_file, f"DECODE_TEST START il={input_len} output_lens={cfg.output_lens}")
    for output_len in cfg.output_lens:
        log(log_file, f"DECODE ol={output_len} START")
        metrics = run_decode_loop(model, input_ids, output_len)
        log(log_file, (
            f"DECODE ol={output_len} DONE "
            f"ttft={metrics.ttft_ms:.1f}ms "
            f"tpt={metrics.tpt_ms:.1f}ms "
            f"total_compile_delta={metrics.total_compile_delta}"
        ))
        save_metrics(
            run_dir / f"metrics_decode_il{input_len}_ol{output_len}.json",
            metrics,
        )
    log(log_file, "DECODE_TEST DONE")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    rerun = False
    cfg = parse_args()
    run_dir, log_file, out_file, counter = setup_run_dir(cfg, rerun)

    if not rerun and counter > 1:
        sys.exit(0)

    set_output_file(out_file)

    config_snapshot = {
        "model":           cfg.model,
        "dtype":           cfg.dtype,
        "compile_backend": cfg.compile_backend,
        "output_lens":     cfg.output_lens,
        "input_lens":      cfg.input_lens,
        "test":            cfg.test,
    }
    with open(run_dir / "config.json", "w", encoding="UTF-8") as f:
        json.dump(config_snapshot, f, indent=2)

    print(f"Run directory : {run_dir}")
    print(f"Log file      : {log_file}")
    print(f"Config        : {run_dir / 'config.json'}")

    monitor, checker = start_memory_profiler(log_file)
    time.sleep(10)

    drop_caches(log_file)
    time.sleep(10)

    log(log_file, "PROGRAM STARTED")

    log(log_file, "TOKENIZER START")
    token_corpus = build_token_corpus(cfg.model)
    log(log_file, f"TOKENIZER DONE corpus_len={len(token_corpus)}")

    model = load_model(cfg, log_file)
    time.sleep(5)

    if cfg.test in ("prefill", "both"):
        torch_prefill_test(model, token_corpus, cfg, log_file, run_dir)
        time.sleep(5)

    if cfg.test in ("decode", "both"):
        torch_decode_test(model, token_corpus, cfg, log_file, run_dir)
        time.sleep(5)

    drop_caches(log_file)
    time.sleep(10)

    log(log_file, "PROGRAM ENDED")

    time.sleep(10)
    stop_memory_profiler(monitor, checker)


if __name__ == "__main__":
    main()
