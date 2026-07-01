"""
torch_rollout_benchmark.py – harness for the pluggable Neuron backend benchmark.

Run with:
    python torch_rollout_benchmark.py --backend openxla
    python torch_rollout_benchmark.py --backend nxdi
    python torch_rollout_benchmark.py --backend trace

The backend name is embedded in every log directory name so runs from different
backends never collide and results are easy to tell apart.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch._dynamo

# Allow Dynamo to recompile as many times as needed so openxla can fully
# exercise the dynamic-shape recompilation path.
torch._dynamo.config.recompile_limit = 10_000

from backends import (
    BACKENDS,
    _DTYPE_ABBREV,
    get_backend,
    log,
    save_metrics,
)
from prompts import prompts


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

DEFAULTS = {
    "model":       "Qwen/Qwen2.5-1.5B-Instruct",
    "dtype":       "bfloat16",
    "backend":     "openxla",
    "output_lens": [64, 127, 128, 129, 255, 256, 257, 512],
    "input_lens":  [64, 127, 128, 129, 255, 256, 257, 512],
    "test":        "both",
}

# Fixed output length used in the prefill sweep (so TTFT comparisons across
# different input lengths are apples-to-apples).
_PREFILL_OUTPUT_LEN = 1

# Fixed input length used in the decode sweep.
_DECODE_INPUT_LEN = 512


# ─── TEE: duplicate stdout/stderr to a file ───────────────────────────────────

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


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Neuron backend benchmark: openxla | nxdi | trace"
    )
    p.add_argument("--model",   default=DEFAULTS["model"])
    p.add_argument("--dtype",   default=DEFAULTS["dtype"])
    p.add_argument(
        "--backend",
        default=DEFAULTS["backend"],
        choices=sorted(BACKENDS),
        help="Execution backend. Reflected in log directory name.",
    )
    p.add_argument(
        "--output-lens",
        nargs="+", type=int,
        default=DEFAULTS["output_lens"],
    )
    p.add_argument(
        "--input-lens",
        nargs="+", type=int,
        default=DEFAULTS["input_lens"],
    )
    p.add_argument(
        "--test",
        choices=["prefill", "decode", "both"],
        default=DEFAULTS["test"],
    )
    return p.parse_args()


# ─── LOGGING DIRECTORY ────────────────────────────────────────────────────────

def _dtype_abbrev(dtype: str) -> str:
    return _DTYPE_ABBREV.get(dtype.lower(), dtype)


def setup_run_dir(cfg, rerun: bool = True):
    model_short = cfg.model.split("/")[-1]
    il_vals = "-".join(str(x) for x in cfg.input_lens)
    ol_vals = "-".join(str(x) for x in cfg.output_lens)
    base_name = (
        f"{cfg.backend}_{model_short}_{_dtype_abbrev(cfg.dtype)}"
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


# ─── SYSTEM HELPERS ───────────────────────────────────────────────────────────

def set_output_file(out_log):
    f = open(out_log, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
    os.dup2(f.fileno(), sys.__stdout__.fileno())
    os.dup2(f.fileno(), sys.__stderr__.fileno())


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


# ─── TOKEN CORPUS ─────────────────────────────────────────────────────────────

def build_token_corpus(model_name: str) -> list[int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return tok.encode(" ".join(prompts), add_special_tokens=False)


def make_input_ids(token_corpus: list[int], input_len: int) -> torch.Tensor:
    tiled = token_corpus * ((input_len // len(token_corpus)) + 1)
    return torch.tensor([tiled[:input_len]], dtype=torch.long)


# ─── TEST 1: PREFILL SWEEP ────────────────────────────────────────────────────
#
# For each input_len, run generate(input_ids, _PREFILL_OUTPUT_LEN) twice:
#   run 1 = COLD: first encounter of this input length → may compile
#   run 2 = WARM: repeated call → compilation cache hit
#
# _PREFILL_OUTPUT_LEN = 1 keeps the decode portion constant so TTFT comparisons
# across input lengths are apples-to-apples.

def torch_prefill_test(backend, token_corpus, cfg, log_file, run_dir):
    log(log_file, "PREFILL_TEST START")
    for input_len in cfg.input_lens:
        input_ids = make_input_ids(token_corpus, input_len)
        for run_idx in (1, 2):
            tag = "COLD" if run_idx == 1 else "WARM"
            label = f"PREFILL_{input_len}_{tag}"
            log(log_file, f"{label} START")
            metrics = backend.generate(input_ids, _PREFILL_OUTPUT_LEN, log_file)
            log(log_file, f"{label} DONE")
            save_metrics(
                run_dir / f"metrics_prefill_il{input_len}_run{run_idx}.json",
                metrics,
            )
    log(log_file, "PREFILL_TEST DONE")


# ─── TEST 2: DECODE SWEEP ─────────────────────────────────────────────────────
#
# For each output_len, run generate(input_ids_512, output_len) twice:
#   run 1 = COLD: may compile for each new (input_len + step) shape
#   run 2 = WARM: all graphs already compiled
#
# _DECODE_INPUT_LEN = 512 is fixed so the prefill shape is constant and only
# the KV-cache extension varies across output lengths.

def torch_decode_test(backend, token_corpus, cfg, log_file, run_dir):
    input_ids = make_input_ids(token_corpus, _DECODE_INPUT_LEN)

    log(log_file, "DECODE_TEST START")
    for output_len in cfg.output_lens:
        for run_idx in (1, 2):
            tag = "COLD" if run_idx == 1 else "WARM"
            label = f"DECODE_{output_len}_{tag}"
            log(log_file, f"{label} START")
            metrics = backend.generate(input_ids, output_len, log_file)
            log(log_file, f"{label} DONE")
            save_metrics(
                run_dir / f"metrics_decode_il{_DECODE_INPUT_LEN}_ol{output_len}_run{run_idx}.json",
                metrics,
            )
    log(log_file, "DECODE_TEST DONE")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    cfg = parse_args()
    run_dir, log_file, out_file, _ = setup_run_dir(cfg, rerun=True)

    set_output_file(out_file)

    config_snapshot = {
        "model":              cfg.model,
        "dtype":              cfg.dtype,
        "backend":            cfg.backend,
        "output_lens":        cfg.output_lens,
        "input_lens":         cfg.input_lens,
        "test":               cfg.test,
        "prefill_output_len": _PREFILL_OUTPUT_LEN,
        "decode_input_len":   _DECODE_INPUT_LEN,
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

    log(log_file, "PROGRAM START")

    log(log_file, "TOKENIZER START")
    token_corpus = build_token_corpus(cfg.model)
    log(log_file, "TOKENIZER DONE")

    backend = get_backend(cfg.backend)
    backend.prepare(cfg, log_file)
    time.sleep(5)

    if cfg.test in ("prefill", "both"):
        torch_prefill_test(backend, token_corpus, cfg, log_file, run_dir)
        time.sleep(5)

    if cfg.test in ("decode", "both"):
        torch_decode_test(backend, token_corpus, cfg, log_file, run_dir)
        time.sleep(5)

    backend.teardown()

    drop_caches(log_file)
    time.sleep(10)

    log(log_file, "PROGRAM DONE")

    time.sleep(10)
    stop_memory_profiler(monitor, checker)


if __name__ == "__main__":
    main()
