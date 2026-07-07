import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from prompts import prompts


DEFAULTS = {
    "model": "/home/ubuntu/trn2-rl-rollout/local-models/Qwen/Qwen2.5-1.5B-Instruct",
    "dtype":                  "bfloat16",
    "tensor_parallel_size":   1,
    "max_num_seqs":           1,
    "max_model_len":          1024,
    "enable_prefix_caching":  False,
    "enable_chunked_prefill": False,
    "disable_log_stats":      True,
    "output_len":             256,
    "output_lens":            [64, 127, 128, 129, 255, 256, 257, 512],
    "input_lens":             [64, 127, 128, 129, 255, 256, 257, 512],
    "decode_input_len":       512,
    "test":                   "both",
    "sweep_mode":             "sequential",
    "sweep_seed":             0,
}

# Fixed output length used in the prefill sweep (keeps TTFT apples-to-apples).
_PREFILL_OUTPUT_LEN = 1


# ─── SWEEP ORDERERS ───────────────────────────────────────────────────────────

def order_sequential(lengths: list[int], seed: int = 0) -> list[int]:
    """Visit each length exactly once, in the order given."""
    return list(lengths)


def order_random(lengths: list[int], seed: int = 0) -> list[int]:
    """Visit each length exactly once, in a reproducibly shuffled order."""
    out = list(lengths)
    random.Random(seed).shuffle(out)
    return out


def order_alternating(lengths: list[int], seed: int = 0) -> list[int]:
    """Interleave new lengths with the previous one.

    For [L0, L1, L2, L3, L4] produces:
        L0, L1, L0, L2, L1, L3, L2, L4, L3

    Each new length is immediately followed by a revisit of the previous so
    the compiled graph for the prior shape stays warm before the next size.
    """
    if len(lengths) <= 1:
        return list(lengths)
    result = [lengths[0]]
    for i in range(1, len(lengths)):
        result.append(lengths[i])
        result.append(lengths[i - 1])
    return result


SWEEP_ORDERERS = {
    "sequential":  order_sequential,
    "random":      order_random,
    "alternating": order_alternating,
}


def build_sweep_order(lengths: list[int], mode: str, seed: int) -> list[int]:
    return SWEEP_ORDERERS[mode](lengths, seed)


def _visit_suffix(visit_count: int) -> str:
    """Return '' for the first visit, '_v2' for the second, '_v3' for third, etc."""
    return "" if visit_count == 1 else f"_v{visit_count}"


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
            
@dataclass
class TokenMetrics:
    batch_size: int = 0
    prompt_idx: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0
    tpt_ms: float = 0.0
    total_time_ms: float = 0.0


def parse_args():
    p = argparse.ArgumentParser(description="vLLM Neuron benchmark")
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--dtype", default=DEFAULTS["dtype"])
    p.add_argument("--tensor-parallel-size", type=int, default=DEFAULTS["tensor_parallel_size"])
    p.add_argument("--max-num-seqs", type=int, default=DEFAULTS["max_num_seqs"])
    p.add_argument("--max-model-len", type=int, default=DEFAULTS["max_model_len"])
    p.add_argument("--output-len", type=int, default=DEFAULTS["output_len"])
    p.add_argument("--output-lens", nargs="+", type=int, default=DEFAULTS["output_lens"])
    p.add_argument("--input-lens",  nargs="+", type=int, default=DEFAULTS["input_lens"])
    p.add_argument("--decode-input-len", type=int, default=DEFAULTS["decode_input_len"],
                   help="Fixed input length used throughout the decode sweep.")
    p.add_argument("--test", choices=["prefill", "decode", "both"], default=DEFAULTS["test"])
    p.add_argument("--sweep-mode", choices=list(SWEEP_ORDERERS), default=DEFAULTS["sweep_mode"],
                   help="Order in which lengths are visited: sequential (default), "
                        "random (shuffled with --sweep-seed), or alternating "
                        "(each new length is followed by a revisit of the previous).")
    p.add_argument("--sweep-seed", type=int, default=DEFAULTS["sweep_seed"],
                   help="RNG seed for --sweep-mode random (no effect on other modes).")
    p.add_argument("--enable-prefix-caching", action="store_true", default=DEFAULTS["enable_prefix_caching"])
    p.add_argument("--enable-chunked-prefill", action="store_true", default=DEFAULTS["enable_chunked_prefill"])
    p.add_argument("--enable-log-stats", action="store_false", dest="disable_log_stats", default=DEFAULTS["disable_log_stats"])
    return p.parse_args()


_DTYPE_ABBREV = {
    "bfloat16": "bf16",
    "float16":  "fp16",
    "float32":  "fp32",
    "float8":   "fp8",
    "int8":     "int8",
    "int4":     "int4",
}

def dtype_abbrev(dtype: str) -> str:
    return _DTYPE_ABBREV.get(dtype.lower(), dtype)


def setup_run_dir(cfg, rerun=True, sweep=False):
    model_short = cfg.model.split("/")[-1]
    if sweep:
        il_vals = "-".join(str(x) for x in cfg.input_lens)
        ol_vals = "-".join(str(x) for x in cfg.output_lens)
        base_name = (
            f"{model_short}_{dtype_abbrev(cfg.dtype)}"
            f"_tp{cfg.tensor_parallel_size}"
            f"_bs{cfg.max_num_seqs}"
            f"_test-{cfg.test}"
            f"_sweep-{cfg.sweep_mode}"
            f"_ol[{ol_vals}]"
            f"_il[{il_vals}]"
        )
    else:
        base_name = (
            f"{model_short}_{dtype_abbrev(cfg.dtype)}"
            f"_tp{cfg.tensor_parallel_size}"
            f"_bs{cfg.max_num_seqs}"
            f"_ol{cfg.output_len}"
        )

    logs_dir = Path("logs")
    candidate = logs_dir / base_name
    counter = 1
    while candidate.exists():
        candidate = logs_dir / f"{base_name} ({counter})"
        counter += 1

    if rerun or counter == 1:
        candidate.mkdir(parents=True)
    return candidate, candidate / "run.log", candidate / "out.log", candidate / "metrics.json", counter


def get_post(label):
    ts = datetime.now(timezone.utc).isoformat()
    return f"\n>>> [POST] {ts} {label} <<<\n"


def log(log_file, text):
    with open(log_file, "a", encoding="UTF-8") as f:
        f.write(get_post(text))

def save_metrics(metrics_file, metrics: list[TokenMetrics]):
    payload = {
        "batch_size": metrics[0].batch_size if metrics else 0,
        "prompts": [
            {
                "prompt_idx": m.prompt_idx,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "ttft_ms": round(m.ttft_ms, 2),
                "tpt_ms": round(m.tpt_ms, 2),
                "total_time_ms": round(m.total_time_ms, 2),
            }
            for m in metrics
        ],
    }
    with open(metrics_file, "w", encoding="UTF-8") as f:
        json.dump(payload, f, indent=2)


def save_sweep_metrics(metrics_file, cold: list[TokenMetrics], warm: list[TokenMetrics]):
    """Save COLD and WARM runs together in one file (mirrors run_mini_verl_benchmark.py)."""
    def _ser(metrics, tag):
        return {
            "tag":        tag,
            "batch_size": metrics[0].batch_size if metrics else 0,
            "prompts": [
                {
                    "prompt_idx":    m.prompt_idx,
                    "input_tokens":  m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "ttft_ms":       round(m.ttft_ms, 2),
                    "tpt_ms":        round(m.tpt_ms, 2),
                    "total_time_ms": round(m.total_time_ms, 2),
                }
                for m in metrics
            ],
        }
    payload = {"cold": _ser(cold, "COLD"), "warm": _ser(warm, "WARM")}
    with open(metrics_file, "w", encoding="UTF-8") as f:
        json.dump(payload, f, indent=2)


def drop_caches(log_file):
    log(log_file, "CACHE_DROP START")
    subprocess.run(["sync"], check=True)
    subprocess.run(
        ["sudo", "tee", "/proc/sys/vm/drop_caches"],
        input=b"3\n", check=True, capture_output=True,
    )
    log(log_file, "CACHE_DROP DONE")



# ─── TOKEN CORPUS ─────────────────────────────────────────────────────────────

def build_token_corpus(model_name: str) -> list[int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return tok.encode(" ".join(prompts), add_special_tokens=False)


def make_prompt_token_ids(token_corpus: list[int], input_len: int) -> list[int]:
    tiled = token_corpus * ((input_len // len(token_corpus)) + 1)
    return tiled[:input_len]


# SYNC GENERATION

def llm_engine(cfg, log_file):
    log(log_file, "COMPILATION START")
    from vllm import LLM
    llm = LLM(
        model = cfg.model,
        dtype = cfg.dtype,
        tensor_parallel_size = cfg.tensor_parallel_size,
        max_num_seqs = cfg.max_num_seqs,
        max_model_len = cfg.max_model_len,
        enable_prefix_caching = cfg.enable_prefix_caching,
        enable_chunked_prefill = cfg.enable_chunked_prefill,
        disable_log_stats = cfg.disable_log_stats
    )
    log(log_file, "COMPILATION DONE")
    return llm

def async_engine(cfg, log_file):
    log(log_file, "COMPILATION START")
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model = cfg.model,
        dtype = cfg.dtype,
        tensor_parallel_size = cfg.tensor_parallel_size,
        max_num_seqs = cfg.max_num_seqs,
        max_model_len = cfg.max_model_len,
        enable_prefix_caching = cfg.enable_prefix_caching,
        enable_chunked_prefill = cfg.enable_chunked_prefill,
        disable_log_stats = cfg.disable_log_stats,
    ))
    log(log_file, "COMPILATION DONE")
    return engine

async def _stream_one(engine, prompt: str, sampling_params, prompt_idx: int, batch_size: int) -> TokenMetrics:
    m = TokenMetrics(prompt_idx=prompt_idx, batch_size=batch_size)
    token_times: list[float] = []

    t0 = time.monotonic()
    async for output in engine.generate(
        request_id = str(uuid.uuid4()),
        prompt = prompt,
        sampling_params = sampling_params,
    ):
        now = time.monotonic()

        if not token_times:
            m.ttft_ms = (now - t0) * 1000
            m.input_tokens = len(output.prompt_token_ids)

        m.output_tokens += len(output.outputs[0].token_ids)
        token_times.append(now)

    m.total_time_ms = (token_times[-1] - t0) * 1000 if token_times else 0.0
    if len(token_times) > 1:
        intervals = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
        m.tpt_ms = sum(intervals) / len(intervals) * 1000
    return m




def llm_generation(llm, output_len, log_file, cache_remove=False):
    from vllm import SamplingParams

    log(log_file, f"GENERATION START")
    llm.generate(prompts, SamplingParams(max_tokens=output_len))
    log(log_file, f"GENERATION DONE")

async def async_warmup(engine, batch_size: int, log_file) -> None:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    log(log_file, "WARMUP START")
    sampling_params = SamplingParams(max_tokens=32, output_kind=RequestOutputKind.DELTA)
    await asyncio.gather(*[
        _stream_one(engine, "Hello", sampling_params, i, batch_size)
        for i in range(batch_size)
    ])
    log(log_file, "WARMUP DONE")


async def async_generation(engine, output_len, batch_size: int, log_file) -> list[TokenMetrics]:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    log(log_file, "GENERATION START")
    sampling_params = SamplingParams(
        min_tokens  = output_len,
        max_tokens  = output_len,
        output_kind = RequestOutputKind.DELTA,
    )

    # Tile prompts to fill exactly batch_size concurrent slots
    batch_prompts = [prompts[i % len(prompts)] for i in range(batch_size)]

    results = await asyncio.gather(*[
        _stream_one(engine, prompt, sampling_params, idx, batch_size)
        for idx, prompt in enumerate(batch_prompts)
    ])

    log(log_file, "GENERATION DONE")
    return list(results)


async def async_generation_sweep(engine, token_corpus: list[int], input_len: int,
                                 output_len: int, batch_size: int, log_file) -> list[TokenMetrics]:
    """Like async_generation but uses token IDs of a specific length as the prompt."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    log(log_file, "GENERATION START")
    sampling_params = SamplingParams(
        min_tokens  = output_len,
        max_tokens  = output_len,
        output_kind = RequestOutputKind.DELTA,
    )

    token_ids = make_prompt_token_ids(token_corpus, input_len)
    prompt = {"prompt_token_ids": token_ids}

    results = await asyncio.gather(*[
        _stream_one(engine, prompt, sampling_params, idx, batch_size)
        for idx in range(batch_size)
    ])

    log(log_file, "GENERATION DONE")
    return list(results)


# ─── TEST 1: PREFILL SWEEP ────────────────────────────────────────────────────
#
# For each input_len, run generate(token_ids, _PREFILL_OUTPUT_LEN) twice:
#   run 1 = COLD, run 2 = WARM

async def vllm_prefill_test(engine, token_corpus, cfg, log_file, run_dir):
    """Prefill sweep ordered by cfg.sweep_mode.

    Revisited input lengths get _v2, _v3, ... suffixes on labels and filenames.
    """
    log(log_file, "PREFILL_TEST START")
    sweep_order = build_sweep_order(cfg.input_lens, cfg.sweep_mode, cfg.sweep_seed)
    visit_counts: dict[int, int] = {}

    for input_len in sweep_order:
        visit_counts[input_len] = visit_counts.get(input_len, 0) + 1
        vsuf = _visit_suffix(visit_counts[input_len])

        cold = warm = None
        for run_idx in (1, 2):
            tag = "COLD" if run_idx == 1 else "WARM"
            label = f"PREFILL_{input_len}_{tag}{vsuf}"
            log(log_file, f"{label} START")
            metrics = await async_generation_sweep(
                engine, token_corpus, input_len, _PREFILL_OUTPUT_LEN, cfg.max_num_seqs, log_file,
            )
            log(log_file, f"{label} DONE")
            if run_idx == 1:
                cold = metrics
            else:
                warm = metrics
        save_sweep_metrics(run_dir / f"metrics_prefill_il{input_len}{vsuf}.json", cold, warm)
        await asyncio.sleep(2)
    log(log_file, "PREFILL_TEST DONE")


# ─── TEST 2: DECODE SWEEP ─────────────────────────────────────────────────────
#
# For each output_len, run generate(token_ids, output_len) twice:
#   run 1 = COLD, run 2 = WARM

async def vllm_decode_test(engine, token_corpus, cfg, log_file, run_dir):
    """Decode sweep ordered by cfg.sweep_mode.

    Revisited output lengths get _v2, _v3, ... suffixes on labels and filenames.
    """
    log(log_file, "DECODE_TEST START")
    dil = cfg.decode_input_len
    sweep_order = build_sweep_order(cfg.output_lens, cfg.sweep_mode, cfg.sweep_seed)
    visit_counts: dict[int, int] = {}

    for output_len in sweep_order:
        visit_counts[output_len] = visit_counts.get(output_len, 0) + 1
        vsuf = _visit_suffix(visit_counts[output_len])

        cold = warm = None
        for run_idx in (1, 2):
            tag = "COLD" if run_idx == 1 else "WARM"
            label = f"DECODE_{output_len}_{tag}{vsuf}"
            log(log_file, f"{label} START")
            metrics = await async_generation_sweep(
                engine, token_corpus, dil, output_len, cfg.max_num_seqs, log_file,
            )
            log(log_file, f"{label} DONE")
            if run_idx == 1:
                cold = metrics
            else:
                warm = metrics
        save_sweep_metrics(run_dir / f"metrics_decode_il{dil}_ol{output_len}{vsuf}.json", cold, warm)
        await asyncio.sleep(2)
    log(log_file, "DECODE_TEST DONE")


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


def vllm_run(log_file, cfg):
    drop_caches(log_file)
    time.sleep(10)

    log(log_file, "PROGRAM STARTED")
    log(log_file, "VLLM_IMPORT START")
    import vllm
    log(log_file, "VLLM_IMPORT DONE")

    llm = llm_engine(cfg, log_file)
    time.sleep(10)

    llm_generation(llm, cfg.output_len, log_file)
    time.sleep(10)

    drop_caches(log_file)
    time.sleep(10)

    log(log_file, "PROGRAM ENDED")
    
async def async_vllm_run(log_file, cfg, metrics_file):
    drop_caches(log_file)
    await asyncio.sleep(10)

    log(log_file, "PROGRAM STARTED")
    log(log_file, "VLLM_IMPORT START")
    import vllm
    log(log_file, "VLLM_IMPORT DONE")

    engine = async_engine(cfg, log_file)
    await asyncio.sleep(10)

    # warm up run
    await async_warmup(engine, cfg.max_num_seqs, log_file)
    await asyncio.sleep(10)

    metrics = await async_generation(engine, cfg.output_len, cfg.max_num_seqs, log_file)
    save_metrics(metrics_file, metrics)

    engine.shutdown()
    await asyncio.sleep(10)

    drop_caches(log_file)
    await asyncio.sleep(10)

    log(log_file, "PROGRAM ENDED")
    return metrics

async def async_vllm_custom_run(log_file, cfg, run_dir):
    drop_caches(log_file)
    await asyncio.sleep(10)

    log(log_file, "PROGRAM STARTED")
    log(log_file, "VLLM_IMPORT START")
    import vllm
    log(log_file, "VLLM_IMPORT DONE")

    log(log_file, "TOKENIZER START")
    token_corpus = build_token_corpus(cfg.model)
    log(log_file, "TOKENIZER DONE")

    engine = async_engine(cfg, log_file)
    await asyncio.sleep(10)

    await async_warmup(engine, cfg.max_num_seqs, log_file)
    await asyncio.sleep(10)

    if cfg.test in ("prefill", "both"):
        await vllm_prefill_test(engine, token_corpus, cfg, log_file, run_dir)
        await asyncio.sleep(10)

    if cfg.test in ("decode", "both"):
        await vllm_decode_test(engine, token_corpus, cfg, log_file, run_dir)
        await asyncio.sleep(10)

    engine.shutdown()
    await asyncio.sleep(10)

    drop_caches(log_file)
    await asyncio.sleep(10)

    log(log_file, "PROGRAM ENDED")


def set_output_file(out_log):
    f = open(out_log, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)

    os.dup2(f.fileno(), sys.__stdout__.fileno())
    os.dup2(f.fileno(), sys.__stderr__.fileno())



def main():
    rerun = True
    cfg = parse_args()
    run_dir, log_file, out_file, metrics_file, counter = setup_run_dir(cfg, rerun, sweep=True)

    if not rerun and counter > 1:
        sys.exit(0)

    set_output_file(out_file)

    config_snapshot = {
        "model":                  cfg.model,
        "dtype":                  cfg.dtype,
        "tensor_parallel_size":   cfg.tensor_parallel_size,
        "max_num_seqs":           cfg.max_num_seqs,
        "max_model_len":          cfg.max_model_len,
        "enable_prefix_caching":  cfg.enable_prefix_caching,
        "enable_chunked_prefill": cfg.enable_chunked_prefill,
        "disable_log_stats":      cfg.disable_log_stats,
        "output_len":             cfg.output_len,
        "output_lens":            cfg.output_lens,
        "input_lens":             cfg.input_lens,
        "test":                   cfg.test,
        "prefill_output_len":     _PREFILL_OUTPUT_LEN,
        "decode_input_len":       cfg.decode_input_len,
        "sweep_mode":             cfg.sweep_mode,
        "sweep_seed":             cfg.sweep_seed,
        "prefill_order":          build_sweep_order(cfg.input_lens,  cfg.sweep_mode, cfg.sweep_seed),
        "decode_order":           build_sweep_order(cfg.output_lens, cfg.sweep_mode, cfg.sweep_seed),
    }
    with open(run_dir / "config.json", "w", encoding="UTF-8") as f:
        json.dump(config_snapshot, f, indent=2)

    print(f"Run directory : {run_dir}")
    print(f"Log file      : {log_file}")
    print(f"Config        : {run_dir / 'config.json'}")

    monitor, checker = start_memory_profiler(log_file)
    time.sleep(10)

    asyncio.run(async_vllm_custom_run(log_file, cfg, run_dir))

    time.sleep(10)
    stop_memory_profiler(monitor, checker)



if __name__ == "__main__":
    main()
