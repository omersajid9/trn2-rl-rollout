import argparse
import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid
from dataclasses import dataclass, field

from prompts import prompts


DEFAULTS = {
    "model":                  "Qwen/Qwen2.5-1.5B-Instruct",
    "dtype":                  "bfloat16",
    "tensor_parallel_size":   1,
    "max_num_seqs":           1,
    "max_model_len":          768,
    "enable_prefix_caching":  False,
    "enable_chunked_prefill": False,
    "disable_log_stats":      True,
    "output_len": 256
}

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


def setup_run_dir(cfg):
    model_short = cfg.model.split("/")[-1]
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

    candidate.mkdir(parents=True)
    return candidate, candidate / "run.log", candidate / "out.log", candidate / "metrics.json"


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


def drop_caches(log_file):
    log(log_file, "CACHE_DROP START")
    subprocess.run(["sync"], check=True)
    subprocess.run(
        ["sudo", "tee", "/proc/sys/vm/drop_caches"],
        input=b"3\n", check=True, capture_output=True,
    )
    log(log_file, "CACHE_DROP DONE")



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
    
async def async_vllm_run(log_file, cfg):
    drop_caches(log_file)
    time.sleep(10)

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
    engine.shutdown()
    await asyncio.sleep(10)

    drop_caches(log_file)
    await asyncio.sleep(10)

    log(log_file, "PROGRAM ENDED")
    return metrics


def set_output_file(out_log):
    f = open(out_log, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)

    os.dup2(f.fileno(), sys.__stdout__.fileno())
    os.dup2(f.fileno(), sys.__stderr__.fileno())



def main():
    cfg = parse_args()
    run_dir, log_file, out_file, metrics_file = setup_run_dir(cfg)
    
    set_output_file(out_file)

    # Save a config snapshot next to the log so every run is self-documenting
    config_snapshot = {
        "model": cfg.model,
        "dtype": cfg.dtype,
        "tensor_parallel_size": cfg.tensor_parallel_size,
        "max_num_seqs": cfg.max_num_seqs,
        "max_model_len": cfg.max_model_len,
        "enable_prefix_caching": cfg.enable_prefix_caching,
        "enable_chunked_prefill": cfg.enable_chunked_prefill,
        "disable_log_stats": cfg.disable_log_stats,
        "output_len": cfg.output_len,
    }
    with open(run_dir / "config.json", "w", encoding="UTF-8") as f:
        json.dump(config_snapshot, f, indent=2)

    print(f"Run directory : {run_dir}")
    print(f"Log file : {log_file}")
    print(f"Config : {run_dir / 'config.json'}")

    monitor, checker = start_memory_profiler(log_file)
    time.sleep(10)

    # vllm_run(log_file, cfg)

    metrics = asyncio.run(async_vllm_run(log_file, cfg))
    save_metrics(metrics_file, metrics)

    time.sleep(10)
    stop_memory_profiler(monitor, checker)



if __name__ == "__main__":
    main()
