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
    "output_len":             256,
    "input_lens":             [127, 128, 256],
    "same_input_len":         False,
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
    def isatty(self):
        return False
            
@dataclass
class TokenMetrics:
    batch_size: int = 0
    prompt_idx: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0
    tpt_ms: float = 0.0
    total_time_ms: float = 0.0
    per_token_latencies_ms: list[float] = field(default_factory=list)


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
    p.add_argument("--input-lens", nargs="+", type=int, default=DEFAULTS["input_lens"],
                   help="Prefill lengths to sweep (default: 127 128 256)")
    p.add_argument("--same-input-len", action="store_true", default=DEFAULTS["same_input_len"],
                   help="All requests in a batch share the same input length (uniform sweep); "
                        "default is mixed: batch slots cycle through all --input-lens values")
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


def setup_run_dir(cfg, rerun = True):
    model_short = cfg.model.split("/")[-1]
    il_tag = "ilsame" if cfg.same_input_len else "ilmixed"
    il_vals = "-".join(str(x) for x in cfg.input_lens)
    base_name = (
        f"{model_short}_{dtype_abbrev(cfg.dtype)}"
        f"_tp{cfg.tensor_parallel_size}"
        f"_bs{cfg.max_num_seqs}"
        f"_ol{cfg.output_len}"
        f"_{il_tag}[{il_vals}]"
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
                "per_token_latencies_ms": m.per_token_latencies_ms,
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


# INPUT-LENGTH HELPERS

def build_token_corpus(model_name: str) -> list[int]:
    """Tokenize the full prompts list into one long token sequence for slicing."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    combined = " ".join(prompts)
    return tok.encode(combined, add_special_tokens=False)

def make_prompt_tokens(token_corpus: list[int], input_len: int) -> list[int]:
    """Return exactly input_len token IDs, tiling the corpus if needed."""
    tiled = token_corpus * ((input_len // len(token_corpus)) + 1)
    return tiled[:input_len]

def make_batch_prompts(token_corpus: list[int], input_lens: list[int],
                       batch_size: int, current_len: int | None,
                       same_input_len: bool) -> list[dict]:
    """
    Build batch_size prompt dicts for engine.generate().
    same_input_len=True  → all slots get current_len tokens (uniform batch).
    same_input_len=False → slots cycle through input_lens (heterogeneous batch).
    """
    if same_input_len:
        ids = make_prompt_tokens(token_corpus, current_len)
        return [{"prompt_token_ids": ids} for _ in range(batch_size)]
    return [
        {"prompt_token_ids": make_prompt_tokens(token_corpus, input_lens[i % len(input_lens)])}
        for i in range(batch_size)
    ]


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

async def _stream_one(engine, prompt, sampling_params, prompt_idx: int, batch_size: int) -> TokenMetrics:
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
        m.per_token_latencies_ms = [round(x * 1000, 3) for x in intervals]
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


async def async_generation(engine, output_len, batch_size: int, log_file,
                           batch_prompts=None) -> list[TokenMetrics]:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    log(log_file, "GENERATION START")
    sampling_params = SamplingParams(
        min_tokens  = output_len,
        max_tokens  = output_len,
        output_kind = RequestOutputKind.DELTA,
    )

    if batch_prompts is None:
        # Default: tile string prompts from the corpus to fill batch_size slots
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

async def async_vllm_custom_run(log_file, cfg, metrics_file):
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
    save_metrics(metrics_file.with_stem(f"metrics_ol{cfg.output_len}_A"), metrics)

    await asyncio.sleep(10)

    metrics = await async_generation(engine, cfg.output_len + 1, cfg.max_num_seqs, log_file)
    save_metrics(metrics_file.with_stem(f"metrics_ol{cfg.output_len + 1}_B"), metrics)

    await asyncio.sleep(10)

    metrics = await async_generation(engine, cfg.output_len, cfg.max_num_seqs, log_file)
    save_metrics(metrics_file.with_stem(f"metrics_ol{cfg.output_len}_C"), metrics)

    await asyncio.sleep(10)

    metrics = await async_generation(engine, cfg.output_len + 1, cfg.max_num_seqs, log_file)
    save_metrics(metrics_file.with_stem(f"metrics_ol{cfg.output_len + 1}_D"), metrics)

    engine.shutdown()
    await asyncio.sleep(10)

    drop_caches(log_file)
    await asyncio.sleep(10)

    log(log_file, "PROGRAM ENDED")
    return metrics


async def async_input_len_run(log_file, cfg, metrics_file):
    drop_caches(log_file)
    await asyncio.sleep(10)

    log(log_file, "PROGRAM STARTED")
    log(log_file, "VLLM_IMPORT START")
    import vllm
    log(log_file, "VLLM_IMPORT DONE")

    engine = async_engine(cfg, log_file)
    await asyncio.sleep(10)

    await async_warmup(engine, cfg.max_num_seqs, log_file)
    await asyncio.sleep(10)

    log(log_file, "TOKENIZER START")
    token_corpus = build_token_corpus(cfg.model)
    log(log_file, f"TOKENIZER DONE corpus_len={len(token_corpus)}")

    if cfg.same_input_len:
        # One generation run per input length; entire batch is uniform at that length.
        for input_len in cfg.input_lens:
            log(log_file, f"INPUT_LEN {input_len} START")
            batch_prompts = make_batch_prompts(
                token_corpus, cfg.input_lens, cfg.max_num_seqs, input_len, same_input_len=True
            )
            metrics = await async_generation(
                engine, cfg.output_len, cfg.max_num_seqs, log_file, batch_prompts=batch_prompts
            )
            save_metrics(metrics_file.with_stem(f"metrics_il{input_len}"), metrics)
            log(log_file, f"INPUT_LEN {input_len} DONE")
            await asyncio.sleep(10)
    else:
        # Single generation run; batch slots cycle through all input_lens values.
        log(log_file, f"INPUT_LEN mixed{cfg.input_lens} START")
        batch_prompts = make_batch_prompts(
            token_corpus, cfg.input_lens, cfg.max_num_seqs, None, same_input_len=False
        )
        metrics = await async_generation(
            engine, cfg.output_len, cfg.max_num_seqs, log_file, batch_prompts=batch_prompts
        )
        save_metrics(metrics_file.with_stem("metrics_il_mixed"), metrics)
        log(log_file, f"INPUT_LEN mixed DONE")
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
    rerun = False
    cfg = parse_args()
    run_dir, log_file, out_file, metrics_file, counter = setup_run_dir(cfg, rerun)

    if not rerun and counter > 1:
        sys.exit(0)
    
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
        "input_lens": cfg.input_lens,
        "same_input_len": cfg.same_input_len,
    }
    with open(run_dir / "config.json", "w", encoding="UTF-8") as f:
        json.dump(config_snapshot, f, indent=2)

    print(f"Run directory : {run_dir}")
    print(f"Log file : {log_file}")
    print(f"Config : {run_dir / 'config.json'}")

    monitor, checker = start_memory_profiler(log_file)
    time.sleep(10)

    # vllm_run(log_file, cfg)
    # asyncio.run(async_vllm_run(log_file, cfg, metrics_file))

    asyncio.run(async_input_len_run(log_file, cfg, metrics_file))

    time.sleep(10)
    stop_memory_profiler(monitor, checker)



if __name__ == "__main__":
    main()
