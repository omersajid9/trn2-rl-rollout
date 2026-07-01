"""
Pluggable Neuron execution backends for torch_rollout_benchmark.

Backends
--------
openxla  – torch.compile(backend='openxla'): dynamic JIT-per-shape baseline.
             Model stays CPU-resident; Dynamo recompiles a fresh NEFF for every
             new input shape it encounters.
nxdi     – NxD Inference: AOT-compile with automatic context/token bucketing
             and a static on-device KV cache.  Weights live in HBM.
trace    – torch_neuronx.trace: one AOT-compiled prefill graph per input-length
             bucket.  TTFT comes from the Neuron NEFF; subsequent decode steps
             run on CPU with the base HF model (hybrid; demonstrates prefill
             benefit only).

Usage
-----
    from backends import get_backend, RunMetrics, save_metrics, log

    backend = get_backend("openxla")          # or "nxdi" / "trace"
    backend.prepare(cfg, log_file)
    metrics = backend.generate(input_ids, max_new_tokens, log_file)
    save_metrics(run_dir / "metrics.json", metrics)
    backend.teardown()
"""

import gc
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM


# ─── SHARED TYPES ─────────────────────────────────────────────────────────────

_DTYPE_MAP: Dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}

_DTYPE_ABBREV: Dict[str, str] = {
    "bfloat16": "bf16",
    "float16":  "fp16",
    "float32":  "fp32",
}


@dataclass
class RunMetrics:
    """Metrics for a single test point (prefill call or full decode generation)."""
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0
    tpt_ms: float = 0.0
    total_time_ms: float = 0.0
    # Wall-clock time of the AOT compilation / trace phase in prepare().
    # 0.0 for openxla (compilation is lazy, tracked via per_step_compile_deltas).
    compile_ms: float = 0.0
    per_token_latencies_ms: list[float] = field(default_factory=list)
    per_step_compile_deltas: list[int] = field(default_factory=list)
    total_compile_delta: int = 0


def save_metrics(path: Path, metrics: RunMetrics) -> None:
    payload = {
        "input_tokens":            metrics.input_tokens,
        "output_tokens":           metrics.output_tokens,
        "ttft_ms":                 round(metrics.ttft_ms, 2),
        "tpt_ms":                  round(metrics.tpt_ms, 2),
        "total_time_ms":           round(metrics.total_time_ms, 2),
        "compile_ms":              round(metrics.compile_ms, 2),
        "total_compile_delta":     metrics.total_compile_delta,
        "per_token_latencies_ms":  metrics.per_token_latencies_ms,
        "per_step_compile_deltas": metrics.per_step_compile_deltas,
    }
    with open(path, "w", encoding="UTF-8") as f:
        json.dump(payload, f, indent=2)


# ─── LOGGING ──────────────────────────────────────────────────────────────────

def get_post(label: str) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    return f"\n>>> [POST] {ts} {label} <<<\n"


def log(log_file, text: str) -> None:
    with open(log_file, "a", encoding="UTF-8") as f:
        f.write(get_post(text))


# ─── NEURON COMPILE-CACHE COUNTING ────────────────────────────────────────────
# Used by OpenXLABackend to detect recompilation events (heuristic: new .neff
# files appearing in the Neuron cache directories).

_NEURON_CACHE_DIRS = [
    Path.home() / ".cache" / "neuron",
    Path("/var/tmp/neuron-compile-cache"),
]


def _count_neuron_neff_files() -> int:
    count = 0
    for base in _NEURON_CACHE_DIRS:
        if base.exists():
            count += sum(1 for _ in base.rglob("*.neff"))
    count += sum(1 for p in Path("/tmp").glob("neuronxcc-*") if p.is_dir())
    return count


def cache_delta(before: int, after: int) -> int:
    return max(0, after - before)


# ─── BACKEND REGISTRY ─────────────────────────────────────────────────────────

BACKENDS: Dict[str, type] = {}


def _register(cls):
    """Class decorator: add cls to the BACKENDS registry under cls.name."""
    BACKENDS[cls.name] = cls
    return cls


def get_backend(name: str):
    """Return a fresh instance of the named backend."""
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {sorted(BACKENDS)}"
        )
    return BACKENDS[name]()


# ─── OPENXLA BACKEND ──────────────────────────────────────────────────────────

@_register
class OpenXLABackend:
    """
    Baseline: torch.compile(backend='openxla').

    The model stays CPU-resident.  openxla JIT-compiles a new NEFF graph each
    time Dynamo encounters a fresh (input_len, kv_len) shape pair.  Because the
    KV cache grows by one position per decode step, every step triggers a
    recompilation and the compiled-graph cache in host RAM grows monotonically.
    This backend captures that dynamic-shape worst-case behaviour so it can be
    compared against the AOT approaches below.
    """
    name = "openxla"

    def __init__(self) -> None:
        self.model = None
        self.device = None

    def prepare(self, cfg, log_file) -> None:
        import torch_xla.core.xla_model as xm

        torch_dtype = _DTYPE_MAP.get(cfg.dtype, torch.bfloat16)

        log(log_file, "MODEL_LOAD START")
        model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch_dtype)
        model.eval()
        log(log_file, "MODEL_LOAD DONE")

        log(log_file, "MODEL_DEVICE START")
        self.device = xm.xla_device()
        log(log_file, "MODEL_DEVICE DONE")

        log(log_file, "TORCH_COMPILE START")
        self.model = torch.compile(model, backend="openxla")
        log(log_file, "TORCH_COMPILE DONE")

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        log_file=None,
    ) -> RunMetrics:
        m = RunMetrics(input_tokens=input_ids.shape[1])

        # ── prefill ───────────────────────────────────────────────
        before = _count_neuron_neff_files()
        t0 = time.monotonic()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, past_key_values=None, use_cache=True)
        prefill_ms = (time.monotonic() - t0) * 1000
        m.ttft_ms = prefill_ms
        m.per_token_latencies_ms.append(round(prefill_ms, 3))
        m.per_step_compile_deltas.append(
            cache_delta(before, _count_neuron_neff_files())
        )

        past = out.past_key_values
        next_tok = out.logits[:, -1].argmax(-1, keepdim=True)

        # ── decode ────────────────────────────────────────────────
        for _ in range(max_new_tokens):
            before = _count_neuron_neff_files()
            t_step = time.monotonic()
            with torch.no_grad():
                out = self.model(
                    input_ids=next_tok, past_key_values=past, use_cache=True
                )
            step_ms = (time.monotonic() - t_step) * 1000
            m.per_token_latencies_ms.append(round(step_ms, 3))
            m.per_step_compile_deltas.append(
                cache_delta(before, _count_neuron_neff_files())
            )
            past = out.past_key_values
            next_tok = out.logits[:, -1].argmax(-1, keepdim=True)

        m.output_tokens = max_new_tokens
        m.total_time_ms = sum(m.per_token_latencies_ms)
        decode_steps = m.per_token_latencies_ms[1:]
        m.tpt_ms = (
            sum(decode_steps) / len(decode_steps) if decode_steps else 0.0
        )
        m.total_compile_delta = sum(m.per_step_compile_deltas)

        # Release KV cache / graphs before the next run.
        del out, past, next_tok, input_ids
        gc.collect()
        import torch_xla.core.xla_model as xm
        xm.mark_step()
        return m

    def teardown(self) -> None:
        del self.model
        self.model = None
        gc.collect()


# ─── NXD INFERENCE BACKEND ────────────────────────────────────────────────────

@_register
class NxDIBackend:
    """
    NxD Inference (neuronx_distributed_inference).

    compile() traces all context-encoding and token-generation graphs for
    every bucket size and saves the compiled NEFFs to
    ./compiled/nxdi/{model_safe_name}/.  Subsequent runs skip compilation and
    jump straight to load().  Weights are loaded into HBM and stay there for
    the lifetime of this backend instance.

    TTFT is measured by calling model.forward() directly for the prefill step.
    The full generation (prefill + decode) is timed via the HF generation
    adapter, giving us total_time_ms.  tpt_ms is derived as
    (total - ttft) / (n_tokens - 1).

    Neuron config
    -------------
    tp_degree   = 2   (safe for all Qwen2.5 variants on trn2.3xlarge)
    seq_len     = max(input_lens) + max(output_lens)
    context_encoding_buckets = sorted(input_lens)  (one compiled graph per il)
    token_generation_buckets = auto (NxD generates powers-of-two up to seq_len)
    """
    name = "nxdi"

    def __init__(self) -> None:
        self.model = None
        self.compile_ms: float = 0.0
        self._compiled_path: Optional[str] = None

    def prepare(self, cfg, log_file) -> None:
        from neuronx_distributed_inference.models.qwen2.modeling_qwen2 import (
            NeuronQwen2ForCausalLM,
            Qwen2NeuronConfig,
            Qwen2InferenceConfig,
        )
        from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

        torch_dtype = _DTYPE_MAP.get(cfg.dtype, torch.bfloat16)

        max_ctx = max(cfg.input_lens)
        max_new = max(cfg.output_lens)
        seq_len = max_ctx + max_new

        neuron_config = Qwen2NeuronConfig(
            tp_degree=2,
            batch_size=1,
            seq_len=seq_len,
            max_context_length=max_ctx,
            max_length=seq_len,
            torch_dtype=torch_dtype,
            context_encoding_buckets=sorted(set(cfg.input_lens)),
            enable_bucketing=True,
        )

        config = Qwen2InferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(cfg.model),
        )

        model_safe = cfg.model.replace("/", "_")
        compiled_path = str(Path("compiled/nxdi") / model_safe)
        self._compiled_path = compiled_path

        t0 = time.monotonic()

        self.model = NeuronQwen2ForCausalLM(cfg.model, config)
        # NxDI's get_state_dict() only takes the local-directory fast path when
        # model_path passes os.path.isdir().  When given a HF repo ID it falls
        # into init_on_device(meta) + from_pretrained, which then crashes on
        # tie_weights() for tied-embedding models ("Cannot copy out of meta
        # tensor").  Fix: resolve the repo ID to its local HF snapshot cache so
        # get_state_dict uses load_state_dict() directly.
        # normalize_path() also appended a trailing slash which broke
        # HFValidationError, so we override model_path unconditionally.
        try:
            from huggingface_hub import snapshot_download
            local_model_path = snapshot_download(cfg.model, local_files_only=True)
            self.model.model_path = local_model_path
        except Exception:
            # Fall back to plain repo name (trailing slash already stripped by
            # normalize_path → rstrip ensures HF validator doesn't reject it).
            self.model.model_path = cfg.model

        if not (Path(compiled_path) / "model.pt").exists():
            log(log_file, "NXDI_COMPILE START")
            self.model.compile(compiled_path)
            log(log_file, "NXDI_COMPILE DONE")
        else:
            log(log_file, "NXDI_LOAD START")

        self.model.load(compiled_path)
        self.compile_ms = (time.monotonic() - t0) * 1000
        log(log_file, "NXDI_READY DONE")

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        log_file=None,
    ) -> RunMetrics:
        from neuronx_distributed_inference.utils.hf_adapter import (
            HuggingFaceGenerationAdapter,
        )

        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

        # ── TTFT: one prefill forward call ────────────────────────
        self.model.reset()
        t0 = time.monotonic()
        self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        ttft_ms = (time.monotonic() - t0) * 1000

        # ── full generation (prefill + decode) ────────────────────
        # HuggingFaceGenerationAdapter.generate() calls model.reset() internally,
        # so we get a clean run starting from an empty KV cache.
        adapter = HuggingFaceGenerationAdapter(self.model)
        t1 = time.monotonic()
        output_ids = adapter.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        total_ms = (time.monotonic() - t1) * 1000

        self.model.reset()

        n_new = output_ids.shape[1] - input_ids.shape[1]
        m = RunMetrics(
            input_tokens=input_ids.shape[1],
            output_tokens=n_new,
            ttft_ms=ttft_ms,
            total_time_ms=total_ms,
            compile_ms=self.compile_ms,
        )
        if n_new > 1:
            m.tpt_ms = (total_ms - ttft_ms) / (n_new - 1)
        return m

    def teardown(self) -> None:
        del self.model
        self.model = None
        gc.collect()


# ─── TRACE BACKEND ────────────────────────────────────────────────────────────

class _PrefillModule(torch.nn.Module):
    """Thin Module wrapper used by TraceBackend so that torch_neuronx.trace
    invokes its move_state_to_device context manager, which replaces all
    parameters with XLA PlaceholderParameter tensors for the duration of the
    trace.  Without this, passing a plain closure keeps weights on CPU while
    inputs are moved to XLA, causing Int32PermissiveEmbedding to raise
    'Expected XLA tensor. Got: CPUBFloat16Type'.
    """
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids)
        return out.logits[:, -1, :]  # [1, vocab_size]


@_register
class TraceBackend:
    """
    torch_neuronx.trace: one AOT-compiled prefill graph per input-length bucket.

    Compiled graphs are cached under ./compiled/trace/{model_safe_name}/ and
    re-used across runs.  compile_ms records the total wall-clock time of all
    trace() calls (or 0 if all graphs were loaded from cache).

    TTFT comes from the Neuron NEFF (fast, no JIT delay, HBM-resident model).
    Decode steps run on CPU with the base HF model.  tpt_ms therefore reflects
    CPU-side decode performance, not Neuron throughput.

    This is intentional: the TraceBackend isolates the TTFT improvement from
    static-graph compilation, making it directly comparable with openxla COLD
    vs WARM runs.  A production decode path would use NxDI instead.
    """
    name = "trace"

    def __init__(self) -> None:
        self.prefill_graphs: Dict[int, object] = {}   # bucket_len -> ScriptModule
        self.base_model = None
        self.compile_ms: float = 0.0
        self._bucket_lens: list = []

    def prepare(self, cfg, log_file) -> None:
        import torch_neuronx  # noqa: F401  (triggers neuron runtime init)

        torch_dtype = _DTYPE_MAP.get(cfg.dtype, torch.bfloat16)

        log(log_file, "MODEL_LOAD START")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model, torch_dtype=torch_dtype
        )
        self.base_model.eval()
        log(log_file, "MODEL_LOAD DONE")

        model_safe = cfg.model.replace("/", "_")
        compiled_dir = Path("compiled/trace") / model_safe
        compiled_dir.mkdir(parents=True, exist_ok=True)

        bucket_lens = sorted(set(cfg.input_lens))
        self._bucket_lens = bucket_lens

        t0 = time.monotonic()
        for blen in bucket_lens:
            cache_path = compiled_dir / f"prefill_b{blen}.pt"

            if cache_path.exists():
                log(log_file, f"TRACE_LOAD START b{blen}")
                graph = torch.jit.load(str(cache_path))
                log(log_file, f"TRACE_LOAD DONE b{blen}")
            else:
                log(log_file, f"TRACE_COMPILE START b{blen}")
                example = torch.zeros(1, blen, dtype=torch.long)
                # torch_neuronx.trace only moves model parameters to XLA
                # (via its internal move_state_to_device context manager) when
                # func is a torch.nn.Module.  Passing a plain closure keeps
                # weights on CPU while inputs become XLA tensors, causing
                # Int32PermissiveEmbedding to raise "Expected XLA tensor".
                # Use a Module wrapper so move_state_to_device handles the
                # parameter placement and restoration automatically.
                graph = torch_neuronx.trace(
                    _PrefillModule(self.base_model),
                    (example,),
                )
                torch.jit.save(graph, str(cache_path))
                log(log_file, f"TRACE_COMPILE DONE b{blen}")

            self.prefill_graphs[blen] = graph

        self.compile_ms = (time.monotonic() - t0) * 1000
        log(log_file, "TRACE_READY DONE")

    def _find_bucket(self, input_len: int) -> int:
        """Return the smallest bucket length that is >= input_len."""
        for blen in self._bucket_lens:
            if blen >= input_len:
                return blen
        return self._bucket_lens[-1]

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        log_file=None,
    ) -> RunMetrics:
        input_len = input_ids.shape[1]
        bucket_len = self._find_bucket(input_len)
        m = RunMetrics(
            input_tokens=input_len,
            compile_ms=self.compile_ms,
        )

        # ── TTFT: traced prefill on Neuron ────────────────────────
        # Pad to the bucket length so the compiled static shape matches.
        padded_ids = torch.zeros(1, bucket_len, dtype=torch.long)
        padded_ids[0, :input_len] = input_ids[0, :input_len]

        before = _count_neuron_neff_files()
        t0 = time.monotonic()
        _ = self.prefill_graphs[bucket_len](padded_ids)
        ttft_ms = (time.monotonic() - t0) * 1000
        m.ttft_ms = ttft_ms
        m.per_token_latencies_ms.append(round(ttft_ms, 3))
        m.per_step_compile_deltas.append(
            cache_delta(before, _count_neuron_neff_files())
        )

        if max_new_tokens == 0:
            m.output_tokens = 0
            m.total_time_ms = ttft_ms
            return m

        # ── decode: CPU base model ────────────────────────────────
        # Re-run prefill on CPU to obtain the KV cache for the decode loop.
        # This CPU prefill overhead is NOT counted in per_token_latencies_ms[0]
        # (that slot holds the Neuron trace TTFT); it is absorbed into the
        # first decode step below.
        with torch.no_grad():
            cpu_out = self.base_model(input_ids=input_ids, use_cache=True)
        past = cpu_out.past_key_values
        next_tok = cpu_out.logits[:, -1].argmax(-1, keepdim=True)

        for _ in range(max_new_tokens):
            before = _count_neuron_neff_files()
            t_step = time.monotonic()
            with torch.no_grad():
                step_out = self.base_model(
                    input_ids=next_tok, past_key_values=past, use_cache=True
                )
            step_ms = (time.monotonic() - t_step) * 1000
            m.per_token_latencies_ms.append(round(step_ms, 3))
            m.per_step_compile_deltas.append(
                cache_delta(before, _count_neuron_neff_files())
            )
            past = step_out.past_key_values
            next_tok = step_out.logits[:, -1].argmax(-1, keepdim=True)

        m.output_tokens = max_new_tokens
        m.total_time_ms = sum(m.per_token_latencies_ms)
        decode_steps = m.per_token_latencies_ms[1:]
        m.tpt_ms = (
            sum(decode_steps) / len(decode_steps) if decode_steps else 0.0
        )
        m.total_compile_delta = sum(m.per_step_compile_deltas)

        del past, next_tok, cpu_out
        gc.collect()
        return m

    def teardown(self) -> None:
        del self.base_model, self.prefill_graphs
        self.base_model = None
        self.prefill_graphs = {}
        gc.collect()
