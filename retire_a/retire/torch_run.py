from neuronx_distributed_inference.models.qwen2.modeling_qwen2 import (
    NeuronQwen2ForCausalLM,
    Qwen2NeuronConfig,
    Qwen2InferenceConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

class NxDIBackend:
    name = "nxdi"

    def __init__(self) -> None:
        self.model = None
        self.compile_ms: float = 0.0
        self._compiled_path: Optional[str] = None

    def prepare(self, cfg, log_file) -> None:

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
        # normalize_path() inside NxDI appends a trailing slash, which turns a
        # HF model ID like "Qwen/Qwen2.5-1.5B-Instruct" into an invalid repo id
        # "Qwen/Qwen2.5-1.5B-Instruct/" and causes HFValidationError on load.
        self.model.model_path = self.model.model_path.rstrip("/")

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




if __name__ == "__main__":
    main()