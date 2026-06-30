import torch, torch_neuronx
from transformers import AutoModelForCausalLM, AutoTokenizer


# manual prefill+decode loop — this is what TRL's HF server does
def generate(model, tok, prompt, max_new):
    ids = tok(prompt, return_tensors="pt").input_ids
    past = None
    for step in range(max_new):
        out = model(input_ids=ids, past_key_values=past, use_cache=True)
        past = out.past_key_values        # grows by 1 every step → shape changes
        next_tok = out.logits[:, -1].argmax(-1, keepdim=True)
        ids = next_tok
    return ids


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.bfloat16)
    model = torch.compile(model, backend="openxla")   # or whatever Neuron's torch.compile entry point is
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    generate(model, tok, "Hello jee", 3)