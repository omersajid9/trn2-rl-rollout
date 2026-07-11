import torch
import torch_neuronx  # MUST come before torch.compile(backend="neuron")
from mini_verl.workers.generation import GenEngine, GenerationParams  # or from mini_verl.workers.generation
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = model.to("neuron")
