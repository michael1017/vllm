from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer

llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", gpu_memory_utilization=0.4,
          enable_prefix_caching=True, max_model_len=5000,
          )
#tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
sampling_params = SamplingParams(temperature=0, max_tokens=1)

input_prompt = "Hello how are you?"*100
llm.generate([input_prompt], sampling_params)

input_prompt = "Hello how are you?"*101
llm.generate([input_prompt], sampling_params)
