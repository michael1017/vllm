#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export VLLM_PORT=12345

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# vllm instance
VLLM_LOGGING_LEVEL=DEBUG VLLM_RPC_PORT=5570 VLLM_DISAGG_PREFILL_ROLE=lmc CUDA_VISIBLE_DEVICES=0 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8100 \
    --max-model-len 5000 \
    --gpu-memory-utilization 0.4 &


# lmc driver
python lmcache_main.py &
sleep 10
wait_for_server 8100
# serve an example request
#curl http://localhost:8100/v1/completions \
#-H "Content-Type: application/json" \
#-d '{
#"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
#"prompt": "San Francisco is a",
#"max_tokens": 10,
#"temperature": 0
#}'

# clean up
#ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9