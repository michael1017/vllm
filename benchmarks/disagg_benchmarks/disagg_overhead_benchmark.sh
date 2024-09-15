#!/bin/bash

# Requirement: 8x H100 GPUs.


# Model: neuralmagic/Meta-Llama-3-70B-Instruct-FP8-KV 
# Query: 2048 input tokens, 11 output tokens, QPS 4, 500 requests
# Resource: 8x H100
# Approaches:
# 1. Chunked prefill: 1 vllm instance with tp=8
# 2. Chunked prefill: 2 vllm instance with tp=4, equivalent to 1 tp=4 instance with QPS 4
# 3. Disaggregated prefill: 1 prefilling instance and 1 decoding instance
# Prefilling instance: max_output_token=1
# Decoding instance: force the input tokens be the same across requests to bypass prefilling

set -ex

kill_gpu_processes() {
  # kill all processes on GPU.
  pkill pt_main_thread
  sleep 10

  # remove vllm config file
  rm -rf ~/.config/vllm

  # Print the GPU memory usage
  # so that we know if all GPU processes are killed.
  gpu_memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  # The memory usage should be 0 MB.
  echo "GPU 0 Memory Usage: $gpu_memory_usage MB"
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


benchmark() {

  export VLLM_LOGGING_LEVEL=DEBUG
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  export VLLM_PORT=12345

  # compare chunked prefill with disaggregated prefill

  results_folder="./results"
  model="meta-llama/Meta-Llama-3.1-8B-Instruct"
  dataset_name="sonnet"
  dataset_path="../sonnet_4x.txt"
  num_prompts=10
  qps=$1
  prefix_len=50
  input_len=2048
  output_len=$2

  # large model
  # VLLM_RPC_PORT=5570 VLLM_DISAGG_PREFILL_ROLE=prefill CUDA_VISIBLE_DEVICES=0,1,2,3 python3 \
  #     -m vllm.entrypoints.openai.api_server \
  #     --model $model \
  #     --port 8100 \
  #     -tp 4 \
  #     --max-model-len 30000 \
  #     --gpu-memory-utilization 0.8 &
  # VLLM_RPC_PORT=5580 VLLM_DISAGG_PREFILL_ROLE=decode CUDA_VISIBLE_DEVICES=4,5,6,7 python3 \
  #   -m vllm.entrypoints.openai.api_server \
  #   --model $model \
  #   --port 8200 \
  #   -tp 4 \
  #   --max-model-len 30000 \
  #   --gpu-memory-utilization 0.8 &

  VLLM_RPC_PORT=5570 VLLM_DISAGG_PREFILL_ROLE=prefill CUDA_VISIBLE_DEVICES=0 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8100 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.8 &

# decoding instance
VLLM_RPC_PORT=5580 VLLM_DISAGG_PREFILL_ROLE=decode CUDA_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8200 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.8 &

  wait_for_server 8100
  wait_for_server 8200

  # let the prefill instance finish prefill
  python3 ../benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --sonnet-input-len $input_len \
          --sonnet-output-len $output_len \
          --sonnet-prefix-len $prefix_len \
          --num-prompts $num_prompts \
          --port 8100 \
          --save-result \
          --result-dir $results_folder \
          --result-filename disagg_prefill_2xtp4.json \
          --request-rate "inf"


  # send the request to decode.
  # The TTFT of this command will be the overhead of disagg prefill impl.
  python3 ../benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --sonnet-input-len $input_len \
          --sonnet-output-len $output_len \
          --sonnet-prefix-len $prefix_len \
          --num-prompts $num_prompts \
          --port 8200 \
          --save-result \
          --result-dir $results_folder \
          --result-filename disagg_prefill_2xtp4.json \
          --request-rate $qps
  kill_gpu_processes

}


main() {

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)

  pip install quart httpx

  cd "$(dirname "$0")"

  cd ..
  # create sonnet-4x.txt
  echo "" > sonnet_4x.txt
  for _ in {1..4}
  do
    cat sonnet.txt >> sonnet_4x.txt
  done
  cd disagg_benchmarks

  rm -rf results
  mkdir results

  default_qps=1
  default_output_len=1
  benchmark $default_qps $default_output_len

}


main "$@"
