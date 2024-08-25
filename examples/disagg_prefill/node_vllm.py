import torch
import torch.distributed as dist
import time
DISTRIBUTED_KV_GLOO_TAG = 24857323
#backend = 'cpu:gloo, cuda:nccl'
backend = 'nccl'
distributed_init_method = 'tcp://127.0.0.1:23457'
world_size = 2
vllm_rank = 0
ranks = [0, 1]
torch.distributed.init_process_group(
    backend=backend,
    init_method=distributed_init_method,
    world_size=world_size,
    rank=vllm_rank)
print(f"Rank {vllm_rank} initialized")
cpu_group = torch.distributed.new_group(ranks, backend="gloo")
print(f"Rank {vllm_rank} initialized (cpu group)")
def send():
    tensor = torch.zeros(1)#.cuda()
    tensor += 1
    dist.send(tensor, 1, cpu_group, tag=DISTRIBUTED_KV_GLOO_TAG)
    print(tensor)

send()