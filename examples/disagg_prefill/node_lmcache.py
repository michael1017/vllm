import torch
import torch.distributed as dist
import time
DISTRIBUTED_KV_GLOO_TAG = 24857323
#backend = 'nccl'
#backend = 'cpu:gloo, cuda:nccl'
backend = 'nccl'
distributed_init_method = 'tcp://127.0.0.1:23457'
world_size = 2
lmc_rank = 1
ranks = [0, 1]
torch.distributed.init_process_group(
    backend=backend,
    init_method=distributed_init_method,
    world_size=world_size,
    rank=lmc_rank)
print(f"Rank {lmc_rank} initialized")
cpu_group = torch.distributed.new_group(ranks, backend="gloo")
print(f"Rank {lmc_rank} initialized (cpu group)")
time.sleep(2)
def receive():
    tensor = torch.zeros(1)#.cuda()
    dist.recv(tensor, 0, cpu_group, tag=DISTRIBUTED_KV_GLOO_TAG)
    print(tensor)

receive()