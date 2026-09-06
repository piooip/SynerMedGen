# ddp_smoke.py
import os, torch, torch.distributed as dist
from datetime import timedelta
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", init_method="env://",
    rank=rank, world_size=world, timeout=timedelta(minutes=5))
print(f"[rank {rank}] pg inited on cuda:{local_rank}", flush=True)
dist.barrier()
if rank == 0: print("all good!", flush=True)
dist.destroy_process_group()
