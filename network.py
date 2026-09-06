import os, socket, torch, torch.distributed as dist
from datetime import timedelta

def main():
    backend = "nccl"
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=5))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())

    # 每个 rank 放不同值
    x = torch.tensor([rank + 1.0], device=device)
    dist.barrier()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    # 期望值：world_size*(world_size+1)/2
    expected = world_size * (world_size + 1) / 2
    ok = torch.allclose(x, torch.tensor([expected], device=device))
    print(f"[host={socket.gethostname()} rank={rank}] x={x.item()} expected={expected} ok={ok}", flush=True)

    # 做几轮，验证稳定性
    for i in range(5):
        x = torch.full((1,), float(rank), device=device)
        dist.all_reduce(x)
        if rank == 0:
            print(f"round {i}: sum={x.item()}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    # 建议的诊断环境变量，可按需在外部导出
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")       # 单机常用
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")      # 若拓扑复杂/多 root-complex
    main()