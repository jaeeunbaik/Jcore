import os
import torch
import torch.distributed as dist

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)   # ← 이 줄 추가
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[R{rank}/{world_size}] init done")

    tensor = torch.ones(1).cuda()
    for i in range(10):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(f"iter {i}: tensor={tensor.item()}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()