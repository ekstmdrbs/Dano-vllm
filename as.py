# test_allreduce.py
import torch
import torch.distributed as dist
import os
import argparse

def run(rank, world_size):
    """
    단순한 All-Reduce 연산을 테스트하여 분산 통신 환경을 검증합니다.
    """
    print(f"--> [Rank {rank}] Process starting.")
    
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '23333'
        
        dist.init_process_group(
            backend="nccl", 
            rank=rank, 
            world_size=world_size
        )
        print(f"✅ [Rank {rank}] dist.init_process_group SUCCEEDED.")
    except Exception as e:
        print(f"❌ [Rank {rank}] dist.init_process_group FAILED: {e}")
        return

    # 각 rank가 다른 값을 가진 텐서를 생성
    tensor = torch.tensor([rank + 1], dtype=torch.float32).cuda(rank)
    print(f"[Rank {rank}] Initial tensor: {tensor.item()}")

    # All-Reduce 연산: 모든 텐서의 값을 더해서 모든 rank에 결과를 공유
    print(f"[Rank {rank}] Performing All-Reduce...")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"✅ [Rank {rank}] All-Reduce COMPLETED.")

    # 모든 rank는 동일한 결과값을 가져야 함 (e.g., world_size=2 -> 1+2=3)
    print(f"[Rank {rank}] Final tensor after all_reduce: {tensor.item()}")
    
    dist.destroy_process_group()
    print(f"[Rank {rank}] Process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()
    
    run(rank=args.rank, world_size=args.world_size)
