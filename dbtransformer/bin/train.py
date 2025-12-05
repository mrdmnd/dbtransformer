"""
Minimal PyTorch Distributed Training Demo
==========================================
Run with torchrun:
    Single node, CPU:      torchrun --nproc_per_node=2 train.py
    Single node, 2 GPUs:   torchrun --nproc_per_node=2 train.py
    Multi-node (node 0):   torchrun --nnodes=2 --node_rank=0 --master_addr=<IP> --master_port=29500 --nproc_per_node=2 train.py
    Multi-node (node 1):   torchrun --nnodes=2 --node_rank=1 --master_addr=<IP> --master_port=29500 --nproc_per_node=2 train.py
"""

import os

import torch
import torch.distributed as dist
from loguru import logger
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from tqdm import tqdm


def setup() -> None:
    use_cuda = torch.cuda.is_available()
    logger.info(f"Using CUDA: {use_cuda}")
    backend = "nccl" if use_cuda else "gloo"
    logger.info(f"Using backend: {backend}")
    if not use_cuda:
        logger.warning("CUDA is not available, using CPU + Gloo backend")

    """Initialize distributed process group (torchrun sets env vars automatically)."""
    dist.init_process_group(backend=backend)
    if use_cuda:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup() -> None:
    dist.destroy_process_group()


def get_device_and_rank() -> tuple[torch.device, int | None]:
    """Return appropriate device + rank (GPU if available, otherwise CPU + None)"""
    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        return torch.device(f"cuda:{local_rank}"), local_rank
    return torch.device("cpu"), None


def main() -> None:
    setup()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device, local_rank = get_device_and_rank()

    logger.info(f"[Rank {rank}/{world_size}] Running on device {device}")

    # Simple model
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)

    # Wrap with DDP if using cuda
    if device.type == "cuda":
        wrapped_model = DistributedDataParallel(model, device_ids=[local_rank])
    else:
        wrapped_model = DistributedDataParallel(model)

    # Dummy dataset (each rank sees different subset)
    x = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(x, y)
    sampler: DistributedSampler[TensorDataset] = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in tqdm(range(50)):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        total_loss = 0.0

        for batch_x, batch_y in loader:
            x, y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            loss = loss_fn(wrapped_model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Only rank 0 prints
        if rank == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}")

    cleanup()
    if rank == 0:
        print("Training complete!")


if __name__ == "__main__":
    main()
