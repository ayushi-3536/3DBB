import random
import numpy as np
import torch

import torch.distributed as dist
import os
def get_dist_info():
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment is not initialized.")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    return world_size, rank
    
def set_random_seed(seed, rank=0):
    """
    Set the random seed for reproducibility across various libraries.
    
    Args:
        seed (int): The base seed value.
        rank (int): The rank of the current process in the distributed setup.
    """
    # Calculate a unique seed for this process
    unique_seed = seed + rank

    # Set seed for Python's built-in random module
    random.seed(unique_seed)
    
    # Set seed for NumPy
    np.random.seed(unique_seed)
    
    # Set seed for PyTorch
    torch.manual_seed(unique_seed)
    
    # Ensure deterministic behavior in CUDA (if using GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(unique_seed)
        torch.cuda.manual_seed_all(unique_seed)  # If using multi-GPU

    # Optionally set additional settings for PyTorch (to ensure full determinism)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False