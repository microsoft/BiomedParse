"""Device assignment and file sharding for ordinary Python and torchrun."""

import os

import torch


def inference_worker(files):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid torchrun rank configuration.")
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise ValueError("Launch at most one inference worker per visible GPU.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    files = sorted(files)
    if not files:
        raise ValueError("No supported input files found.")
    return device, files[rank::world_size]