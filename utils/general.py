import torch
import numpy as np
import os
import random
from pathlib import Path


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def seed_everything(seed: int):
    """ Set global seed for full reproducibility. """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Per-worker seed function for DataLoader reproducibility."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# HELPER: AUTO-INCREMENT OUTPUT DIRECTORY
# =============================================================================
def increment_path(path):
    """ Auto-increment directory name to avoid overwriting existing runs.

        Example: ``runs/train/exp`` → ``runs/train/exp2`` → ``runs/train/exp3``
        """
    path = Path(path)
    if not path.exists():
        return path
    for n in range(2, 9999):
        p = Path(f"{path}{n}")
        if not p.exists():
            return p