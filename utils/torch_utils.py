import numpy as np
import torch
import torch.nn as nn


def worker_init_fn(worker_id):
    """Function to set random numpy seed, see:
    https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/"""
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([worker_id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
