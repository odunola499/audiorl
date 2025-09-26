import torch
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer

def get_cosine_schedule_with_warmup_by_tokens(optimizer:torch.optim.Optimizer, total_tokens,
                                   global_batch_size:int,
                                   seq_len:int,
                                   warmup_ratio:float = None,
                                   warmup_steps:int  = None,
                                   min_lr_ratio = 0.1):
    steps = total_tokens // (global_batch_size * seq_len)
    assert warmup_steps is not None or warmup_ratio is not None, "Either warmup_steps or warmup_ratio must be provided"
    if warmup_steps is None:
        warmup_steps = int(steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)

