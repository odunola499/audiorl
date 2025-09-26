import torch
from bitsandbytes import optim as bnb_optim
from src.trainers.base import BaseTrainer
from src.optimization import get_cosine_schedule_with_warmup_by_tokens
from transformers import get_cosine_schedule_with_warmup


OPTIMIZERS = {
    'adamw': torch.optim.AdamW,
    'bnb_adamw': bnb_optim.AdamW8bit,
}

SCHEDULERS = {
    'cosine_warmup': get_cosine_schedule_with_warmup,
    'token_cosine_warmup': get_cosine_schedule_with_warmup_by_tokens
}
