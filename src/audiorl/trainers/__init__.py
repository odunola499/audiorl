import torch
from bitsandbytes import optim as bnb_optim
from src.audiorl.trainers.base_depreciated import BaseTrainer
from src.audiorl.optimization import get_cosine_schedule_with_warmup_by_tokens
from .utils import RunningMoments
from transformers import get_cosine_schedule_with_warmup


OPTIMIZERS = {
    'adamw': torch.optim.AdamW,
    'bnb_adamw': bnb_optim.AdamW8bit,
}

SCHEDULERS = {
    'cosine_warmup': get_cosine_schedule_with_warmup,
    'token_cosine_warmup': get_cosine_schedule_with_warmup_by_tokens
}
