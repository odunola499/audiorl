import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    pass

@dataclass
class TrainConfig:
    num_epochs:int
    max_steps:int = None
    lr:int = None
    batch_size:int = 2
    save_steps:int = 1000
    log_steps:int = 10
    eval_steps:int = 100
    max_completion_length:int = 500





@dataclass
class DataConfig:
    prompt_text_column:str = 'prompt'
    chosen_text_column:str = 'chosen'
    rejected_text_column:str = 'rejected'



