import torch
from dataclasses import dataclass
from typing import List, Literal
from datetime import datetime

@dataclass
class ModelConfig:
    pass

now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
@dataclass
class TrainConfig:
    accelerator:str = 'auto' # 'cpu', 'gpu', 'tpu', 'auto'
    devices:List[int] = -1
    num_epochs:int = 1
    max_steps:int = None
    max_steps_in_audio_secs:int = None
    max_train_tokens:int = 1_000_000_000 # 1 billion tokens
    lr:int = None
    batch_size:int = 2
    save_every:int = 1000
    snapshot_path:str = f'./checkpoints_{now_str}.ckpt'
    save_num_checkpoints:int = 1 # todo: Only keep the last n checkpoints
    log_every:int = 10
    eval_every:int = 100
    num_eval_steps:int = None
    max_completion_length:int = 500
    optimizer:str = 'bnb_adamw'
    scheduler:str = 'linear'
    weight_decay:float = 0.01
    log_to:Literal['wandb','csv'] = 'csv'
    warmup_steps:int = 0
    warmup_ratio:float = 0.2
    gradient_accumulation_steps:int = 1
    push_to_hub:bool = False
    hub_model_id:str = None





@dataclass
class DataConfig:
    prompt_text_column:str = 'prompt'
    chosen_text_column:str = 'chosen'
    rejected_text_column:str = 'rejected'



