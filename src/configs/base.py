import torch
from dataclasses import dataclass
from typing import List, Literal
from datetime import datetime


@dataclass
class ModelConfig:
    """Configuration for model architecture and parameters.

    """
    pass


now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class TrainArgs:
    """Training configuration arguments.

    These parameters control training behavior, logging, saving, and optimization.
    """

    accelerator: str = 'auto'
    """Device to run training on: 'cpu', 'gpu', or 'auto'."""

    precision: Literal['bf16_mixed', 'float32'] = 'float32'
    """Numerical precision used during training."""

    devices: List[int] = -1
    """Which devices (GPU indices) to use. -1 means use all available devices."""

    num_epochs: int = 1
    """Number of full passes through the dataset."""

    max_steps: int = None
    """Maximum number of training steps (overrides num_epochs if set)."""

    max_train_tokens: int = 1_000_000_000
    """Maximum number of tokens to process during training."""

    lr: int = None
    """Learning rate for the optimizer."""

    batch_size: int = 2
    """Number of samples per training batch."""

    save_every: int = 1000
    """How often (in steps) to save checkpoints."""

    snapshot_path: str = f'./checkpoints_{now_str}.ckpt'
    """Default path for saving checkpoint snapshots."""

    save_num_checkpoints: int = 1
    """Number of recent checkpoints to keep on disk."""

    log_every: int = 10
    """How often (in steps) to log training metrics."""

    log_audio_every: int = 500
    """How often (in steps) to log audio samples (expensive, so less frequent)."""

    log_n_audio: int = 4
    """Number of audio samples to log each time audio logging is triggered."""

    log_dir: str = './logs'
    """Directory to store log files (for CSV or other loggers)."""

    eval_every: int = 100
    """How often (in steps) to run evaluation."""

    num_eval_steps: int = None
    """Number of steps to use during evaluation (None = full eval set)."""

    max_completion_length: int = 500
    """Maximum length of generated output during evaluation."""

    optimizer: str = 'bnb_adamw'
    """Optimizer choice, e.g. 'adamw', 'bnb_adamw'."""

    scheduler: str = 'linear'
    """Learning rate scheduler type."""

    weight_decay: float = 0.01
    """Weight decay coefficient for regularization."""

    log_to: Literal['wandb', 'csv'] = 'csv'
    """Logging backend to use. Defaults to CSV if wandb is not specified."""

    warmup_steps: int = 0
    """Number of learning rate warmup steps."""

    warmup_ratio: float = 0.2
    """Fraction of total steps used for learning rate warmup."""

    gradient_accumulation_steps: int = 1
    """Accumulate gradients over this many steps before an optimizer update."""

    push_to_hub: bool = False
    """Whether to push trained model to Hugging Face Hub."""

    hub_model_id: str = None
    """Model ID to use when pushing to Hugging Face Hub."""

    project_name: str = 'ASR_Pref'
    """Project name for experiment tracking/logging."""

    experiment_name: str = f'ASR_Pref-{now_str}'
    """Unique experiment run name, defaults to timestamped project name."""

    wandb_entity: str = None
    """Weights & Biases entity (team or username)."""


@dataclass
class DataConfig:
    """Configuration for dataset columns used in training."""

    prompt_text_column: str = 'prompt'
    """Name of the column containing input prompts."""

    chosen_text_column: str = 'chosen'
    """Name of the column with preferred/completion responses."""

    rejected_text_column: str = 'rejected'
    """Name of the column with rejected/comparison responses."""
