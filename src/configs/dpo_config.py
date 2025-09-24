from src.configs import TrainConfig
from typing import Optional

class DPOConfig(TrainConfig):
    ref_model: Optional[str] = None
    beta:float = 0.1
    target_kl:float = None
    adaptive_beta:bool = False
    kl_clip:float = 5.0
    label_smoothing:float = 0.1