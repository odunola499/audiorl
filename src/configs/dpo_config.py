from src.configs import TrainArgs
from typing import Optional

class DPOConfig(TrainArgs):
    """Training configuration arguments for DPO, inherits from TrainArgs.
        """
    ref_model: Optional[str] = None
    """Reference model for baseline log-ratio. Inherits from ModelASR. if None, a copy of base model is used."""
    beta:float = 0.1
    """Hyperparameter: Temperature scaling factor to control how strong the model is pushed to
        prefer chosen responses over rejected ones. Higher values increase the preference for chosen responses."""
