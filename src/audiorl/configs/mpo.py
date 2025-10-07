from .dpo import DPOArgs

class MPOArgs(DPOArgs):
    """Training configuration arguments for MPO.
        """
    loss_weights:list = [1.0, 1.0, 1.0]