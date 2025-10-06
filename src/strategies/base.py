import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseCollator(ABC):
    def __init__(self, processor):
        self.processor = processor

    @abstractmethod
    def __call__(self, batch:List[Dict[str, Any]]) -> Dict[str, Any]:
        pass



class BaseStrategy(ABC):
    pass
