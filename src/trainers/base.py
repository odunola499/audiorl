import torch
from abc import ABC, abstractmethod
from torch import nn, Tensor

class BaseTrainer(ABC, nn.Module):
    pass