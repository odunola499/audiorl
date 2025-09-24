import torch
from abc import ABC, abstractmethod
from torch import nn, Tensor
from src.models import ASRBaseModel
from typing import Iterable, List, Union, Tuple, Literal
from src.configs import TrainConfig

class BaseTrainer(ABC, nn.Module):
    def __init__(self,
                 model:ASRBaseModel,
                 loaders:Union[Tuple[Iterable], Iterable],
                 config:TrainConfig,
                 optimizer:Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None):
        super().__init__()
        self.rank = 0
        self.world_size = torch.cuda.device_count()
        self.model = model
        if isinstance(loaders, Iterable):
            self.train_loader = loaders
        else:
            if len(loaders) == 1:
                self.train_loader = loaders
            elif len(loaders) == 2:
                self.train_loader, self.valid_loader = loaders
            elif len(loaders) == 3:
                self.train_loader, self.valid_loader, self.test_loader = loaders
            else:
                raise ValueError(f'loaders should either be a single iterator or a tuple of length <=3, got {len(loaders)}')

        self.train_loader, self.valid_loader = loaders
        self.config = config
        self.optimizer, self.scheduler = optimizer

    @abstractmethod
    def train_step(self, batch):
        """A single batch step, batch should be from collate fn"""
        pass

    @abstractmethod
    def validation_step(self, batch):
        """A single batch step, batch should be fro collate n"""
        pass

    @abstractmethod
    def optimizer_step(self):
        pass

    @abstractmethod
    def compute_loss(self,model = None, batch = None, mode:Literal['train', 'valid', 'test'] = 'valid', return_outputs = False, num_items_in_batch = None,logits:torch.FloatTensor = None, labels:torch.LongTensor = None):
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def log_progress(self):
        pass

    @abstractmethod
    def configure_optimizer(self):
        pass

    @abstractmethod
    def compute_metrics(self, model:nn.Module = None, batch = None, model_output = None,mode:Literal['train', 'valid', 'test'] = 'valid') -> dict:
        pass

    def setup_train_dataloader(self, train_loader):
        if not hasattr(train_loader, "__iter__"):
            raise TypeError("train_loader must be iterable")
        self.train_loader = train_loader

    def setup_valid_dataloader(self, valid_loader):
        if not hasattr(valid_loader, "__iter__"):
            raise TypeError("valid_loader must be iterable")
        self.valid_loader = valid_loader




