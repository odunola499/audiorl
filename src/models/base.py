import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch import nn
from typing import Union, List
import numpy as np

@dataclass
class ModelForwardOutput:
    loss:torch.Tensor = None
    logits:torch.Tensor = None

@dataclass
class GenerationInput:
    temperature:int = None
    do_sample:bool = True
    top_k:int = None
    max_length:int = 100
    num_beams:int = 2


@dataclass
class GenerationOutput:
    sequences:torch.LongTensor
    hidden_states:torch.FloatTensor = None

@dataclass
class FeaturesRequest:
    input_features:torch.FloatTensor = None
    input_ids:torch.LongTensor = None
    text_attention_mask:torch.LongTensor = None
    audio_attention_mask:torch.LongTensor = None

class ASRBaseModel(ABC, nn.Module):

    @abstractmethod
    def forward(self,
                input_ids:torch.Tensor = None,
                attention_mask:torch.Tensor = None,
                audio_features:torch.Tensor = None) -> ModelForwardOutput:
        """Forward Function"""
        pass

    @abstractmethod
    def generate(self,
                 input_ids:torch.Tensor = None,
                 attention_mask:torch.Tensor = None,
                 audio_features:torch.Tensor = None,
                 gen_inputs:GenerationInput = None) -> GenerationOutput:
        """Generation Function"""
        pass

    @abstractmethod
    def compute_features(self, arrays:List[np.ndarray], texts:List[str] = None) -> FeaturesRequest:
        """Convert audio and text to audio features, input_ids attention mask, text is expected to already have been formatted to prompt template"""
        pass
