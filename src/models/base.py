import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch import nn
from typing import Union, List, Optional
import numpy as np
from io import BytesIO
import soundfile as sf
import librosa

@dataclass
class ModelForwardOutput:
    loss:torch.Tensor = None
    logits:torch.Tensor = None

@dataclass
class GenerationInput:
    temperature:int = 0
    do_sample:bool = False
    top_k:int = 1
    max_length:int = 100
    num_beams:int = 2


@dataclass
class GenerationOutput:
    sequences:torch.LongTensor = None
    hidden_states:torch.FloatTensor = None
    texts: List[str] = None

@dataclass
class FeaturesRequest:
    audio_features:torch.FloatTensor = None
    input_ids:torch.LongTensor = None
    text_attention_mask:torch.LongTensor = None
    audio_attention_mask:torch.LongTensor = None

@dataclass
class DPOFeaturesRequest:
    audio_features:torch.FloatTensor
    prompt_input_ids:torch.LongTensor = None
    prompt_attention_mask:torch.LongTensor = None
    chosen_input_ids:torch.LongTensor = None
    chosen_attention_mask:torch.LongTensor = None
    rejected_input_ids:torch.LongTensor = None
    rejected_attention_mask:torch.LongTensor = None

class ReferenceModel(ABC):

    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

    @abstractmethod
    def generate(self, audio:Union[bytes, List[bytes]], prompt:Optional[List[str]], generation_params:GenerationInput = None) -> Union[List[str], str]:
        pass

    def bytes_to_array(self, bytes, sample_rate):
        audio = BytesIO(bytes)
        array, sr = sf.read(audio)
        if sr != sample_rate:
            array = librosa.resample(array.T, orig_sr=sr, target_sr=sample_rate).T
        return array

class PolicyModel(ABC, nn.Module):

    @abstractmethod
    def forward(self,
                input_ids:torch.Tensor = None,
                attention_mask:torch.Tensor = None,
                audio_features:torch.Tensor = None,
                compute_log_probs:bool = False) -> ModelForwardOutput:
        """Forward Function"""
        # Remember to clamp logits to match seq len of input_ids if model concatenated audio and text embeds
        pass

    @abstractmethod
    def generate(self,
                 input_ids:torch.Tensor = None,
                 attention_mask:torch.Tensor = None,
                 audio_features:torch.Tensor = None,
                 gen_inputs:GenerationInput = None,
                 audio:Union[bytes, List[bytes]] = None,
                 prompt:Optional[List[str]] = None
                 ) -> GenerationOutput:
        """Generation Function"""
        pass


    @abstractmethod
    def compute_features(self, arrays:List[np.ndarray], texts:List[str] = None) -> FeaturesRequest:
        """Convert audio and text to audio features, input_ids attention mask, text is expected to already have been formatted to prompt template"""
        pass

    @property
    @abstractmethod
    def model_name(self):
        """Name Associated with model"""
        pass

    @property
    @abstractmethod
    def pad_token_id(self):
        """Pad token id"""
        pass

    @property
    @abstractmethod
    def pad_token(self):
        """Pad token id"""
        pass

    @property
    @abstractmethod
    def eos_token_id(self):
        """Pad token id"""
        pass

    @property
    @abstractmethod
    def bos_token_id(self):
        """Pad token id"""
        pass

    @property
    @abstractmethod
    def eos_token(self):
        """Pad token id"""
        pass

    @property
    @abstractmethod
    def bos_token(self):
        """Pad token id"""
        pass


    @property
    @abstractmethod
    def model_size(self):
        """Model size"""
        pass
