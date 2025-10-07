from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from dataclasses import dataclass
from src.audiorl.models.base import PolicyModel, ReferenceModel

@dataclass
class PreferenceDataPoint:
    """Single preference data point containing audio and text pairs"""
    audio: np.ndarray
    chosen: str
    rejected: str
    prompt: str = None

@dataclass 
class GenerationConfig:
    """Configuration for data generation process"""
    batch_size: int = 16
    max_samples: int = 1000
    temperature: float = 1.0
    num_samples_per_audio: int = 2

class BaseDataGenerator(ABC):
    """
    Base class for off-policy preference data generation strategies.
    
    Generates preference pairs from audio inputs using different sampling strategies
    to create training data for preference tuning methods like DPO.
    """
    
    def __init__(self, 
                 policy_model: PolicyModel,
                 config: GenerationConfig = None):
        self.policy_model = policy_model
        self.config = config or GenerationConfig()
    
    @abstractmethod
    def generate_preference_pairs(self, 
                                audio_batch: List[np.ndarray],
                                prompts: List[str] = None) -> List[PreferenceDataPoint]:
        """
        Generate preference pairs from audio inputs.
        
        Args:
            audio_batch: List of audio arrays
            prompts: Optional prompts for each audio
            
        Returns:
            List of preference data points with chosen/rejected pairs
        """
        pass
    
    def _transcribe_batch(self, 
                         model: Union[PolicyModel, ReferenceModel],
                         audio_batch: List[np.ndarray],
                         prompts: List[str] = None,
                         **gen_kwargs) -> List[str]:
        """Helper method to transcribe a batch of audio with given model"""
        transcriptions = []
        for i, audio in enumerate(audio_batch):
            prompt = prompts[i] if prompts else None
            transcription = model.generate(audio, prompt, **gen_kwargs)
            transcriptions.append(transcription)
        return transcriptions