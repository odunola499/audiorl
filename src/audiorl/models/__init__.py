from src.audiorl.models.base import PolicyModel, ReferenceModel
from src.audiorl.models.policy.whisper import WhisperModel
from src.audiorl.models.offline_reference.openai import OpenAIReferenceModel
from src.audiorl.models.offline_reference.whisper import WhisperReferenceModel

MAPPING = {
    'whisper': WhisperModel
}
def get_model(name, size = 'large') -> PolicyModel:
    return MAPPING[name](size = size)