from src.models.base import PolicyModel, ReferenceModel
from src.models.policy.whisper import WhisperModel
from src.models.offline_reference.openai import OpenAIReferenceModel
from src.models.offline_reference.whisper import WhisperReferenceModel

MAPPING = {
    'whisper': WhisperModel
}
def get_model(name, size = 'large') -> PolicyModel:
    return MAPPING[name](size = size)