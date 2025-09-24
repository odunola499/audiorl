from src.models.base import ASRBaseModel
from src.models.whisper import WhisperModel

MAPPING = {
    'whisper': WhisperModel
}
def get_model(name, size = 'large') -> ASRBaseModel:
    return MAPPING[name](size = size)