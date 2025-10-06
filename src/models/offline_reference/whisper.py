from src.models.base import ReferenceModel, GenerationInput
from typing import Optional,Union, List
from transformers import WhisperForConditionalGeneration, AutoFeatureExtractor
import torch


class WhisperReferenceModel(ReferenceModel):
    def __init__(self, device = 'cuda'):

        super().__init__(tokenizer=None)
        self.model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3',
                                                                     attn_implementation='flash_attention_2')

        self.model.encoder = torch.compile(self.model.encoder)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained('openai/whisper-large-v3')
        self.model = self.model.to(device)
        self.device = device

    def generate(self, audio:Union[bytes, List[bytes]], prompt: Optional[List[str]], generation_params:GenerationInput = None):

        if isinstance(audio, bytes):
            array = [self.bytes_to_array(audio, self.feature_extractor.sampling_rate)]
        elif isinstance(audio, list):
            array = [self.bytes_to_array(a, self.feature_extractor.sampling_rate) for a in audio]
        else:
            raise ValueError("Audio must be bytes or list of bytes")

        if not generation_params:
            generation_params = GenerationInput()

        inputs = self.feature_extractor(array, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        generated_ids = self.model.generate(input_features=input_features,
                                            max_length=448,
                                            temperature = generation_params.temperature,
                                            do_sample = generation_params.do_sample,
                                            top_k = generation_params.top_k,
                                            num_beams = generation_params.num_beams)

        transcription = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        if len(transcription) == 1:
            transcription = transcription[0]
        return transcription