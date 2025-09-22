import numpy as np
from src.models.base import ASRBaseModel, ModelForwardOutput, GenerationOutput, GenerationInput, FeaturesRequest
from typing import Literal, List
from transformers import WhisperForConditionalGeneration, AutoTokenizer, AutoFeatureExtractor

from torch import Tensor

model_sizes = {
    'small':'openai/whisper-small',
    'medium':'openai/whisper-medium',
    'large': 'openai/whisper-large'
}
class WhisperModel(ASRBaseModel):
    def __init__(self, size:Literal['small', 'medium', 'large']):
        super().__init__()
        self.model = WhisperForConditionalGeneration.from_pretrained(model_sizes[size])
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_sizes[size])
        self.tokenizer = AutoTokenizer.from_pretrained(model_sizes[size])

    def forward(self, input_ids:Tensor = None,
                attention_mask:Tensor = None,
                audio_features:Tensor = None):
        model_output = self.model(
            input_features = audio_features,
        )
        logits = model_output.logits
        loss = model_output.loss
        output = ModelForwardOutput(
            loss = loss,
            logits = logits
        )
        return output

    def generate(self,
                 input_ids:Tensor = None,
                 attention_mask:Tensor = None,
                 audio_features:Tensor = None,
                 gen_inputs: GenerationInput = None
                 ):

        if gen_inputs is None:
            gen_inputs = GenerationInput(
                temperature= 1,
                do_sample=False
            )

        outputs = self.model.generate(
            input_features=audio_features,
            temperature=gen_inputs.temperature,
            do_sample=gen_inputs.do_sample,
            top_k=gen_inputs.top_k,
            max_length=getattr(gen_inputs, "max_length", None),
            num_beams=getattr(gen_inputs, "num_beams", None),
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

        return GenerationOutput(
            sequences=outputs.sequences,
            hidden_states=getattr(outputs, "hidden_states", None),
        )

    def compute_features(self, arrays:List[np.ndarray], texts:List[str] = None) -> FeaturesRequest:
        audio_features = self.feature_extractor(arrays, return_tensors = 'pt').input_features
        input_features = self.tokenizer(texts, return_tensors = 'pt')
        input_ids = input_features.input_ids
        attention_mask = input_features.attention_mask
        return FeaturesRequest(
            input_features=audio_features,
            input_ids= input_ids,
            text_attention_mask= attention_mask
        )


