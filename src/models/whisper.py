import numpy as np
from src.models.base import (ASRBaseModel,
                             ModelForwardOutput,
                             GenerationOutput,
                             GenerationInput,
                             FeaturesRequest,
                             DPOFeaturesRequest)
from typing import Literal, List
from transformers import WhisperForConditionalGeneration, AutoTokenizer, AutoFeatureExtractor

from typing import List, Optional
from torch import Tensor

model_sizes = {
    'small':'openai/whisper-small',
    'medium':'openai/whisper-medium',
    'large': 'openai/whisper-large'
}
class WhisperModel(ASRBaseModel):
    def __init__(self, size:str):
        super().__init__()
        if size not in ['small', 'medium', 'large']:
            raise ValueError(f'Invalid model size {size}, choose from small, medium, large')

        self.model = WhisperForConditionalGeneration.from_pretrained(model_sizes[size],
                                                                     attn_implementation = 'flash_attention_2')
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_sizes[size])
        self.tokenizer = AutoTokenizer.from_pretrained(model_sizes[size])

    def forward(self, input_ids:Tensor = None,
                attention_mask:Tensor = None,
                audio_features:Tensor = None,
                compute_log_probs:bool = False,
                use_cache:bool = False):
        model_output = self.model(
            input_features = audio_features, use_cache = use_cache,
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
            audio_features=audio_features,
            input_ids= input_ids,
            text_attention_mask= attention_mask
        )

    def compute_dpo_features(self,
                             audio:List[np.ndarray],
                             prompt:Optional[List[str]],
                             chosen_completions:List[str],
                             rejected_completions:List[str]):
        # TODO: Needs to be moved to the collator
        if prompt:
            prompt_tokens = self.tokenizer(prompt, return_tensors = 'pt')
            prompt_input_ids = prompt_tokens.input_ids
            prompt_attention_mask = prompt_tokens.attention_mask
        else:
            prompt_input_ids = None
            prompt_attention_mask = None

        audio = self.feature_extractor(audio, return_tensors = 'pt').input_features
        chosen = self.tokenizer(chosen_completions, return_tensors = 'pt')
        rejected = self.tokenizer(rejected_completions, return_tensors = 'pt')

        chosen_input_ids = chosen.input_ids
        chosen_attention_mask = chosen.attention_mask
        rejected_input_ids = rejected.input_ids
        rejected_attention_mask = rejected.attention_mask

        return DPOFeaturesRequest(
            audio_features=audio,
            chosen_input_ids=chosen_input_ids,
            prompt_input_ids = prompt_input_ids,
            prompt_attention_mask = prompt_attention_mask,
            chosen_attention_mask=chosen_attention_mask,
            rejected_input_ids = rejected_input_ids,
            rejected_attention_mask = rejected_attention_mask
        )

    def model_name(self):
        return 'whisper'

    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def model_size(self):
        return 'large'

    def pad_token(self):
        return self.tokenizer.pad_token

    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    def eos_token(self):
        return self.tokenizer.eos_token

    def bos_token(self):
        return self.tokenizer.bos_token


