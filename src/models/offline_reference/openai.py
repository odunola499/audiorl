import openai
from src.models.base import ReferenceModel, GenerationInput
from typing import Optional,Union, List
import warnings

class OpenAIReferenceModel(ReferenceModel):
    def __init__(self, api_key: str, model_name: str = "gpt-4o-audio-preview"):
        super().__init__(tokenizer=None)
        openai.api_key = api_key
        self.model_name = model_name

    def generate(self, audio:Union[bytes, List[bytes]], prompt: Optional[List[str]], generation_params:GenerationInput = None):
        if isinstance(audio, bytes):
            audio = [audio]
        elif not isinstance(audio, list):
            raise ValueError("Audio must be bytes or list of bytes")

        if prompt is None:
            prompt = [""] * len(audio)
        elif isinstance(prompt, str):
            prompt = [prompt]
        elif not isinstance(prompt, list):
            raise ValueError("Prompt must be a string or list of strings")

        if len(audio) != len(prompt):
            warnings.warn(f"Length of audio {len(audio)} and prompt {len(prompt)} lists dont match, all audio would share same prompt", UserWarning)
            prompt = [prompt[0]] * len(audio)


        transcriptions = []
        for audio_bytes, p in zip(audio, prompt):
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that transcribes audio."},
                    {"role": "user", "content": p}
                ],
                files=[
                    {
                        "file": ("audio.wav", audio_bytes, "audio/wav")
                    }
                ],
                max_tokens=generation_params.max_length if generation_params else 1000,
                temperature=generation_params.temperature if generation_params else 0,
                n=1,
                stop=None
            )
            transcription = response['choices'][0]['message']['content']
            transcriptions.append(transcription)
        if len(transcriptions) == 1:
            return transcriptions[0]
        return transcriptions