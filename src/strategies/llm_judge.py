from typing import List, Callable, Optional
import numpy as np
from data_generation_base import BaseDataGenerator, PreferenceDataPoint, GenerationConfig
from src.models.base import GenerationInput

class LLMJudgeGenerator(BaseDataGenerator):
    """
    Off-policy preference data generation using LLM as judge to select best/worst samples.
    
    This strategy:
    1. Samples multiple transcriptions from the policy model
    2. Uses an LLM judge to evaluate and rank the transcriptions
    3. Selects best as chosen and worst as rejected
    
    Most cost-effective approach as it only requires text evaluation by LLM,
    not audio processing.
    """
    
    def __init__(self, 
                 policy_model,
                 llm_judge: Callable[[List[str], Optional[str]], tuple],
                 config: GenerationConfig = None):
        super().__init__(policy_model, config)
        self.llm_judge = llm_judge
    
    def generate_preference_pairs(self, 
                                audio_batch: List[np.ndarray],
                                prompts: List[str] = None) -> List[PreferenceDataPoint]:
        
        preference_pairs = []
        
        for i, audio in enumerate(audio_batch):
            prompt = prompts[i] if prompts else None
            
            candidate_transcriptions = []
            for _ in range(self.config.num_samples_per_audio):
                gen_input = GenerationInput(
                    temperature=self.config.temperature,
                    do_sample=True
                )
                transcription = self.policy_model.generate(
                    audio=audio, 
                    text=prompt,
                    gen_inputs=gen_input
                )
                candidate_transcriptions.append(transcription)
            
            if len(set(candidate_transcriptions)) < 2:
                continue
            
            best_transcription, worst_transcription = self.llm_judge(
                candidate_transcriptions, 
                prompt
            )
            
            preference_pairs.append(PreferenceDataPoint(
                audio=audio,
                chosen=best_transcription,
                rejected=worst_transcription,
                prompt=prompt
            ))
        
        return preference_pairs
    
    def _default_judge_prompt(self, transcriptions: List[str], context: str = None) -> str:
        """Default prompt template for LLM judge evaluation"""
        context_str = f"Audio context: {context}\n" if context else ""
        transcriptions_str = "\n".join([f"{i+1}. {t}" for i, t in enumerate(transcriptions)])
        
        return f"""
{context_str}
Rate these transcriptions for accuracy, clarity, and natural language quality:

{transcriptions_str}

Return only two numbers: the index of the BEST transcription and the index of the WORST transcription.
Format: BEST: X, WORST: Y
"""