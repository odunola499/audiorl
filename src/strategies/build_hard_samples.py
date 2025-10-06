from src.models import PolicyModel, ReferenceModel
from typing import List, Optional, Union
import jiwer
import random

class HardSamplesStrategy:
    def __init__(self, policy_model: PolicyModel, references: List[ReferenceModel], device: str = 'cuda', batch_size = 8):
        self.policy_model = policy_model.to(device)
        self.references = references
        self.device = device
        self.batch_size = batch_size

    def _policy_and_ref_transcripts(self, audio_samples: List[bytes], prompts: Optional[List[str]] = None) -> dict:

        if prompts is None:
            prompts = [None] * len(audio_samples)

        results = {'policy': []}

        for index in range(0, len(audio_samples), self.batch_size):
            batch_audio = audio_samples[index:index + self.batch_size]
            batch_prompts = prompts[index:index + self.batch_size]

            policy_transcriptions = self.policy_model.generate(audio = audio_samples, prompt = prompts)
            results['policy'].extend(policy_transcriptions.texts)


            for ref in self.references:
                ref_transcriptions = ref.generate(audio = batch_audio, prompt = batch_prompts)
                name = ref.__class__.__name__
                if name in results:
                    results[name] = []
                results[name].extend(ref_transcriptions)

        return results

    def get_hard_samples(self, results: dict, cutoff: float = 0.15):
        policies = results['policy']
        ref_dict = {k: v for k, v in results.items() if k != 'policy'}

        chosen_samples, rejected_samples = [], []

        for i, policy_text in enumerate(policies):
            wers = []
            refs_for_sample = []
            names = []

            for name, ref_texts in ref_dict.items():
                ref_text = ref_texts[i]
                refs_for_sample.append((name, ref_text))
                wers.append(jiwer.wer(ref_text, policy_text))
                names.append(name)

            if all(w > cutoff for w in wers):
                priority_refs = [
                    ref_text for (name, ref_text) in refs_for_sample
                    if any(x in name.lower() for x in ["gemini", "openai", "elevenlabs"])
                ]

                if priority_refs:
                    chosen = priority_refs[0]
                else:
                    chosen = random.choice([r for _, r in refs_for_sample])

                chosen_samples.append(chosen)
                rejected_samples.append(policy_text)

        return chosen_samples, rejected_samples
















