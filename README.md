# AudioRL

## Preference Tuning For Speech Recognition and Audio Models

AudioRL is a lightweight library for preference-tuning speech recognition and audio language models.
It is partly inspired by the reward functions introduced in the FunASR model, designed to enhance word error rate (WER) performance on hard instances of speech recognition.

Big Shout-out to the HuggingFace team for their amazing work on the `transformers` and `trl` libraries, which served as a foundation for this project.
The Framework also draws design inspiration from `Pytorch Lightning` for its modular design and ease of use.

We plan to support Direct Preference Optimization (DPO) and GRPO out of the box, with a design that makes it easy to extend the library to more models and new preference-tuning techniques as the field evolves.

### Features
1. Preference Tuning for ASR and audio llms with hf-like api design.
2. Modular interface for plugging in new models or objectives
3. Support for reward shaping inspired by FunASR

### Roadmap

#### Models
- [ ] Whisper
- [ ] Qwen2 Audio
- [ ] Stepfun Audio

#### Tuning Techniques
- [ ] Direct Preference Optimization
- [ ] Grouped Reward Policy Optimization

#### Dataset Preparation
- [ ] Pairwise Preference Dataset
- [ ] Reward Shaping Dataset

#### Training
