# AudioRL

## Preference Tuning For Speech Recognition and Audio Models

AudioRL is a lightweight library for preference-tuning speech recognition and audio language models.
It is partly inspired by the reward functions introduced in the FunASR model, designed to enhance word error rate (WER) performance on hard instances of speech recognition.

It turns out that a lot of preference tuning research done in Vision language models are applicable to audio models as well, and AudioRL aims to bring some of these techniques to the audio community. The more novel research has gone into preparing datasets and reward functions for audio models, and we would be sharing some soon.

Big Shout-out to the HuggingFace team for their amazing work on the `transformers` and `trl` libraries, which served as a foundation for this project.
The Framework also draws design inspiration from `Pytorch Lightning` for its modular design and ease of use.

We plan to support Direct Preference Optimization (DPO) and other offline RL techniques popularized by multimodal lms like mpo from internvl and stepfun audio out of the box, with a design that makes it easy to extend the library to more models and new preference-tuning techniques as the field evolves.

Sometime in the future, we would support online RL techniques, speech output models(omni type models and text-to-speech) and a <7b Audio language model built from the ground up with open source data, code and infra.


### Features
1. Preference Tuning for ASR and audio llms with hf and pytorch lightning api design.
2. Distributed Training (DDP) support with torchrun.
2. Modular interface for plugging in new models or objectives
3. Support for reward shaping inspired by FunASR

### Roadmap

#### Models
- [x] [Whisper](https://cdn.openai.com/papers/whisper.pdf)
- [ ] [Qwen2 Audio](https://arxiv.org/abs/2407.10759)
- [ ] [Stepfun Audio](https://arxiv.org/abs/2506.08967)

#### Tuning Techniques
- [x] [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [ ] [Mixed Preference Optimization](https://huggingface.co/papers/2411.10442)
- [ ] [Grouped Reward Policy Optimization](https://arxiv.org/abs/2402.03300)

#### Dataset Preparation
- [ ] Pairwise Preference Dataset
- [ ] Reward Shaping Dataset


#### Training
- [x] Distributed Data Parallel
- [ ] Fully Sharded Data Parallel
- 