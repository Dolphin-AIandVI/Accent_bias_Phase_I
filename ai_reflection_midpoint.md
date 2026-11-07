Introduction

Our project investigates whether wav2vec 2.0 (Meta) and Whisper (OpenAI) behave differently across English accents, using the EdAcc dataset. In Phase I we treated both models as black boxes. We wanted to build a reproducible pipeline that scores both models on the same clips, then find systematic differences in errors and confidence across accents, beyond a single WER number. As the AI-aided pair, we used Claude AI to speed up pipeline design and code, and to broaden the analysis.”

Why these two models, and why these accents
1. Same task, very different training data

wav2vec2-base-960h is pretrained on unlabeled audio and fine-tuned on LibriSpeech (960 hours of read audiobooks). That data is fairly narrow, mostly North American, and has little accent variety.
Whisper was trained on a very large, diverse set of web audio (roughly 680k hours, weakly supervised). This is the opposite of the narrow-data case.
That gives you a testable hypothesis: does narrow training data show up as accent bias, and does diverse data reduce it?. LibriSpeech is open, so you can inspect it. Whisper’s data is not public.

2. Different architectures, different kinds of confidence

wav2vec2 is a CTC model that emits a label per audio frame, with no decoder.
Whisper is an encoder–decoder that generates tokens autoregressively, so it can also drop or invent words.
This is why we used8 two different log-probability definitions. It also means the two models should fail in different ways, such as garbled characters from wav2vec2 versus deletions or hallucinations from Whisper.

3. Phase II would need open weights
Both are open and available on Hugging Face. we intend to extract encoder embeddings for UMAP or t-SNE (Lecture 3) and, for Whisper, look at attention maps. SHAP or LIME on derived features (Lecture 4) works for either. A closed API model would not allow that.

4. Practical reasons
Wav2vec2 and Whisper are widely used, they run in Colab with pinned checkpoints for reproducibility, and the small variants are feasible on limited compute (about 4 hours on CPU). Accent disparities in speech recognition are also a well-known topic.



