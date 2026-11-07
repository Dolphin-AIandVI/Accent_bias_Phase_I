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
Wav2vec2 and Whisper are widely used, they run in Colab with pinned checkpoints for reproducibility, and the small variants are feasible on limited compute (about 4 hours on CPU). Accent disparities in speech recognition are also a well-known topic.1. Understanding requirements

Prompt: “Here is our assignment brief [paste]. Phase I is black-box only. Our model task is accent bias in wav2vec 2.0 and Whisper on EdAcc. Propose 3 concrete research questions and what data each needs.”

Gist of reply: Compare WER and confidence by accent against a baseline, test whether errors concentrate in certain speakers or words, and check whether confidence tracks correctness per accent.

Verify: Does each question stay black-box? Is the sample large enough to call a pattern systematic? Drop or reshape anything that needs model internals, which belongs in Phase II.

2. Concepts: getting a confidence score

Prompt: “How can I get a confidence score from Whisper and from wav2vec2-CTC for one transcript? Are they comparable?”

Gist of reply:

Whisper generates tokens, so you can average the log-probabilities of the generated tokens (excluding special tokens).
wav2vec2 outputs per-frame logits, so you take the greedy path, collapse repeats, drop blanks and average the chosen tokens’ log-probabilities.
The two scores are not directly comparable, because they are computed over different units.

Verify: Read the Hugging Face docs for the generation scoring function, and hand-check the scores on 2–3 clips. Check whether confidence actually drops when the transcript is wrong.

3. Code generation

Prompt: “Write code to load edinburghcstr/edacc (test split), keep audio undecoded, show clip counts per accent and speakers per accent, then sample up to N clips per accent with a fixed seed and a duration cap.”

Gist of reply: Load with Audio(decode=False), convert metadata to pandas, group and sample, and read duration from audio metadata without decoding.

Prompt (scoring loop): “Run both models over the sample and save results to CSV every 25 clips so a crashed session can resume.”

Verify: Check the clip counts against the dataset page, and kill the run midway to confirm it resumes without duplicating rows. Also ask whether the sampling balances clips or speakers, because the first version often balances only clips.

4. Debugging

Prompt: “This warning appears for every clip: [paste]. Is it harmful, and how do I fix it?”

Gist of reply: The warning is harmless, but it means conflicting generation settings, so set only max_new_tokens or clean up the generation config, and lower the logging verbosity.

Prompt (runtime): “Scoring 1,400 clips takes hours on CPU. Options?”

Gist of reply: Switch the Colab runtime to GPU, batch the audio, or reduce the sample.

Verify: Re-run on a few clips to confirm the output is unchanged after the fix.

5. Interpreting results

Prompt: “Reference: [text]. Whisper output: [text], confidence X. wav2vec2 output: [text], confidence Y. What does this suggest, and what should we test next?”

Gist of reply: A good answer notes the differing error types (deletion versus garbled characters), points out that one clip is not evidence, and proposes a systematic test such as confidence against WER for each model.

Verify: Don’t treat the interpretation as a finding. Turn it into a hypothesis and test it on all clips, and tell the AI what the plot actually shows rather than what you expect.
