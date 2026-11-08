# Accent Bias in Speech Recognition: wav2vec 2.0 vs. Whisper

> **Status:** Phase I, data collection done; analysis in progress.

## 1. Task

We investigate whether automatic speech recognition (ASR) models behave differently
on different **English accents**, and, if so, how and why.

The assignment has two phases:

| Phase       | Question                               | Constraint |

| **I**       | How does the model behave?             | **Black-box only.** Use inputs and outputs, no model internals. |
| **II**      | What did the model learn?              | Open the box (representations, attention, gradients). Use at                                                               least two explanation techniques and compare them. |

**AI4Vis track.** The group is split into two pairs working on the same model and task:
an *AI-aided* pair and a *human-only* pair. The pairs switch roles at the end of Phase I,
and the AI-aided pair keeps a record of its AI use (prompts, accepted/rejected suggestions,
failure patterns, best practices).

### Phase I goal

Go beyond a table of benchmark scores. Show behaviors that aggregate performance hides
and explain them in depth, for example:

- Do errors concentrate in particular accents, speakers, words or clip types?
- Do the two models fail in different ways (substitutions vs. deletions vs. insertions)?
- Does model confidence track correctness equally well for every accent?

### Research questions (draft, to be agreed by the group)

1. [Q1: e.g. WER by accent, controlling for speaker]
2. [Q2: e.g. where do the two models disagree?]
3. [Q3: e.g. is confidence calibrated across accents?]

---

## 2. Models

Both models are open-weight and loaded from Hugging Face, with pinned checkpoints for reproducibility.

|                   | wav2vec 2.0                       | Whisper |

| **Checkpoint**    | `facebook/wav2vec2-base-960h`     | `openai/whisper-small` |
| **Developer**     | Meta                              | OpenAI |
| **Architecture**  | Encoder with CTC head (frame-level predictions, no decoder) | Encoder–decoder transformer (autoregressive token generation) |
| **Training data** | Self-supervised pretraining, fine-tuned on LibriSpeech 960 h (read audiobooks) | Large, diverse, weakly supervised web audio |
| **Class in code** | `AutoModelForCTC` + `AutoProcessor` | `WhisperForConditionalGeneration` + `WhisperProcessor` |

**Why this pair:** same task, very different training data and architecture.
That makes it possible to ask whether narrow training data shows up as accent bias, and
whether the two designs fail differently. Both are open, which keeps Phase II (embeddings,
attention) possible.

**Caveats to state in the report:**
- Models are not size-matched (base vs. small).
- wav2vec2 was fine-tuned on read speech; EdAcc is conversational.
- Whisper's training data is not public.

### Inference settings

- Audio converted to mono, resampled to 16 kHz.
- **Whisper:** English forced (`language="en"`, `task="transcribe"`), greedy decoding
  (`do_sample=False`), `max_new_tokens=200`.
- **wav2vec2:** greedy CTC decoding (argmax per frame).

### Confidence scores (output-level only)

| Model                   | Definition |

| Whisper                 | Mean log-probability of generated tokens (special tokens excluded). |
| wav2vec2                 | Approximate: mean log-probability of tokens on the greedy CTC path (repeats collapsed, blanks                               dropped). |

The two scores are computed over different units and are **not directly comparable
across models**. Compare within a model across accents.

---

## 3. Dataset

**EdAcc**, the Edinburgh International Accents of English Corpus
(`edinburghcstr/edacc` on Hugging Face), `test` split. Conversational English from
speakers with many different accents, with speaker, L1 and gender metadata.


### Accents included (8)

| Role | Accents |
|---|---|
| **Focus** | Nigerian English |
| **Baseline** | Mainstream US English |
| Others | Kenyan, Ghanaian, Jamaican, Indian, Southern British, Irish |

Other accents in EdAcc (e.g. Italian, Romanian, Indonesian, Catalan, French) were left
out so that the project stays on English varieties. [Confirm this was intentional.]

### Sampling

| Parameter | Value |
|---|---|
| Clips per accent | up to 200 (`N_PER_ACCENT`) |
| Minimum reference length | 3 words (`MIN_REF_WORDS`) |
| Maximum duration | 30 s (`MAX_DURATION_SEC`) |
| Random seed | 42 |

Clips are sampled **per accent**. This balances clip counts, **not speakers**.

### Resulting sample (1,422 clips)

| Accent | Clips | Speakers | Median clips / speaker |
|---|---:|---:|---:|
| Nigerian English | 190 | 5 | 20.0 |
| Kenyan English | 196 | 2 | 98.0 |
| Ghanaian English | 131 | 1 | 131.0 |
| Jamaican English | 194 | 4 | 47.0 |
| Indian English | 198 | 3 | 86.0 |
| Mainstream US English | 179 | 9 | 24.0 |
| Southern British English | 153 | 2 | 76.5 |
| Irish English | 181 | 3 | 67.0 |

> **Known confound:** speakers per accent range from 1 to 9. An "accent effect"
> may really be a speaker effect, so report speaker-level results and uncertainty.

---

## 4. Pipeline

1. Load EdAcc `test` with `Audio(decode=False)`. Inspect metadata without decoding audio.
2. Filter (accent list, ≥3 reference words), sample per accent, apply the duration cap.
3. Load both models and processors.
4. Score every clip with both models. The run saves every 25 clips and can resume.
5. Save results to `edacc_two_model_predictions.csv`.
6. [Analysis: see section 6.]

Runtime: about 4 hours on CPU for 1,422 clips (a GPU runtime is recommended).

### Output columns

`row`, `accent`, `speaker`, `l1`, `gender`, `duration_s`, `reference`,
`whisper_text`, `whisper_mean_logprob`, `wav2vec_text`, `wav2vec_mean_logprob`

> Texts are saved **raw** (no normalization). Normalize case, punctuation and
> disfluency handling consistently on both reference and hypothesis **before** computing WER.

---

## 5. Repository layout (suggested)

```
.
├── README.md
├── ai_interaction.md/
│   └── AI_aided_accent_bias.ipynb       
│   └── Human_only_accent_bias.ipynb
├── data/
│   └── edacc_two_model_predictions.csv
├── analysis/                       
└── ai_use_log.md                   # [AI-aided pair: prompts, decisions, failures]
```

## 6. Phase I analysis plan (black-box only)

- [ ] Normalize texts, compute WER by accent **and by speaker**, with uncertainty
- [ ] Error-type breakdown (substitution / deletion / insertion), disfluency handling
- [ ] Per-clip disagreement between the two models
- [ ] Whisper insertions / hallucinations
- [ ] Confidence vs. correctness, per accent (output-level scores only)
- [ ] Input-side properties: duration, speech rate, audio quality, gender, L1, word rarity
- [ ] Explain *why* error-heavy groups differ, e.g. sub-groups and what distinguishes them

Phase II (not started): representation analysis and at least two explanation techniques.

## 7. Setup

```bash
pip install -q "transformers>=4.44" datasets jiwer wordfreq librosa soundfile accelerate
```

Key libraries: `transformers`, `datasets`, `torch`, `librosa`, `soundfile`, `jiwer`,
`wordfreq`, `pandas`, `seaborn`, `matplotlib`.

## 8. Team

- Pair A: [Sheikh Farzah, Justice Ezirim Ndubuisi], role in Phase I: [AI-aided]
- Pair B: [Ebba Wakgari Tilahun, Maria Roychan], role in Phase I: [human-only]
