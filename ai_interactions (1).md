# AI Interactions Record: AI-aided pair, Phase I

**AI-aided pair:** [First Last, Student ID] · [First Last, Student ID]
**Project:** Accent bias in wav2vec 2.0 and Whisper on the EdAcc dataset

This file records the GenAI interactions that influenced the AI-aided pair's Phase I analysis, and what they show about the strengths and limits of GenAI assistance. It combines the records of both pair members: one member's saved ChatGPT conversation (research planning) and the other member's prompts, reply summaries and verification notes for Claude (pipeline and analysis). It is checked against the AI-aided notebook and the Phase I visual report.

---

## 1. Context

In Phase I we treated wav2vec 2.0 and Whisper as black boxes. We wanted a reproducible pipeline that scores both models on the same clips, and then to find systematic differences in errors and confidence across accents, beyond a single WER number. As the AI-aided pair we used GenAI to speed up pipeline design and code and to broaden the analysis.

**Why these two models.**
- *Different training data.* `wav2vec2-base-960h` is pretrained on unlabeled audio and fine-tuned on LibriSpeech (960 hours of read audiobooks), which is fairly narrow and mostly North American. Whisper was trained on a very large, diverse set of web audio (roughly 680k hours, weakly supervised). This gives a testable question: does narrow training data show up as accent bias, and does diverse data reduce it? LibriSpeech is open; Whisper's data is not public.
- *Different architectures.* wav2vec 2.0 is a CTC model that emits a label per audio frame, with no decoder. Whisper is an encoder-decoder that generates tokens one at a time, so it can also drop or invent words. We therefore expect different failure modes (garbled characters versus deletions or hallucinations) and used two different log-probability definitions for confidence.
- *Phase II.* Both are open-weight models on Hugging Face, so encoder embeddings (UMAP or t-SNE), attention maps for Whisper, and SHAP or LIME on derived features remain possible.
- *Practical.* Both are widely used, run in Colab with pinned checkpoints, and the small variants are feasible on limited compute (about 4 hours on CPU).

## 2. Research planning with ChatGPT

- **Tool and model:** ChatGPT [FILL IN: model]
- **Used by:** [FILL IN: member name]
- **Date:** [FILL IN]

### Prompts

The saved conversation does not include the opening prompt. ChatGPT's first reply restates the idea as investigating whether speech recognition systems understand people equally well across regional English accents, and which model makes fewer mistakes.

ChatGPT then proposed a direction and asked two design questions: the primary objective (comparative evaluation, fairness improvement, or deeper investigation of internals) and the number of models (two, three or four). The conversation does not show which options were selected. The plan that followed assumes the fairness-improvement direction and three models.

The member then asked for a full plan:

> develop a complete research plan. Include: Project title. Abstract. Research objectives. Research questions. Hypotheses, where appropriate. Dataset description. Model selection and justification. Experimental methodology. Evaluation metrics. Visualization plan. Limitations. Expected contributions. The research must be measurable and scientifically defensible.

### What ChatGPT produced

A Master's-level plan titled *AccentFair*, with three models (Whisper, wav2vec 2.0, a NVIDIA Conformer), two datasets (EdAcc for evaluation, Common Voice for fine-tuning), metrics (WER, CER, macro-average WER, worst-group WER, group gap, semantic similarity), speaker-level bootstrap intervals and mixed-effects statistics, eight visualizations, a six-phase method including accent-balanced fine-tuning, a 12-week schedule, and a warning that accent disparities in speech recognition are already well studied, with a list of related papers.

### How the plan influenced the analysis

| Suggestion in the plan | Outcome in the AI-aided notebook |
|---|---|
| EdAcc as the main dataset | Adopted (test split) |
| `facebook/wav2vec2-base-960h` | Adopted unchanged |
| `openai/whisper-small.en` | Modified: `openai/whisper-small` (multilingual, English forced) |
| Report corpus-level WER, not an average of per-clip WER | Adopted (corpus WER) |
| Substitution, deletion and insertion analysis | Adopted |
| Speaker-level WER distribution | Adopted (speaker-level strip plot) |
| Speaker-level confidence intervals, statistical tests | Not used |
| Model-by-accent heatmap | Not made; point and dumbbell views used instead |
| Interactive dashboard | Partly adopted (plotly dropdown filters) |
| Conformer model, CER, semantic similarity | Not used in Phase I |
| Minimum of 15 speakers and 100 utterances per accent group | Not applied; groups have 1 to 9 speakers |
| Common Voice and accent-balanced fine-tuning | Not part of Phase I, which analyses model behaviour as a black box |

[FILL IN: the team's own reasons for each choice, where they differ from the scope note in the last row.]

## 3. Pipeline and analysis with Claude

- **Tool and model:** Claude [FILL IN: model]
- **Used by:** [FILL IN: member name]

The prompts below are reproduced as recorded by the member, with pasted material (assignment text, warnings, outputs) left out. The "reply" lines are the member's summaries, not the full replies.

### 3.1 Understanding requirements

**Prompt:** "Here is our assignment brief [paste]. Phase I is black-box only. Our model task is accent bias in wav2vec 2.0 and Whisper on EdAcc. Propose 3 concrete research questions and what data each needs."

**Reply (summary):** Compare WER and confidence by accent against a baseline; test whether errors concentrate in certain speakers or words; check whether confidence tracks correctness per accent.

**Verification:** Does each question stay black-box? Is the sample large enough to call a pattern systematic? Anything that needs model internals was dropped or reshaped, because it belongs in Phase II.

**Effect on the analysis:** the notebook contains a speaker-level WER view, a confidence-versus-error view, and a review of unusually long outputs, matching these questions.

### 3.2 Concepts: getting a confidence score

**Prompt:** "How can I get a confidence score from Whisper and from wav2vec2-CTC for one transcript? Are they comparable?"

**Reply (summary):** For Whisper, average the log-probabilities of the generated tokens (excluding special tokens). For wav2vec 2.0, take the greedy path, collapse repeats, drop blanks, and average the chosen tokens' log-probabilities. The two scores are not directly comparable, because they are computed over different units.

**Verification:** Read the Hugging Face documentation for the generation scoring function, hand-check the scores on 2 to 3 clips, and check whether confidence actually drops when the transcript is wrong.

**Effect on the analysis:** the notebook uses a mean token log-probability for each model, and the visual report states that the two confidence scales differ, so only relative comparison within a model is appropriate.

### 3.3 Code generation

**Prompt:** "Write code to load edinburghcstr/edacc (test split), keep audio undecoded, show clip counts per accent and speakers per accent, then sample up to N clips per accent with a fixed seed and a duration cap."

**Reply (summary):** Load with `Audio(decode=False)`, convert the metadata to pandas, group and sample, and read durations from audio metadata without decoding.

**Prompt (scoring loop):** "Run both models over the sample and save results to CSV every 25 clips so a crashed session can resume."

**Verification:** Check the clip counts against the dataset page, and stop the run midway to confirm it resumes without duplicating rows. Ask whether the sampling balances clips or speakers, because the first version often balances only clips.

**Effect on the analysis:** the notebook samples up to 200 clips per accent with a fixed seed and a duration cap, and resumes from a saved predictions file. A code comment states that the sampling balances clip counts, not speakers, and the speaker-level plot was added to show the resulting imbalance (1 speaker for Ghanain English, 9 for Mainstream US English).

### 3.4 Debugging

**Prompt:** "This warning appears for every clip: [paste]. Is it harmful, and how do I fix it?"

**Reply (summary):** The warning is harmless but means conflicting generation settings; set only `max_new_tokens` or clean up the generation config, and lower the logging verbosity.

**Prompt (runtime):** "Scoring 1,400 clips takes hours on CPU. Options?"

**Reply (summary):** Switch the Colab runtime to GPU, batch the audio, or reduce the sample.

**Verification:** Re-run on a few clips to confirm the output is unchanged after the fix.

**Effect on the analysis:** [FILL IN: which option was chosen. The saved notebook output shows a CPU run.]

### 3.5 Interpreting results

**Prompt:** "Reference: [text]. Whisper output: [text], confidence X. wav2vec2 output: [text], confidence Y. What does this suggest, and what should we test next?"

**Reply (summary):** The answer noted the differing error types (deletion versus garbled characters), pointed out that one clip is not evidence, and proposed a systematic test such as confidence against WER for each model.

**Verification:** Do not treat the interpretation as a finding. Turn it into a hypothesis and test it on all clips, and tell the AI what the plot actually shows, not what you expect.

**Effect on the analysis:** the notebook labels unusually long outputs as hallucination candidates "for review, not proof", and the visual report says the audio must be checked before using that label.

## 4. Other GenAI use in the project (Claude, separate conversation)

Team-level use of Claude for planning and writing. [FILL IN: which members used it, and which of these influenced the AI-aided notebook.]

| Request | What Claude provided |
|---|---|
| Assignment text, the course example and the accent-bias idea; how to do the project | Explanation of the Phase I and Phase II tasks, the midpoint deliverables and a timeline |
| Discussion material for the group | Decision points and three possible routes |
| The EdAcc dataset link | A summary of the dataset page (splits, columns) and points to check: conversational speech, few speakers, annotation tags, clip lengths |
| Confusion about the two pairs | A clarification that "AI-aided" means using GenAI, not the speech models |
| A sample plan for both pairs, then a working solution | A sample plan, a script-based pipeline tested on synthetic data, and a single-notebook version tested only on synthetic data (not handed over) |
| Both notebooks and the visual report | Drafts of the reflection and README, revised at the team's direction |

The Phase I visual report (`Vis4AI_Phase_I_Visual_Report.html`): [FILL IN: how GenAI was used].

## 5. Strengths and limitations observed

**Strengths**
- Fast, structured starting points: research questions from the assignment brief, a full research plan, and working loading, sampling and scoring code.
- Clear explanations of concepts we needed, such as why the two models' confidence scores are not comparable.
- Useful methodological cautions: a performance gap is not by itself evidence of unfairness; accent labels should come from documented metadata; speaker-level analysis is needed; pretrained models may already have seen the test data.
- The AI itself warned that a single clip is not evidence and suggested testing hypotheses on all clips. It also warned that accent disparity in speech recognition is already well studied.

**Limitations**
- **Defaults that needed checking.** The first version of the sampling code often balanced only clips, not speakers. We caught this by asking, and flagged it, but the final sample still balances clips.
- **Scope mismatch.** The ChatGPT plan was a 12-week thesis design with fine-tuning and three models, whereas Phase I asks for black-box analysis. It also followed the AI's own preferred direction (fairness improvement), which is not part of the course task.
- **Unverified claims.** The plan summarizes the EdAcc paper as reporting lower performance for Indian, Jamaican and Nigerian English. In our sample, Jamaican and Nigerian English had the highest error rates, but Indian English was second-lowest of 8 for both models. We did not check the AI's literature claims against the sources, so they remain unverified. Differences in sample and speakers could also explain the mismatch.
- **Thresholds not met.** The plan's suggested minimum group sizes were labeled provisional, and our own sample does not meet them.
- **No code or data in the planning conversation.** Its visualization suggestions were descriptions, so the code had to come from the Claude interactions.
- **Verification effort.** Every suggestion needed checking (clip counts, resume behaviour, hand-checked scores, unchanged output after a fix), and interpretations had to be turned into hypotheses and tested.

## 6. What is not recorded

- The opening prompt of the ChatGPT planning conversation.
- Full replies from Claude (only summaries by the member are kept).
- The prompts used to write the visual report and the visualization code, and the team's reasons for individual accept, modify and reject decisions.
- Dates and exact model versions.
