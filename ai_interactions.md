# AI Interactions Record: AI-aided pair, Phase I

**AI-aided pair:** [First Last, Student ID] · [First Last, Student ID]
**Project:** Accent bias in wav2vec 2.0 and Whisper on the EdAcc dataset

This file records the GenAI interactions that influenced the AI-aided pair's Phase I analysis, and what they show about the strengths and limits of GenAI assistance. It is based on a saved ChatGPT conversation from a member of the pair, on the AI-aided notebook, and on the pair's Phase I visual report. The last section lists what is not recorded.

---

## 1. Research planning with ChatGPT

- **Tool and model:** ChatGPT [FILL IN: model]
- **Used by:** [FILL IN: member name]
- **Date:** [FILL IN]
- **Purpose:** turn the project idea (compare speech recognition models across English accents) into a research plan.

### Prompts

The saved conversation does not include the opening prompt. ChatGPT's first reply restates the idea as investigating whether speech recognition systems understand people equally well across regional English accents, and which model makes fewer mistakes.

ChatGPT then proposed a research direction and asked two design questions: the primary objective (comparative evaluation, fairness improvement, or deeper investigation of internals) and the number of models (two, three or four). The conversation does not show which options were selected. The plan that followed assumes the fairness-improvement direction and three models.

The member then asked for a full plan:

> develop a complete research plan. Include: Project title. Abstract. Research objectives. Research questions. Hypotheses, where appropriate. Dataset description. Model selection and justification. Experimental methodology. Evaluation metrics. Visualization plan. Limitations. Expected contributions. The research must be measurable and scientifically defensible.

### What ChatGPT produced

A Master's-level plan titled *AccentFair*, containing:
- three models (Whisper, wav2vec 2.0, a NVIDIA Conformer) and two datasets (EdAcc for evaluation, Common Voice for fine-tuning)
- metrics: WER, CER, macro-average WER, worst-group WER, group gap, semantic similarity
- speaker-level bootstrap confidence intervals and mixed-effects statistics
- eight visualizations, including a model-by-accent heatmap, speaker-level WER distributions, an error-type view and an interactive dashboard
- a six-phase method including accent-balanced fine-tuning, and a 12-week schedule
- a warning that accent disparities in speech recognition are already well studied, with a list of related papers

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

### Strengths observed

- Fast, structured plan that turned a loose idea into concrete models, datasets and metrics.
- Practical checkpoint names that could be used directly.
- Relevant methodological cautions: a performance gap is not by itself evidence of unfairness; accent labels should come from documented metadata; speaker-level analysis is needed; pretrained models may already have seen the test data.
- An explicit warning about novelty, which kept the project from presenting accent disparity as a new discovery.

### Limitations observed

- **Scope mismatch.** The plan was a 12-week thesis design with fine-tuning and three models. Phase I asks for black-box analysis of behaviour, so most of the plan was outside the task.
- **Direction set by the AI.** The plan followed the AI's stated preference for a fairness-improvement direction, which is not part of the course task.
- **No code or data.** The visualization suggestions were descriptions accompanied by decorative images from other websites. The notebook code had to come from other interactions (Section 2).
- **Unverified literature claims.** The plan summarizes the EdAcc paper as reporting lower performance for Indian, Jamaican and Nigerian English. In our sample, Jamaican and Nigerian English had the highest error rates, but Indian English was second-lowest of 8 for both models. The record does not show that the AI's literature claims were checked against the sources, so we treat them as unverified. Differences in sample and number of speakers could also explain the mismatch.
- **Thresholds not tested.** The suggested minimum group sizes were labeled provisional, and our own sample does not meet them.

## 2. Notebook and visual report development

The saved ChatGPT conversation does not include the prompts used to write the notebook or the visual report. What the notebook shows:

- a resumable inference loop, a one-clip test before the full run, and a check that stops the run if Whisper scores are missing
- speaker-aware sampling and a wide set of views: paired clip-level differences, speaker-level WER, error composition, speech rate against WER, long-output candidates, confidence against error, and UMAP and PCA projections
- a comment in the code ("uses your earlier load_audio() helper") that suggests the code was written over several turns of a conversation

Prompts and decisions: [FILL IN: key prompts reconstructed from the chat history, and which suggestions were accepted, modified or rejected, and why]

Visual report (`Vis4AI_Phase_I_Visual_Report.html`): [FILL IN: how GenAI was used]

## 3. Other GenAI use in the project (Claude)

Team-level use of Claude during planning and writing. [FILL IN: which members used it, and which of these influenced the AI-aided notebook.]

| Request | What Claude provided |
|---|---|
| Assignment text, the course example and the accent-bias idea; how to do the project | Explanation of the Phase I and Phase II tasks, the midpoint deliverables and a timeline |
| Discussion material for the group | Decision points and three possible routes |
| The EdAcc dataset link | A summary of the dataset page (splits, columns) and points to check: conversational speech, few speakers, annotation tags, clip lengths |
| Confusion about the two pairs | A clarification that "AI-aided" means using GenAI, not the speech models |
| A sample plan for both pairs, then a working solution | A sample plan, a script-based pipeline tested on synthetic data, and a single-notebook version tested only on synthetic data (not handed over) |
| Both notebooks and the visual report | Drafts of the reflection and README, revised at the team's direction |

## 4. What is not recorded

- The opening prompt of the ChatGPT planning conversation.
- The prompts used to write the notebook code and the visual report.
- The team's reasons for accepting, modifying or rejecting individual suggestions.
