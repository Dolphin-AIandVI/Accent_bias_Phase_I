# AI Reflection and Comparative Analysis: Phase I (Midpoint)

**AI-aided pair:** [Sheikh Farzah, K12557094] · [Justice Ezirim Ndubuisi, K12341139]
**Human-only pair:** [Ebba Wakgari Tilahun, K52207605] · [Maria Roychan K12440895]

**Project:** Accent bias in wav2vec 2.0 and Whisper on the EdAcc dataset

This reflection compares the Phase I work of our two pairs. Both pairs studied English accents in wav2vec 2.0 and Whisper on EdAcc; the AI-aided pair used Generative AI actively and the human-only pair did not. The comparison is based on our notebooks (code, outputs and written text) and on our experience of the work.

---

## 1. What each pair did

|                       | Human-only pair                                                           | AI-aided pair                                              |
|-----------------------|---------------------------------------------------------------------------|------------------------------------------------------------|
| Models                | wav2vec2-base-960h, Whisper-tiny                                          | wav2vec2-base-960h, Whisper-small |
| Data                  | first 500 rows of the validation split (streamed)                         | test split, random sample of up to 200 clips per accent                                                                                                           (1,422 clips after filters) |
| Accent groups         | 4 (Indian 65 clips, Southern British 86, European 130, Mainstream US 219) | 8 (Nigerian, Kenyan, Ghanain, Jamaican, Indian, Mainstream                                                                                                         US, Southern British, Irish) |
| Speaker information   | not stored                                                                | stored and used in the analysis|
| Text normalization    | lower-casing                                                              | lower-casing and punctuation removal |
| Error rate            | mean of per-clip WER, median alongside                                    | corpus WER (errors divided by reference words) |
| Confidence            | wav2vec confidence stored; Whisper confidence a fixed placeholder (0.85)  | mean token log-probability for both models |
| Visualizations        | grouped bar chart of mean WER; two stacked bar charts of mean error counts| seven static views plus interactive dropdown plots and                                                                                                            PCA/UMAP projections |
| Written conclusions   | two markdown cells                                                        | none in the notebook |
| Compute               | about 29 minutes for 500 clips on CPU                                     | runtime for 1,422 clips |

## 2. Record of AI use (AI-aided pair)

The AI-aided pair used GenAI as a collaborator for pipeline code, analysis ideas and visualization code. Our prompt record for Phase I is incomplete: the prompts were not saved as we worked, so `ai-interactions/` does not cover every interaction that shaped the analysis.

- Tool and model used: Claude Sonnet 5.5

## 3. Where the results agree and differ

**Agreement.** For wav2vec 2.0, substitutions are the largest error type in both pairs. In both pair Mainstream US English is not the lowest-error group for wav2vec 2.0.

**Differences.**

| Question                           | Human-only                                                      | AI-aided                                                |
|------------------------------------|-----------------------------------------------------------------|---------------------------------------------------------|
| Indian English                     | highest error of 4 groups (mean WER 3.47 wav2vec, 3.00 Whisper) | among the lowest: 45.1% wav2vec (2nd lowest of 8), 21.1%                                                                                                          Whisper (2nd lowest of 8) |
| Mainstream US English with Whisper | lowest of the 4 groups                                          | middle of the 8 groups (24.8%) |
| Dominant Whisper error type        | substitutions for all 4 accents                                 | deletions exceed substitutions for 7 of 8 accents                                                                                                                 (Jamaican is the exception) |

Both pairs differ in several design choices that can each affect these numbers: the Whisper model size (tiny vs small), the normalization (without punctuation removal, a correct word with a comma or capital letter counts as a substitution), the sample (500 consecutive rows vs a random sample per accent), and the aggregation (a mean of per-clip WER is strongly affected by very short clips with long outputs, giving means above 1).

## 4. Insights found by one pair and not the other

**Found by the AI-aided pair**

- **Speaker variation.** The speaker-level plot shows large spread within accents (for example, one Indian English speaker near 0.67 Whisper WER and the other two near 0.2). Some groups consist of one or two speakers: Ghanain English 1, Kenyan 2, Southern British 2.
- **Additional accents.** Nigerian, Kenyan, Ghanain and Jamaican English appear only in the AI-aided sample. Jamaican English has the highest WER for both models (37.5% Whisper, 67.4% wav2vec) and Nigerian English the second highest for wav2vec (66.5%).
- **Whisper insertion loops.** Whisper-small inserted words at about 7.8% of reference words for Jamaican English, versus roughly 0.3 to 3.3% for the other accents. The flagged examples show repetition loops ("no, no, no, ..." and "I, I, I, ...").
- **Weak agreement between models.** The correlation of clip-level WER between the two models is 0.28, so they often fail on different clips.

**Found by the human-only pair**

- **A written interpretation.** The human-only pair stated findings in words and proposed an explanation (a training-data hypothesis).
- **Median alongside mean.** Printing medians showed how far the means were pulled by outlier clips.

**Not covered by either pipeline**

- Annotation tags (`<LAUGH>`, `<DTMF>`) and fillers (um, uh) remain in the reference text after normalization, which can raise deletion counts and affects absolute WER values.
- Segments marked to be ignored in scoring were not excluded.
- Neither AI_aided nor Human_only reported confidence intervals for accent-level error rates.

## 5. Visualization choices

**Human-only pair.** Three static plots: grouped bars of mean WER and two stacked bars of mean error counts per clip. They are simple to read and give a quick overview of the four accents. They show means (some above 1) without spread, error counts are not divided by reference length, and the two stacked panels use different colormaps.

**AI-aided pair.** An accent-by-model point plot, a paired clip-level WER difference plot, a speaker-level strip plot, normalized error composition, speech rate versus WER, hallucination-candidate bars, confidence-versus-error lines, and UMAP and PCA projections with plotly dropdown filters. The paired and speaker-level views show things the bar charts cannot, and code comments label the hallucination view as "a flag for review, not proof". The interactivity is dropdown filtering rather than linked brushing, the projections are colored by accent without a follow-up on what distinguishes each cluster, and accent-level points have no error bars.

## 6. Breadth and depth of investigation

AI assistance increased **breadth**: 8 accents instead of 4, and about ten analyses instead of three. It increased **depth of evidence** (speaker level, paired comparison, hallucination candidates). It did not by itself increase **depth of interpretation**: many views were produced but not examined in writing. The human-only pair's narrower analysis reached an interpretation sooner. The depth of reasoning was therefore higher in the human-only pair, and the depth of evidence higher in the AI-aided one.

## 7. Confidence and the evidence behind it

- **Human-only pair: low confidence.** The pair felt that they worked with less data and that their analysis was limited to calculating error rates, so they are not confident in their work. This matches the evidence: 500 consecutive clips, four accent groups, error rates and error-type counts as the only measures, and no speaker information to check the group results against.
- **AI-aided pair: high confidence.** The pair felt very confident. they believed their visualizations showed the data more deeply and explained more patterns. This rests on a larger sample (8 accents, 1,422 clips) and on views beyond error rates: speaker-level WER, paired clip-level comparison, error composition, hallucination candidates, confidence against error, and acoustic-feature projections.
- **What the evidence supports.** The AI-aided pair's confidence is better supported by breadth and variety of evidence. their accent-level conclusions are still limited by groups of 1 to 3 speakers and by the absence of confidence intervals, so descriptive statements about our samples are justified more than general claims about accents.

## 8. Did AI assistance save time or add verification work?

- **Time saved.** The AI-aided notebook is larger and includes resumable checkpointing, a one-clip smoke test before the full run, a check that stops the run if Whisper scores are missing, speaker-aware sampling, and a wide set of analyses. The human-only run of 500 clips took about 29 minutes on CPU.
- 
- **Verification work.** The AI-aided notebook shows warnings from the transformers library about generation arguments, a Colab widget-manager step needed for the interactive plots (re-run out of order), and Colab-specific cells.
- 
- **Net effect.** AI saved time on code and widened the investigation, but moved effort from writing to checking. The checks that mattered most for the results (normalization rules, sample design, speaker variation, written interpretation) had to come from us.

## 9. Failure patterns observed in AI-assisted work

1. **Plausible pipeline with unexamined analytic choices.** The code runs and looks complete, but the normalization keeps annotation tags and fillers.
2. **Many views, little interpretation.** Producing plots is cheap, so views accumulated without a question for each and without written conclusions.
3. **Environment assumptions.** Colab-only cells make the notebook less portable.

## 10. Emerging best practices for using AI in this work

1. Treat each AI-suggested analytic choice (normalization, sampling, metrics) as a hypothesis, and check it on a few real examples or a toy case.
2. Keep speaker identity visible from the start, because it changes how group-level plots should be read.
3. Ask for several alternative visualizations with trade-offs, then select one per question.
4. Write each finding in one sentence next to its plot, with the evidence it rests on.
5. Save the prompt log from the first prompt.

## 11. Summary

1. The two pairs did not run identical experiments: both used wav2vec2-base-960h, but the human-only pair used Whisper-tiny and the AI-aided pair Whisper-small, and they differed in sampling, normalization and aggregation.
2. The pairs reached opposite results for Indian English, which shows how sensitive group-level conclusions are when groups contain only a few speakers.
3. The AI-aided pair covered more accents and more kinds of analysis (speaker level, hallucination candidates, confidence, acoustic features, interactive views) but did not write interpretations.
4. The human-only pair covered fewer accents and wrote explicit interpretations, including a training-data hypothesis that Phase I evidence does not test.
5. AI assistance saved time on code and increased breadth; verification and interpretation remained human work.
