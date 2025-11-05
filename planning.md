## Research Question
Do large language models distinguish between epistemic beliefs that aim at truth-tracking and non-epistemic beliefs that serve symbolic or attitudinal functions when interpreting scenarios adapted from human belief attribution studies?

## Background and Motivation
- Vesga et al. (2025) show that human observers differentiate belief types when predicting agents’ actions; their stimuli (truth-dependent vs. symbolic actions across multiple propositions) provide grounded test cases.
- Computational work on epistemic stance (e.g., Gupta et al., 2023) classifies belief commitment in political text, suggesting feature cues (hedging, evidentials) that may transfer to our setup.
- No existing benchmark directly measures LLM sensitivity to epistemic vs. non-epistemic beliefs, limiting our ability to assess whether models mirror human theory-of-mind distinctions.

## Hypothesis Decomposition
1. **H1 (Categorical alignment):** Given paired agent descriptions, an LLM will more frequently identify the truth-dependent (epistemic) agent as holding an epistemic belief than the symbolic agent, exceeding a majority-class baseline.
2. **H2 (Probability alignment):** LLM confidence scores (converted to probabilities) for epistemic vs. non-epistemic judgments will correlate with human endorsement rates from Vesga et al. Study 3.
3. **H3 (Generalization across signatures):** LLM performance will remain consistent across the belief-signature manipulations in the stimuli (believe vs. think; binary vs. probabilistic; directional vs. nondirectional).

Independent variables: scenario signature (belief verb, certainty framing, directional cue), action type (truth-dependent vs. symbolic), prompt format.  
Dependent variables: model classification (epistemic vs. non-epistemic), confidence score, agreement with human majority, correlation metrics.

Success criteria: statistically significant improvement over baselines (H1) and positive correlation with human judgments (H2) at α = 0.05.

## Proposed Methodology

### Approach
Repurpose Vesga et al.’s Study 3 vignettes as evaluation prompts. For each scenario, present both agent/action descriptions to candidate models and request epistemic classification. Compare outputs with human majority choices and action-type ground truth. Complement with quantitative baselines (majority, lexical heuristic, logistic regression on bag-of-words features).

### Experimental Steps
1. **Data acquisition & parsing** – Download OSF Study 3 materials (CSV + docx). Extract vignette text, map agent actions to labels (truth-dependent vs. symbolic), compute human endorsement proportions.
2. **Dataset construction** – Build a tidy evaluation dataframe with columns: vignette_id, belief_signature metadata, agent texts, human epistemic endorsement rate, ground-truth action_type.
3. **Baseline implementation**  
   a. Majority-class baseline.  
   b. Keyword heuristic (e.g., epistemic verbs vs. social signaling verbs).  
   c. Logistic regression on TF-IDF features using scikit-learn (train via leave-one-vignette-out due to small sample).
4. **LLM evaluation** – Run zero-shot prompts with selected instruction-tuned models (e.g., `google/flan-t5-large`, optional second model if time). Capture categorical predictions and decoded confidence (via verbal probabilities or logit scores).
5. **Scoring & analysis** – Compute accuracy, balanced accuracy, and macro-F1 vs. baselines; correlate LLM confidence with human endorsement percentages; perform ANOVA or mixed-effects analysis over scenario factors.
6. **Error analysis** – Qualitative review of misclassified scenarios to detect systematic failures (e.g., directional cues, probabilistic language).
7. **Documentation** – Record methods, results, and limitations in REPORT.md; update README with reproduction steps.

### Baselines
- **Majority class:** Always predict the more frequent action type (truth-dependent) found in human data.
- **Keyword heuristic:** Label as epistemic if text contains knowledge-related verbs (“know,” “verify,” “evidence”) and as non-epistemic if it highlights signaling (“signal,” “impress,” “show support”), tuned on development subset.
- **TF-IDF Logistic regression:** Train classifier over agent texts using scikit-learn with stratified cross-validation to provide a competitive non-LLM baseline.

### Evaluation Metrics
- Accuracy and balanced accuracy for binary classification (addresses any class imbalance).
- Macro F1 to penalize asymmetric performance.
- Pearson/Spearman correlation between model confidence scores and human epistemic endorsement rates.
- Calibration plots/Brier score if confidence extraction is reliable.

### Statistical Analysis Plan
- Test H1 using binomial exact test comparing LLM accuracy against baseline accuracy; alternatively, McNemar’s test comparing paired predictions vs. majority heuristic.
- Test H2 using Pearson correlation (with Fisher z-transform CI) between model confidence and human proportions; confirm robustness with Spearman correlation.
- For H3, run repeated-measures ANOVA or mixed-effects logistic regression with scenario factors as fixed effects and model prediction as outcome; assess interaction terms.
- Control significance at α = 0.05; adjust for multiple comparisons (Holm-Bonferroni) where applicable.

## Expected Outcomes
- Support for hypothesis: LLM shows ≥10 percentage-point accuracy gain over baselines and meaningful positive correlation with human data.
- Null result: LLM fails to exceed baselines or shows weak correlation, suggesting limited differentiation of belief types.
- Partial support: Performance varies by belief signature, highlighting nuanced model strengths/weaknesses.

## Timeline and Milestones
- **0.5 h** – Finish data extraction and schema inspection.
- **0.5 h** – Construct clean evaluation dataset, compute human aggregates.
- **0.5 h** – Implement baselines and validation harness.
- **0.75 h** – Run LLM evaluations (two models if feasible), capture outputs.
- **0.5 h** – Analyze metrics, run statistical tests, perform error analysis.
- **0.75 h** – Prepare visualizations, REPORT.md, README updates.
- *Buffer (~30%)* reserved for debugging parsing issues or slow model inference.

## Potential Challenges
- **Data parsing complexity:** R scripts and docx materials may require custom parsing; mitigating by scripting extraction early and validating with sample outputs.
- **Model compute/time limits:** Large models may be slow on CPU; prioritize lightweight instruction-tuned models (Flan-T5 variants) and batch inference carefully.
- **Confidence extraction:** Generative models may not expose raw probabilities; plan fallback using verbalized likelihood scales or logits (for seq2seq models via softmax scores).
- **Class imbalance or small sample size:** Use cross-validation and report confidence intervals; incorporate bootstrap resampling to estimate variance.

## Success Criteria
- Clean, reproducible dataset and code pipeline.
- LLM evaluation results with statistical comparison against baselines.
- Comprehensive documentation (REPORT.md, README.md) detailing methodology, findings, and limitations.
