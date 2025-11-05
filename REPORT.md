## 1. Executive Summary
Large language models were evaluated on whether they can distinguish epistemic (truth-tracking) versus non-epistemic (symbolic/signaling) beliefs in the Vesga et al. (2025) “Check Your Attitudes” stimuli. After reconstructing 30 agent–action scenarios from the authors’ OSF release, we compared simple baselines, a TF-IDF logistic classifier, and the instruction-tuned `google/flan-t5-large` model.  

Across 30 scenario–action pairs (15 truth-dependent, 15 symbolic), the logistic baseline achieved 70% accuracy (balanced accuracy 0.74), while Flan-T5 underperformed drastically at 30% accuracy and 0.50 balanced accuracy, indicating predictions near random. Human endorsements showed strong, systematic preferences (e.g., symbolic actions universally attributed to Alex), but LLM confidence was uncorrelated with human ratings (Pearson r ≈ 0.01).  

These results suggest that, under zero-shot prompting, Flan-T5 does **not** differentiate epistemic from non-epistemic beliefs in the structured Vesga et al. tasks, whereas lightweight lexical models partially capture the signal. Practically, this implies additional supervision or fine-tuning is required before relying on current LLMs to emulate human-style belief-type reasoning in social cognition applications.

## 2. Goal
- **Hypothesis tested:** LLMs can distinguish epistemic (truth-dependent) from non-epistemic (symbolic) belief types analogously to humans.  
- **Importance:** Establishing this capability is critical for theory-of-mind style reasoning and downstream safety analyses.  
- **Problem addressed:** Lack of an LLM-oriented benchmark mirroring human studies on belief differentiation.  
- **Expected impact:** Provide a reproducible evaluation harness grounding future investigations or fine-tuning efforts.

## 3. Data Construction

### Dataset Description
- **Source:** Vesga et al. (2025) OSF repository (`https://doi.org/10.17605/OSF.IO/38YGN`).  
- **Participants:** 751 U.S. adults in Study 3.  
- **Stimuli:** Five topical claims (politics, religion, extraterrestrials, animal welfare, AI) crossed with three framing manipulations (Binary/Probabilistic, Think/Believe, Directionality) and two action types (truth-dependent vs. symbolic).  
- **Size:** 30 scenario–action datapoints (15 per action class) after aggregation; each point aggregates 20–28 human ratings.  
- **Accessibility:** Raw CSV (`data/raw/Check_your_attitudes_Study3_Raw_Wide.csv`) and Word materials (`data/raw/Materials_Study3.docx`).

### Example Samples
```
BinaryProbable · AI · Symbolic action
Intro: Alex and Blaine were both called by the same survey company ... about AI consciousness.
Statements:
  Alex: "Artificial intelligence will never achieve human-like consciousness."
  Blaine: "There’s a 98% chance that Artificial intelligence will never achieve human-like consciousness."
Action: Organized a public lecture so people could hear a philosopher arguing AI will never gain consciousness.
Human mean rating (Alex < 0 → Blaine): -1.00 → majority attribute symbolic action to Alex.
```
```
Directionality · Jesus · Truth-dependent action
Intro: Alex and Blaine were asked about the birthplace of Jesus ... did any decision guide their belief?
Statements:
  Alex said, "I decided to believe that Jesus was born in a manger near Bethlehem."
  Blaine said, "I decided whether to believe that Jesus was born in a manger near Bethlehem."
Action: Anonymously donated $1000 to an archaeological organization seeking historical evidence.
Human mean rating: +0.70 → majority expect Blaine (epistemic orientation) to perform truth-dependent action.
```

### Data Quality
- Rating scale recoded from survey outputs (0–7) to -3…+3 following original analysis scripts.  
- **Symbolic actions:** mean human rating -1.27 (SD 0.41); participants unanimously preferred Alex.  
- **Truth-dependent actions:** mean -0.50 (SD 0.69); 5/15 scenarios favored Blaine.  
- No missing values post-melt; demographic columns retained separately for potential covariate analyses.

### Preprocessing Steps
1. Parsed Word materials (`python-docx`) to extract claim text, action descriptions, and framing templates.
2. Reshaped raw wide-format CSV to a tidy participant-level table (question, framing, action type, rating).
3. Converted rating codes via Vesga et al. R pipeline mapping; filtered to action judgments.
4. Aggregated by scenario to compute mean/median ratings, preference strengths, and class labels.
5. Generated evaluation dataframe with prompt-ready text fields plus human statistics (`data/processed/study3_scenarios.csv`).

### Train/Val/Test Splits
- Dataset too small for fixed splits; models evaluated via full-set prediction.  
- Logistic baseline uses 5-fold stratified cross-validation to obtain unbiased accuracy estimates.  
- LLM and rule baselines evaluated zero-shot on the full set.

## 4. Experiment Description

### Methodology
- **High-Level Approach:** Reconstruct Vesga et al. stimuli, derive human ground-truth labels, and test whether models align with human action predictions distinguishing epistemic vs. symbolic belief manifestations.
- **Rationale:** Aligns evaluation directly with empirical human data rather than synthetic prompts. Crossed conditions allow probing robustness across belief signatures.

### Implementation Details
- **Tools & Libraries**
  - Python 3.12.2
  - pandas 2.3.3, numpy 2.3.4, scikit-learn 1.7.2
  - torch 2.9.0 (CPU), transformers 4.57.1, sentencepiece 0.2.1
  - matplotlib 3.10.7, seaborn 0.13.2
- **Algorithms/Models**
  - Majority baseline (predict most common agent).
  - Keyword heuristic (epistemic cue count across agent statements).
  - TF-IDF + logistic regression (`LogisticRegression`, class-weight balanced).
  - `google/flan-t5-large` (1-shot log-likelihood comparison of “Alex” vs. “Blaine”).
- **Hyperparameters**

| Parameter                     | Value                      | Selection Method |
|-------------------------------|----------------------------|------------------|
| TF-IDF n-gram range           | (1, 2)                     | Prior literature |
| Logistic regularization       | L2 (default)               | Default          |
| Logistic max_iter             | 500                        | Convergence check|
| CV folds                      | 5 (stratified)             | Data size limit  |
| Flan-T5 decoding              | Log-likelihood scoring     | Analytical choice|

### Training / Analysis Pipeline
1. `src/data_prep.py`: parse raw assets, output participant-level and scenario-level CSVs.
2. `src/run_experiments.py`: compute baselines, run Flan-T5 scoring (log-probabilities), and store predictions.
3. `src/analyze_results.py`: derive supplemental metrics, correlations, McNemar tests, and generate figures.
4. Results saved to `results/metrics/`; figures in `figures/`.

### Experimental Protocol & Reproducibility
- **Random seed:** 42 (numpy, torch, sklearn).  
- **Runs per model:** one deterministic pass; logistic evaluated via cross-val predictions.  
- **Hardware:** CPU-only Linux workstation; Flan-T5 inference ~3.5 s for 30 prompts.  
- **Execution time:** data prep <1 s; experiment runner ~17 s; analysis ~2 s.  
- **Evaluation metrics:** accuracy, balanced accuracy, macro F1; Pearson/Spearman correlations for confidence alignment; McNemar tests for paired comparisons.  
- **Outputs:** `results/metrics/metrics_summary.json`, `predictions.csv`, `llm_detailed_predictions.json`, `additional_metrics.json`, `correlation_analysis.json`, `mcnemar_tests.json`.

## 5. Result Analysis

### Raw Results
| Method                   | Accuracy | Balanced Acc. | Macro F1 |
|--------------------------|----------|----------------|----------|
| Majority baseline        | 0.83     | 0.50           | 0.45     |
| Keyword heuristic        | 0.50     | 0.70           | 0.49     |
| TF-IDF Logistic (CV)     | 0.70     | 0.74           | 0.63     |
| Flan-T5 Large (LLM)      | 0.30     | 0.50           | 0.30     |

- Per-action accuracies (`results/metrics/additional_metrics.json`):  
  - Logistic: 0.67 (truth-dependent) / 0.73 (symbolic).  
  - Flan-T5: 0.40 (truth-dependent) / 0.20 (symbolic).
- Flan-T5 vs logistic McNemar χ² = 7.56 (p ≈ 0.006); majority vs Flan χ² = 9.38 (p ≈ 0.002).
- Human vs LLM correlation (`results/metrics/correlation_analysis.json`): Pearson r ≈ 0.01 overall (p ≈ 0.96).

### Visualizations
- `figures/performance_comparison.png`: baseline vs. LLM accuracy/balanced accuracy bars.
- `figures/human_vs_model.png`: scatter of human mean ratings vs. Flan-T5 Blaine probability (horizontal/vertical decision boundaries).

### Key Findings
1. Flan-T5 failed to track human belief-type distinctions (accuracy 30%, balanced accuracy 50%), frequently predicting Blaine regardless of action type.  
2. Simple TF-IDF logistics outperformed the majority baseline without fine-tuning, implying lexical cues suffice for partial differentiation (supporting H1 for classical models, refuting it for LLM).  
3. Model confidence bore no correlation with human endorsement strength (H2 rejected).  
4. Performance varied by framing (H3 rejected): Flan-T5 achieved 60% on Directionality but 0% on Think/Believe scenarios.

### Comparison to Baselines
- Logistic vs majority: modest +0.10 absolute accuracy improvement (non-significant, p ≈ 0.39).  
- Flan-T5 significantly worse than both baselines (p < 0.01 via McNemar).

### Surprises & Insights
- Humans strongly favored Alex for symbolic actions, producing a highly imbalanced label distribution that Flan-T5 ignored.  
- Flan-T5’s 90%+ confidence on incorrect symbolic predictions indicates overconfidence rather than calibrated uncertainty.  
- Heuristic balanced accuracy (0.70) exceeded overall accuracy because it exploits class imbalance—highlighting the importance of balanced metrics.

### Error Analysis
- Flan-T5 predicted Blaine for 21/30 scenarios, including all symbolic actions, even when Blaine’s statement contained more epistemic nuance (“98% chance”).  
- Logistic mispredictions concentrated in Think/Believe framing, suggesting lexical overlap between “believe” and “think” limits bag-of-words discrimination.  
- Misclassified examples (e.g., AI symbolic action) show Flan-T5 attends to epistemic phrases (“there’s a 98% chance”) even when the action is symbolic, implying reliance on agent statements without integrating action context.

### Limitations
- Small evaluation set (30 items) limits statistical power; logistic vs majority difference is not significant.  
- Only one LLM (Flan-T5 large) evaluated; results may differ for larger or instruction-optimized models.  
- Zero-shot prompting only; few-shot exemplars or calibration prompts might improve behavior.  
- Human data aggregated; individual-level variance not modeled (e.g., hierarchical modeling).  
- Actions focus on two extremes (truth vs symbolic); intermediate cases untested.

## 6. Conclusions
- **Summary:** In reconstructed Vesga et al. scenarios, Flan-T5 fails to separate epistemic from non-epistemic beliefs, while classical lexical models partially align with human expectations.  
- **Implications:** Current zero-shot LLM reasoning is insufficient for belief-type theory-of-mind tasks; reliability requires task-specific alignment or additional supervision.  
- **Confidence:** Moderate—results are consistent across metrics and supported by statistical tests against baselines, but limited by small sample size and single-model coverage.

## 7. Next Steps

### Immediate Follow-ups
1. Evaluate additional LLMs (e.g., Llama-3 Instruct, GPT-4 family) using the same harness to benchmark broader capabilities.  
2. Incorporate few-shot exemplars derived from human annotations to test whether in-context learning improves alignment.

### Alternative Approaches
1. Fine-tune smaller models on synthetic epistemic vs symbolic annotations created from the Study 3 templates.  
2. Explore discriminative LLM scoring (e.g., logistic regression on embeddings) to reduce overconfidence.

### Broader Extensions
1. Expand dataset with synthetic intermediate belief types (emotion regulation, social alignment) to test gradient sensitivity.  
2. Combine with epistemic stance corpora (Gupta et al., 2023) for multi-domain evaluation.

### Open Questions
- Can calibration techniques (temperature scaling, conformal prediction) mitigate overconfident symbolic misclassifications?  
- How do human certainty judgments (also recorded in the dataset) correlate with LLM predictions when modeled jointly?  
- Are prompts referencing explicit belief definitions sufficient to elicit better LLM discrimination?

## References
- Vesga, A., Van Leeuwen, N., & Lombrozo, T. (2025). *Evidence for Multiple Kinds of Belief in Theory of Mind*. Journal of Experimental Psychology: General. OSF: https://doi.org/10.17605/OSF.IO/38YGN  
- Gupta, A., Blodgett, S. L., Gross, J. H., & O'Connor, B. (2023). *Examining Political Rhetoric with Epistemic Stance Detection*. arXiv:2212.14486.  
- Flan-T5 model documentation: Chung et al., 2022.  
