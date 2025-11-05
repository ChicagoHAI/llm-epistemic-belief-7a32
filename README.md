## LLM Epistemic Belief Study
Reimplementation of Vesga et al. (2025) Study 3 stimuli to probe whether large language models distinguish epistemic (truth-tracking) from non-epistemic (symbolic) beliefs. The repo contains data preparation scripts, baseline models, and zero-shot evaluations of `google/flan-t5-large`.

### Key Findings
- TF-IDF logistic regression reaches 70% accuracy (balanced accuracy 0.74) on 30 scenario–action pairs.
- Flan-T5 large performs near chance (30% accuracy) and shows no correlation with human endorsement strength.
- Logistic vs. Flan-T5 difference is significant (McNemar χ² ≈ 7.56, p ≈ 0.006); logistic vs. majority baseline is not.
- Symbolic actions remain challenging: Flan-T5 predicts Blaine 80% of the time despite humans unanimously choosing Alex.

### Reproduction Guide
1. **Environment**
   ```bash
   uv venv
   source .venv/bin/activate
   uv add --active pandas numpy scikit-learn matplotlib seaborn scipy transformers accelerate torch sentencepiece datasets python-docx tqdm
   ```
2. **Data preparation**
   ```bash
   source .venv/bin/activate
   python src/data_prep.py
   ```
3. **Run experiments & analysis**
   ```bash
   source .venv/bin/activate
   python src/run_experiments.py
   python src/analyze_results.py
   ```
4. Outputs appear under `results/metrics/` and plots in `figures/`.

### Repository Layout
- `data/raw/` – Original OSF downloads (Study 3 CSV, materials docx, R markdown).
- `data/processed/` – Tidy participant responses, scenario table, metadata.
- `src/data_prep.py` – Raw data parser and scenario assembler.
- `src/run_experiments.py` – Baseline + Flan-T5 evaluation and metric logging.
- `src/analyze_results.py` – Supplemental metrics, statistical tests, visualizations.
- `results/metrics/` – JSON summaries, prediction tables, classification report.
- `figures/` – Performance and correlation plots.
- `REPORT.md` – Full technical report (see below).

### Full Report
- See `REPORT.md` for methodology, statistical analysis, visualizations, and discussion.
