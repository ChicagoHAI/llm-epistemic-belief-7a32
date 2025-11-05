## Overview
- **Date:** 2025-05-xx
- **Goal:** Identify literature, datasets, and code resources relevant to probing whether LLMs distinguish epistemic beliefs (truth-tracking) from non-epistemic beliefs (attitudinal, normative, or social-signaling).

## Provided Materials (Workspace)
- `papers/Evidence-for-Multiple-Kinds-of-Belief-in-Theory-of-Mind.pdf` – Vesga et al. (2025). Three human studies showing that laypeople attribute epistemic vs. nonepistemic beliefs differently. Includes stimuli and survey instruments; provides conceptual framing and potential scenario templates.
- `logs/` – System prompt history; no additional research assets.
- No pre-existing datasets, notebooks, or code specific to this project were found.

## External Literature & Benchmarks
- **Gupta et al. (2023)** – “Examining Political Rhetoric with Epistemic Stance Detection” ([arXiv:2212.14486](https://arxiv.org/abs/2212.14486)). Introduces epistemic stance detection in political texts; categorizes claims by belief commitment (asserted, denied, ambivalent). Offers methodological cues (stance classification, hedging detection) and potential feature sets for epistemic/non-epistemic separation.
- **Related concepts:** factual vs. opinion classification datasets (e.g., FEVER, LIAR) capture truthfulness but not preference attitudes; may serve as partial baselines but lack explicit non-epistemic belief labels.

## Human Study Data (Vesga et al.)
- OSF repository: [https://doi.org/10.17605/OSF.IO/38YGN](https://doi.org/10.17605/OSF.IO/38YGN)
  - Navigated via OSF API. Located CSV files in `Data and Code/` (e.g., `Check your attitudes_Study 3 - Raw-Wide.csv`, `Study 2_ThinkBelieve-BinaryProbable_Raw-Wide.csv`, etc.).
  - Files contain scenario-based survey responses distinguishing epistemic vs. non-epistemic belief attributions. These can be repurposed as prompts or evaluation stimuli for LLMs.
  - Example direct download: `https://osf.io/download/wauq7/` (Study 3 raw wide data).

## Additional Dataset Search
- HuggingFace datasets queried for keywords “epistemic,” “belief,” “stance”; no ready-made LLM evaluation set differentiating epistemic vs. non-epistemic beliefs discovered.
- Identified `DKYoon/r1-triviaqa-epistemic` (1000 QA items with “model_think” traces) but lacks explicit non-epistemic belief labels; likely unsuitable as-is.
- Considered adapting opinion/fact benchmarks (e.g., factuality detection) or generating synthetic scenarios if human-study stimuli insufficient.

## Gaps & Proposed Direction
- No existing LLM evaluation benchmark directly targets epistemic vs. non-epistemic belief differentiation.
- Vesga et al. stimuli provide a grounded starting point for constructing evaluation prompts (epistemic vs. social/attitudinal contexts). Need to inspect CSV structure to extract scenario text and labels.
- For baselines, plan to include simple heuristics (e.g., lexical cues for epistemic modality) and potentially compare to factuality classification datasets to illustrate difference.

## Next Actions
1. Parse OSF CSVs to understand available scenario texts, belief-type labels, and annotator judgments.
2. Determine whether to supplement with synthetic or external data (e.g., generate preference statements) to balance classes.
3. Select computational baselines informed by epistemic stance literature (hedging detection, factuality classifiers) for comparison against LLM judgements.
