"""
Experiment runner for evaluating baselines and instruction-tuned models
on the epistemic vs. non-epistemic belief stimuli derived from Vesga et al. (2025).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from data_prep import aggregate_actions, assemble_scenarios, extract_claims_and_actions, extract_topics, tidy_responses, build_framing_templates  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results" / "metrics"


LABELS = ["Alex", "Blaine"]
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def ensure_data_ready() -> pd.DataFrame:
    """Run data preparation if needed and return the scenario dataframe."""

    scenarios_path = PROCESSED_DIR / "study3_scenarios.csv"
    if not scenarios_path.exists():
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        # Re-run data prep snapshot to ensure deterministic outputs
        doc = extract_docs()
        tidy = tidy_responses(RAW_DIR / "Check_your_attitudes_Study3_Raw_Wide.csv")
        tidy.to_csv(PROCESSED_DIR / "study3_participant_responses.csv", index=False)
        aggregated = aggregate_actions(tidy)
        scenarios = assemble_scenarios(
            aggregated,
            build_framing_templates(doc),
            extract_claims_and_actions(doc),
            extract_topics(doc),
        )
        scenarios.to_csv(scenarios_path, index=False)

    return pd.read_csv(scenarios_path)


def extract_docs():
    """Load the Study 3 Word document once to reuse across calls."""

    from docx import Document

    return Document(RAW_DIR / "Materials_Study3.docx")


def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_text_feature(row: pd.Series) -> str:
    """Concatenate scenario components for text-based models."""

    parts = [
        row["intro_text"],
        row["alex_statement"],
        row["blaine_statement"],
        f"Action: {row['action_text']}",
        f"Framing: {row['framing']}",
        f"ActionType: {row['action_type']}",
    ]
    return "\n".join(parts)


def majority_baseline(scenarios: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
    """Always predict the most frequent human-preferred agent."""

    majority_label = scenarios["human_preferred_agent"].mode().iat[0]
    preds = np.array([majority_label] * len(scenarios))
    smry = compute_metrics(scenarios["human_preferred_agent"].to_numpy(), preds)
    return preds, smry


def heuristic_baseline(scenarios: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
    """Rule-based baseline using simple lexical cues."""

    predictions: List[str] = []
    for _, row in scenarios.iterrows():
        alex_text = row["alex_statement"].lower()
        blaine_text = row["blaine_statement"].lower()
        action_type = row["action_type"]

        # default bias towards Alex (symbolic preference in human data)
        choice = "Alex"

        epistemic_markers = ["98% chance", "chance", "% chance", "whether to believe"]
        symbolic_markers = ["i believe", "decided to believe", "we won", "spokesperson", "organized"]

        blaine_score = sum(marker in blaine_text for marker in epistemic_markers)
        alex_score = sum(marker in alex_text for marker in epistemic_markers)

        # For symbolic actions, keep Alex unless Blaine has markedly higher epistemic cues,
        # which would imply Alex is comparatively symbolic.
        if action_type == "Truth Dependent" and blaine_score > alex_score:
            choice = "Blaine"
        elif action_type == "Truth Dependent" and ("decided whether" in blaine_text):
            choice = "Blaine"
        elif action_type == "Symbolic" and alex_score < blaine_score:
            choice = "Blaine"
        elif action_type == "Truth Dependent" and alex_score > blaine_score:
            choice = "Alex"
        else:
            # fall back to symbolic markers
            alex_symbolic = sum(marker in alex_text for marker in symbolic_markers)
            blaine_symbolic = sum(marker in blaine_text for marker in symbolic_markers)
            if blaine_symbolic > alex_symbolic:
                choice = "Alex"

        predictions.append(choice)

    preds = np.array(predictions)
    smry = compute_metrics(scenarios["human_preferred_agent"].to_numpy(), preds)
    return preds, smry


def logistic_baseline(scenarios: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
    """TF-IDF + Logistic Regression with stratified CV predictions."""

    texts = scenarios.apply(build_text_feature, axis=1)
    labels = scenarios["human_preferred_agent"]

    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_SEED),
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    preds = cross_val_predict(pipeline, texts, labels, cv=cv)
    smry = compute_metrics(labels.to_numpy(), preds)
    return preds, smry


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard classification metrics."""

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    return metrics


# ---------------------------------------------------------------------------
# LLM Scoring
# ---------------------------------------------------------------------------

@dataclass
class ModelPrediction:
    label: str
    probability: float
    log_probs: Dict[str, float]
    prompt: str


def build_prompt(row: pd.Series) -> str:
    """Construct the textual prompt for the seq2seq model."""

    intro = row["intro_text"]
    statements = f"{row['alex_statement']}\n{row['blaine_statement']}"
    action = row["action_text"]

    prompt = (
        f"{intro}\n\n"
        f"{statements}\n\n"
        f"We later learned that one of them did this action:\n"
        f"{action}\n\n"
        "Question: Based on their expressed beliefs, who is more likely to have done this action?\n"
        'Answer with either "Alex" or "Blaine".\n'
        "Answer:"
    )
    return prompt


def load_seq2seq_model(model_name: str, device: torch.device) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def log_likelihood(model, tokenizer, prompt: str, label: str, device: torch.device) -> float:
    """Compute the log probability of a label given the prompt."""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Prepend space to ensure proper tokenization alignment
    target = tokenizer(" " + label, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=target)
        loss = outputs.loss.item()
    length = target.shape[-1]
    return -loss * length


def evaluate_model(model_name: str, scenarios: pd.DataFrame, device: torch.device) -> Tuple[np.ndarray, Dict[str, float], List[ModelPrediction]]:
    """Score each scenario using log-likelihood comparison between labels."""

    tokenizer, model = load_seq2seq_model(model_name, device)
    predictions: List[ModelPrediction] = []
    predicted_labels: List[str] = []

    for _, row in tqdm(scenarios.iterrows(), total=len(scenarios), desc=f"Evaluating {model_name}"):
        prompt = build_prompt(row)
        logps = {label: log_likelihood(model, tokenizer, prompt, label, device) for label in LABELS}
        labels_tensor = torch.tensor([logps[label] for label in LABELS])
        probs = torch.softmax(labels_tensor, dim=0).cpu().numpy()
        best_idx = int(probs.argmax())
        predicted = LABELS[best_idx]

        predictions.append(
            ModelPrediction(
                label=predicted,
                probability=float(probs[best_idx]),
                log_probs={label: float(logps[label]) for label in LABELS},
                prompt=prompt,
            )
        )
        predicted_labels.append(predicted)

    y_true = scenarios["human_preferred_agent"].to_numpy()
    y_pred = np.array(predicted_labels)
    metrics = compute_metrics(y_true, y_pred)

    return y_pred, metrics, predictions


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed()
    scenarios = ensure_data_ready()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_summary: Dict[str, Dict[str, float]] = {}
    predictions_summary: Dict[str, List[str]] = {}

    # Majority baseline
    maj_pred, maj_metrics = majority_baseline(scenarios)
    metrics_summary["majority"] = maj_metrics
    predictions_summary["majority"] = maj_pred.tolist()

    # Heuristic baseline
    heur_pred, heur_metrics = heuristic_baseline(scenarios)
    metrics_summary["heuristic"] = heur_metrics
    predictions_summary["heuristic"] = heur_pred.tolist()

    # Logistic baseline
    log_pred, log_metrics = logistic_baseline(scenarios)
    metrics_summary["logistic_tfidf"] = log_metrics
    predictions_summary["logistic_tfidf"] = log_pred.tolist()

    # LLM evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "google/flan-t5-large"
    llm_pred, llm_metrics, llm_details = evaluate_model(model_name, scenarios, device)
    metrics_summary[model_name] = llm_metrics
    predictions_summary[model_name] = llm_pred.tolist()

    # Persist metrics and per-scenario outputs
    (RESULTS_DIR / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2))

    predictions_df = scenarios.copy()
    for key, preds in predictions_summary.items():
        predictions_df[f"pred_{key}"] = preds

    predictions_df[f"{model_name}_prob"] = [detail.probability for detail in llm_details]
    predictions_df[f"{model_name}_logprob_alex"] = [detail.log_probs["Alex"] for detail in llm_details]
    predictions_df[f"{model_name}_logprob_blaine"] = [detail.log_probs["Blaine"] for detail in llm_details]
    predictions_df.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    # Save detailed prompts for reproducibility / qualitative analysis
    detailed_records = [
        {
            "index": idx,
            "prompt": detail.prompt,
            "prediction": detail.label,
            "probability": detail.probability,
            "log_probs": detail.log_probs,
        }
        for idx, detail in enumerate(llm_details)
    ]
    (RESULTS_DIR / "llm_detailed_predictions.json").write_text(json.dumps(detailed_records, indent=2))

    # Print summary to console for quick inspection
    print("=== Metrics Summary ===")
    for name, metrics in metrics_summary.items():
        formatted = ", ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
        print(f"{name}: {formatted}")


if __name__ == "__main__":
    main()
