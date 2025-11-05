"""
Post-hoc analysis and visualization for epistemic belief experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, chi2


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "figures"


def load_data() -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    preds = pd.read_csv(RESULTS_DIR / "predictions.csv")
    metrics = json.loads((RESULTS_DIR / "metrics_summary.json").read_text())
    return preds, metrics


def compute_additional_metrics(preds: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    methods = [
        ("majority", "pred_majority"),
        ("heuristic", "pred_heuristic"),
        ("logistic_tfidf", "pred_logistic_tfidf"),
        ("google/flan-t5-large", "pred_google/flan-t5-large"),
    ]

    results: Dict[str, Dict[str, float]] = {}
    for method_name, column in methods:
        method_metrics: Dict[str, float] = {}
        for action_type in ["Truth Dependent", "Symbolic"]:
            subset = preds[preds["action_type"] == action_type]
            accuracy = (subset["human_preferred_agent"] == subset[column]).mean()
            method_metrics[f"accuracy_{action_type.replace(' ', '_').lower()}"] = float(accuracy)
        results[method_name] = method_metrics

    return results


def extract_blaine_probability(preds: pd.DataFrame) -> np.ndarray:
    logp_blaine = preds["google/flan-t5-large_logprob_blaine"].to_numpy()
    logp_alex = preds["google/flan-t5-large_logprob_alex"].to_numpy()
    max_logp = np.maximum(logp_blaine, logp_alex)
    prob_blaine = np.exp(logp_blaine - max_logp)
    prob_alex = np.exp(logp_alex - max_logp)
    return prob_blaine / (prob_blaine + prob_alex)


def correlation_analysis(preds: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    prob_blaine = extract_blaine_probability(preds)
    preds = preds.assign(flans_prob_blaine=prob_blaine)

    analyses: Dict[str, Dict[str, float]] = {}
    for subset_name, subset in {
        "all": preds,
        "truth_dependent": preds[preds["action_type"] == "Truth Dependent"],
        "symbolic": preds[preds["action_type"] == "Symbolic"],
    }.items():
        if subset.empty:
            continue
        pearson_corr, pearson_p = pearsonr(subset["flans_prob_blaine"], subset["human_mean_rating"])
        spearman_corr, spearman_p = spearmanr(subset["flans_prob_blaine"], subset["human_mean_rating"])
        analyses[subset_name] = {
            "pearson_r": float(pearson_corr),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_corr),
            "spearman_p": float(spearman_p),
        }

    return analyses


def mcnemar_stat(y_true: pd.Series, pred_a: pd.Series, pred_b: pd.Series) -> Dict[str, float]:
    """Compute McNemar's test between two classifiers."""

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    n10 = ((correct_a) & (~correct_b)).sum()  # A correct, B wrong
    n01 = ((~correct_a) & (correct_b)).sum()  # A wrong, B correct

    if n10 + n01 == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n10": int(n10), "n01": int(n01)}

    chi2_stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    p_value = chi2.sf(chi2_stat, df=1)
    return {"chi2": float(chi2_stat), "p_value": float(p_value), "n10": int(n10), "n01": int(n01)}


def plot_performance(metrics: Dict[str, Dict[str, float]]) -> None:
    methods = ["majority", "heuristic", "logistic_tfidf", "google/flan-t5-large"]
    accuracy = [metrics[m]["accuracy"] for m in methods]
    balanced = [metrics[m]["balanced_accuracy"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, accuracy, width, label="Accuracy")
    ax.bar(x + width / 2, balanced, width, label="Balanced Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model performance on epistemic vs. non-epistemic actions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "performance_comparison.png", dpi=200)
    plt.close(fig)


def plot_human_vs_model(preds: pd.DataFrame) -> None:
    prob_blaine = extract_blaine_probability(preds)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        preds["human_mean_rating"],
        prob_blaine,
        c=preds["action_type"].map({"Truth Dependent": "#1f77b4", "Symbolic": "#ff7f0e"}),
        alpha=0.75,
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Human mean rating (Alex < 0 → Blaine)")
    ax.set_ylabel("Flan-T5 probability for Blaine")
    ax.set_title("Human vs. Flan-T5 beliefs about action ownership")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Truth Dependent", markerfacecolor="#1f77b4", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="Symbolic", markerfacecolor="#ff7f0e", markersize=8),
    ]
    ax.legend(handles=handles, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "human_vs_model.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    preds, metrics = load_data()

    additional_metrics = compute_additional_metrics(preds)
    correlation_stats = correlation_analysis(preds)

    # Pairwise McNemar comparisons
    y_true = preds["human_preferred_agent"]
    comparisons = {
        "logistic_vs_majority": mcnemar_stat(y_true, preds["pred_logistic_tfidf"], preds["pred_majority"]),
        "logistic_vs_flan": mcnemar_stat(y_true, preds["pred_logistic_tfidf"], preds["pred_google/flan-t5-large"]),
        "majority_vs_flan": mcnemar_stat(y_true, preds["pred_majority"], preds["pred_google/flan-t5-large"]),
    }

    (RESULTS_DIR / "additional_metrics.json").write_text(json.dumps(additional_metrics, indent=2))
    (RESULTS_DIR / "correlation_analysis.json").write_text(json.dumps(correlation_stats, indent=2))
    (RESULTS_DIR / "mcnemar_tests.json").write_text(json.dumps(comparisons, indent=2))

    plot_performance(metrics)
    plot_human_vs_model(preds)


if __name__ == "__main__":
    main()
