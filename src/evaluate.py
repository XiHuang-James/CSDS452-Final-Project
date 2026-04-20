from pathlib import Path
from typing import Dict, Any
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score


# ============================================================
# Logging helpers
# ============================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def log_step(step_no: int, total_steps: int, title: str) -> None:
    log("\n" + "=" * 80)
    log(f"[Step {step_no}/{total_steps}] {title}")
    log("=" * 80)


def log_sub(msg: str) -> None:
    log(f"  -> {msg}")


# ============================================================
# Core metric functions
# ============================================================

def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 1e-12, 1 - 1e-12)
    try:
        return float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        return float("nan")


def compute_basic_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int32)

    return {
        "auc": safe_auc(y_true, y_prob),
        "log_loss": safe_log_loss(y_true, y_prob),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mean_pred_prob": float(np.mean(y_prob)),
        "positive_rate_true": float(np.mean(y_true)),
    }


def compute_reward_proxy(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """
    Simplified reward proxy for this course project:
        reward_proxy_i = y_i * p_hat_i

    Interpretation:
    - if a clicked sample also has high predicted probability, it contributes more
    - if no click, contribution is zero
    """
    return y_true.astype(np.float64) * y_prob.astype(np.float64)


def compute_ips_style_estimate(
    reward_proxy: np.ndarray,
    w_ips: np.ndarray,
) -> float:
    """
    IPS-style weighted reward proxy:
        (1/n) * sum_i w_i * reward_proxy_i
    """
    return float(np.mean(w_ips.astype(np.float64) * reward_proxy))


def compute_snips_style_estimate(
    reward_proxy: np.ndarray,
    w_ips: np.ndarray,
) -> float:
    """
    SNIPS-style weighted reward proxy:
        sum_i w_i * reward_proxy_i / sum_i w_i
    """
    w = w_ips.astype(np.float64)
    denom = np.sum(w)
    if denom <= 0:
        return float("nan")
    return float(np.sum(w * reward_proxy) / denom)


def evaluate_single_model(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w_ips: np.ndarray,
    w_snips: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate one model.

    Notes:
    - AUC / log_loss are standard classification metrics
    - IPS-style and SNIPS-style here are simplified weighted reward proxies
      under the current project setup
    """
    metrics = compute_basic_metrics(y_true, y_prob)

    reward_proxy = compute_reward_proxy(y_true, y_prob)

    # Use raw IPS weights for both IPS-style and SNIPS-style formulas.
    # w_snips is still reported as metadata / sanity reference if needed.
    metrics["reward_proxy_mean"] = float(np.mean(reward_proxy))
    metrics["ips_style_reward_estimate"] = compute_ips_style_estimate(reward_proxy, w_ips)
    metrics["snips_style_reward_estimate"] = compute_snips_style_estimate(reward_proxy, w_ips)

    metrics["w_ips_mean"] = float(np.mean(w_ips))
    metrics["w_ips_min"] = float(np.min(w_ips))
    metrics["w_ips_max"] = float(np.max(w_ips))

    metrics["w_snips_mean"] = float(np.mean(w_snips))
    metrics["w_snips_min"] = float(np.min(w_snips))
    metrics["w_snips_max"] = float(np.max(w_snips))

    return metrics


# ============================================================
# Plot helpers
# ============================================================

def save_bar_chart(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df[metric_col])
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(metric_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_summary_table_csv(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    df.to_csv(output_path, index=False)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["bts", "random"],
        default="bts",
        help="Which dataset outputs to evaluate.",
    )
    args = parser.parse_args()

    total_steps = 5

    script_path = Path(__file__).resolve()
    src_dir = script_path.parent
    project_root = src_dir.parent

    processed_dir = project_root / "outputs" / f"processed_{args.dataset}"
    model_root_dir = project_root / "outputs" / f"models_{args.dataset}"
    eval_dir = project_root / "outputs" / f"evaluation_{args.dataset}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log("\n" + "#" * 80)
    log("MODEL EVALUATION STARTED")
    log("#" * 80)
    log(f"Project root  : {project_root}")
    log(f"Processed dir : {processed_dir}")
    log(f"Models dir    : {model_root_dir}")
    log(f"Eval dir      : {eval_dir}")
    log(f"Dataset       : {args.dataset}")

    # --------------------------------------------------------
    # Step 1: Load ground-truth labels and weights
    # --------------------------------------------------------
    log_step(1, total_steps, "Loading processed test labels and weights")

    y_test_path = processed_dir / "y_test.npy"
    w_ips_test_path = processed_dir / "w_ips_test.npy"
    w_snips_test_path = processed_dir / "w_snips_test.npy"

    if not y_test_path.exists():
        raise FileNotFoundError(f"Missing file: {y_test_path}")
    if not w_ips_test_path.exists():
        raise FileNotFoundError(f"Missing file: {w_ips_test_path}")
    if not w_snips_test_path.exists():
        raise FileNotFoundError(f"Missing file: {w_snips_test_path}")

    y_test = np.load(y_test_path)
    w_ips_test = np.load(w_ips_test_path)
    w_snips_test = np.load(w_snips_test_path)

    log_sub(f"y_test shape      = {y_test.shape}")
    log_sub(f"w_ips_test shape  = {w_ips_test.shape}")
    log_sub(f"w_snips_test shape= {w_snips_test.shape}")
    log_sub(f"test positive rate= {np.mean(y_test):.6f}")

    # --------------------------------------------------------
    # Step 2: Load model predictions
    # --------------------------------------------------------
    log_step(2, total_steps, "Loading predictions from three trained models")

    model_names = ["baseline", "ips", "snips"]
    y_prob_dict = {}

    for model_name in model_names:
        pred_path = model_root_dir / model_name / "y_prob_test.npy"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")

        y_prob = np.load(pred_path)
        y_prob_dict[model_name] = y_prob

        log_sub(f"{model_name}: loaded {pred_path}")
        log_sub(f"{model_name}: y_prob_test shape = {y_prob.shape}")
        log_sub(f"{model_name}: mean predicted probability = {np.mean(y_prob):.6f}")

    # --------------------------------------------------------
    # Step 3: Compute metrics
    # --------------------------------------------------------
    log_step(3, total_steps, "Computing evaluation metrics")

    rows = []
    detailed_results = {}

    for model_name in model_names:
        log_sub(f"Evaluating model: {model_name}")

        metrics = evaluate_single_model(
            model_name=model_name,
            y_true=y_test,
            y_prob=y_prob_dict[model_name],
            w_ips=w_ips_test,
            w_snips=w_snips_test,
        )

        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)
        detailed_results[model_name] = metrics

        log_sub(
            f"{model_name} | "
            f"AUC={metrics['auc']:.6f}, "
            f"log_loss={metrics['log_loss']:.6f}, "
            f"IPS-style={metrics['ips_style_reward_estimate']:.6f}, "
            f"SNIPS-style={metrics['snips_style_reward_estimate']:.6f}"
        )

    results_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Step 4: Save tables and json
    # --------------------------------------------------------
    log_step(4, total_steps, "Saving evaluation tables and JSON summaries")

    results_csv_path = eval_dir / "evaluation_summary.csv"
    results_json_path = eval_dir / "evaluation_summary.json"

    save_summary_table_csv(results_df, results_csv_path)

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    log_sub(f"Saved CSV summary  -> {results_csv_path}")
    log_sub(f"Saved JSON summary -> {results_json_path}")

    # --------------------------------------------------------
    # Step 5: Save plots
    # --------------------------------------------------------
    log_step(5, total_steps, "Generating bar charts")

    auc_plot_path = eval_dir / "bar_auc.png"
    logloss_plot_path = eval_dir / "bar_log_loss.png"
    ips_plot_path = eval_dir / "bar_ips_style_reward.png"
    snips_plot_path = eval_dir / "bar_snips_style_reward.png"
    prob_plot_path = eval_dir / "bar_mean_pred_prob.png"

    save_bar_chart(
        results_df,
        metric_col="auc",
        title=f"AUC Comparison ({args.dataset})",
        output_path=auc_plot_path,
    )
    save_bar_chart(
        results_df,
        metric_col="log_loss",
        title=f"Log Loss Comparison ({args.dataset})",
        output_path=logloss_plot_path,
    )
    save_bar_chart(
        results_df,
        metric_col="ips_style_reward_estimate",
        title=f"IPS-style Weighted Reward Estimate ({args.dataset})",
        output_path=ips_plot_path,
    )
    save_bar_chart(
        results_df,
        metric_col="snips_style_reward_estimate",
        title=f"SNIPS-style Weighted Reward Estimate ({args.dataset})",
        output_path=snips_plot_path,
    )
    save_bar_chart(
        results_df,
        metric_col="mean_pred_prob",
        title=f"Mean Predicted Probability ({args.dataset})",
        output_path=prob_plot_path,
    )

    log_sub(f"Saved plot -> {auc_plot_path}")
    log_sub(f"Saved plot -> {logloss_plot_path}")
    log_sub(f"Saved plot -> {ips_plot_path}")
    log_sub(f"Saved plot -> {snips_plot_path}")
    log_sub(f"Saved plot -> {prob_plot_path}")

    log("\n" + "#" * 80)
    log("MODEL EVALUATION FINISHED")
    log("#" * 80)
    log(results_df.to_string(index=False))
    log("#" * 80 + "\n")


if __name__ == "__main__":
    main()