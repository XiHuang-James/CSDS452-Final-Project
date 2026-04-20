from pathlib import Path
from typing import Dict, Any, List
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


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
# Utility metrics
# ============================================================

def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def correlation_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def compute_entropy_from_ids(ids: np.ndarray) -> float:
    """
    Shannon entropy of item distribution.
    Higher entropy => recommendations are less concentrated on a few items.
    """
    if len(ids) == 0:
        return float("nan")

    value_counts = pd.Series(ids).value_counts(normalize=True).values
    value_counts = value_counts[value_counts > 0]
    return float(-(value_counts * np.log(value_counts)).sum())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# Plot helpers
# ============================================================

def save_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Path,
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_grouped_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    output_path: Path,
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    plt.figure(figsize=(8, 5))

    groups = df[group_col].unique()
    for g in groups:
        sub = df[df[group_col] == g].copy()
        plt.plot(sub[x_col].astype(str), sub[y_col], marker="o", label=str(g))

    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================
# Data loading
# ============================================================

def load_analysis_inputs(project_root: Path, dataset: str = "bts") -> Dict[str, Any]:
    processed_dir = project_root / "outputs" / f"processed_{dataset}"
    models_dir = project_root / "outputs" / f"models_{dataset}"

    merged_path = processed_dir / "merged_engineered_dataframe.csv"
    idx_test_path = processed_dir / "idx_test.npy"

    baseline_pred_path = models_dir / "baseline" / "y_prob_test.npy"
    ips_pred_path = models_dir / "ips" / "y_prob_test.npy"
    snips_pred_path = models_dir / "snips" / "y_prob_test.npy"

    required = [
        merged_path,
        idx_test_path,
        baseline_pred_path,
        ips_pred_path,
        snips_pred_path,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    log_sub(f"Loading merged dataframe -> {merged_path}")
    df_merged = pd.read_csv(merged_path)

    log_sub(f"Loading idx_test -> {idx_test_path}")
    idx_test = np.load(idx_test_path)

    log_sub("Loading model predictions")
    y_prob_baseline = np.load(baseline_pred_path)
    y_prob_ips = np.load(ips_pred_path)
    y_prob_snips = np.load(snips_pred_path)

    return {
        "df_merged": df_merged,
        "idx_test": idx_test,
        "y_prob_baseline": y_prob_baseline,
        "y_prob_ips": y_prob_ips,
        "y_prob_snips": y_prob_snips,
    }


# ============================================================
# Build test dataframe
# ============================================================

def build_test_analysis_df(
    df_merged: pd.DataFrame,
    idx_test: np.ndarray,
    y_prob_baseline: np.ndarray,
    y_prob_ips: np.ndarray,
    y_prob_snips: np.ndarray,
) -> pd.DataFrame:
    df_test = df_merged.iloc[idx_test].copy().reset_index(drop=True)

    if len(df_test) != len(y_prob_baseline):
        raise ValueError(
            f"Length mismatch: len(df_test)={len(df_test)} "
            f"but len(y_prob_baseline)={len(y_prob_baseline)}"
        )

    df_test["pred_baseline"] = y_prob_baseline
    df_test["pred_ips"] = y_prob_ips
    df_test["pred_snips"] = y_prob_snips

    if "item_id" not in df_merged.columns:
        raise ValueError("item_id not found in merged dataframe")
    if "position" not in df_test.columns:
        raise ValueError("position not found in merged dataframe")
    if "click" not in df_test.columns:
        raise ValueError("click not found in merged dataframe")
    if "propensity_score" not in df_test.columns:
        raise ValueError("propensity_score not found in merged dataframe")

    # Exposure count from the full merged dataset
    exposure_count_map = df_merged.groupby("item_id").size().rename("item_exposure_count")
    df_test = df_test.merge(
        exposure_count_map,
        left_on="item_id",
        right_index=True,
        how="left",
    )

    # Popularity quantile bucket
    df_test["item_exposure_count"] = df_test["item_exposure_count"].fillna(0).astype(np.int64)

    # define "high exposure item" as top 20% by exposure count over unique items
    exposure_per_item = exposure_count_map.reset_index()
    threshold = exposure_per_item["item_exposure_count"].quantile(0.8)

    df_test["is_high_exposure_item"] = (
        df_test["item_exposure_count"] >= threshold
    ).astype(int)

    return df_test


# ============================================================
# Experiment 1: Popularity / exposure bias analysis
# ============================================================

def experiment_1_popularity_bias(
    df_test: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Quantify whether a model is more biased toward historically over-exposed items.

    Metrics:
    1) corr(pred_score, item_exposure_count)
    2) share of high-exposure items in top 10% highest-scored samples
    3) entropy of item distribution among top 10% highest-scored samples
    """
    model_cols = {
        "baseline": "pred_baseline",
        "ips": "pred_ips",
        "snips": "pred_snips",
    }

    rows = []
    top_fraction = 0.10
    top_k = max(1, int(len(df_test) * top_fraction))

    for model_name, pred_col in model_cols.items():
        pred = df_test[pred_col].values
        exposure = df_test["item_exposure_count"].values

        corr = correlation_safe(pred, exposure)

        # top 10% by predicted score
        top_idx = np.argsort(-pred)[:top_k]
        df_top = df_test.iloc[top_idx]

        high_exposure_share = float(df_top["is_high_exposure_item"].mean())
        top_item_entropy = compute_entropy_from_ids(df_top["item_id"].values)

        rows.append({
            "model": model_name,
            "corr_pred_vs_item_exposure": corr,
            "top10pct_high_exposure_share": high_exposure_share,
            "top10pct_item_entropy": top_item_entropy,
            "top10pct_mean_pred_score": float(df_top[pred_col].mean()),
        })

    result_df = pd.DataFrame(rows)

    csv_path = output_dir / "exp1_popularity_bias_summary.csv"
    json_path = output_dir / "exp1_popularity_bias_summary.json"
    result_df.to_csv(csv_path, index=False)
    result_df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    # Plots
    save_bar_chart(
        result_df,
        x_col="model",
        y_col="corr_pred_vs_item_exposure",
        title="Experiment 1: Correlation Between Prediction and Historical Exposure",
        output_path=output_dir / "exp1_corr_pred_vs_exposure.png",
        xlabel="Model",
        ylabel="Correlation",
    )
    save_bar_chart(
        result_df,
        x_col="model",
        y_col="top10pct_high_exposure_share",
        title="Experiment 1: Share of High-Exposure Items in Top 10% Predictions",
        output_path=output_dir / "exp1_top10_high_exposure_share.png",
        xlabel="Model",
        ylabel="Share",
    )
    save_bar_chart(
        result_df,
        x_col="model",
        y_col="top10pct_item_entropy",
        title="Experiment 1: Item Entropy in Top 10% Predictions",
        output_path=output_dir / "exp1_top10_item_entropy.png",
        xlabel="Model",
        ylabel="Entropy",
    )

    return {
        "summary_df": result_df,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }


# ============================================================
# Experiment 2: Propensity bucket analysis
# ============================================================

def assign_propensity_buckets(
    df_test: pd.DataFrame,
    n_buckets: int = 5,
) -> pd.DataFrame:
    df = df_test.copy()

    # qcut with duplicates handling
    df["propensity_bucket"] = pd.qcut(
        df["propensity_score"],
        q=n_buckets,
        labels=[f"Q{i+1}" for i in range(n_buckets)],
        duplicates="drop",
    )

    return df


def experiment_2_propensity_bucket(
    df_test: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Analyze model behavior across propensity buckets.
    Goal:
    - show whether debiased models behave differently in low-propensity regions
    """
    model_cols = {
        "baseline": "pred_baseline",
        "ips": "pred_ips",
        "snips": "pred_snips",
    }

    df = assign_propensity_buckets(df_test, n_buckets=5)

    rows = []
    for bucket in sorted(df["propensity_bucket"].dropna().unique(), key=lambda x: str(x)):
        sub = df[df["propensity_bucket"] == bucket].copy()

        for model_name, pred_col in model_cols.items():
            y_true = sub["click"].values
            y_pred = sub[pred_col].values

            reward_proxy = y_true * y_pred

            rows.append({
                "propensity_bucket": str(bucket),
                "model": model_name,
                "n_samples": int(len(sub)),
                "positive_rate": float(np.mean(y_true)),
                "mean_propensity": float(np.mean(sub["propensity_score"])),
                "mean_pred_prob": float(np.mean(y_pred)),
                "auc": safe_auc(y_true, y_pred),
                "reward_proxy_mean": float(np.mean(reward_proxy)),
            })

    result_df = pd.DataFrame(rows)

    csv_path = output_dir / "exp2_propensity_bucket_summary.csv"
    json_path = output_dir / "exp2_propensity_bucket_summary.json"
    result_df.to_csv(csv_path, index=False)
    result_df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    # Plots
    save_grouped_line_chart(
        result_df,
        x_col="propensity_bucket",
        y_col="auc",
        group_col="model",
        title="Experiment 2: AUC Across Propensity Buckets",
        output_path=output_dir / "exp2_auc_by_propensity_bucket.png",
        xlabel="Propensity Bucket",
        ylabel="AUC",
    )
    save_grouped_line_chart(
        result_df,
        x_col="propensity_bucket",
        y_col="mean_pred_prob",
        group_col="model",
        title="Experiment 2: Mean Predicted Probability Across Propensity Buckets",
        output_path=output_dir / "exp2_mean_pred_by_propensity_bucket.png",
        xlabel="Propensity Bucket",
        ylabel="Mean Predicted Probability",
    )
    save_grouped_line_chart(
        result_df,
        x_col="propensity_bucket",
        y_col="reward_proxy_mean",
        group_col="model",
        title="Experiment 2: Reward Proxy Across Propensity Buckets",
        output_path=output_dir / "exp2_reward_proxy_by_propensity_bucket.png",
        xlabel="Propensity Bucket",
        ylabel="Reward Proxy Mean",
    )

    return {
        "summary_df": result_df,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }


# ============================================================
# Experiment 3: Position bias analysis
# ============================================================

def experiment_3_position_bias(
    df_test: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Analyze model behavior across display positions.
    Goal:
    - show whether baseline absorbs position bias more strongly
    - see if IPS/SNIPS reduce position-driven prediction artifacts
    """
    model_cols = {
        "baseline": "pred_baseline",
        "ips": "pred_ips",
        "snips": "pred_snips",
    }

    rows = []
    positions = sorted(df_test["position"].dropna().unique())

    for pos in positions:
        sub = df_test[df_test["position"] == pos].copy()

        for model_name, pred_col in model_cols.items():
            y_true = sub["click"].values
            y_pred = sub[pred_col].values
            reward_proxy = y_true * y_pred

            rows.append({
                "position": str(pos),
                "model": model_name,
                "n_samples": int(len(sub)),
                "positive_rate": float(np.mean(y_true)),
                "mean_pred_prob": float(np.mean(y_pred)),
                "auc": safe_auc(y_true, y_pred),
                "reward_proxy_mean": float(np.mean(reward_proxy)),
            })

    result_df = pd.DataFrame(rows)

    csv_path = output_dir / "exp3_position_bias_summary.csv"
    json_path = output_dir / "exp3_position_bias_summary.json"
    result_df.to_csv(csv_path, index=False)
    result_df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    save_grouped_line_chart(
        result_df,
        x_col="position",
        y_col="mean_pred_prob",
        group_col="model",
        title="Experiment 3: Mean Predicted Probability Across Positions",
        output_path=output_dir / "exp3_mean_pred_by_position.png",
        xlabel="Position",
        ylabel="Mean Predicted Probability",
    )
    save_grouped_line_chart(
        result_df,
        x_col="position",
        y_col="auc",
        group_col="model",
        title="Experiment 3: AUC Across Positions",
        output_path=output_dir / "exp3_auc_by_position.png",
        xlabel="Position",
        ylabel="AUC",
    )
    save_grouped_line_chart(
        result_df,
        x_col="position",
        y_col="reward_proxy_mean",
        group_col="model",
        title="Experiment 3: Reward Proxy Across Positions",
        output_path=output_dir / "exp3_reward_proxy_by_position.png",
        xlabel="Position",
        ylabel="Reward Proxy Mean",
    )

    return {
        "summary_df": result_df,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["bts", "random"],
        default="bts",
        help="Which dataset outputs to analyze. For debiasing analysis, bts is recommended.",
    )
    args = parser.parse_args()

    total_steps = 5

    script_path = Path(__file__).resolve()
    src_dir = script_path.parent
    project_root = src_dir.parent

    analysis_dir = project_root / "outputs" / f"debias_analysis_{args.dataset}"
    ensure_dir(analysis_dir)

    log("\n" + "#" * 80)
    log("DEBIAS ANALYSIS STARTED")
    log("#" * 80)
    log(f"Project root : {project_root}")
    log(f"Dataset      : {args.dataset}")
    log(f"Output dir   : {analysis_dir}")

    # --------------------------------------------------------
    # Step 1: Load inputs
    # --------------------------------------------------------
    log_step(1, total_steps, "Loading processed data and model predictions")
    data = load_analysis_inputs(project_root=project_root, dataset=args.dataset)

    # --------------------------------------------------------
    # Step 2: Build test analysis dataframe
    # --------------------------------------------------------
    log_step(2, total_steps, "Building test analysis dataframe")
    df_test = build_test_analysis_df(
        df_merged=data["df_merged"],
        idx_test=data["idx_test"],
        y_prob_baseline=data["y_prob_baseline"],
        y_prob_ips=data["y_prob_ips"],
        y_prob_snips=data["y_prob_snips"],
    )

    df_test_path = analysis_dir / "analysis_test_dataframe.csv"
    df_test.to_csv(df_test_path, index=False)

    log_sub(f"df_test shape = {df_test.shape}")
    log_sub(f"Saved merged analysis test dataframe -> {df_test_path}")

    # --------------------------------------------------------
    # Step 3: Experiment 1
    # --------------------------------------------------------
    log_step(3, total_steps, "Experiment 1: Popularity / exposure bias analysis")
    exp1 = experiment_1_popularity_bias(df_test=df_test, output_dir=analysis_dir)
    log_sub(f"Saved Experiment 1 summary -> {exp1['csv_path']}")

    # --------------------------------------------------------
    # Step 4: Experiment 2
    # --------------------------------------------------------
    log_step(4, total_steps, "Experiment 2: Propensity bucket analysis")
    exp2 = experiment_2_propensity_bucket(df_test=df_test, output_dir=analysis_dir)
    log_sub(f"Saved Experiment 2 summary -> {exp2['csv_path']}")

    # --------------------------------------------------------
    # Step 5: Experiment 3
    # --------------------------------------------------------
    log_step(5, total_steps, "Experiment 3: Position bias analysis")
    exp3 = experiment_3_position_bias(df_test=df_test, output_dir=analysis_dir)
    log_sub(f"Saved Experiment 3 summary -> {exp3['csv_path']}")

    # master summary
    master_summary = {
        "dataset": args.dataset,
        "output_dir": str(analysis_dir),
        "experiment_1": {
            "csv_path": exp1["csv_path"],
            "records": exp1["summary_df"].to_dict(orient="records"),
        },
        "experiment_2": {
            "csv_path": exp2["csv_path"],
            "records": exp2["summary_df"].to_dict(orient="records"),
        },
        "experiment_3": {
            "csv_path": exp3["csv_path"],
            "records": exp3["summary_df"].to_dict(orient="records"),
        },
    }

    master_summary_path = analysis_dir / "debias_analysis_summary.json"
    with open(master_summary_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, ensure_ascii=False, indent=2)

    log("\n" + "#" * 80)
    log("DEBIAS ANALYSIS FINISHED")
    log("#" * 80)
    log(f"Saved master summary -> {master_summary_path}")
    log("#" * 80 + "\n")


if __name__ == "__main__":
    main()