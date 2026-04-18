import os
import json
from typing import List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, log_loss

from data_pipeline import (
    build_clean_dataset,
    summarize_csv,
    reservoir_sample_csv,
    prepare_sample_for_baseline,
)


# =========================
# Config
# =========================

ITEM_CONTEXT_PATH = "data/item_context.csv"
RAW_BTS_PATH = "data/bts_all.csv"
RAW_RANDOM_PATH = "data/random_all.csv"

CLEAN_BTS_PATH = "data/bts_clean.csv"
CLEAN_RANDOM_PATH = "data/random_clean.csv"

SAMPLE_BTS_PATH = "data/bts_sample.csv"
SAMPLE_RANDOM_PATH = "data/random_sample.csv"
SAMPLE_BOTH_PATH = "data/both_sample.csv"

OUTPUT_DIR = "outputs"

CHUNKSIZE = 200_000
SAMPLE_SIZE_PER_POLICY = 150_000
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 可选：None / "bts" / "random" / "both"
TRAIN_POLICY = "both"

# 可选：None / 1 / 2 / 3
FILTER_POSITION = None


# =========================
# Helpers
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", MaxAbsScaler()),
    ])

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=True)

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        sparse_threshold=1.0,
    )
    return preprocessor


def train_sgd_logistic(X_train, y_train) -> SGDClassifier:
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=20,
        tol=1e-3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    results = {
        "n_test": int(len(y_test)),
        "avg_pred_ctr": float(np.mean(y_prob)),
    }
    if len(np.unique(y_test)) > 1:
        results["auc"] = float(roc_auc_score(y_test, y_prob))
        results["logloss"] = float(log_loss(y_test, y_prob))
    else:
        results["auc"] = None
        results["logloss"] = None
    return results


def concat_samples(sample_paths: List[str], output_path: str) -> pd.DataFrame:
    dfs = []
    for p in sample_paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p, low_memory=True))
    if not dfs:
        raise ValueError("No sample files found to concatenate.")
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_csv(output_path, index=False)
    return df


# =========================
# Main
# =========================

if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)
    ensure_dir("data")

    # 1) Build clean BTS
    if os.path.exists(RAW_BTS_PATH):
        print("[1/8] Building clean BTS dataset...")
        bts_build_summary = build_clean_dataset(
            raw_csv_path=RAW_BTS_PATH,
            item_context_path=ITEM_CONTEXT_PATH,
            output_csv_path=CLEAN_BTS_PATH,
            behavior_policy="bts",
            chunksize=CHUNKSIZE,
            filter_position=FILTER_POSITION,
        )
        save_json(bts_build_summary, os.path.join(OUTPUT_DIR, "build_summary_bts.json"))
    else:
        print("[1/8] BTS raw file not found, skip.")

    # 2) Build clean random
    if os.path.exists(RAW_RANDOM_PATH):
        print("[2/8] Building clean random dataset...")
        random_build_summary = build_clean_dataset(
            raw_csv_path=RAW_RANDOM_PATH,
            item_context_path=ITEM_CONTEXT_PATH,
            output_csv_path=CLEAN_RANDOM_PATH,
            behavior_policy="random",
            chunksize=CHUNKSIZE,
            filter_position=FILTER_POSITION,
        )
        save_json(random_build_summary, os.path.join(OUTPUT_DIR, "build_summary_random.json"))
    else:
        print("[2/8] random raw file not found, skip.")

    # 3) Summaries
    if os.path.exists(CLEAN_BTS_PATH):
        print("[3/8] Summarizing BTS clean dataset...")
        bts_summary = summarize_csv(
            CLEAN_BTS_PATH,
            output_json_path=os.path.join(OUTPUT_DIR, "dataset_summary_bts.json"),
            chunksize=CHUNKSIZE,
        )
        print(json.dumps(bts_summary, ensure_ascii=False, indent=2))

    if os.path.exists(CLEAN_RANDOM_PATH):
        print("[4/8] Summarizing random clean dataset...")
        random_summary = summarize_csv(
            CLEAN_RANDOM_PATH,
            output_json_path=os.path.join(OUTPUT_DIR, "dataset_summary_random.json"),
            chunksize=CHUNKSIZE,
        )
        print(json.dumps(random_summary, ensure_ascii=False, indent=2))

    # 4) Sampling
    if os.path.exists(CLEAN_BTS_PATH):
        print("[5/8] Sampling BTS...")
        reservoir_sample_csv(
            input_csv_path=CLEAN_BTS_PATH,
            output_sample_csv_path=SAMPLE_BTS_PATH,
            sample_size=SAMPLE_SIZE_PER_POLICY,
            seed=RANDOM_STATE,
            chunksize=CHUNKSIZE,
        )

    if os.path.exists(CLEAN_RANDOM_PATH):
        print("[6/8] Sampling random...")
        reservoir_sample_csv(
            input_csv_path=CLEAN_RANDOM_PATH,
            output_sample_csv_path=SAMPLE_RANDOM_PATH,
            sample_size=SAMPLE_SIZE_PER_POLICY,
            seed=RANDOM_STATE + 1,
            chunksize=CHUNKSIZE,
        )

    # 5) Choose training source
    if TRAIN_POLICY == "bts":
        sample_csv_path = SAMPLE_BTS_PATH
    elif TRAIN_POLICY == "random":
        sample_csv_path = SAMPLE_RANDOM_PATH
    elif TRAIN_POLICY == "both":
        print("[7/8] Concatenating BTS + random samples...")
        concat_samples([SAMPLE_BTS_PATH, SAMPLE_RANDOM_PATH], SAMPLE_BOTH_PATH)
        sample_csv_path = SAMPLE_BOTH_PATH
    else:
        raise ValueError("TRAIN_POLICY must be one of: 'bts', 'random', 'both'")

    if not os.path.exists(sample_csv_path):
        raise FileNotFoundError(f"Sample file not found: {sample_csv_path}")

    # 6) Prepare sample for baseline
    prepared = prepare_sample_for_baseline(
        sample_csv_path=sample_csv_path,
        test_size=TEST_SIZE,
        seed=RANDOM_STATE,
    )

    prepared.train_df.to_csv(os.path.join(OUTPUT_DIR, "train_clean_sample.csv"), index=False)
    prepared.test_df.to_csv(os.path.join(OUTPUT_DIR, "test_clean_sample.csv"), index=False)
    save_json(prepared.summary, os.path.join(OUTPUT_DIR, "dataset_summary_sample.json"))

    print(json.dumps(prepared.summary, ensure_ascii=False, indent=2))

    # 7) Train baseline
    print("[8/8] Training baseline...")
    X_train_df = prepared.train_df[prepared.feature_cols].copy()
    X_test_df = prepared.test_df[prepared.feature_cols].copy()
    y_train = prepared.train_df["click"].to_numpy()
    y_test = prepared.test_df["click"].to_numpy()

    if len(np.unique(y_train)) < 2:
        save_json(
            {
                "error": "Training set has only one class in click.",
                "train_policy": TRAIN_POLICY,
                "filter_position": FILTER_POSITION,
                "sample_summary": prepared.summary,
            },
            os.path.join(OUTPUT_DIR, "baseline_results.json"),
        )
        print("Training set has only one class in click.")
    else:
        preprocessor = build_preprocessor(
            numeric_cols=prepared.numeric_cols,
            categorical_cols=prepared.categorical_cols,
        )

        print("Fitting preprocessor...")
        X_train = preprocessor.fit_transform(X_train_df)
        X_test = preprocessor.transform(X_test_df)

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")

        model = train_sgd_logistic(X_train, y_train)
        results = evaluate_model(model, X_test, y_test)
        results["train_policy"] = TRAIN_POLICY
        results["filter_position"] = FILTER_POSITION
        results["feature_count"] = int(len(prepared.feature_cols))
        results["sample_size_used"] = int(len(prepared.train_df) + len(prepared.test_df))

        save_json(results, os.path.join(OUTPUT_DIR, "baseline_results.json"))
        print(json.dumps(results, ensure_ascii=False, indent=2))

    print("Done.")