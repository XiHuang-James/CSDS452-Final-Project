from pathlib import Path
from typing import Dict, Any
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# Utility: logging / progress printing
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
# Core preprocessing function
# ============================================================

def build_obd_dataset(
    all_csv_path: Path,
    item_context_csv_path: Path,
    output_dir: Path,
    dataset_name: str,
    propensity_clip_min: float = 1e-3,
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Preprocess one OBD dataset (e.g., bts or random) and save processed
    model inputs to local disk.

    Final outputs can be directly used for:
    - baseline model
    - IPS-weighted model
    - SNIPS-weighted model
    """

    total_steps = 12
    t0 = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load raw CSVs
    # ------------------------------------------------------------------
    log_step(1, total_steps, f"Loading raw CSV files for dataset = {dataset_name}")
    log_sub(f"all.csv path           : {all_csv_path}")
    log_sub(f"item_context.csv path  : {item_context_csv_path}")

    all_df = pd.read_csv(all_csv_path)
    item_df = pd.read_csv(item_context_csv_path)

    log_sub(f"Loaded all.csv shape          = {all_df.shape}")
    log_sub(f"Loaded item_context.csv shape = {item_df.shape}")

    # ------------------------------------------------------------------
    # Step 2: Drop useless unnamed index columns
    # ------------------------------------------------------------------
    log_step(2, total_steps, "Dropping auto-generated unnamed columns")

    def drop_unnamed(df: pd.DataFrame, name: str) -> pd.DataFrame:
        unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
        if unnamed_cols:
            log_sub(f"{name}: dropping unnamed columns -> {unnamed_cols}")
            df = df.drop(columns=unnamed_cols)
        else:
            log_sub(f"{name}: no unnamed columns found")
        return df

    all_df = drop_unnamed(all_df, "all.csv")
    item_df = drop_unnamed(item_df, "item_context.csv")

    log_sub(f"all.csv shape after cleanup          = {all_df.shape}")
    log_sub(f"item_context.csv shape after cleanup = {item_df.shape}")

    # ------------------------------------------------------------------
    # Step 3: Merge item_context into log data
    # ------------------------------------------------------------------
    log_step(3, total_steps, "Merging item_context.csv into all.csv by item_id")

    if "item_id" not in all_df.columns:
        raise ValueError("`item_id` not found in all.csv")
    if "item_id" not in item_df.columns:
        raise ValueError("`item_id` not found in item_context.csv")

    df = all_df.merge(item_df, on="item_id", how="left", validate="many_to_one")

    log_sub(f"Merged dataframe shape = {df.shape}")
    missing_item_context_rows = df["item_feature_0"].isna().sum() if "item_feature_0" in df.columns else -1
    if missing_item_context_rows >= 0:
        log_sub(f"Rows with missing merged item features = {missing_item_context_rows}")

    # ------------------------------------------------------------------
    # Step 4: Remove timestamp from model features
    # ------------------------------------------------------------------
    log_step(4, total_steps, "Dropping timestamp (not used in current project)")

    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
        log_sub("Dropped column: timestamp")
    else:
        log_sub("timestamp not found; skipping")

    # ------------------------------------------------------------------
    # Step 5: Check expected columns
    # ------------------------------------------------------------------
    log_step(5, total_steps, "Checking required columns and defining feature groups")

    required_cols = ["item_id", "position", "click", "propensity_score"]
    user_feature_cols = [f"user_feature_{i}" for i in range(4)]
    affinity_cols = [f"user-item_affinity_{i}" for i in range(80)]
    item_feature_numeric_cols = ["item_feature_0"]
    item_feature_categorical_cols = [f"item_feature_{i}" for i in range(1, 4)]

    expected_cols = (
        required_cols
        + user_feature_cols
        + affinity_cols
        + item_feature_numeric_cols
        + item_feature_categorical_cols
    )

    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Expected columns missing after merge: {missing_cols}")

    onehot_cols = ["item_id", "position"]

    high_cardinality_cat_cols = user_feature_cols + item_feature_categorical_cols

    base_numeric_cols = affinity_cols + item_feature_numeric_cols

    encoded_high_card_cols = [f"{col}_code" for col in high_cardinality_cat_cols]
    numeric_cols = base_numeric_cols + encoded_high_card_cols

    log_sub(f"One-hot feature count              = {len(onehot_cols)}")
    log_sub(f"High-cardinality categorical count = {len(high_cardinality_cat_cols)}")
    log_sub(f"Numeric feature count              = {len(base_numeric_cols)}")
    log_sub(f"Encoded high-card numeric count    = {len(encoded_high_card_cols)}")
    log_sub(f"Total raw model features           = {len(onehot_cols) + len(numeric_cols)}")

    # ------------------------------------------------------------------
    # Step 6: Clean dtypes
    # ------------------------------------------------------------------
    log_step(6, total_steps, "Cleaning data types")

    df["click"] = pd.to_numeric(df["click"], errors="coerce").fillna(0).astype(np.int8)

    df["propensity_score"] = pd.to_numeric(
        df["propensity_score"], errors="coerce"
    ).astype(np.float32)

    for col in onehot_cols:
        df[col] = df[col].astype(str)

    for col in base_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    for col in high_cardinality_cat_cols:
        df[col] = df[col].astype(str)
        df[f"{col}_code"] = df[col].astype("category").cat.codes.astype(np.int32)

    log_sub("Converted click to int8")
    log_sub("Converted propensity_score to float32")
    log_sub("Converted one-hot columns to string")
    log_sub("Converted base numeric features to float32")
    log_sub("Converted high-cardinality categorical columns to int32 category codes")

    # ------------------------------------------------------------------
    # Step 7: Missing value summary
    # ------------------------------------------------------------------
    log_step(7, total_steps, "Inspecting missing values and basic dataset statistics")

    missing_summary = df.isna().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

    log_sub(f"Number of rows = {len(df)}")
    log_sub(f"Click positive rate = {df['click'].mean():.6f}")

    if len(missing_summary) > 0:
        log_sub("Columns with missing values:")
        for col, cnt in missing_summary.head(20).items():
            log_sub(f"  {col}: {cnt}")
        if len(missing_summary) > 20:
            log_sub(f"  ... and {len(missing_summary) - 20} more columns")
    else:
        log_sub("No missing values detected")

    # ------------------------------------------------------------------
    # Step 8: Create propensity-based weights
    # ------------------------------------------------------------------
    log_step(8, total_steps, "Creating propensity-based training weights")

    df["propensity_missing_or_nonpositive"] = (
        df["propensity_score"].isna() | (df["propensity_score"] <= 0)
    ).astype(np.int8)

    num_bad_propensity = int(df["propensity_missing_or_nonpositive"].sum())
    log_sub(f"Rows with missing/non-positive propensity = {num_bad_propensity}")

    df["propensity_score_filled"] = df["propensity_score"].fillna(propensity_clip_min)
    df["propensity_score_clipped"] = df["propensity_score_filled"].clip(
        lower=propensity_clip_min
    )

    w_baseline = np.ones(len(df), dtype=np.float32)
    w_ips = (1.0 / df["propensity_score_clipped"].values).astype(np.float32)
    w_snips = (w_ips / np.mean(w_ips)).astype(np.float32)

    log_sub(f"propensity clip min = {propensity_clip_min}")
    log_sub(f"IPS weight stats: min={w_ips.min():.6f}, max={w_ips.max():.6f}, mean={w_ips.mean():.6f}")
    log_sub(f"SNIPS weight stats: min={w_snips.min():.6f}, max={w_snips.max():.6f}, mean={w_snips.mean():.6f}")

    # ------------------------------------------------------------------
    # Step 9: Build raw X and y
    # ------------------------------------------------------------------
    log_step(9, total_steps, "Building raw feature table X_raw and label y")

    final_feature_cols = onehot_cols + numeric_cols
    X_raw = df[final_feature_cols].copy()
    y = df["click"].values.astype(np.float32)

    log_sub(f"X_raw shape = {X_raw.shape}")
    log_sub(f"y shape     = {y.shape}")

    # ------------------------------------------------------------------
    # Step 10: Fit preprocessing pipeline and transform features
    # ------------------------------------------------------------------
    log_step(10, total_steps, "Applying feature engineering pipeline (detailed progress)")

    # ------------------------------------------------
    # Step 10.1 Extract feature subsets
    # ------------------------------------------------
    log_sub("[10.1] Splitting raw feature table into one-hot and numeric parts")

    t0 = time.time()

    X_onehot_raw = X_raw[onehot_cols]
    X_numeric_raw = X_raw[numeric_cols]

    log_sub(f"one-hot raw shape  = {X_onehot_raw.shape}")
    log_sub(f"numeric raw shape  = {X_numeric_raw.shape}")

    log_sub(f"[10.1] Done in {time.time() - t0:.2f} seconds")


    # ------------------------------------------------
    # Step 10.2 Fit one-hot encoder
    # ------------------------------------------------
    log_sub("[10.2] Fitting one-hot encoder")

    t0 = time.time()

    onehot_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    onehot_pipeline.fit(X_onehot_raw)

    log_sub(f"[10.2] Finished fitting one-hot encoder in {time.time() - t0:.2f} seconds")


    # ------------------------------------------------
    # Step 10.3 Transform one-hot features
    # ------------------------------------------------
    log_sub("[10.3] Transforming one-hot features")

    t0 = time.time()

    X_onehot = onehot_pipeline.transform(X_onehot_raw)

    log_sub(f"one-hot matrix shape = {X_onehot.shape}")
    log_sub(f"[10.3] Finished transforming one-hot features in {time.time() - t0:.2f} seconds")
    del X_onehot_raw

    # ------------------------------------------------
    # Step 10.4 Fit numeric transformer
    # ------------------------------------------------
    log_sub("[10.4] Preparing numeric features without sklearn fit (memory-safe mode)")

    t0 = time.time()

    X_numeric_array = X_numeric_raw.to_numpy(dtype=np.float32, copy=True)
    X_numeric_array = np.nan_to_num(X_numeric_array, nan=0.0, posinf=0.0, neginf=0.0)

    log_sub(f"[10.4] Finished numeric array preparation in {time.time() - t0:.2f} seconds")
    log_sub(f"numeric dense array shape = {X_numeric_array.shape}")

    # ------------------------------------------------
    # Step 10.5 Transform numeric features
    # ------------------------------------------------
    log_sub("[10.5] Converting numeric array to sparse matrix")

    t0 = time.time()

    X_numeric = sparse.csr_matrix(X_numeric_array, dtype=np.float32)

    log_sub(f"numeric matrix shape = {X_numeric.shape}")
    log_sub(f"[10.5] Finished converting numeric features in {time.time() - t0:.2f} seconds")
    del X_numeric_array
    del X_numeric_raw

    # ------------------------------------------------
    # Step 10.6 Combine feature matrices
    # ------------------------------------------------
    log_sub("[10.6] Combining one-hot and numeric matrices")

    t0 = time.time()

    X = sparse.hstack([X_onehot, X_numeric], format="csr", dtype=np.float32)

    log_sub(f"final feature matrix shape = {X.shape}")
    log_sub(f"[10.6] Finished combining matrices in {time.time() - t0:.2f} seconds")
    del X_onehot
    del X_numeric

    # ------------------------------------------------
    # Step 10.7 Generate feature names
    # ------------------------------------------------
    log_sub("[10.7] Generating feature names")

    try:
        onehot_feature_names = onehot_pipeline.named_steps["onehot"].get_feature_names_out(onehot_cols)
    except Exception:
        onehot_feature_names = [f"onehot_{i}" for i in range(X_onehot.shape[1])]

    numeric_feature_names = numeric_cols

    feature_names = np.concatenate([onehot_feature_names, numeric_feature_names])

    log_sub(f"total feature count = {len(feature_names)}")

    # ------------------------------------------------------------------
    # Step 11: Train/test split
    # ------------------------------------------------------------------
    log_step(11, total_steps, "Splitting processed data into train/test sets")

    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    X_train = X[idx_train]
    X_test = X[idx_test]

    y_train = y[idx_train]
    y_test = y[idx_test]

    w_baseline_train = w_baseline[idx_train]
    w_baseline_test = w_baseline[idx_test]

    w_ips_train = w_ips[idx_train]
    w_ips_test = w_ips[idx_test]

    w_snips_train = w_snips[idx_train]
    w_snips_test = w_snips[idx_test]

    log_sub(f"Train size = {len(idx_train)}")
    log_sub(f"Test size  = {len(idx_test)}")
    log_sub(f"Train click rate = {y_train.mean():.6f}")
    log_sub(f"Test click rate  = {y_test.mean():.6f}")

    # ------------------------------------------------------------------
    # Step 12: Save all outputs
    # ------------------------------------------------------------------
    log_step(12, total_steps, f"Saving processed outputs to local directory: {output_dir}")

    metadata = {
        "dataset_name": dataset_name,
        "n_rows": int(len(df)),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "n_features_after_transform": int(X.shape[1]),
        "click_rate_overall": float(df["click"].mean()),
        "click_rate_train": float(y_train.mean()),
        "click_rate_test": float(y_test.mean()),
        "propensity_clip_min": float(propensity_clip_min),
        "num_rows_propensity_missing_or_nonpositive": num_bad_propensity,
        "onehot_cols": onehot_cols,
        "high_cardinality_cat_cols": high_cardinality_cat_cols,
        "base_numeric_cols": base_numeric_cols,
        "encoded_high_card_cols": encoded_high_card_cols,
        "numeric_cols": numeric_cols,
        "final_feature_cols": final_feature_cols,
        "all_csv_path": str(all_csv_path),
        "item_context_csv_path": str(item_context_csv_path),
        "output_dir": str(output_dir),
        "random_state": random_state,
        "test_size": test_size,
    }

    if save_artifacts:
        # Save merged engineered dataframe
        merged_csv_path = output_dir / "merged_engineered_dataframe.csv"
        df.to_csv(merged_csv_path, index=False)
        log_sub(f"Saved merged dataframe -> {merged_csv_path}")

        # Save full feature matrix
        X_path = output_dir / "X_full.npz"
        sparse.save_npz(X_path, X if sparse.issparse(X) else sparse.csr_matrix(X))
        log_sub(f"Saved full transformed X -> {X_path}")

        # Save train/test matrices
        X_train_path = output_dir / "X_train.npz"
        X_test_path = output_dir / "X_test.npz"
        sparse.save_npz(X_train_path, X_train if sparse.issparse(X_train) else sparse.csr_matrix(X_train))
        sparse.save_npz(X_test_path, X_test if sparse.issparse(X_test) else sparse.csr_matrix(X_test))
        log_sub(f"Saved X_train -> {X_train_path}")
        log_sub(f"Saved X_test  -> {X_test_path}")

        # Save labels
        np.save(output_dir / "y_full.npy", y)
        np.save(output_dir / "y_train.npy", y_train)
        np.save(output_dir / "y_test.npy", y_test)
        log_sub("Saved y_full.npy / y_train.npy / y_test.npy")

        # Save weights
        np.save(output_dir / "w_baseline_full.npy", w_baseline)
        np.save(output_dir / "w_baseline_train.npy", w_baseline_train)
        np.save(output_dir / "w_baseline_test.npy", w_baseline_test)

        np.save(output_dir / "w_ips_full.npy", w_ips)
        np.save(output_dir / "w_ips_train.npy", w_ips_train)
        np.save(output_dir / "w_ips_test.npy", w_ips_test)

        np.save(output_dir / "w_snips_full.npy", w_snips)
        np.save(output_dir / "w_snips_train.npy", w_snips_train)
        np.save(output_dir / "w_snips_test.npy", w_snips_test)
        log_sub("Saved baseline / IPS / SNIPS weight files")

        # Save indices
        np.save(output_dir / "idx_train.npy", idx_train)
        np.save(output_dir / "idx_test.npy", idx_test)
        log_sub("Saved train/test indices")

        # Save feature names
        feature_names_df = pd.DataFrame({"feature_name": feature_names})
        feature_names_path = output_dir / "feature_names.csv"
        feature_names_df.to_csv(feature_names_path, index=False)
        log_sub(f"Saved feature names -> {feature_names_path}")

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        log_sub(f"Saved metadata -> {metadata_path}")

    elapsed = time.time() - t0
    log_sub(f"Finished preprocessing dataset [{dataset_name}] in {elapsed:.2f} seconds")

    return {
        "df_merged": df,
        "X": X,
        "X_train": X_train,
        "X_test": X_test,
        "y": y,
        "y_train": y_train,
        "y_test": y_test,
        "w_baseline": w_baseline,
        "w_baseline_train": w_baseline_train,
        "w_baseline_test": w_baseline_test,
        "w_ips": w_ips,
        "w_ips_train": w_ips_train,
        "w_ips_test": w_ips_test,
        "w_snips": w_snips,
        "w_snips_train": w_snips_train,
        "w_snips_test": w_snips_test,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "preprocessor": {
            "onehot_pipeline": onehot_pipeline,
            "numeric_pipeline": None,
            "onehot_cols": onehot_cols,
            "numeric_cols": numeric_cols,
        },
        "feature_names": feature_names,
        "metadata": metadata,
    }


# ============================================================
# Helper: process one dataset folder
# ============================================================

def process_dataset_folder(
    dataset_dir: Path,
    output_base_dir: Path,
    dataset_name: str,
    propensity_clip_min: float = 1e-3,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Process one dataset folder such as:
        data/bts/
        data/random/
    """
    all_csv_path = dataset_dir / "all.csv"
    item_context_csv_path = dataset_dir / "item_context.csv"
    output_dir = output_base_dir / f"processed_{dataset_name}"

    if not all_csv_path.exists():
        raise FileNotFoundError(f"File not found: {all_csv_path}")
    if not item_context_csv_path.exists():
        raise FileNotFoundError(f"File not found: {item_context_csv_path}")

    return build_obd_dataset(
        all_csv_path=all_csv_path,
        item_context_csv_path=item_context_csv_path,
        output_dir=output_dir,
        dataset_name=dataset_name,
        propensity_clip_min=propensity_clip_min,
        test_size=test_size,
        random_state=random_state,
        save_artifacts=True,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Expected project structure:

    CSDS452-FINAL-PROJECT/
    ├── data/
    │   ├── bts/
    │   │   ├── all.csv
    │   │   └── item_context.csv
    │   └── random/
    │       ├── all.csv
    │       └── item_context.csv
    ├── outputs/
    └── src/
        └── preprocess.py
    """

    script_path = Path(__file__).resolve()
    src_dir = script_path.parent
    project_root = src_dir.parent

    data_dir = project_root / "data"
    output_base_dir = project_root / "outputs"

    bts_dir = data_dir / "bts"
    random_dir = data_dir / "random"

    output_base_dir.mkdir(parents=True, exist_ok=True)

    log("\n" + "#" * 80)
    log("OBD PREPROCESSING STARTED")
    log("#" * 80)
    log(f"Script path   : {script_path}")
    log(f"Project root  : {project_root}")
    log(f"Data dir      : {data_dir}")
    log(f"Output dir    : {output_base_dir}")

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # You can choose whether to process both datasets or only bts
    datasets_to_process = [
        ("bts", bts_dir),
        ("random", random_dir),
    ]

    summary_rows = []
    total_start = time.time()

    for dataset_name, dataset_dir in datasets_to_process:
        log("\n" + "#" * 80)
        log(f"NOW PROCESSING DATASET: {dataset_name}")
        log("#" * 80)

        result = process_dataset_folder(
            dataset_dir=dataset_dir,
            output_base_dir=output_base_dir,
            dataset_name=dataset_name,
            propensity_clip_min=1e-3,
            test_size=0.2,
            random_state=42,
        )

        summary_rows.append({
            "dataset_name": dataset_name,
            "n_rows": result["metadata"]["n_rows"],
            "n_train": result["metadata"]["n_train"],
            "n_test": result["metadata"]["n_test"],
            "n_features_after_transform": result["metadata"]["n_features_after_transform"],
            "click_rate_overall": result["metadata"]["click_rate_overall"],
            "bad_propensity_rows": result["metadata"]["num_rows_propensity_missing_or_nonpositive"],
            "output_dir": result["metadata"]["output_dir"],
        })

    total_elapsed = time.time() - total_start

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_base_dir / "preprocess_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    log("\n" + "#" * 80)
    log("ALL PREPROCESSING FINISHED")
    log("#" * 80)
    log(summary_df.to_string(index=False))
    log(f"\nSaved summary -> {summary_csv_path}")
    log(f"Total elapsed time: {total_elapsed:.2f} seconds")
    log("#" * 80 + "\n")


if __name__ == "__main__":
    main()