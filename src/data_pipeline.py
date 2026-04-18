import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# =========================
# Column schema
# =========================

BASE_REQUIRED_COLS = ["timestamp", "item_id", "position", "click", "propensity_score"]
USER_FEATURE_COLS = [f"user_feature_{i}" for i in range(4)]
AFFINITY_COLS = [f"user-item_affinity_{i}" for i in range(80)]
ITEM_FEATURE_COLS = [f"item_feature_{i}" for i in range(4)]

# 这些列本来就应该是“数值型”，如果出现哈希字符串，就删整行
NUMERIC_REQUIRED_COLS = (
        ["item_id", "position", "click", "propensity_score"]
        + AFFINITY_COLS
        + ["item_feature_0"]
)

# 这些列即使是哈希字符串也保留，因为它们本来就是类别值
CATEGORICAL_ALLOWED_COLS = (
        USER_FEATURE_COLS
        + ["item_feature_1", "item_feature_2", "item_feature_3", "behavior_policy"]
)


# =========================
# Utilities
# =========================

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_all_usecols() -> List[str]:
    return BASE_REQUIRED_COLS + USER_FEATURE_COLS + AFFINITY_COLS


def clean_item_context(item_context_path: str) -> pd.DataFrame:
    item_df = pd.read_csv(item_context_path)

    unnamed_cols = [c for c in item_df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        item_df = item_df.drop(columns=unnamed_cols)

    # item_id 必须是数值；如果 item_context 里 item_id 本身非法，就删掉
    item_df["item_id_num"] = pd.to_numeric(item_df["item_id"], errors="coerce")
    item_df = item_df[item_df["item_id_num"].notna()].copy()
    item_df["item_id"] = item_df["item_id_num"].astype("int32")
    item_df = item_df.drop(columns=["item_id_num"])

    # item_feature_0 应该是数值；非法设为 NaN，后面可填 0
    if "item_feature_0" in item_df.columns:
        item_df["item_feature_0"] = pd.to_numeric(item_df["item_feature_0"], errors="coerce").astype("float32")

    # item_feature_1/2/3 视为类别，转成字符串保留
    for c in ["item_feature_1", "item_feature_2", "item_feature_3"]:
        if c in item_df.columns:
            item_df[c] = item_df[c].astype(str)

    return item_df


def coerce_numeric_and_drop_bad_rows(
        df: pd.DataFrame,
        numeric_cols: List[str],
) -> Tuple[pd.DataFrame, int]:
    """
    只对应该为数值的列做 coercion。
    如果某一行在这些列中出现非法字符串（例如哈希），整行删除。
    不会错位，因为删除的是整行。
    """
    work = df.copy()
    bad_mask = pd.Series(False, index=work.index)

    for col in numeric_cols:
        if col not in work.columns:
            continue
        coerced = pd.to_numeric(work[col], errors="coerce")
        # 原值非空，但转数值失败 => 这行对该数值列是非法的
        col_bad = work[col].notna() & coerced.isna()
        bad_mask = bad_mask | col_bad
        work[col] = coerced

    dropped = int(bad_mask.sum())
    work = work.loc[~bad_mask].copy()

    return work, dropped


def cast_clean_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "item_id" in out.columns:
        out["item_id"] = out["item_id"].astype("int32")
    if "position" in out.columns:
        out["position"] = out["position"].astype("int8")
    if "click" in out.columns:
        out["click"] = out["click"].astype("int8")
    if "propensity_score" in out.columns:
        out["propensity_score"] = out["propensity_score"].astype("float32")

    for c in AFFINITY_COLS:
        if c in out.columns:
            out[c] = out[c].astype("float32")

    if "item_feature_0" in out.columns:
        out["item_feature_0"] = out["item_feature_0"].astype("float32")

    # user_feature_* / item_feature_1/2/3 作为类别保留字符串
    for c in USER_FEATURE_COLS:
        if c in out.columns:
            out[c] = out[c].astype(str)

    for c in ["item_feature_1", "item_feature_2", "item_feature_3"]:
        if c in out.columns:
            out[c] = out[c].astype(str)

    if "behavior_policy" in out.columns:
        out["behavior_policy"] = out["behavior_policy"].astype(str)

    return out


# =========================
# Build clean dataset for large raw CSV
# =========================

def build_clean_dataset(
        raw_csv_path: str,
        item_context_path: str,
        output_csv_path: str,
        behavior_policy: str,
        chunksize: int = 200_000,
        filter_position: Optional[int] = None,
) -> Dict[str, Any]:
    """
    分块处理超大 CSV：
    - 只读必要列
    - 仅对“应为数值”的列检查非法值
    - 数值列里出现哈希字符串 => 删除整行
    - 类别列里的哈希字符串保留
    - merge item_context
    """
    ensure_parent_dir(output_csv_path)

    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    item_df = clean_item_context(item_context_path)

    usecols = get_all_usecols()
    first_chunk = True

    total_in = 0
    total_out = 0
    total_dropped_bad_numeric = 0
    total_dropped_missing_item_context = 0

    reader = pd.read_csv(
        raw_csv_path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=True,
    )

    for chunk_idx, chunk in enumerate(reader, start=1):
        total_in += len(chunk)

        unnamed_cols = [c for c in chunk.columns if c.startswith("Unnamed")]
        if unnamed_cols:
            chunk = chunk.drop(columns=unnamed_cols)

        if filter_position is not None:
            chunk = chunk[chunk["position"].astype(str) == str(filter_position)].copy()

        if len(chunk) == 0:
            print(f"[build_clean_dataset] {behavior_policy} chunk={chunk_idx}, skipped after position filter")
            continue

        # 这里只检查真正应为数值的原始列
        raw_numeric_cols = ["item_id", "position", "click", "propensity_score"] + AFFINITY_COLS
        chunk, dropped_bad = coerce_numeric_and_drop_bad_rows(chunk, raw_numeric_cols)
        total_dropped_bad_numeric += dropped_bad

        if len(chunk) == 0:
            print(f"[build_clean_dataset] {behavior_policy} chunk={chunk_idx}, all rows dropped due to bad numeric values")
            continue

        # merge item context
        chunk["item_id"] = chunk["item_id"].astype("int32")
        chunk = chunk.merge(item_df, on="item_id", how="left")

        # 如果 item_context 没 merge 上，也删掉，避免后面特征不完整
        before_merge_drop = len(chunk)
        chunk = chunk[chunk["item_feature_1"].notna()].copy()
        total_dropped_missing_item_context += (before_merge_drop - len(chunk))

        if len(chunk) == 0:
            print(f"[build_clean_dataset] {behavior_policy} chunk={chunk_idx}, all rows dropped after item_context merge")
            continue

        # 再检查 merge 后新增的 item_feature_0 是否非法；非法整行删
        chunk, dropped_item_f0 = coerce_numeric_and_drop_bad_rows(chunk, ["item_feature_0"])
        total_dropped_bad_numeric += dropped_item_f0

        if len(chunk) == 0:
            print(f"[build_clean_dataset] {behavior_policy} chunk={chunk_idx}, all rows dropped after item_feature_0 check")
            continue

        chunk["behavior_policy"] = behavior_policy
        chunk = cast_clean_dtypes(chunk)

        chunk.to_csv(
            output_csv_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False
        total_out += len(chunk)

        print(
            f"[build_clean_dataset] {behavior_policy} chunk={chunk_idx}, "
            f"in={total_in}, out={total_out}, dropped_bad_numeric={total_dropped_bad_numeric}, "
            f"dropped_missing_item_context={total_dropped_missing_item_context}"
        )

    summary = {
        "behavior_policy": behavior_policy,
        "raw_rows_seen": int(total_in),
        "clean_rows_written": int(total_out),
        "dropped_bad_numeric_rows": int(total_dropped_bad_numeric),
        "dropped_missing_item_context_rows": int(total_dropped_missing_item_context),
        "output_csv_path": output_csv_path,
    }
    return summary


# =========================
# Summary helpers
# =========================

def summarize_csv(
        input_csv_path: str,
        output_json_path: Optional[str] = None,
        chunksize: int = 200_000,
) -> Dict[str, Any]:
    total_rows = 0
    click_sum = 0.0
    pscore_sum = 0.0
    pscore_min = None
    pscore_max = None
    position_counts: Dict[str, int] = {}
    policy_counts: Dict[str, int] = {}
    item_ids = set()

    for chunk in pd.read_csv(input_csv_path, chunksize=chunksize, low_memory=True):
        total_rows += len(chunk)
        click_sum += pd.to_numeric(chunk["click"], errors="coerce").fillna(0).sum()
        p = pd.to_numeric(chunk["propensity_score"], errors="coerce")
        pscore_sum += p.fillna(0).sum()

        cmin = float(p.min())
        cmax = float(p.max())
        pscore_min = cmin if pscore_min is None else min(pscore_min, cmin)
        pscore_max = cmax if pscore_max is None else max(pscore_max, cmax)

        item_ids.update(pd.to_numeric(chunk["item_id"], errors="coerce").dropna().astype("int32").unique().tolist())

        for k, v in chunk["position"].value_counts().to_dict().items():
            position_counts[str(k)] = position_counts.get(str(k), 0) + int(v)

        for k, v in chunk["behavior_policy"].value_counts().to_dict().items():
            policy_counts[str(k)] = policy_counts.get(str(k), 0) + int(v)

    summary = {
        "n_rows": int(total_rows),
        "ctr": float(click_sum / total_rows) if total_rows > 0 else None,
        "n_unique_items": int(len(item_ids)),
        "position_counts": position_counts,
        "policy_counts": policy_counts,
        "propensity_min": pscore_min,
        "propensity_max": pscore_max,
        "propensity_mean": float(pscore_sum / total_rows) if total_rows > 0 else None,
    }

    if output_json_path is not None:
        ensure_parent_dir(output_json_path)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# =========================
# Reservoir sampling
# =========================

def reservoir_sample_csv(
        input_csv_path: str,
        output_sample_csv_path: str,
        sample_size: int,
        seed: int = 42,
        chunksize: int = 200_000,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    reservoir = None
    seen = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(input_csv_path, chunksize=chunksize, low_memory=True), start=1):
        chunk = chunk.reset_index(drop=True)
        n_chunk = len(chunk)

        if reservoir is None:
            take = min(sample_size, n_chunk)
            reservoir = chunk.iloc[:take].copy()
            seen += take
            start_idx = take
        else:
            start_idx = 0

        for i in range(start_idx, n_chunk):
            seen += 1
            j = rng.integers(0, seen)
            if j < sample_size:
                reservoir.iloc[j] = chunk.iloc[i]

        print(f"[reservoir_sample_csv] chunk={chunk_idx}, seen={seen}")

    if reservoir is None:
        raise ValueError(f"No data found in {input_csv_path}")

    reservoir = reservoir.reset_index(drop=True)
    ensure_parent_dir(output_sample_csv_path)
    reservoir.to_csv(output_sample_csv_path, index=False)
    return reservoir


# =========================
# Sample preparation
# =========================

@dataclass
class SamplePreparedData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    numeric_cols: List[str]
    categorical_cols: List[str]
    feature_cols: List[str]
    summary: Dict[str, Any]


def build_feature_lists_from_df(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = [c for c in AFFINITY_COLS if c in df.columns]
    if "item_feature_0" in df.columns:
        numeric_cols.append("item_feature_0")

    categorical_cols = [c for c in USER_FEATURE_COLS if c in df.columns]

    for c in ["position", "behavior_policy", "item_feature_1", "item_feature_2", "item_feature_3"]:
        if c in df.columns:
            categorical_cols.append(c)

    return numeric_cols, categorical_cols


def split_sample_df(
        sample_df: pd.DataFrame,
        test_size: float = 0.2,
        seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = sample_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if df["click"].nunique() < 2:
        n_test = int(len(df) * test_size)
        test_df = df.iloc[:n_test].copy()
        train_df = df.iloc[n_test:].copy()
        return train_df, test_df

    pos_df = df[df["click"] == 1].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    neg_df = df[df["click"] == 0].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_test_pos = int(len(pos_df) * test_size)
    n_test_neg = int(len(neg_df) * test_size)

    test_df = pd.concat(
        [pos_df.iloc[:n_test_pos], neg_df.iloc[:n_test_neg]],
        axis=0,
        ignore_index=True,
    ).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_df = pd.concat(
        [pos_df.iloc[n_test_pos:], neg_df.iloc[n_test_neg:]],
        axis=0,
        ignore_index=True,
    ).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, test_df


def prepare_sample_for_baseline(
        sample_csv_path: str,
        test_size: float = 0.2,
        seed: int = 42,
) -> SamplePreparedData:
    df = pd.read_csv(sample_csv_path, low_memory=True)
    df.insert(0, "sample_id", np.arange(len(df), dtype=np.int32))

    # 强制数值列转数值，类别列保留字符串
    for c in ["click", "position", "item_id", "propensity_score"] + AFFINITY_COLS + ["item_feature_0"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["click"] = df["click"].fillna(0).astype("int8")
    df["position"] = df["position"].astype("int8")
    df["item_id"] = df["item_id"].astype("int32")
    df["propensity_score"] = df["propensity_score"].astype("float32")

    for c in AFFINITY_COLS:
        if c in df.columns:
            df[c] = df[c].astype("float32")

    if "item_feature_0" in df.columns:
        df["item_feature_0"] = df["item_feature_0"].fillna(0).astype("float32")

    for c in USER_FEATURE_COLS + ["item_feature_1", "item_feature_2", "item_feature_3", "behavior_policy"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    numeric_cols, categorical_cols = build_feature_lists_from_df(df)
    feature_cols = numeric_cols + categorical_cols
    train_df, test_df = split_sample_df(df, test_size=test_size, seed=seed)

    summary = {
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "ctr_full": float(df["click"].mean()),
        "ctr_train": float(train_df["click"].mean()) if len(train_df) > 0 else None,
        "ctr_test": float(test_df["click"].mean()) if len(test_df) > 0 else None,
        "feature_count": int(len(feature_cols)),
        "numeric_feature_count": int(len(numeric_cols)),
        "categorical_feature_count": int(len(categorical_cols)),
    }

    return SamplePreparedData(
        train_df=train_df,
        test_df=test_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        feature_cols=feature_cols,
        summary=summary,
    )