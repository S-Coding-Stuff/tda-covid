from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1]
ENRICHED_INPUT = DATA_DIR / "foi-02477-with-obesity.csv"
CLEAN_OUTPUT = DATA_DIR / "tda_obesity_prescribing_clean.csv"
BALLMAPPER_OUTPUT = DATA_DIR / "tda_obesity_ballmapper_input.csv"

TARGET_YEAR = "Dec 22 - Nov 23"
COMPARISON_YEAR = "Dec 23 - Nov 24"
NUMERIC_COLUMNS = {
    "Items": "items",
    "Total number of unique identified patients": "patients",
    "Net Ingredient Cost (£)": "net_cost",
    "Adult obesity rate 2022/23 (%)": "obesity_rate",
}
GROUP_KEYS = ["Region Code", "Region", "Patient Age Band (Years old)", "Gender"]


def _clean_numeric(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("�", "", regex=False)
            .str.strip(),
            errors="coerce",
        )
        .fillna(0)
    )


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    for col in NUMERIC_COLUMNS:
        df[col] = _clean_numeric(df[col])
    df["Region"] = df["Region"].str.strip().str.upper()
    df["Gender"] = df["Gender"].str.strip()
    df["Patient Age Band (Years old)"] = df["Patient Age Band (Years old)"].str.strip()
    return df


def aggregate_base_metrics(df: pd.DataFrame) -> pd.DataFrame:
    sum_targets = {
        col: "sum"
        for col in NUMERIC_COLUMNS
        if col != "Adult obesity rate 2022/23 (%)"
    }
    agg_map = sum_targets | {"Adult obesity rate 2022/23 (%)": "mean"}
    grouped = (
        df.groupby(GROUP_KEYS, as_index=False)
        .agg(agg_map)
        .rename(columns={old: new for old, new in NUMERIC_COLUMNS.items()})
    )
    grouped["items_per_patient"] = grouped.apply(
        lambda row: row["items"] / row["patients"] if row["patients"] > 0 else math.nan,
        axis=1,
    )
    grouped["cost_per_item"] = grouped.apply(
        lambda row: row["net_cost"] / row["items"] if row["items"] > 0 else math.nan,
        axis=1,
    )
    grouped["cost_per_patient"] = grouped.apply(
        lambda row: row["net_cost"] / row["patients"] if row["patients"] > 0 else math.nan,
        axis=1,
    )
    totals = (
        grouped.groupby(["Region", "Patient Age Band (Years old)"])["items"]
        .transform("sum")
        .replace(0, np.nan)
    )
    grouped["gender_items_share"] = (grouped["items"] / totals).fillna(0)
    grouped["gender_is_female"] = grouped["Gender"].str.lower().eq("female").astype(float)
    grouped["gender_is_male"] = grouped["Gender"].str.lower().eq("male").astype(float)
    grouped[["items_per_patient", "cost_per_patient", "cost_per_item"]] = grouped[
        ["items_per_patient", "cost_per_patient", "cost_per_item"]
    ].fillna(0)
    return grouped


def _share(features: Dict[str, float], total: float) -> Dict[str, float]:
    if total <= 0:
        return {key: math.nan for key in features}
    return {key: value / total for key, value in features.items()}


def min_max_scale(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    scaled = df.copy()
    for col in columns:
        min_val = scaled[col].min()
        max_val = scaled[col].max()
        if pd.isna(min_val) or pd.isna(max_val) or math.isclose(min_val, max_val):
            scaled[f"norm_{col}"] = 0.5
        else:
            scaled[f"norm_{col}"] = (scaled[col] - min_val) / (max_val - min_val)
    return scaled


def compute_yoy_changes(
    current: pd.DataFrame,
    future: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    future_lookup = future.set_index(GROUP_KEYS)
    records = []
    for _, row in current.iterrows():
        key = tuple(row[col] for col in GROUP_KEYS)
        record = {col: row[col] for col in GROUP_KEYS}
        if key in future_lookup.index:
            future_row = future_lookup.loc[key]
            for col in columns:
                current_value = row[col]
                future_value = future_row[col]
                if current_value == 0:
                    record[f"{col}_yoy_pct"] = 0.0
                else:
                    record[f"{col}_yoy_pct"] = (future_value - current_value) / current_value
        else:
            for col in columns:
                record[f"{col}_yoy_pct"] = 0.0
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    df_all = load_data(ENRICHED_INPUT)
    df_current = df_all[df_all["Year"] == TARGET_YEAR].copy()
    if df_current.empty:
        raise RuntimeError(f"No rows found for {TARGET_YEAR}")
    df_future = df_all[df_all["Year"] == COMPARISON_YEAR].copy()

    base_current = aggregate_base_metrics(df_current)
    base_future = (
        aggregate_base_metrics(df_future)
        if not df_future.empty
        else base_current.iloc[:0].copy()
    )
    yoy_columns = ["items", "patients", "net_cost"]
    yoy = compute_yoy_changes(base_current, base_future, yoy_columns)

    clean = (
        base_current.merge(yoy, on=GROUP_KEYS, how="left")
        .sort_values(GROUP_KEYS)
        .reset_index(drop=True)
    )
    clean.to_csv(CLEAN_OUTPUT, index=False)

    feature_columns = [
        "items",
        "patients",
        "net_cost",
        "items_per_patient",
        "cost_per_patient",
        "cost_per_item",
        "obesity_rate",
        "items_yoy_pct",
        "patients_yoy_pct",
        "net_cost_yoy_pct",
        "gender_items_share",
    ]
    normalized = min_max_scale(clean, feature_columns)
    normalized[
        GROUP_KEYS
        + feature_columns
        + [f"norm_{c}" for c in feature_columns]
    ].to_csv(BALLMAPPER_OUTPUT, index=False)
    print(f"Wrote {CLEAN_OUTPUT} and {BALLMAPPER_OUTPUT}")


if __name__ == "__main__":
    main()
