from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1]
CLEAN_INPUT = DATA_DIR / "tda_obesity_prescribing_clean.csv"
FIG_DIR = DATA_DIR / "figures"


def load_dataset() -> pd.DataFrame:
    if not CLEAN_INPUT.exists():
        raise FileNotFoundError(
            f"{CLEAN_INPUT} is missing. Run dataset_experiments/scripts/preprocess_data.py first."
        )
    df = pd.read_csv(CLEAN_INPUT)
    yoy_cols = [col for col in df.columns if col.endswith("_yoy_pct")]
    df[yoy_cols] = df[yoy_cols].fillna(0)
    return df.sort_values(["Region", "Patient Age Band (Years old)", "Gender"]).reset_index(drop=True)


def ensure_fig_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def scatter_cost_vs_obesity(df: pd.DataFrame) -> None:
    ensure_fig_dir()
    sizes = 250 + 1500 * (df["patients"] / df["patients"].max())
    cmap_values = df["items_per_patient"]
    labels = (
        df["Region"]
        + " / "
        + df["Patient Age Band (Years old)"]
        + " / "
        + df["Gender"]
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        df["obesity_rate"],
        df["cost_per_patient"],
        s=sizes,
        c=cmap_values,
        cmap="plasma",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    for text, x, y in zip(labels, df["obesity_rate"], df["cost_per_patient"]):
        ax.annotate(text, (x, y), fontsize=7, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Adult obesity rate (%)")
    ax.set_ylabel("Cost per patient (£)")
    ax.set_title("Cost vs obesity (size = patients, color = items/patient)")
    cbar = fig.colorbar(scatter, ax=ax, label="Items per patient")
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "scatter_cost_vs_obesity.png", dpi=300)
    plt.close(fig)


def stacked_gender_shares(df: pd.DataFrame) -> None:
    ensure_fig_dir()
    pivot = (
        df.groupby(["Region", "Gender"])["items"]
        .sum()
        .reset_index()
        .pivot(index="Region", columns="Gender", values="items")
        .fillna(0)
    )
    shares = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
    genders = list(shares.columns)
    colors = plt.cm.Set2(np.linspace(0, 1, len(genders)))

    fig, ax = plt.subplots(figsize=(8, 4))
    bottoms = np.zeros(len(shares))
    x = np.arange(len(shares))
    for gender, color in zip(genders, colors):
        ax.bar(
            x,
            shares[gender].to_numpy() * 100,
            bottom=bottoms,
            label=gender,
            color=color,
        )
        bottoms += shares[gender].to_numpy() * 100
    ax.set_xticks(x, shares.index, rotation=45, ha="right")
    ax.set_ylabel("Share of prescription items (%)")
    ax.set_title("Gender distribution of prescription items by region")
    ax.legend(frameon=False, ncols=3, loc="upper center")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "stacked_gender_shares.png", dpi=300)
    plt.close(fig)


def age_share_heatmap(df: pd.DataFrame) -> None:
    ensure_fig_dir()
    table = (
        df.groupby(["Region", "Patient Age Band (Years old)"])["items"]
        .sum()
        .unstack(fill_value=0)
    )
    shares = table.div(table.sum(axis=1), axis=0).fillna(0)
    age_labels = shares.columns.tolist()
    matrix = shares.to_numpy() * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(age_labels)))
    ax.set_xticklabels(age_labels, rotation=45)
    ax.set_yticks(range(len(shares.index)))
    ax.set_yticklabels(shares.index)
    ax.set_title("Prescription item share by age band (%) per region")
    cbar = fig.colorbar(im, ax=ax, label="Share (%)")
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "age_share_heatmap.png", dpi=300)
    plt.close(fig)


def patients_vs_items(df: pd.DataFrame) -> None:
    ensure_fig_dir()
    agg = df.groupby(["Region", "Patient Age Band (Years old)", "Gender"], as_index=False)[
        ["patients", "items"]
    ].sum()
    labels = (
        agg["Region"]
        + " / "
        + agg["Patient Age Band (Years old)"]
        + " / "
        + agg["Gender"]
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(agg["patients"], agg["items"], color="#3182bd", s=120)
    coeffs = np.polyfit(agg["patients"], agg["items"], deg=1)
    x_vals = np.linspace(agg["patients"].min(), agg["patients"].max(), 100)
    ax.plot(x_vals, coeffs[0] * x_vals + coeffs[1], color="#9e9ac8", linestyle="--", label="Best fit")
    for text, x, y in zip(labels, agg["patients"], agg["items"]):
        ax.annotate(text, (x, y), fontsize=7, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Unique patients (region-age-gender)")
    ax.set_ylabel("Prescription items")
    ax.set_title("Patients vs items by region-age-gender")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "patients_vs_items.png", dpi=300)
    plt.close(fig)


def obesity_vs_items_per_patient(df: pd.DataFrame) -> None:
    ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(
        df["obesity_rate"],
        df["items_per_patient"],
        c=df["net_cost"],
        cmap="inferno",
        s=140,
    )
    labels = (
        df["Region"]
        + " / "
        + df["Patient Age Band (Years old)"]
        + " / "
        + df["Gender"]
    )
    for text, x, y in zip(labels, df["obesity_rate"], df["items_per_patient"]):
        ax.annotate(text, (x, y), fontsize=7, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Adult obesity rate (%)")
    ax.set_ylabel("Items per patient")
    ax.set_title("Items per patient vs obesity rate (color = total cost)")
    fig.colorbar(scatter, ax=ax, label="Net cost (£)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "items_per_patient_vs_obesity.png", dpi=300)
    plt.close(fig)


def yoy_bar_chart(df: pd.DataFrame) -> None:
    ensure_fig_dir()
    metrics = ["items_yoy_pct", "patients_yoy_pct", "net_cost_yoy_pct"]
    labels = ["Items YoY %", "Patients YoY %", "Net Cost YoY %"]
    grouped = (
        df.groupby(["Region", "Patient Age Band (Years old)"], as_index=False)[metrics].mean()
    )
    grouped["label"] = grouped["Region"] + " / " + grouped["Patient Age Band (Years old)"]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    x = np.arange(len(grouped))
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(
            x + idx * width - width,
            grouped[metric] * 100,
            width=width,
            label=label,
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, grouped["label"], rotation=90, ha="right")
    ax.set_ylabel("Year-on-year change (%)")
    ax.set_title("Year-on-year shifts by region & age")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "yoy_changes.png", dpi=300)
    plt.close(fig)


def gender_share_chart(df: pd.DataFrame) -> None:
    ensure_fig_dir()
    female = df[df["Gender"].str.lower() == "female"].copy()
    if female.empty:
        return
    female["label"] = (
        female["Region"] + " / " + female["Patient Age Band (Years old)"]
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(female["label"], female["gender_items_share"] * 100, color="#c51b8a")
    ax.set_xlabel("Share of items for female patients (%)")
    ax.set_title("Female contribution within each region-age group")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "gender_items_share.png", dpi=300)
    plt.close(fig)


def main() -> None:
    df = load_dataset()
    scatter_cost_vs_obesity(df)
    stacked_gender_shares(df)
    age_share_heatmap(df)
    patients_vs_items(df)
    obesity_vs_items_per_patient(df)
    yoy_bar_chart(df)
    gender_share_chart(df)
    print(f"Generated figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
