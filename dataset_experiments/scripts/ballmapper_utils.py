from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1]
BALLMAPPER_INPUT = DATA_DIR / "tda_obesity_ballmapper_input.csv"
GROUP_KEYS = ["Region Code", "Region", "Patient Age Band (Years old)", "Gender"]

SUMMARY_COLUMNS = [
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
    "share_dulaglutide",
    "share_exenatide",
    "share_insulin_combo",
    "share_liraglutide",
    "share_lixisenatide",
    "share_semaglutide",
    "share_tirzepatide",
]

FEATURE_COLUMNS = [
    "norm_items",
    "norm_patients",
    "norm_net_cost",
    "norm_items_per_patient",
    "norm_cost_per_patient",
    "norm_cost_per_item",
    "norm_obesity_rate",
    "norm_items_yoy_pct",
    "norm_patients_yoy_pct",
    "norm_net_cost_yoy_pct",
    "norm_gender_items_share",
]

SUMMARY_COLUMNS = [
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


@dataclass
class BallMapperNode:
    index: int
    center_idx: int
    members: List[int]
    size: int
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_obesity_rate(self) -> float:
        return self.metrics.get("obesity_rate", float("nan"))

    @property
    def mean_cost_per_patient(self) -> float:
        return self.metrics.get("cost_per_patient", float("nan"))


def load_features(
    path: Path | None = None, feature_columns: Sequence[str] | None = None
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    dataset_path = path or BALLMAPPER_INPUT
    df = pd.read_csv(dataset_path)
    for col in SUMMARY_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    cols = list(feature_columns or FEATURE_COLUMNS)
    if not set(cols).issubset(df.columns):
        raise ValueError("Normalized columns are missing; run preprocess_data.py first.")
    X = df[cols].to_numpy(dtype=float)
    def compose_label(row: pd.Series) -> str:
        region = str(row.get("Region", "")).split()[0].title() if row.get("Region") else ""
        age_band = str(row.get("Patient Age Band (Years old)", "")).replace("Years old", "").strip()
        gender_val = str(row.get("Gender", ""))
        if gender_val.lower().startswith("f"):
            gender = "F"
        elif gender_val.lower().startswith("m"):
            gender = "M"
        elif gender_val:
            gender = "U"
        else:
            gender = ""
        left = "-".join(part for part in [region, gender] if part)
        if left and age_band:
            return f"{left}/{age_band}"
        return left or age_band or "Node"

    labels = df.apply(compose_label, axis=1).tolist()
    return df, X, labels


def compute_ball(points: np.ndarray, center: np.ndarray, radius: float) -> List[int]:
    distances = np.linalg.norm(points - center, axis=1)
    return sorted(np.where(distances <= radius)[0].tolist())


def build_cover(points: np.ndarray, radius: float) -> Tuple[List[int], List[List[int]]]:
    centers: List[int] = []
    cover: List[List[int]] = []
    for idx in range(points.shape[0]):
        if not centers:
            centers.append(idx)
            cover.append(compute_ball(points, points[idx], radius))
            continue
        distances = np.linalg.norm(points[centers] - points[idx], axis=1)
        if (distances <= radius).any():
            continue
        centers.append(idx)
        cover.append(compute_ball(points, points[idx], radius))
    return centers, cover


def build_nodes(
    centers: Sequence[int],
    cover: Sequence[Sequence[int]],
    df: pd.DataFrame,
    summary_columns: Iterable[str] | None = None,
) -> List[BallMapperNode]:
    if summary_columns is None:
        numeric_cols = [col for col in df.columns if col not in GROUP_KEYS and pd.api.types.is_numeric_dtype(df[col])]
    else:
        numeric_cols = list(summary_columns)
    nodes: List[BallMapperNode] = []
    for node_idx, (center_idx, members) in enumerate(zip(centers, cover)):
        member_array = np.asarray(members, dtype=int)
        metrics = {}
        for col in numeric_cols:
            if col not in df.columns:
                metrics[col] = 0.0
                continue
            values = df.iloc[member_array][col]
            try:
                numeric = pd.to_numeric(values, errors="coerce")
                metrics[col] = float(np.nanmean(numeric))
            except Exception:
                metrics[col] = 0.0
        row = df.iloc[center_idx]
        metadata = {
            "region_name": row.get("Region", ""),
            "age_band": row.get("Patient Age Band (Years old)", ""),
            "gender_label": row.get("Gender", ""),
        }
        nodes.append(
            BallMapperNode(
                index=node_idx,
                center_idx=center_idx,
                members=list(members),
                size=len(members),
                metrics=metrics,
                metadata=metadata,
            )
        )
    return nodes


def build_graph(nodes: Sequence[BallMapperNode], labels: Sequence[str]) -> nx.Graph:
    G = nx.Graph()
    for node in nodes:
        attrs = {
            "members": node.members,
            "size": node.size,
            "label": labels[node.center_idx],
        }
        attrs.update(node.metrics)
        attrs.update(node.metadata)
        G.add_node(node.index, **attrs)
    for i in range(len(nodes)):
        members_i = set(nodes[i].members)
        for j in range(i + 1, len(nodes)):
            if members_i.intersection(nodes[j].members):
                G.add_edge(i, j)
    return G


def nodes_to_dataframe(nodes: Sequence[BallMapperNode], labels: Sequence[str]) -> pd.DataFrame:
    records = []
    for node in nodes:
        record = {
            "node_id": node.index,
            "region": labels[node.center_idx],
            "size": node.size,
            "members": node.members,
        }
        record.update(node.metrics)
        record.update(node.metadata)
        records.append(record)
    return pd.DataFrame(records)
