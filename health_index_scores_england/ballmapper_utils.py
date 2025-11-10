from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "health_index_combined_2021.csv"
DEFAULT_FEATURES = [
    "Healthy People Domain",
    "Healthy Lives Domain",
    "Healthy Places Domain",
]


@dataclass
class BallMapperNode:
    index: int
    center_idx: int
    members: List[int]
    size: int
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_health_index_dataset(path: Path | None = None) -> pd.DataFrame:
    dataset_path = path or DATA_PATH
    return pd.read_csv(dataset_path)


def normalize_features(df: pd.DataFrame, feature_cols: Sequence[str], method: str = "minmax") -> Tuple[np.ndarray, List[str]]:
    if not feature_cols:
        raise ValueError("Select at least one feature for the Ball Mapper point cloud.")
    numeric_df = df.copy()
    for col in feature_cols:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
    data = numeric_df[feature_cols].to_numpy(dtype=float)
    if method == "zscore":
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        std[std == 0] = 1.0
        normalized = (data - mean) / std
    else:
        min_vals = np.nanmin(data, axis=0)
        max_vals = np.nanmax(data, axis=0)
        denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
        normalized = (data - min_vals) / denom
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized, list(feature_cols)


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
) -> List[BallMapperNode]:
    numeric_cols = [
        col
        for col in df.columns
        if col not in {"Source", "Year", "Area Code", "Area Name", "Area Type"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    nodes: List[BallMapperNode] = []
    for node_idx, (center_idx, members) in enumerate(zip(centers, cover)):
        member_array = np.asarray(members, dtype=int)
        metrics: Dict[str, float] = {}
        for col in numeric_cols:
            values = pd.to_numeric(df.iloc[member_array][col], errors="coerce")
            metrics[col] = float(np.nanmean(values)) if not values.isna().all() else 0.0
        row = df.iloc[center_idx]
        metadata = {
            "area_code": row.get("Area Code", ""),
            "area_name": row.get("Area Name", ""),
            "area_type": row.get("Area Type", ""),
            "source": row.get("Source", ""),
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


def compute_layout(G: nx.Graph) -> dict[int, tuple[float, float]]:
    if len(G) == 0:
        return {}
    counts = [G.nodes[n]["size"] for n in G.nodes]
    avg = np.mean(counts) if counts else 1.0
    max_count = max(counts) if counts else 1.0
    base_k = 1 / np.sqrt(len(G))
    k = base_k * (1 + avg / max_count)
    try:
        return nx.spring_layout(G, seed=42, k=k, weight=None)
    except Exception:
        return nx.spectral_layout(G)


def nodes_to_dataframe(nodes: Sequence[BallMapperNode], labels: Sequence[str]) -> pd.DataFrame:
    records = []
    for node in nodes:
        record = {
            "node_id": node.index,
            "label": labels[node.center_idx],
            "size": node.size,
            "members": node.members,
        }
        record.update(node.metrics)
        record.update(node.metadata)
        records.append(record)
    return pd.DataFrame(records)


def compose_labels(df: pd.DataFrame) -> List[str]:
    def _label(row: pd.Series) -> str:
        base = row.get("Area Name", "")
        type_part = row.get("Area Type", "")
        return f"{base} ({type_part})" if type_part else base

    return df.apply(_label, axis=1).tolist()
