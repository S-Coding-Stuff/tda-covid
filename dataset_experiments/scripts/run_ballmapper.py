from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from dataset_experiments.scripts.ballmapper_utils import (
    DATA_DIR,
    build_cover,
    build_graph,
    build_nodes,
    load_features,
    nodes_to_dataframe,
)

FIGURES_DIR = DATA_DIR / "figures"
GRAPH_JSON = DATA_DIR / "ballmapper_graph.json"


def save_graph_json(node_df: pd.DataFrame, path: Path) -> None:
    payload = {
        "nodes": [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"members"} or isinstance(value, list)
                },
                "members": row["members"],
            }
            for row in node_df.to_dict(orient="records")
        ]
    }
    path.write_text(json.dumps(payload, indent=2))


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


def plot_graph(G: nx.Graph, metric: str, output_path: Path, title: str) -> None:
    pos = compute_layout(G)
    colors = [G.nodes[n][metric] for n in G.nodes]
    counts = [G.nodes[n]["size"] for n in G.nodes]
    if counts:
        sizes = np.interp(counts, [min(counts), max(counts)], [60, 300])
    else:
        sizes = []

    fig, ax = plt.subplots(figsize=(7, 6))
    dark_bg = (0.06, 0.06, 0.06)
    fig.patch.set_facecolor(dark_bg)
    ax.set_facecolor(dark_bg)
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=sizes,
        node_color=colors,
        cmap="viridis",
        ax=ax,
        vmin=min(colors),
        vmax=max(colors),
        linewidths=0.5,
        edgecolors="white",
    )
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, edge_color="#cccccc")
    label_candidates = sorted(G.nodes, key=lambda n: G.nodes[n]["size"], reverse=True)
    label_limit = max(5, int(0.15 * len(label_candidates)))
    label_nodes = label_candidates[:label_limit]
    label_dict = {n: G.nodes[n]["label"] for n in label_nodes}
    nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=8, font_color="white")
    ax.grid(True, color=(1, 1, 1, 0.08))
    cbar = fig.colorbar(nodes, ax=ax, label=title)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title(title, color="white")
    ax.set_axis_off()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ball Mapper on regional obesity data.")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Ball radius (0-1 scale).")
    parser.add_argument(
        "--color-metric",
        choices=[
            "obesity_rate",
            "net_cost",
            "items",
            "cost_per_patient",
            "patients",
            "items_per_patient",
            "cost_per_item",
            "items_yoy_pct",
            "patients_yoy_pct",
            "net_cost_yoy_pct",
            "gender_items_share",
            "share_semaglutide",
            "share_liraglutide",
            "share_tirzepatide",
            "share_dulaglutide",
            "share_exenatide",
            "share_insulin_combo",
            "share_lixisenatide",
        ],
        default="obesity_rate",
        help="Metric used for node coloring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, X, labels = load_features()
    centers, cover = build_cover(X, args.epsilon)
    nodes = build_nodes(centers, cover, df)
    G = build_graph(nodes, labels)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    node_df = nodes_to_dataframe(nodes, labels)
    save_graph_json(node_df, GRAPH_JSON)
    plot_path = FIGURES_DIR / f"ballmapper_{args.color_metric}_eps{args.epsilon:.2f}.png"
    plot_graph(
        G,
        args.color_metric,
        plot_path,
        f"Ball Mapper (ε={args.epsilon:.2f}) - {args.color_metric.replace('_', ' ').title()}",
    )
    print(f"Graph nodes: {len(G.nodes)}, edges: {len(G.edges)}")
    print(f"Saved figure to {plot_path}")
    print(f"Saved node summary to {GRAPH_JSON}")


if __name__ == "__main__":
    main()
