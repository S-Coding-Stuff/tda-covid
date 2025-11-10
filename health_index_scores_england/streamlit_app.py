from __future__ import annotations

import json
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import health_index_scores_england.ballmapper_utils as bm

DATA_PATH = Path(__file__).resolve().parent / "health_index_combined_2021.csv"
DEFAULT_FEATURES = [
    "Healthy People Domain",
    "Healthy Lives Domain",
    "Healthy Places Domain",
]
PRESET_CONFIGS = {
    "Custom": {
        "features": DEFAULT_FEATURES,
        "color": "Healthy People Domain",
        "notes": "Manually selected features/colour.",
    },
    "🧠 Mental health & deprivation": {
        "features": ["Child poverty [Pl4]", "Unemployment [Pl4]", "Mental health [Pe]"],
        "color": "Life expectancy [Pe3]",
        "notes": "Health expectancy differences driven by economic hardship + mental wellbeing.",
    },
    "🏃 Lifestyle gradient": {
        "features": ["Smoking [L1]", "Physical activity [L1]", "Healthy eating [L1]"],
        "color": "Healthy People Domain",
        "notes": "Behavioural risk clusters vs overall health.",
    },
    "⚖️ Lifestyle Risk & Mental Wellbeing": {
        "features": [
            "Overweight and obesity in adults [L3]",
            "Alcohol misuse [L1]",
            "Drug misuse [L1]",
        ],
        "color": "Mental health [Pe]",
        "notes": "Topological link between substance use, obesity, and mental wellbeing.",
    },
    "🌳 Environment & access": {
        "features": ["Access to green space [Pl]", "Air pollution [Pl5]", "Distance to GP services [Pl2]"],
        "color": "Mental health [Pe]",
        "notes": "Urban–rural separation linked to mental health.",
    },
    "🧒 Education & youth wellbeing": {
        "features": ["Early years development [L2]", "Pupil attainment [L2]", "Teenage pregnancy [L2]"],
        "color": "Healthy Lives Domain",
        "notes": "Generational gradients through youth outcomes.",
    },
    "❤️ Preventive health": {
        "features": [
            "Cancer screening attendance [L4]",
            "Child vaccination coverage [L4]",
            "Overweight and obesity in adults [L3]",
        ],
        "color": "Life expectancy [Pe3]",
        "notes": "Preventive engagement vs longevity.",
    },
    "🔄 Cross-domain summary": {
        "features": [
            "Healthy People Domain",
            "Healthy Lives Domain",
            "Healthy Places Domain",
        ],
        "color": "Overweight and obesity in adults [L3]",
        "notes": "Macro-level topology of national health balance.",
    },
}


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return bm.load_health_index_dataset(DATA_PATH)


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Filters")
    source_options = sorted(df["Source"].unique())
    selected_sources = st.sidebar.multiselect("Source", source_options, default=source_options)
    area_types = sorted(df["Area Type"].unique())
    selected_area_types = st.sidebar.multiselect("Area Type", area_types, default=area_types)
    filtered = df[df["Source"].isin(selected_sources) & df["Area Type"].isin(selected_area_types)].copy()
    return filtered


def choose_features(df: pd.DataFrame) -> tuple[List[str], str]:
    st.sidebar.markdown("### Feature Selection")
    numeric_cols = [
        col
        for col in df.columns
        if col not in {"Source", "Year", "Area Code", "Area Name", "Area Type"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    default_features = [f for f in DEFAULT_FEATURES if f in numeric_cols][:4] or numeric_cols[:4]
    feature_cols = st.sidebar.multiselect(
        "Ball Mapper features", numeric_cols, default=default_features, key="feature_cols"
    )
    norm_method = st.sidebar.selectbox("Normalization method", ["minmax", "zscore"], index=0)
    return feature_cols, norm_method


def render_dataset_summary(df: pd.DataFrame) -> None:
    st.subheader("Filtered dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Sources", df["Source"].nunique())
    c3.metric("Area types", df["Area Type"].nunique())
    st.dataframe(df)
    st.download_button(
        "Download filtered CSV",
        data=df.to_csv(index=False),
        file_name="health_index_filtered.csv",
        mime="text/csv",
    )


def build_plotly_graph(G: nx.Graph, color_metric: str, size_metric: str, feature_cols: List[str]) -> go.Figure:
    pos = bm.compute_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    colors = [G.nodes[n][color_metric] for n in G.nodes]
    counts = np.array([G.nodes[n].get(size_metric, G.nodes[n]["size"]) for n in G.nodes], dtype=float)
    if len(counts) == 0:
        node_sizes = []
    else:
        scaled_counts = np.cbrt(np.clip(counts, a_min=0, a_max=None))
        min_px, max_px = 20, 120
        if scaled_counts.max() == scaled_counts.min():
            node_sizes = [((min_px + max_px) / 2) for _ in counts]
        else:
            node_sizes = list(
                np.interp(
                    scaled_counts,
                    [float(scaled_counts.min()), float(scaled_counts.max())],
                    [min_px, max_px],
                )
            )
    hovertext = []
    for n in G.nodes:
        base_lines = [
            f"Label: {G.nodes[n]['label']}",
            f"Area type: {G.nodes[n].get('area_type', '—')}",
            f"Source: {G.nodes[n].get('source', '—')}",
            f"Size: {int(G.nodes[n]['size'])}",
            f"{color_metric}: {G.nodes[n].get(color_metric, 0):.2f}",
        ]
        feature_lines = [
            f"{feat}: {G.nodes[n].get(feat, 0):.2f}"
            for feat in feature_cols
            if feat in G.nodes[n]
        ]
        hovertext.append("<br>".join(base_lines + feature_lines))
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="rgba(200,200,200,0.4)"),
        hoverinfo="none",
        mode="lines",
    )
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes],
        y=[pos[n][1] for n in G.nodes],
        mode="markers",
        marker=dict(
            size=node_sizes,
            color=colors,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=color_metric.replace("_", " ").title()),
            line=dict(color="#333333", width=1),
        ),
        hoverinfo="text",
        hovertext=hovertext,
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Ball Mapper Graph (color={color_metric})",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=720,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="rgba(12,12,12,1)",
        paper_bgcolor="rgba(12,12,12,1)",
        font=dict(color="#f0f0f0"),
    )
    return fig


def summarize_nodes(node_df: pd.DataFrame, color_metric: str) -> str:
    if node_df.empty or color_metric not in node_df.columns:
        return "No nodes to summarise."
    largest = node_df.sort_values("size", ascending=False).iloc[0]
    max_color = node_df.sort_values(color_metric, ascending=False).iloc[0]
    min_color = node_df.sort_values(color_metric, ascending=True).iloc[0]
    return (
        f"Largest ball: **{largest['label']}** (covers {int(largest['size'])} records). "
        f"Highest {color_metric}: **{max_color['label']}** ({max_color[color_metric]:.2f}). "
        f"Lowest {color_metric}: **{min_color['label']}** ({min_color[color_metric]:.2f})."
    )


def main() -> None:
    st.set_page_config(page_title="Health Index Ball Mapper", layout="wide")
    st.title("Health Index Ball Mapper Explorer (2021)")
    st.markdown(
        "Interactively explore the Health Index scores (England, 2021). "
        "Filter sources/area types, choose feature sets, and generate Ball Mapper graphs on the fly."
    )
    df = load_dataset()
    filtered_df = sidebar_filters(df)
    feature_cols, norm_method = choose_features(filtered_df)
    if filtered_df.empty:
        st.warning("No data after filtering. Adjust sidebar filters.")
        return
    render_dataset_summary(filtered_df)

    st.header("Ball Mapper Playground")
    epsilon = st.slider("Ball radius ε", 0.05, 0.5, value=0.2, step=0.01)
    numeric_cols = [
        col
        for col in filtered_df.columns
        if col not in {"Source", "Year", "Area Code", "Area Name", "Area Type"}
        and pd.api.types.is_numeric_dtype(filtered_df[col])
    ]
    if not numeric_cols:
        st.warning("No numeric columns available to colour by.")
        return
    default_color = "Healthy People Domain" if "Healthy People Domain" in numeric_cols else numeric_cols[0]
    color_metric = default_color
    preset_name = st.selectbox("Preset theme", list(PRESET_CONFIGS.keys()), index=0)
    preset_cfg = PRESET_CONFIGS[preset_name]
    if preset_name != "Custom":
        preset_features = [f for f in preset_cfg["features"] if f in numeric_cols]
        if preset_features:
            feature_cols = preset_features
        if preset_cfg["color"] in numeric_cols:
            color_metric = preset_cfg["color"]
        st.caption(f"{preset_cfg['notes']}  \nFeatures: {', '.join(feature_cols)} | Colour: {color_metric}")
    color_metric = st.selectbox(
        "Colour metric",
        numeric_cols,
        index=numeric_cols.index(color_metric) if color_metric in numeric_cols else 0,
    )
    size_options = ["size"] + feature_cols
    size_metric = st.selectbox("Node size metric (display only)", size_options, index=0)

    if st.button("Generate Ball Mapper graph", type="primary"):
        normalized, used_features = bm.normalize_features(filtered_df, feature_cols, norm_method)
        labels = bm.compose_labels(filtered_df)
        centers, cover = bm.build_cover(normalized, epsilon)
        if not centers:
            st.warning("ε too small — no landmarks created. Increase ε.")
            return
        nodes = bm.build_nodes(centers, cover, filtered_df)
        G = bm.build_graph(nodes, labels)
        if color_metric not in nodes[0].metrics and color_metric != "size":
            st.warning(f"{color_metric} is not available as a numeric metric.")
            return
        node_df = bm.nodes_to_dataframe(nodes, labels)
        fig = build_plotly_graph(G, color_metric, size_metric, feature_cols)
        st.plotly_chart(fig, use_container_width=True)
        st.info(summarize_nodes(node_df, color_metric))
        st.subheader("Node summary")
        st.dataframe(node_df)
        st.download_button(
            "Download node summary CSV",
            data=node_df.to_csv(index=False),
            file_name="health_index_ballmapper_nodes.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download node summary JSON",
            data=json.dumps(node_df.to_dict(orient="records"), indent=2),
            file_name="health_index_ballmapper_nodes.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
