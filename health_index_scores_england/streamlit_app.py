from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from ripser import ripser

import sys

try:  # Optional dependency for GeoJSON preprocessing
    from shapely.geometry import mapping, shape
    from shapely.ops import transform
except ImportError:
    mapping = shape = transform = None

try:
    from pyproj import Transformer
except ImportError:
    Transformer = None

SHAPELY_AVAILABLE = all(v is not None for v in (mapping, shape, transform, Transformer))
TRANSFORMER = (
    Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    if SHAPELY_AVAILABLE
    else None
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
sys.path.append(str(Path(__file__).resolve().parents[1] / ".tda_lib"))

import health_index_scores_england.ballmapper_utils as bm

DATA_PATH = Path(__file__).resolve().parent / "health_index_combined_2021.csv"
GEOJSON_CANDIDATES = list(
    Path(__file__).resolve().parent.glob("Local_Authority_Districts*.geojson")
)
GEOJSON_PATH = GEOJSON_CANDIDATES[0] if GEOJSON_CANDIDATES else None
SIMPLIFIED_GEOJSON_PATH = Path(__file__).resolve().parent / "england_la_simplified.geojson"
GEOJSON_SIMPLIFY_TOLERANCE = 500.0  # metres (data in British National Grid)
COORD_DECIMALS = 5
DEFAULT_FEATURES = [
    "Healthy People Domain",
    "Healthy Lives Domain",
    "Healthy Places Domain",
]
MAP_HOVER_FIELDS = [
    "Healthy People Domain",
    "Unemployment [Pl4]",
    "Life expectancy [Pe3]",
    "Mental health [Pe]",
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
    "🧠 Lifestyle Risk & Mental Wellbeing": {
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
    "🧬 Chronic Disease Burden": {
        "features": [
            "Cancer [Pe5]",
            "Cardiovascular conditions [Pe5]",
            "Diabetes [Pe5]",
            "Respiratory conditions [Pe5]",
            "Dementia [Pe5]",
        ],
        "color": "Healthy People Domain",
        "notes": "Overlapping chronic disease patterns and their effect on overall health.",
    },
    "🌆 Urban Environment & Air Quality": {
        "features": [
            "Air pollution [Pl5]",
            "Noise complaints [Pl5]",
            "Household overcrowding [Pl5]",
            "Access to green space [Pl]",
            "Road safety [Pl5]",
        ],
        "color": "Healthy Places Domain",
        "notes": "Urban environmental stressors vs metropolitan/rural health environments.",
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
    "⏳ Social Gradient of Longevity": {
        "features": [
            "Child poverty [Pl4]",
            "Unemployment [Pl4]",
            "Job-related training [Pl4]",
            "Physical activity [L1]",
            "Healthy eating [L1]",
            "Smoking [L1]",
            "Alcohol misuse [L1]",
            "Cancer screening attendance [L4]",
            "Child vaccination coverage [L4]",
            "Access to green space [Pl]",
            "Air pollution [Pl5]",
        ],
        "color": "Life expectancy [Pe3]",
        "notes": "Socioeconomic–behavioural–environmental gradient vs life expectancy.",
    },
    "🌐 Regional Health Inequality": {
        "features": [
            "Life expectancy [Pe3]",
            "Mental health [Pe]",
            "Diabetes [Pe5]",
            "Respiratory conditions [Pe5]",
            "Physical activity [L1]",
            "Smoking [L1]",
            "Alcohol misuse [L1]",
            "Access to green space [Pl]",
            "Air pollution [Pl5]",
            "Unemployment [Pl4]",
        ],
        "color": "Healthy People Domain",
        "notes": (
            "Chronic disease, behaviours, environment, and deprivation jointly shaping health inequality. "
            "Suggested ε≈0.4."
        ),
    },
}


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return bm.load_health_index_dataset(DATA_PATH)


def _round_coords(value: Any, decimals: int) -> Any:
    if isinstance(value, (float, int)):
        return round(float(value), decimals)
    if isinstance(value, (list, tuple)):
        return [_round_coords(item, decimals) for item in value]
    return value


def _preprocess_geojson(source_path: Path) -> Dict[str, Any]:
    with source_path.open("r", encoding="utf-8") as fh:
        raw_data = json.load(fh)
    processed_features: List[Dict[str, Any]] = []
    for feature in raw_data.get("features", []):
        props = feature.get("properties", {}) or {}
        lad_code = str(props.get("LAD21CD", "")).strip()
        if not lad_code.startswith("E"):
            continue
        geometry = feature.get("geometry")
        if geometry and SHAPELY_AVAILABLE:
            try:
                geom = shape(geometry)
                if not geom.is_valid:
                    geom = geom.buffer(0)
                geom = geom.simplify(GEOJSON_SIMPLIFY_TOLERANCE, preserve_topology=True)
                if TRANSFORMER is not None:
                    geom = transform(TRANSFORMER.transform, geom)
                geometry = mapping(geom)
            except Exception:
                geometry = feature.get("geometry")
        if geometry:
            geometry = {
                "type": geometry.get("type"),
                "coordinates": _round_coords(geometry.get("coordinates"), COORD_DECIMALS),
            }
        processed_features.append(
            {
                "type": feature.get("type", "Feature"),
                "properties": {
                    "LAD21CD": lad_code,
                    "LAD21NM": props.get("LAD21NM", ""),
                },
                "geometry": geometry,
            }
        )
    return {"type": raw_data.get("type", "FeatureCollection"), "features": processed_features}


def _ensure_simplified_geojson(base_path: Path | None) -> Tuple[Dict[str, Any], Path]:
    if SIMPLIFIED_GEOJSON_PATH.exists():
        with SIMPLIFIED_GEOJSON_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh), SIMPLIFIED_GEOJSON_PATH
    if base_path is None or not base_path.exists():
        raise FileNotFoundError(
            "Local authority GeoJSON file not found. Place it in `health_index_scores_england/`."
        )
    if SHAPELY_AVAILABLE:
        data = _preprocess_geojson(base_path)
        SIMPLIFIED_GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SIMPLIFIED_GEOJSON_PATH.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"))
        return data, SIMPLIFIED_GEOJSON_PATH
    with base_path.open("r", encoding="utf-8") as fh:
        return json.load(fh), base_path


@st.cache_data(show_spinner=False)
def load_geojson(path: str | None = None) -> Tuple[Dict[str, Any], List[str]]:
    base_path = Path(path) if path else GEOJSON_PATH
    data, _ = _ensure_simplified_geojson(base_path)
    codes = sorted(
        {
            feature.get("properties", {}).get("LAD21CD")
            for feature in data.get("features", [])
            if feature.get("properties", {}).get("LAD21CD")
        }
    )
    return data, codes


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Filters")
    source_options = sorted(df["Source"].unique())
    selected_sources = st.sidebar.multiselect("Source", source_options, default=source_options)
    area_types = sorted(df["Area Type"].unique())
    default_area_types = [atype for atype in area_types if atype.lower() != "country"]
    selected_area_types = st.sidebar.multiselect("Area Type", area_types, default=default_area_types)
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
    st.sidebar.caption("Tip: When selecting many features, try **z-score** normalisation and a larger ε range to maintain connectivity.")
    return feature_cols or default_features, norm_method


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


def build_area_assignment_df(
    df: pd.DataFrame,
    nodes: List[bm.BallMapperNode],
    color_metric: str,
    feature_cols: List[str],
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    metric_columns = set(MAP_HOVER_FIELDS + feature_cols + [color_metric])
    for node in nodes:
        ball_label = f"Ball {node.index}"
        for member_idx in node.members:
            row = df.iloc[member_idx]
            area_code = row.get("Area Code")
            if not isinstance(area_code, str):
                continue
            record: Dict[str, Any] = {
                "Area Code": area_code.strip(),
                "Area Name": row.get("Area Name", ""),
                "Area Type": row.get("Area Type", ""),
                "ball_id": node.index,
                "ball_label": ball_label,
            }
            for col in metric_columns:
                record[col] = node.metrics.get(col)
            records.append(record)
    map_df = pd.DataFrame(records)
    if map_df.empty:
        return map_df
    map_df = map_df.dropna(subset=["Area Code"])
    map_df = map_df.drop_duplicates(subset=["Area Code"], keep="first")
    for col in metric_columns:
        if col in map_df.columns:
            map_df[col] = pd.to_numeric(map_df[col], errors="coerce")
    map_df["ball_label"] = map_df["ball_label"].astype(str)
    return map_df


def render_geo_overlay(
    map_df: pd.DataFrame,
    geojson: Dict[str, Any],
    color_metric: str,
    feature_cols: List[str],
) -> None:
    if map_df.empty:
        st.info("No overlapping local authority records available for the geographic overlay.")
        return
    if color_metric not in map_df.columns:
        st.warning(f"{color_metric} unavailable for the geographic overlay.")
        return
    color_values = pd.to_numeric(map_df[color_metric], errors="coerce")
    map_df = map_df.assign(**{color_metric: color_values})
    if color_values.notna().sum() == 0:
        st.warning(f"{color_metric} has no numeric data to colour the map.")
        return
    hover_fields = []
    if color_metric in map_df.columns:
        hover_fields.append(color_metric)
    for feat in feature_cols:
        if feat in map_df.columns:
            hover_fields.append(feat)
    hover_fields.extend(col for col in MAP_HOVER_FIELDS if col in map_df.columns)
    hover_fields = list(dict.fromkeys(hover_fields))
    hover_data = {col: True for col in hover_fields}
    hover_data["ball_id"] = True
    hover_data["ball_label"] = True
    color_min = float(color_values.min())
    color_max = float(color_values.max())
    fig = px.choropleth_mapbox(
        map_df,
        geojson=geojson,
        locations="Area Code",
        featureidkey="properties.LAD21CD",
        color=color_metric,
        hover_name="Area Name",
        hover_data=hover_data,
        mapbox_style="carto-positron",
        zoom=5.0,
        opacity=0.8,
        center={"lat": 53.4, "lon": -1.6},
        color_continuous_scale="Viridis",
        range_color=(color_min, color_max),
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_colorbar=dict(title=color_metric),
        legend_title_text="",
        height=720,
    )
    st.plotly_chart(fig, use_container_width=True)


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
        margin=dict(l=20, r=20, t=60, b=110),
        plot_bgcolor="rgba(12,12,12,1)",
        paper_bgcolor="rgba(12,12,12,1)",
        font=dict(color="#f0f0f0"),
    )
    return fig


def compute_barcodes(points: np.ndarray, maxdim: int = 1) -> Dict[str, np.ndarray]:
    result = ripser(points, maxdim=maxdim)
    diagrams = result["dgms"]
    return {"H0": diagrams[0], "H1": diagrams[1] if len(diagrams) > 1 else np.empty((0, 2))}


def render_barcodes(barcodes: Dict[str, np.ndarray]) -> None:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("H0 barcode", "H1 barcode"))
    colors = {"H0": "skyblue", "H1": "salmon"}
    for row, dim in enumerate(["H0", "H1"], start=1):
        diagram = barcodes.get(dim, np.empty((0, 2)))
        for idx, (birth, death) in enumerate(diagram):
            death = death if np.isfinite(death) else birth + 1.0
            fig.add_trace(
                go.Scatter(
                    x=[birth, death],
                    y=[idx, idx],
                    mode="lines",
                    line=dict(color=colors[dim], width=3),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
    fig.update_yaxes(title_text="Intervals", showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="Filtration value")
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(12,12,12,1)",
        paper_bgcolor="rgba(12,12,12,1)",
        font=dict(color="#f0f0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)


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
    if "bm_feature_cols" not in st.session_state:
        st.session_state["bm_feature_cols"] = DEFAULT_FEATURES
    if "bm_norm_method" not in st.session_state:
        st.session_state["bm_norm_method"] = "minmax"
    feature_cols, norm_method = choose_features(filtered_df)
    st.session_state["bm_feature_cols"] = feature_cols or DEFAULT_FEATURES
    st.session_state["bm_norm_method"] = norm_method
    if filtered_df.empty:
        st.warning("No data after filtering. Adjust sidebar filters.")
        return
    render_dataset_summary(filtered_df)

    st.header("Ball Mapper Playground")
    range_col, toggle_col = st.columns([0.8, 0.2])
    with toggle_col:
        wide_range = st.checkbox("ε up to 10.0", value=False, key="epsilon_wide")
    max_eps = 10.0 if wide_range else 0.5
    if "epsilon_value" not in st.session_state:
        st.session_state["epsilon_value"] = 0.2
    st.session_state["epsilon_value"] = min(st.session_state["epsilon_value"], max_eps)
    with range_col:
        epsilon = st.slider(
            "Ball radius ε",
            min_value=0.05,
            max_value=max_eps,
            value=st.session_state["epsilon_value"],
            step=0.01,
        )
    st.session_state["epsilon_value"] = epsilon
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
            st.session_state["bm_feature_cols"] = feature_cols
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

    should_generate = st.button("Generate Ball Mapper graph", type="primary")
    if should_generate:
        normalized, used_features = bm.normalize_features(
            filtered_df, st.session_state["bm_feature_cols"], st.session_state["bm_norm_method"]
        )
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
        fig = build_plotly_graph(G, color_metric, size_metric, st.session_state["bm_feature_cols"])
        caption_text = f"Features: {', '.join(st.session_state['bm_feature_cols'])} | Colour: {color_metric} | ε = {epsilon}"
        fig.add_annotation(
            text=caption_text,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.08,
            showarrow=False,
            font=dict(color="#f0f0f0", size=12),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(caption_text)
        fig_html = fig.to_html(include_plotlyjs="cdn")
        png_bytes = None
        png_error = None
        try:
            png_bytes = fig.to_image(format="png", scale=2)
        except Exception as exc:  # Kaleido may be unavailable on some deployments
            png_error = str(exc)
        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            st.download_button(
                "Download plot (HTML)",
                data=fig_html,
                file_name="health_index_ballmapper_plot.html",
                mime="text/html",
                use_container_width=True,
            )
        with plot_col2:
            if png_bytes:
                st.download_button(
                    "Download plot (PNG)",
                    data=png_bytes,
                    file_name="health_index_ballmapper_plot.png",
                    mime="image/png",
                    use_container_width=True,
                )
            else:
                st.info("PNG export unavailable in this environment (Kaleido missing). Download the HTML version instead.")
        st.info(summarize_nodes(node_df, color_metric))
        st.markdown("#### Persistence barcodes")
        landmark_points = normalized[centers] if centers else np.empty((0, normalized.shape[1]))
        barcodes = compute_barcodes(landmark_points)
        render_barcodes(barcodes)
        st.markdown("#### Geographic overlay (local authority view)")
        try:
            geojson_data, geo_codes = load_geojson(str(GEOJSON_PATH) if GEOJSON_PATH else None)
            map_df = build_area_assignment_df(
                filtered_df, nodes, color_metric, st.session_state["bm_feature_cols"]
            )
            if not map_df.empty:
                valid_codes = set(geo_codes)
                map_df = map_df[map_df["Area Code"].isin(valid_codes)]
            if map_df.empty:
                st.info("No local authority rows match the boundary file after filtering.")
            else:
                render_geo_overlay(
                    map_df,
                    geojson_data,
                    color_metric,
                    st.session_state["bm_feature_cols"],
                )
        except FileNotFoundError as exc:
            st.info(str(exc))
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
