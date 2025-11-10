from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import sys

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from dataset_experiments.scripts import ballmapper_utils as bm_utils

DATA_DIR = Path(__file__).resolve().parent
CLEAN_DATA = DATA_DIR / "tda_obesity_prescribing_clean.csv"
BALLMAPPER_DATA = DATA_DIR / "tda_obesity_ballmapper_input.csv"

SCATTER_SIZE_OPTIONS = ["patients", "items"]
BALL_COLOR_OPTIONS = [
    "obesity_rate",
    "net_cost",
    "items",
    "cost_per_patient",
    "cost_per_item",
    "items_per_patient",
    "items_yoy_pct",
    "net_cost_yoy_pct",
    "gender_items_share",
]
DRUG_SHARE_COLUMNS = {
    "share_dulaglutide": "Dulaglutide share",
    "share_exenatide": "Exenatide share",
    "share_insulin_combo": "Insulin combo share",
    "share_liraglutide": "Liraglutide share",
    "share_lixisenatide": "Lixisenatide share",
    "share_semaglutide": "Semaglutide share",
    "share_tirzepatide": "Tirzepatide share",
}
for col in DRUG_SHARE_COLUMNS:
    if col not in BALL_COLOR_OPTIONS:
        BALL_COLOR_OPTIONS.append(col)
CORE_FEATURES = ["norm_items", "norm_patients", "norm_net_cost", "norm_obesity_rate"]
FEATURE_SET_LIBRARY = {
    "core": {
        "label": "Regional–Demographic Core",
        "features": CORE_FEATURES,
        "color": "obesity_rate",
        "epsilon": 0.2,
        "notes": "Default view; all regions, ages, genders with four core metrics.",
    },
    "cost_focus": {
        "label": "Cost Focus",
        "features": [
            "norm_cost_per_patient",
            "norm_cost_per_item",
            "norm_items",
            "norm_obesity_rate",
        ],
        "color": "cost_per_patient",
        "epsilon": 0.2,
        "notes": "Highlights spend hot-spots vs obesity gradient.",
    },
    "outlier_hunt": {
        "label": "Outlier Hunt",
        "features": CORE_FEATURES,
        "color": "obesity_rate",
        "epsilon": 0.1,
        "notes": "Smaller ε to surface isolated demographic clusters.",
    },
    "drug_mix": {
        "label": "Drug Mix",
        "features": [
            "norm_share_semaglutide",
            "norm_share_liraglutide",
            "norm_share_tirzepatide",
            "norm_cost_per_patient",
        ],
        "color": "share_semaglutide",
        "epsilon": 0.22,
        "notes": "Focus on adoption of key GLP-1 therapies vs cost.",
    },
}

def slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


PRESETS: Dict[str, Dict] = {
    "High-cost focus": {
        "description": "Highlights regions combining high obesity and high cost per patient.",
        "scatter": {
            "x": "obesity_rate",
            "y": "cost_per_patient",
            "color": "cost_per_item",
            "size": "patients",
            "regions": None,
        },
        "ball": {"epsilon": 0.18, "color": "cost_per_patient"},
    },
    "Growth focus": {
        "description": "Emphasises 2023/24 growth using YoY deltas.",
        "scatter": {
            "x": "items_yoy_pct",
            "y": "net_cost_yoy_pct",
            "color": "patients_yoy_pct",
            "size": "items",
            "regions": None,
        },
        "ball": {"epsilon": 0.2, "color": "items_yoy_pct"},
    },
    "Demographic focus": {
        "description": "Looks at female-heavy regions and older age prescriptions.",
        "scatter": {
            "x": "share_items_female",
            "y": "share_items_60plus",
            "color": "share_items_female_minus_male",
            "size": "patients",
            "regions": None,
        },
        "ball": {"epsilon": 0.22, "color": "share_items_female"},
    },
}


def load_clean_data() -> pd.DataFrame:
    if not CLEAN_DATA.exists():
        raise FileNotFoundError(
            "Clean dataset missing. Run `python dataset_experiments/scripts/preprocess_data.py` first."
        )
    df = pd.read_csv(CLEAN_DATA)
    return df


def load_ballmapper_arrays(
    feature_columns: List[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, List[str]]:
    if not BALLMAPPER_DATA.exists():
        raise FileNotFoundError(
            "Ball Mapper input missing. Run `python dataset_experiments/scripts/preprocess_data.py` first."
        )
    return bm_utils.load_features(BALLMAPPER_DATA, feature_columns)


def ensure_default_state(numeric_options: List[str], regions: List[str]) -> None:
    defaults = {
        "scatter_x": numeric_options[0],
        "scatter_y": numeric_options[min(1, len(numeric_options) - 1)],
        "scatter_color": numeric_options[min(2, len(numeric_options) - 1)],
        "scatter_size": SCATTER_SIZE_OPTIONS[0],
        "scatter_regions": regions,
        "ball_epsilon": 0.25,
        "ball_color": BALL_COLOR_OPTIONS[0],
        "ball_feature_set": "core",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_preset_to_state(preset_name: str, numeric_options: List[str], regions: List[str]) -> None:
    preset = PRESETS[preset_name]
    scatter = preset.get("scatter", {})
    state = st.session_state
    for key, option_key in [
        ("scatter_x", "x"),
        ("scatter_y", "y"),
        ("scatter_color", "color"),
        ("scatter_size", "size"),
    ]:
        value = scatter.get(option_key)
        if value:
            if key == "scatter_size" and value in SCATTER_SIZE_OPTIONS:
                state[key] = value
            elif key != "scatter_size" and value in numeric_options:
                state[key] = value
    region_override = scatter.get("regions")
    if region_override:
        state["scatter_regions"] = [r for r in region_override if r in regions]
    else:
        state["scatter_regions"] = regions
    ball = preset.get("ball", {})
    epsilon = ball.get("epsilon")
    if epsilon:
        state["ball_epsilon"] = float(epsilon)
    color = ball.get("color")
    if color in BALL_COLOR_OPTIONS:
        state["ball_color"] = color


def handle_pending_preset(numeric_options: List[str], regions: List[str]) -> None:
    pending = st.session_state.pop("pending_preset", None)
    if pending and pending in PRESETS:
        apply_preset_to_state(pending, numeric_options, regions)


def handle_pending_ball_color() -> None:
    pending_color = st.session_state.pop("pending_ball_color", None)
    if pending_color:
        st.session_state["ball_color"] = pending_color


def handle_pending_ball_epsilon() -> None:
    pending_eps = st.session_state.pop("pending_ball_epsilon", None)
    if pending_eps is not None:
        st.session_state["ball_epsilon"] = pending_eps


def handle_pending_ball_features() -> None:
    pending_feat = st.session_state.pop("pending_ball_features", None)
    if pending_feat is not None:
        st.session_state["ball_feature_set"] = pending_feat


def nudge_epsilon(multiplier: float) -> None:
    current = st.session_state.get("ball_epsilon", 0.25)
    new_value = max(0.05, min(0.5, current * multiplier))
    st.session_state["pending_ball_epsilon"] = round(new_value, 3)
    st.rerun()


def apply_ballmapper_preset(preset_key: str) -> None:
    config = FEATURE_SET_LIBRARY[preset_key]
    st.session_state["pending_ball_features"] = preset_key
    st.session_state["pending_ball_color"] = config["color"]
    st.session_state["pending_ball_epsilon"] = config["epsilon"]
    st.rerun()


def render_overview(df: pd.DataFrame) -> None:
    st.header("Project Overview")
    st.markdown(
        """
        This dashboard adapts the **Ball Mapper** approach from Dłotko & Rudkin (2020) to
        weight-loss prescribing data. Each record is a **Region × Age band × Gender** slice with:

        - `Items`, `Patients`, `Net Cost` (summed over the year)
        - `Adult obesity rate 2022/23` (regional mean)
        - Derived ratios such as cost per patient, items per patient, and YoY shifts into 2023/24.

        Use the tabs to explore:
        1. **Explorer** – classic scatter + YoY table to spot outliers quickly.
        2. **Ball Mapper** – the topological view that clusters similar region/age/gender groups.
        3. **Preset Views** – one-click configurations for common stories (high cost, growth, demographics).
        """
    )
    region_count = df["Region"].nunique() if "Region" in df.columns else len(df)
    records = len(df)
    age_count = df["Patient Age Band (Years old)"].nunique() if "Patient Age Band (Years old)" in df.columns else 0
    gender_count = df["Gender"].nunique() if "Gender" in df.columns else 0
    cols = st.columns(4)
    cols[0].metric("Records", f"{records:,}")
    cols[1].metric("Regions", f"{region_count}")
    if age_count:
        cols[2].metric("Age bands", age_count)
    if gender_count:
        cols[3].metric("Genders", gender_count)
    st.markdown("---")


def streamlit_scatter(df: pd.DataFrame, numeric_options: List[str]) -> None:
    current_x = st.session_state["scatter_x"]
    current_y = st.session_state["scatter_y"]
    current_color = st.session_state["scatter_color"]
    current_size = st.session_state["scatter_size"]
    st.sidebar.markdown("### Scatter Controls")
    x_axis = st.sidebar.selectbox(
        "X-axis metric",
        numeric_options,
        index=numeric_options.index(current_x) if current_x in numeric_options else 0,
        key="scatter_x",
    )
    y_axis = st.sidebar.selectbox(
        "Y-axis metric",
        numeric_options,
        index=numeric_options.index(current_y) if current_y in numeric_options else 0,
        key="scatter_y",
    )
    color_metric = st.sidebar.selectbox(
        "Color metric",
        numeric_options,
        index=numeric_options.index(current_color) if current_color in numeric_options else 0,
        key="scatter_color",
    )
    size_metric = st.sidebar.selectbox(
        "Bubble size metric",
        SCATTER_SIZE_OPTIONS,
        index=SCATTER_SIZE_OPTIONS.index(current_size) if current_size in SCATTER_SIZE_OPTIONS else 0,
        key="scatter_size",
    )
    region_filter = st.sidebar.multiselect(
        "Filter regions",
        options=df["Region"].tolist(),
        default=st.session_state.get("scatter_regions", df["Region"].tolist()),
        key="scatter_regions",
    )
    filtered = df[df["Region"].isin(region_filter)]
    fig = px.scatter(
        filtered,
        x=x_axis,
        y=y_axis,
        color=color_metric,
        size=size_metric,
        hover_name="Region",
        size_max=60,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(title="Regional scatter explorer")
    st.plotly_chart(fig, use_container_width=True)


def ballmapper_controls() -> tuple[float, str]:
    st.sidebar.markdown("### Ball Mapper Controls")
    epsilon = st.sidebar.slider(
        "Ball radius ε",
        min_value=0.05,
        max_value=0.5,
        step=0.01,
        value=float(st.session_state.get("ball_epsilon", 0.25)),
        key="ball_epsilon",
    )
    tweak_cols = st.sidebar.columns(2)
    with tweak_cols[0]:
        if st.button("ε -10%", use_container_width=True):
            nudge_epsilon(0.9)
    with tweak_cols[1]:
        if st.button("ε +10%", use_container_width=True):
            nudge_epsilon(1.1)
    current_color = st.session_state.get("ball_color", BALL_COLOR_OPTIONS[0])
    color_choice = st.sidebar.selectbox(
        "Node colour metric",
        BALL_COLOR_OPTIONS,
        index=BALL_COLOR_OPTIONS.index(current_color)
        if current_color in BALL_COLOR_OPTIONS
        else 0,
        key="ball_color",
    )
    return epsilon, color_choice


def build_ballmapper_plot(
    epsilon: float, color_metric: str, feature_columns: List[str] | None = None
) -> tuple[go.Figure, pd.DataFrame]:
    df, X, labels = load_ballmapper_arrays(feature_columns)
    centers, cover = bm_utils.build_cover(X, epsilon)
    nodes = bm_utils.build_nodes(centers, cover, df)
    G = bm_utils.build_graph(nodes, labels)
    node_df = bm_utils.nodes_to_dataframe(nodes, labels)
    if color_metric not in node_df.columns:
        raise ValueError(f"{color_metric} not available for coloring.")
    color_values = [G.nodes[n][color_metric] for n in G.nodes]
    colorscale = "Viridis" if "cost" not in color_metric else "Inferno"
    title = color_metric.replace("_", " ").title()
    pos = compute_interactive_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="rgba(255,255,255,0.4)"),
        hoverinfo="none",
        mode="lines",
    )
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    counts = [G.nodes[n]["size"] for n in G.nodes()]
    if counts:
        sizes = np.sqrt(np.array(counts, dtype=float))
        min_px, max_px = 30, 120
        if float(sizes.max()) == float(sizes.min()):
            node_sizes = [((min_px + max_px) / 2) for _ in counts]
        else:
            node_sizes = list(np.interp(sizes, [float(sizes.min()), float(sizes.max())], [min_px, max_px]))
    else:
        node_sizes = []
    hovertext = []
    for n in G.nodes():
        details = [
            f"Label: {G.nodes[n]['label']}",
            f"Region: {G.nodes[n].get('region_name', '—')}",
            f"Age band: {G.nodes[n].get('age_band', '—')}",
            f"Gender: {G.nodes[n].get('gender_label', '—')}",
            f"Size: {int(G.nodes[n]['size'])}",
            f"Cost per patient: £{G.nodes[n].get('cost_per_patient', 0):.2f}",
            f"Items per patient: {G.nodes[n].get('items_per_patient', 0):.2f}",
            f"Obesity rate: {G.nodes[n].get('obesity_rate', 0):.2f}",
        ]
        hovertext.append("<br>".join(details))
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=node_sizes,
            color=color_values,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=title),
            line=dict(color="#333333", width=1),
        ),
        hoverinfo="text",
        hovertext=hovertext,
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Ball Mapper graph (ε={epsilon:.2f}, color={title})",
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, showticklabels=False),
        height=720,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(15,15,15,1)",
        plot_bgcolor="rgba(15,15,15,1)",
        font=dict(color="#f4f4f4"),
    )
    return fig, node_df


def summarize_topology(node_df: pd.DataFrame, color_metric: str) -> str:
    if node_df.empty or color_metric not in node_df.columns:
        return "No nodes to summarise."
    largest = node_df.sort_values("size", ascending=False).iloc[0]
    hottest = node_df.sort_values(color_metric, ascending=False).iloc[0]
    coolest = node_df.sort_values(color_metric, ascending=True).iloc[0]
    largest_label = largest.get("region", largest.get("node_id"))
    hottest_label = hottest.get("region", hottest.get("node_id"))
    coolest_label = coolest.get("region", coolest.get("node_id"))
    return (
        f"Largest ball: **{largest_label}** (covers {int(largest['size'])} records). "
        f"Highest {color_metric}: **{hottest_label}** ({hottest[color_metric]:.2f}). "
        f"Lowest {color_metric}: **{coolest_label}** ({coolest[color_metric]:.2f})."
    )




def render_ballmapper_presets() -> None:
    st.header("Preset Ideas")
    st.markdown(
        "These curated configurations match the presets above. Click any preset button to apply it instantly."
    )
    for key, config in FEATURE_SET_LIBRARY.items():
        with st.expander(config["label"], expanded=False):
            st.markdown(f"**Use case:** {config['notes']}")
            st.markdown(f"**Features:** `{', '.join(config['features'])}`")
            st.markdown(f"**Colour:** `{config['color']}`  |  **ε:** {config['epsilon']}")

def render_presets_tab(numeric_options: List[str], regions: List[str]) -> None:
    st.subheader("Recommended views")
    st.write(
        "Each preset sets the scatter axes, colour metrics, region filter, and Ball Mapper parameters "
        "for a particular analytical angle. Click **Apply preset** to load it into the sidebar controls."
    )
    for name, config in PRESETS.items():
        scatter_cfg = config.get("scatter", {})
        ball_cfg = config.get("ball", {})
        with st.expander(name, expanded=False):
            st.write(config.get("description", ""))
            st.markdown(
                f"- Scatter: x=`{scatter_cfg.get('x', '—')}`, y=`{scatter_cfg.get('y', '—')}`, "
                f"colour=`{scatter_cfg.get('color', '—')}`, size=`{scatter_cfg.get('size', '—')}`"
            )
            st.markdown(
                f"- Ball Mapper: ε={ball_cfg.get('epsilon', '—')}, colour metric=`{ball_cfg.get('color', '—')}`"
            )
            if st.button(f"Apply preset: {name}", key=f"apply_{slugify(name)}"):
                st.session_state["pending_preset"] = name
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="Regional Obesity & TDA Explorer", layout="wide")
    st.title("Regional Obesity & Weight-loss Prescribing Explorer")
    df = load_clean_data()
    regions = df["Region"].tolist() if "Region" in df.columns else df.index.astype(str).tolist()
    numeric_options = sorted(
        [col for col in df.select_dtypes(include=["number"]).columns if col not in {"Region Code"}]
    )
    ensure_default_state(numeric_options, regions)
    handle_pending_preset(numeric_options, regions)
    handle_pending_ball_color()
    handle_pending_ball_epsilon()
    handle_pending_ball_features()

    tabs = st.tabs(["Overview", "Explorer", "Ball Mapper", "Preset views"])

    with tabs[0]:
        render_overview(df)
        with st.expander("Raw data preview", expanded=False):
            st.dataframe(df)

    with tabs[1]:
        st.header("Scatter Explorer")
        st.markdown(
            "Select metrics in the sidebar to compare Region × Age × Gender slices. "
            "The bubble size reflects patient counts; colours show the chosen metric."
        )
        streamlit_scatter(df, numeric_options)

        st.subheader("Drug Mix Explorer")
        drug_item_cols = [
            col for col in df.columns if col.startswith("items_") and col not in {"items", "items_per_patient"}
        ]
        region_options = sorted(df["Region"].unique())
        region_choice = st.selectbox("Region", region_options, key="drug_region")
        age_options = ["All"] + sorted(df["Patient Age Band (Years old)"].unique())
        age_choice = st.selectbox("Age band", age_options, key="drug_age")
        gender_options = ["All"] + sorted(df["Gender"].unique())
        gender_choice = st.selectbox("Gender", gender_options, key="drug_gender")
        subset = df[df["Region"] == region_choice]
        if age_choice != "All":
            subset = subset[subset["Patient Age Band (Years old)"] == age_choice]
        if gender_choice != "All":
            subset = subset[subset["Gender"] == gender_choice]
        if subset.empty:
            st.info("No prescriptions for the selected slice.")
        else:
            drug_totals = subset[drug_item_cols].sum()
            total_items = drug_totals.sum()
            if total_items <= 0:
                st.info("No prescriptions for the selected slice.")
            else:
                drug_shares = (drug_totals / total_items).fillna(0)
                chart_df = pd.DataFrame(
                    {
                        "Drug": [col.replace("items_", "").replace("_", " ").title() for col in drug_shares.index],
                        "Share": drug_shares.values,
                    }
                )
                fig_drug = px.bar(
                    chart_df,
                    x="Drug",
                    y="Share",
                    color="Drug",
                    text=chart_df["Share"].map(lambda x: f"{x:.1%}"),
                )
                fig_drug.update_layout(yaxis_tickformat=".0%", showlegend=False, title="Prescription share by drug class")
                st.plotly_chart(fig_drug, use_container_width=True)
                st.dataframe(chart_df.sort_values("Share", ascending=False), use_container_width=True)

        st.subheader("Year-on-year summary")
        yoy_cols = ["items_yoy_pct", "patients_yoy_pct", "net_cost_yoy_pct"]
        id_vars = [col for col in ["Region", "Patient Age Band (Years old)", "Gender"] if col in df.columns]
        yoy_df = df[id_vars + yoy_cols].copy()
        yoy_melted = yoy_df.melt(
            id_vars=id_vars,
            var_name="Metric",
            value_name="YoY%",
        )
        yoy_melted["YoY%"] = yoy_melted["YoY%"] * 100
        st.dataframe(yoy_melted)

    with tabs[2]:
        st.header("Ball Mapper Playground")
        st.markdown(
            """
            1. **Point cloud** – we use the four core features `[Items, Patients, Net Cost, Obesity Rate]` (all min-max scaled).
            2. **Ball Mapper** – choose ε (start at 0.1–0.2). Balls capture clusters of similar region/age/gender points.
            3. **Colourings** – use the dropdown or quick buttons to render overlays such as obesity %, cost per patient,
               total items, or female share. Switch colours to explore multiple health-economic dimensions interactively.
            """
        )
        st.code("Features → [norm_items, norm_patients, norm_net_cost, norm_obesity_rate]", language="text")
        st.caption("Tip: save multiple screenshots with different colour metrics to build your figure set.")

        epsilon, color_choice = ballmapper_controls()
        current_feature_key = st.session_state.get("ball_feature_set", "core")
        current_feature_cfg = FEATURE_SET_LIBRARY.get(current_feature_key, FEATURE_SET_LIBRARY["core"])
        st.info(
            f"Current feature set: **{current_feature_cfg['label']}** "
            f"→ `{', '.join(current_feature_cfg['features'])}`"
        )
        st.info(
            f"Current feature set: **{current_feature_cfg['label']}** "
            f"→ `{', '.join(current_feature_cfg['features'])}`"
        )

        preset_colors = {
            "Obesity rate (%)": "obesity_rate",
            "Net ingredient cost (£)": "net_cost",
            "Items": "items",
            "Female share": "gender_items_share",
            "Cost per patient": "cost_per_patient",
            "Semaglutide share": "share_semaglutide",
        }
        button_cols = st.columns(len(preset_colors))
        for col, (label, metric) in zip(button_cols, preset_colors.items()):
            with col:
                if st.button(label, key=f"quick_color_{metric}"):
                    st.session_state["pending_ball_color"] = metric
                    st.rerun()

        st.markdown("**Preset feature sets**")
        preset_cols = st.columns(len(FEATURE_SET_LIBRARY))
        for col, (pkey, cfg) in zip(preset_cols, FEATURE_SET_LIBRARY.items()):
            with col:
                if st.button(cfg["label"], key=f"bm_preset_{pkey}"):
                    apply_ballmapper_preset(pkey)

        if st.button("Generate Ball Mapper graph", key="generate_ballmapper"):
            fig, node_df = build_ballmapper_plot(epsilon, color_choice, current_feature_cfg["features"])
            st.plotly_chart(fig, use_container_width=True)
            st.info(summarize_topology(node_df, color_choice))
            st.subheader("Node summary")
            st.dataframe(node_df)
            with st.expander("Topology notes", expanded=True):
                top_dense = node_df.nlargest(5, "size")[["node_id", "region", "size", "obesity_rate", "cost_per_patient"]]
                isolated = node_df.nsmallest(5, "size")[["node_id", "region", "size", "obesity_rate", "cost_per_patient"]]
                st.markdown("**Densest clusters (largest balls):**")
                st.dataframe(top_dense)
                st.markdown("**Most isolated clusters (smallest balls):**")
                st.dataframe(isolated)
                st.caption(
                    "Use these tables to note which combinations behave like archetypes versus sparsely populated outliers."
                )
        st.divider()
        render_ballmapper_presets()

    with tabs[3]:
        render_presets_tab(numeric_options, regions)


def compute_interactive_layout(G: nx.Graph) -> dict[int, tuple[float, float]]:
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


if __name__ == "__main__":
    main()
