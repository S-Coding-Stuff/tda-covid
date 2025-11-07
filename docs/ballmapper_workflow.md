# Ball Mapper Workflow (Python)

The preprocessing pipeline in `dataset_experiments/scripts/preprocess_data.py` produces two artefacts:

1. `dataset_experiments/tda_obesity_prescribing_clean.csv` — regional aggregates with demographic shares.
2. `dataset_experiments/tda_obesity_ballmapper_input.csv` — the same regions plus min–max scaled core features ready for Ball Mapper.

Use the following workflow to generate Ball Mapper graphs entirely in Python.  
Each point now corresponds to a **Region × Age band × Gender** combination so the point cloud carries far more structure than the original region-only view.

## 1. Load and select features

```python
import pandas as pd
from pathlib import Path

DATA_DIR = Path("dataset_experiments")
ballmapper_df = pd.read_csv(DATA_DIR / "tda_obesity_ballmapper_input.csv")

feature_cols = [
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
    "norm_share_items_female",
    "norm_share_items_60plus",
]
X = ballmapper_df[feature_cols].to_numpy()
color_values = ballmapper_df["obesity_rate"].to_numpy()
labels = ballmapper_df["Region"]
```
> The preprocessing script also stores the raw (non-normalised) counterparts for each `"norm_*"` column, so you can colour nodes in whatever metric you prefer (cost per patient, gender shares, YoY deltas, etc.).

## 2. Build a Ball Mapper cover

If you have the [`tda-ballmapper`](https://pypi.org/project/tda-ballmapper/) package installed:

```python
from tda_ballmapper import BallMapper

epsilon = 0.25  # tweak between 0.1 and 0.3 to change connectivity
bm = BallMapper(X, color_values)
cover = bm.build_cover(epsilon)
```

The cover stores which data points fall inside each ball (node). Each node is associated with:

- `bm.landmarks` — indices of landmark points,
- `cover[i]` — list of original points covered by node `i`,
- `bm.values[i]` — the average coloring value for node `i` (here obesity rate).

If the package is unavailable, you can implement a minimal cover:

```python
import numpy as np

def dense_ball_cover(points, radius):
    landmarks = []
    cover = []
    for idx, point in enumerate(points):
        if not landmarks:
            landmarks.append(point)
            cover.append([idx])
            continue
        dists = np.linalg.norm(np.asarray(landmarks) - point, axis=1)
        if (dists <= radius).any():
            cover[dists.argmin()].append(idx)
        else:
            landmarks.append(point)
            cover.append([idx])
    return np.asarray(landmarks), cover
```

## 3. Build the Ball Mapper graph

```python
import networkx as nx

G = nx.Graph()
for node_id, members in enumerate(cover):
    G.add_node(
        node_id,
        members=members,
        size=len(members),
        color=color_values[members].mean(),
        label=labels.iloc[node_id],
    )

for i in range(len(cover)):
    for j in range(i + 1, len(cover)):
        if set(cover[i]).intersection(cover[j]):
            G.add_edge(i, j)
```

## 4. Visualise

You can render interactively with `plotly` or export static layouts using `networkx` and `matplotlib`:

```python
import matplotlib.pyplot as plt
import numpy as np

pos = nx.spring_layout(G, seed=42)
colors = [G.nodes[n]["color"] for n in G.nodes]
sizes = [G.nodes[n]["size"] * 15 for n in G.nodes]

fig, ax = plt.subplots(figsize=(6, 5))
scatter = nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, cmap="viridis", ax=ax)
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)
nx.draw_networkx_labels(G, pos, labels={n: labels.iloc[n] for n in G.nodes}, font_size=8)
fig.colorbar(scatter, ax=ax, label="Obesity rate (%)")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(DATA_DIR / "figures" / "ballmapper_obesity.png", dpi=300)
```

> Tip: experiment with different ε values and alternative colouring metrics (e.g., `cost_per_patient`) to surface distinct regional patterns.

## 5. Interactive Streamlit App

Launch the interactive explorer (scatter plots + Ball Mapper playground) with:

```bash
streamlit run dataset_experiments/streamlit_app.py
```

The sidebar lets you pick scatter metrics, filter regions, adjust ε, and swap the ball colouring metric between obesity rate and cost per patient.
