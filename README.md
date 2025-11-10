# Regional Weight-Loss Prescribing & TDA Explorer

This branch re-implements the Ball Mapper approach from Dłotko & Rudkin (2020) for a different public-health setting:  
**weight-loss medication prescribing + adult obesity prevalence across NHS regions of England.**  
Each data point now represents a **Region × Age band × Gender** slice so we can inspect demographic structure, cost hotspots, and prescription equity.

---

## 🧱 Repository layout

```
dataset_experiments/
├── foi-02477.csv / foi-02477-with-obesity.csv   # raw + enriched FOI data
├── indicator-93881-all-areas.data.csv          # Fingertips obesity indicator
├── tda_obesity_prescribing_clean.csv           # Region × Age × Gender aggregates
├── tda_obesity_ballmapper_input.csv            # normalised features for Ball Mapper
├── scripts/
│   ├── preprocess_data.py                      # ETL → clean + normalised datasets
│   ├── generate_figures.py                     # static matplotlib insight figures
│   ├── run_ballmapper.py                       # CLI Ball Mapper renderer
│   └── ballmapper_utils.py                     # shared utilities
├── streamlit_app.py                            # interactive dashboard
└── figures/                                    # exported PNGs (scatter, heatmaps, BM)
docs/
└── ballmapper_workflow.md                      # manual workflow notes
```

---

## ⚙️ Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Preprocess / normalise
```bash
python dataset_experiments/scripts/preprocess_data.py
```
Creates:
- `tda_obesity_prescribing_clean.csv` – aggregated metrics (items/patients/cost, obesity rate, YoY deltas, gender share, drug-class totals/shares).
- `tda_obesity_ballmapper_input.csv` – min–max scaled features ready for Ball Mapper.

### 2. Static figures
```bash
python dataset_experiments/scripts/generate_figures.py
```
Writes PNGs (scatter, age heatmap, gender stacks, YoY bars, etc.) to `dataset_experiments/figures/`.

### 3. CLI Ball Mapper
```bash
python dataset_experiments/scripts/run_ballmapper.py --epsilon 0.2 --color-metric obesity_rate
```
Produces a PNG plus `ballmapper_graph.json` describing node memberships/metrics.

### 4. Streamlit app
```bash
streamlit run dataset_experiments/streamlit_app.py
```
Tabs include:
- **Overview** – project context + dataset stats.
- **Explorer** – scatter + YoY table with region/age/gender filters plus a *Drug Mix Explorer* (interactive bar/table of drug shares).
- **Ball Mapper** – interactive BM graph with ε nudges, colour toggles, feature-set presets (including a *Drug Mix* preset and colouring by individual drug shares).
- **Preset views** – quick configurations for the scatter explorer.

### 5. Health Index playground
```bash
streamlit run health_index_scores_england/streamlit_app.py
```
Explore the `health_index_combined_2021.csv` dataset with fully configurable Ball Mapper controls:
- Filter by source (England / Regions / IMD Quintiles) and area type.
- Pick any combination of Health Index metrics and normalisation scheme (min-max or z-score).
- Use preset themes (mental health, lifestyle risk, chronic disease, preventive health, urban environment, etc.).
- Adjust ε with ±10% buttons; plots regenerate automatically.
- Download filtered data, node summaries, and JSON payloads directly from the UI.

---

## 🧮 Feature engineering highlights

- **Aggregation**: Region × Age band × Gender with summed `Items`, `Patients`, `Net Cost`, averaged obesity rate, per-patient/per-item costs, and YoY deltas (Dec 23–Nov 24 vs Dec 22–Nov 23 when available).
- **Gender share**: within each region–age slice we compute the proportion of items attributed to each gender.
- **Drug mix**: class-level totals (`items_semaglutide`, `items_liraglutide`, …) and fractional shares (`share_semaglutide`, …) capture therapy uptake patterns.
- **Ball Mapper**: default point cloud uses `[norm_items, norm_patients, norm_net_cost, norm_obesity_rate]`. Preset buttons in the app swap in other feature sets (cost focus, outlier hunt, *drug mix*) and adjust ε/colour automatically.

---

## 🎨 Suggested figure set

1. **Cost vs Obesity scatter** (bubble size = patients, colour = items/patient).  
2. **Gender stack bars** per region.  
3. **Age-band heatmap** of item share.  
4. **Patients vs Items** scatter (Region × Age × Gender).  
5. **Drug Mix Explorer** bar chart (from Streamlit) highlighting class shares for a selected Region/Age/Gender slice.  
6. **Ball Mapper** screenshots coloured by:
   - Adult obesity rate (%)
   - Net ingredient cost (£)
   - Items
   - Female share
   - Semaglutide share (or other drug classes)

Use the Streamlit quick buttons or the CLI to capture each colouring.

---

## 📖 References

- Dłotko, P., & Rudkin, S. (2020). *Visualising the Evolution of English Covid-19 Cases with Topological Data Analysis Ball Mapper.*
- FOI dataset: NHS England (FOI-02477).  
- Obesity indicator: Public Health England (Indicator 93881 – adult obesity prevalence).

---

Questions or suggestions? Open an issue or tweak the Streamlit app to prototype new TDA views. Happy mapping! 🟣🟠🟡
