# Health Index Ball Mapper Explorer

This branch pivots the project toward England’s **Office for National Statistics (ONS) Health Index**.  
Instead of weight-loss prescribing slices, the interactive tooling now focuses on the multi-domain Health Index (Healthy People / Healthy Lives / Healthy Places) and its 200+ indicators. The Streamlit app lets you filter to **LTLA-only** data, pick arbitrary metric combinations, auto-tune ε, and instantly regenerate Ball Mapper, geographic overlays, persistence barcodes, and PDF reports for **any year between 2015–2021**.

---

## 📂 Repository Layout (branch-specific highlights)

```
dataset_experiments/                # legacy prescribing/obesity ETL + Streamlit (still available)
health_index_scores_england/
├── healthindexscoresengland.xlsx   # raw ONS workbook (Tables 3–10)
├── health_index_combined_<year>.csv# 2015–2021 combined tables (Table_3/4/… + IMD quintile table)
├── ballmapper_utils.py             # shared BM helpers
├── streamlit_app.py                # main Health Index Streamlit experience
└── prepare_health_index_csvs.py    # script that generates all combined CSVs
docs/                               # background notes
requirements.txt                    # Python dependencies
```

The legacy prescribing app + scripts remain untouched under `dataset_experiments/`, but this README, the ETL updates, and new UI work all target the Health Index explorer.

---

## ⚙️ Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Rebuild all Health Index CSVs (2015–2021)
```bash
cd health_index_scores_england
python prepare_health_index_csvs.py
```
This script reads `healthindexscoresengland.xlsx`, pulls the yearly tables (`Table_3_2021_Index`, `Table_4_2020_Index`, …, `Table_9_2015_Index`) plus `Table_10_IMD_quintile`, aligns the headers to the 2021 schema, merges IMD quintiles, and writes `health_index_combined_<year>.csv` for every year.

---

## 🚀 Streamlit App (Health Index)

Launch:
```bash
streamlit run health_index_scores_england/streamlit_app.py
```
Key capabilities:

- **Year range 2015–2021** with LTLA-only filtering (IMD quads still available as Source filters).
- **Sidebar filters** (Source + Area Type) persist across year changes. Any change auto-regenerates the plots once you’ve generated them once.
- **Feature selection**: pick arbitrary numeric indicators + choose min–max or z-score normalisation. Preset themes (mental health, chronic disease, etc.) snap in curated feature lists.
- **Correlation explorer** (sidebar toggle): mix-and-match indicator bundles (mental health, deprivation, physical health) or specific metrics to instantly render Pearson/Spearman heatmaps and surface the strongest relationships without leaving the app.
- **Auto ε** + manual slider with auto-regenerate when ε changes (no extra clicks needed once a plot exists).
- **Global colour bounds** reused across years so Ball Mapper + map scales are consistent when toggling years.
- **Map zoom controls** (scroll-to-zoom, modebar buttons) + hover metadata showing Node IDs and metric values.
- **Downloads**: filtered dataset, node summary CSV/JSON, interactive plot HTML/PNG, map PNG, barcode PNG, and the final PDF report.
- **PDF reports** built with ReportLab containing the current Ball Mapper graph, map, persistence barcodes, and top/bottom node tables.

> Data source: [ONS Health Index scores for England](https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/healthandwellbeing/datasets/healthindexscoresengland)

---

## 🧠 Guardrails & Persistence

The most recent updates focus on ensuring a smooth analytical workflow:

1. **State preservation across years** – filters, feature selections, colour metric, size metric, normalisation choice, and ε are remembered when switching years. Once a plot is generated, flipping the year automatically reruns Ball Mapper with identical settings.
2. **LTLA-only comparisons** – the app automatically filters to LTLA rows so year-on-year cluster comparisons remain apples-to-apples. IMD quintile data is still present (as a Source filter) when needed.
3. **Global colour scales** – min/max bounds for each metric are computed across all available years so Ball Mapper and the map reuse the same scale when viewing different years.
4. **Automatic reruns** – after the first graph render, changing year, ε, filters, feature list, or normalisation triggers an immediate rerun (no need to mash the “Generate” button repeatedly). This ensures iteration speed while keeping context intact.
5. **Zoomable map** – choropleth map uses `scrollZoom=True` and exposes Plotly’s zoom controls, so you can dive into specific LTLA clusters without re-running the app.

---

## 🗺️ Roadmap / Next Steps

- **Year-to-year comparison tab**: select baseline vs current year and display node-level deltas, LTLA transition matrices, and map difference colouring.
- **Topology stability curves**: overlay β₀/β₁ vs ε for every year to highlight fragmentation/shocks.
- **Animated playback**: play through years automatically with locked presets and ε to observe cluster evolution.
- **Report tab**: richer layout summarising top deltas, transition matrices, and stability charts (currently the PDF already captures the essentials).

---

## 🔁 Legacy Prescribing Explorer (optional)

The original prescribing + obesity Ball Mapper tooling is still available:
```bash
streamlit run dataset_experiments/streamlit_app.py
```
That app contains the scatter/gender/age/drug-mix explorer plus the original Ball Mapper for Region × Age × Gender prescribing slices. It’s untouched by this branch but remains handy as a reference.

---

## 📄 Reference

- Dłotko, P., & Rudkin, S. (2020). *Visualising the Evolution of English COVID-19 Cases with Topological Data Analysis Ball Mapper.*  
- ONS Health Index data portal (see link above).

Have ideas for new presets, delta analytics, or report layouts? Open an issue or tweak the Streamlit app—this branch is designed to be easily extensible. Happy mapping! 🟢🟡🟣
