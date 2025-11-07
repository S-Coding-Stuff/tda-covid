# Visualising English COVID-19 Cases with Topological Data Analysis

This repository contains a Python-based implementation and exploration of the study  
**“Visualising the Evolution of English Covid-19 Cases with Topological Data Analysis Ball Mapper”** by Dlotko and Rudkin (2020).  
The project applies *Topological Data Analysis* (TDA) to understand how COVID-19 case patterns evolved across English regions using socio-economic indicators.

---

## 📘 Overview

The Ball Mapper method provides a topological summary of high-dimensional data by representing it as a network of overlapping balls.  
Here, each ball captures a group of regions with similar socio-economic profiles. By colouring these balls according to COVID-19 case intensity, the method reveals structural relationships that are hard to see with traditional plots.

This project reproduces and extends the results of the original paper using an entirely Python toolchain.

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd tda-covid
   ```
2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Running the analysis

- Place the raw COVID-19 case data and socio-economic indicators in a `data/` folder (structure to be documented as the project evolves).
- Open the notebooks or scripts that reproduce the Ball Mapper workflow (notebooks will live under `notebooks/`).
- Launch JupyterLab for an interactive exploration:
  ```bash
  jupyter lab
  ```

---

## 🛠️ Project structure (planned)

- `data/` – source datasets (not tracked in git).
- `notebooks/` – Jupyter notebooks for exploratory analysis and visualisations.
- `src/` – reusable Python modules implementing Ball Mapper utilities and helpers.

This layout will be updated as soon as code and notebooks are added.
