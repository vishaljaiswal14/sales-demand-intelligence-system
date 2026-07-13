# Sales Demand Intelligence System

A retail forecasting and demand intelligence system that automates demand planning, audits transaction logs for operational anomalies, and segments product portfolios.

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-Dataframe-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas"></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-Math-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn"></a>
  <a href="https://www.statsmodels.org/"><img src="https://img.shields.io/badge/Statsmodels-Statistics-blue?style=flat-square" alt="Statsmodels"></a>
  <a href="https://facebook.github.io/prophet/"><img src="https://img.shields.io/badge/Prophet-Time%20Series-blue?style=flat-square" alt="Prophet"></a>
  <a href="https://xgboost.readthedocs.io/"><img src="https://img.shields.io/badge/XGBoost-ML-2F80ED?style=flat-square" alt="XGBoost"></a>
  <a href="https://plotly.com/"><img src="https://img.shields.io/badge/Plotly-Charts-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter Notebook"></a>
</p>

## Table of Contents

- [Executive Overview](#executive-overview)
- [Motivation & Business Context](#motivation--business-context)
- [Design Principles](#design-principles)
- [Key Features](#key-features)
- [Project Architecture & Engineering Decisions](#project-architecture--engineering-decisions)
- [Folder Structure](#folder-structure)
- [Technology Stack](#technology-stack)
- [Installation & Setup](#installation--setup)
- [Usage & Execution Workflow](#usage--execution-workflow)
- [Dashboard & Visualizations](#dashboard--visualizations)
- [Key Analytical Results](#key-analytical-results)
- [Engineering Roadmap](#engineering-roadmap)
- [Data Sources & Methodology Notes](#data-sources--methodology-notes)

---

## Executive Overview

This project implements a demand planning pipeline using four years of historical Superstore retail sales data. To select the production model, the pipeline trains and backtests SARIMA, Prophet, and XGBoost models against a chronological hold-out set. Additionally, the system isolates irregular transaction weeks using rolling Z-scores and Isolation Forests, and clusters product categories with K-Means to suggest stocking rules. All analytics are served through an interactive Streamlit dashboard and summarized in a PDF business report.

![Dashboard Overview](docs/screenshots/overview.png)

---

## Motivation & Business Context

Retailers continuously balance inventory costs against service levels: holding too much stock binds working capital, while holding too little leads to stockouts and lost revenue. Optimizing this balance requires answering three questions:
- **Forecast Scale:** What are the expected sales volumes next quarter by product category and geographic region?
- **Operational Exceptions:** Which weeks showed abnormal demand spikes, and what transaction-level details explain them?
- **Inventory Classification:** How do product lines differ in sales volume and stability, and what stocking policies fit each profile?

This project provides an empirical framework to address these questions using historical transaction logs.

---

## Design Principles

- **Decoupled Compute & UI:** The dashboard operates as a read-only presentation layer. It loads pre-rendered artifacts from disk and never executes time-series training or heavy aggregation at runtime.
- **Methodological Transparency:** Every model parameter, outlier threshold, and clustering configuration is empirically justified in the primary notebook rather than defined by default settings.
- **Strict Temporal Validation:** To prevent data leakage, all model evaluations rely on chronological train-test splits and rolling-origin backtests rather than random cross-validation.

---

## Key Features

- **Forecasting:** Evaluates SARIMA, Prophet, and XGBoost on a 12-month chronological hold-out set and a rolling-origin backtest.
- **Aggregation:** Generates monthly category and regional forecasts, separating baseline trends from seasonal patterns.
- **Anomaly Detection:** Combines rolling Z-scores and Isolation Forest models to isolate irregular transaction weeks, mapping outliers back to the underlying customer orders.
- **Segmentation:** Groups product sub-categories using K-Means clustering based on revenue contribution and demand volatility to recommend target stocking rules.
- **Dashboard Visualization:** Serves pre-computed metrics and Plotly charts via a Streamlit interface, bypassing slow run-time calculations.
- **Executive Reporting:** Summarizes analytical findings and business recommendations in a stakeholder-ready PDF report.

---

## Project Architecture & Engineering Decisions

The project separates source code, documentation, and data assets to ensure clean separation of concerns:

```
raw data ──▶ src/ pipelines ──▶ processed datasets ──▶ analysis.ipynb (notebook + charts)
                   │                                        │
                   └──▶ src/dashboard/artifacts.py ──▶ dashboard artifacts ──▶ Streamlit app
```

### Engineering Rationale

* **Single Source of Truth (`src/`):** To prevent code duplication and runtime drift, all data loaders, model pipelines, and visualization styles live under `src/`. Both the research notebook and the dashboard builder import code from these shared modules.
* **Decoupled Training & Rendering:** Fitting models like Prophet or SARIMA on page load degrades user experience and introduces runtime failures. To avoid this, `src/dashboard/artifacts.py` runs modeling pipelines offline and serializes metrics, forecasts, and cluster profiles into static files. The Streamlit dashboard operates purely as a fast, read-only presentation layer.
* **Separation of Narrative & Interaction:** The Jupyter notebook (`analysis.ipynb`) acts as the auditable project log. It documents exploratory data analysis, stationarity checks, diagnostic plots, and methodology comparisons. The dashboard (`app.py`) is reserved exclusively for interactive stakeholder exploration.

---

## Folder Structure

```
sales-demand-intelligence-system/
├── data/
│   ├── raw/                  # Raw transaction history (train.csv)
│   └── processed/            # Cleaned data files and serialized dashboard artifacts
├── src/
│   ├── data/                 # Ingestion, quality checks, and aggregation pipelines
│   ├── features/             # Feature extraction and EDA helper functions
│   ├── models/               # Forecasting, anomaly detection, and clustering models
│   ├── visualization/        # Static plotting functions and style templates
│   ├── utils/                # Project path resolution helpers
│   └── dashboard/            # Streamlit page layouts, components, and figures
├── charts/                   # Exported high-resolution figures for reports
├── docs/
│   └── screenshots/          # Dashboard application screenshots
├── reports/
│   └── Executive_Business_Report.pdf # Formal PDF report for business stakeholders
├── analysis.ipynb            # Research notebook documenting the analytics workflow
├── app.py                    # Streamlit dashboard entry point
└── requirements.txt          # Production package requirements
```

---

## Technology Stack

| Category | Technologies | Purpose |
| :--- | :--- | :--- |
| **Core Runtime** | Python (3.9+) | Base programming language for pipeline and interface development. |
| **Data Engineering** | pandas, NumPy, SciPy | Ingestion, data validation, temporal aggregation, and numeric computations. |
| **Time Series Forecasting** | statsmodels, Prophet, XGBoost | SARIMA, seasonal-trend decomposition, additive models, and gradient boosting. |
| **Unsupervised Learning** | scikit-learn | Isolation Forest for anomaly detection; K-Means for demand segmentation. |
| **Visualization** | Plotly, Matplotlib, Seaborn | Custom static narrative figures and interactive dashboard plots. |
| **Application Layer** | Streamlit | Multi-page dashboard framework and theme engine. |
| **Environment** | Jupyter Notebook | Interactive model validation and reproducible research narrative. |

---

## Installation & Setup

Ensure Python 3.9 or higher is installed. 

### 1. Clone the Repository
Clone the source code to your local machine:
```bash
git clone https://github.com/vishaljaiswal14/sales-demand-intelligence-system.git
cd sales-demand-intelligence-system
```

### 2. Configure the Virtual Environment
Set up an isolated environment and install the required dependencies:
```bash
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Add Raw Data
Download the raw [Superstore Sales dataset](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) from Kaggle and copy `train.csv` into the `data/raw/` folder.

---

## Usage & Execution Workflow

Execute the codebase as a step-by-step pipeline: run the analysis notebook to inspect modeling metrics, compile the static outputs, and then launch the dashboard.

### 1. Research and Model Validation
Launch Jupyter to review EDA, stationarity diagnostics (ADF tests), autocorrelation plots, residuals behavior, and baseline model comparisons:
```bash
jupyter notebook analysis.ipynb
```
Executing the cells regenerates files under `data/processed/` and exports static figures to the `charts/` directory.

### 2. Serialize Pipeline Artifacts
Train the models, score anomalies, and fit K-Means offline, caching the results to `data/processed/dashboard/`:
```bash
python3 -m src.dashboard.artifacts
```
*Note: The dashboard requires these precomputed artifacts. Running `app.py` before building artifacts will result in loading errors.*

### 3. Launch the Dashboard
Run the Streamlit application to explore findings interactively:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to access the multi-page user interface.

---

## Dashboard & Visualizations

The dashboard contains distinct view pages matching specific operational goals:

- **Sales Forecasting Page:** Side-by-side performance evaluation of SARIMA, Prophet, and XGBoost on hold-out periods, complete with prediction intervals.
  
  ![Forecasting Screenshot](docs/screenshots/forecasting.png)

- **Anomaly Detection Page:** Weekly sales irregularities identified via rolling Z-score and Isolation Forest pipelines, showing order-level audit logs.
  
  ![Anomaly Detection Screenshot](docs/screenshots/anomalies.png)

- **Product Clustering Page:** Product sub-categories partitioned into demand segments using K-Means, mapping profiles directly to operational stocking rules.
  
  ![Clustering Screenshot](docs/screenshots/clustering.png)

---

## Key Analytical Results

| Area | Finding / Metric | Operational Impact |
| --- | --- | --- |
| **Scale** | 9,800 transactions, 4,922 unique orders, $2.26M revenue | Establishes the demand baseline. |
| **Growth** | +50.5% annual revenue (2018 vs. 2015), concentrated in 2017–2018 | Signals expanding market share and scale. |
| **Top Category** | Technology (36.6% of revenue); West ($710K) top region | High-value inventory focus area. |
| **Seasonality** | September, November, and December account for the seasonal peak | Guides peak-season stock preparation. |
| **Forecasting** | SARIMA(1,0,1)(0,1,1)₁₂ selected; rolling-origin backtest MAPE: 17.7% | Minimizes planning error over Prophet (18.0%) and XGBoost (20.4%). |
| **Next Quarter** | Projected 2019 Q1: $50.1K (Jan), $35.7K (Feb), $69.7K (Mar) | Direct target metrics for purchasing. |
| **Anomalies** | 19 unusual weeks flagged (11 by Z-score, 11 by Isolation Forest, 3 by both) | Triggers audit of bulk orders and data entry. |
| **Segments** | 4 product clusters (Revenue Backbone, Volatile Tier, Consistent Low-Volume, Slow-Movers) | Maps specific stocking policies (e.g., JIT vs. Min-Max) to each product. |

---

## Engineering Roadmap

1. **Exogenous Variables:** Integrate regional marketing schedules and promotional events into Prophet and XGBoost to capture sudden demand spikes.
2. **Performance Monitoring & Retraining:** Establish a monitoring script to compute prediction drift (e.g., tracking MAPE degradation over time) and trigger pipeline runs when accuracy slips.
3. **Containerized Deployment:** Package the application using Docker for cloud hosting (AWS ECS or GCP Cloud Run), storing serialized artifacts in shared object storage.
4. **Automated Audits:** Transition the batch anomaly detection pipeline to run daily on incoming transaction streams, enabling near-real-time discovery of data entry errors or order spikes.

---

## Data Sources & Methodology Notes

### Excluded Supplementary Datasets
During the exploratory and planning phases, a supplementary Video Game Sales dataset was evaluated for cross-domain integration. It was excluded from the final model pipeline because it shares no logical join keys or temporal overlap with the Superstore transaction log. Merging the datasets would have been forced, introducing synthetic noise rather than adding explanatory power. The evaluation and rationale for using a single-source dataset are documented in the methodology discussion within [analysis.ipynb](file:///Users/vishaljaiswal/Desktop/SalesForecasting_VishalJaiswal/analysis.ipynb).
