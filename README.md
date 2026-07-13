# Sales Demand Intelligence System

A retail forecasting and demand intelligence system that processes historical transaction logs to automate demand planning, isolate operational anomalies, and segment product portfolios.

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

This system processes historical retail transaction logs to generate product-level forecasts, audit operations for anomalous order spikes, and classify inventory into target stocking cohorts. 

To maximize forecast accuracy, the project evaluates and backtests three competing models (SARIMA, Prophet, and XGBoost) on a chronological hold-out set. The statistical outputs, clustering properties, and anomaly reports are served through a lightweight, pre-rendered Streamlit dashboard alongside a PDF business report.

![Dashboard Overview](docs/screenshots/overview.png)

---

## Motivation & Business Context

Retailers face a continuous optimization challenge: over-ordering binds working capital in inventory, while under-ordering triggers stockouts and customer churn. Balancing this trade-off requires answering concrete questions:
- **Baseline Forecasting:** What are the expected sales volumes for next quarter across product categories and geographic regions?
- **Exception Auditing:** Which weeks exhibited irregular demand spikes, and which customer orders drove those anomalies?
- **Stocking Segmentation:** Which product sub-categories drive the bulk of revenue, and which are too volatile to stock under standard Min-Max rules?

This repository addresses these operational questions by mapping transaction logs to statistical forecasts, anomaly scores, and product classifications.

---

## Design Principles

- **Decoupled Compute & Presentation:** The frontend dashboard never runs models or heavy aggregations. The analytical pipeline runs once, serializing results to disk for read-only consumption.
- **Methodological Transparency:** Every parameter choice, anomaly threshold, and clustering hyperparameter is documented and validated inside the notebook, rather than assumed.
- **Strict Temporal Validation:** To guarantee realistic performance, all models are evaluated using chronological hold-outs and rolling-origin backtests instead of random k-fold splits.

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

The directory structure and execution workflow are designed around three principles: a single source of truth, decoupled runtimes, and reproducible narrative.

- **Single Source of Truth (`src/`):** All modeling, loading, feature engineering, and styling logic is written once in pure Python modules under `src/`. The Jupyter notebook and the dashboard builder both import these identical packages. This eliminates duplicate business logic and ensures that research code and production dashboard metrics cannot drift.
- **Decoupled Compute & Presentation (Artifact Serialization):** Running time-series modeling (like fitting SARIMA or Prophet) at page load is a bad engineering practice. It introduces performance lag and state instability. To solve this, the pipeline separates model fitting from user interaction. `artifacts.py` runs the models offline, serializes outputs into lightweight JSON/CSV files, and the Streamlit app reads these precomputed files. The dashboard acts as a pure, fast presentation layer.
- **Separation of Narrative and Code:** The Jupyter notebook (`analysis.ipynb`) serves as the research history, documenting EDA, statistical diagnostic checks (such as residuals tests), and methodology validation. It is designed to be executed from top to bottom for auditability. Meanwhile, the dashboard (`app.py`) is optimized for interactive stakeholder interrogation.

---

## Folder Structure

```
sales-demand-intelligence-system/
├── data/
│   ├── raw/                  # Raw transaction history (train.csv)
│   └── processed/            # Cleaned data files and serialized dashboard artifacts
├── src/
│   ├── data/                 # Ingestion, quality audits, and aggregation pipelines
│   ├── features/             # Feature engineering and EDA calculations
│   ├── models/               # Forecasting, anomaly, and clustering implementations
│   ├── visualization/        # Matplotlib/Seaborn custom plotting functions
│   ├── utils/                # Shared path mapping helpers
│   └── dashboard/            # Streamlit view rendering, theme, and Plotly figures
├── charts/                   # Exported high-resolution figures for reports
├── docs/
│   └── screenshots/          # Dashboard layout screenshots
├── reports/
│   └── Executive_Business_Report.pdf # Formal PDF report for business stakeholders
├── analysis.ipynb            # Analytical notebook documenting the research workflow
├── app.py                    # Streamlit entry point script
└── requirements.txt          # Python dependency specifications
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

### 3. Retrieve and Place the Dataset
The raw transaction dataset is not checked into version control. Download the raw data:
1. Visit [Superstore Sales dataset on Kaggle](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting).
2. Download the dataset and copy the `train.csv` file into the `data/raw/` directory.

---

## Usage & Execution Workflow

The codebase is designed as a pipeline. First, run the notebook for narrative research and validation. Next, compile the modeling outputs into static files. Finally, run the Streamlit app to visualize them.

### 1. Narrative Analysis & Model Validation
Start Jupyter and execute the narrative notebook to review the exploratory analysis, statistical diagnostics (ADF, ACF/PACF, residuals tests), and baseline model comparisons:
```bash
jupyter notebook analysis.ipynb
```
Running the cells exports high-resolution figures to the `charts/` folder and saves processed features.

### 2. Build Dashboard Artifacts
Generate and cache the predictions, anomaly labels, and product segments. This execution runs the models offline, saving the results under `data/processed/dashboard/`:
```bash
python3 -m src.dashboard.artifacts
```
*Note: The Streamlit app relies on these precomputed artifacts. You must run this command before launching the dashboard.*

### 3. Launch the Streamlit Dashboard
Start the local web server to explore the results interactively:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to view the multi-page application.

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

1. **Exogenous Feature Expansion:** Integrate promotional calendars and public holidays into the Prophet and XGBoost pipelines to improve forecast response during peak promotional events.
2. **Continuous Training & Performance Monitoring:** Build a scheduled execution script (cron) to monitor forecast accuracy drift, computing error metrics (MAPE/RMSE) on incoming data and triggering model refitting if performance falls below a set baseline.
3. **Containerized Deployment:** Package the dashboard application into a Docker container and set up a deployment configuration for cloud hosting (e.g., AWS ECS or GCP Cloud Run), utilizing shared cloud storage for the precomputed artifacts.
4. **Low-Latency Anomaly Monitoring:** Shift the anomaly detection script from batch execution to a daily pipeline, alerting operations planners to bulk order entry errors or sudden demand shifts within 24 hours.

---

## Data Sources & Methodology Notes

### Excluded Supplementary Datasets
During the exploratory and planning phases, a supplementary Video Game Sales dataset was evaluated for cross-domain integration. It was excluded from the final model pipeline because it shares no logical join keys or temporal overlap with the Superstore transaction log. Merging the datasets would have been forced, introducing synthetic noise rather than adding explanatory power. The evaluation and rationale for using a single-source dataset are documented in the methodology discussion within [analysis.ipynb](file:///Users/vishaljaiswal/Desktop/SalesForecasting_VishalJaiswal/analysis.ipynb).
