# Sales Demand Intelligence System

An end-to-end retail forecasting and demand intelligence system. Implements statistical time-series models, machine learning anomaly detection, and product segmentation on four years of historical transaction data, served via an interactive dashboard and an executive business report.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-ORANGE?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-2F80ED?style=flat-square)](https://xgboost.readthedocs.io/)
[![Prophet](https://img.shields.io/badge/Prophet-Forecasting-blue?style=flat-square)](https://facebook.github.io/prophet/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-Time%20Series-green?style=flat-square)](https://www.statsmodels.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)

---

## Technologies

`Python` `Pandas` `NumPy` `scikit-learn` `Statsmodels` `SARIMA` `Prophet` `XGBoost` `Isolation Forest` `K-Means` `PCA` `Plotly` `Matplotlib` `Seaborn` `Streamlit` `Jupyter Notebook` `Machine Learning` `Time Series Forecasting` `Retail Analytics` `Business Intelligence` `Data Visualization` `Feature Engineering`

---

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

This project implements a demand planning and business intelligence pipeline using four years of Superstore retail transaction data. Instead of relying on a single forecasting model, the system evaluates three distinct modeling methodologies (SARIMA, Prophet, and XGBoost) on chronological hold-out and rolling-origin backtests to select the most accurate model for production. 

Beyond high-level trends, the pipeline processes the transaction history to identify weekly anomalies via statistical and machine learning estimators (Z-score and Isolation Forest) and categorizes products into distinct stocking segments using K-Means clustering. The outputs are served through a sub-second load Streamlit dashboard and summarized in an executive report written for business stakeholders.

![Dashboard Overview](docs/screenshots/overview.png)

---

## Motivation & Business Context

In retail operations, inventory management is a major driver of cost and capital efficiency. Carrying excess inventory ties up working capital and increases storage costs, while stockouts result in lost revenue and reduced customer retention. 

To balance this trade-off, planners require tools that answer specific operational questions:
- **Trend and Scale:** What is the expected sales volume for the upcoming quarter by category and region?
- **Operational Exceptions:** Which weeks showed abnormal transaction behavior, and what orders drove those anomalies?
- **Inventory Classification:** How do different product lines behave in terms of volume and stability, and how should stocking policies adapt?

This repository provides an empirical, data-driven approach to answering these questions. It replaces intuition-based planning with model-backed intervals and auditable statistical thresholds.

---

## Design Principles

- **Decoupled Computation and Presentation:** The interactive application never runs model training or complex aggregations. All analytical steps are executed once, serialized to disk, and loaded read-only by the interface.
- **Methodological Transparency:** Every model selection, anomaly threshold, and clustering configuration is supported by validation metrics in the primary Jupyter notebook.
- **Strict Temporal Validation:** To prevent data leakage, all model evaluation uses time-based train-test splits and rolling-origin validation rather than random cross-validation.

---

## Key Features

- **Chronological Model Evaluation:** Evaluates SARIMA, Prophet, and XGBoost on a 12-month hold-out set and a rolling-origin backtest.
- **Multi-Granular Forecasting:** Generates monthly category and region forecasts, separating baseline trends from seasonal variations.
- **Dual-Estimator Anomaly Detection:** Combines Z-scores on rolling averages with an Isolation Forest model to isolate irregular weeks, mapping flags back to line-item orders.
- **Demand Segmentation:** Clusters product sub-categories based on revenue contribution and demand volatility, generating concrete stocking rules.
- **High-Performance App:** A Streamlit dashboard that serves pre-computed JSON and CSV artifacts, ensuring fast loads and UI consistency.
- **Executive Summary:** A formal report (`reports/Executive_Business_Report.pdf`) focused on business findings and operational recommendations.

---

## Project Architecture & Engineering Decisions

The project separates source code, documentation, and data assets to ensure clean separation of concerns:

```
raw data ──▶ src/ pipelines ──▶ processed datasets ──▶ analysis.ipynb (notebook + charts)
                   │                                        │
                   └──▶ src/dashboard/artifacts.py ──▶ dashboard artifacts ──▶ Streamlit app
```

### Engineering Decisions:

1. **Why `src/` Exists:** To avoid notebook bloat, all core algorithms, data loaders, feature engineering pipelines, and visualization styles are defined in clean Python modules. This allows the same logic to be shared by the analysis notebook and the dashboard builder, ensuring consistency.
2. **Why the Notebook Exists:** `analysis.ipynb` serves as the research and narrative layer. It documents the exploratory data analysis, parameter search space, diagnostic checks (such as ACF/PACF plots), and model comparison details.
3. **Decoupling Modeling from Streamlit:** Streamlit is designed for quick rendering but lacks a built-in state machine for long-running mathematical operations. Running models like Prophet or SARIMA on page load would cause slow execution and introduce runtime instabilities. Instead, the model artifacts are serialized beforehand by `src/dashboard/artifacts.py` into lightweight JSON/CSV formats, allowing the Streamlit app to load instantly.

---

## Folder Structure

```
sales-demand-intelligence-system/
├── data/
│   ├── raw/                     # Raw transaction history (train.csv)
│   └── processed/               # Cleaned datasets and serialized dashboard artifacts
├── src/
│   ├── data/                     # Ingestion, quality checks, and aggregation pipelines
│   ├── features/                 # Temporal feature engineering and EDA helpers
│   ├── models/                   # SARIMA, Prophet, XGBoost, anomalies, and K-Means models
│   ├── visualization/            # Custom matplotlib/seaborn chart builders and styling
│   ├── utils/                    # Shared system path helpers
│   └── dashboard/                # Streamlit views, layouts, custom themes, and figures
├── charts/                       # PNG figures generated by the analysis notebook
├── docs/
│   └── screenshots/             # Reference screenshots of the Streamlit dashboard
├── reports/
│   └── Executive_Business_Report.pdf  # Executive summary and business report
├── analysis.ipynb                # Main analysis notebook (tasks 1-6)
├── app.py                        # Streamlit dashboard entry point
└── requirements.txt              # Project dependency file
```

---

## Technology Stack

### Programming Language & Core Libraries
- **Python (3.9+)**: Base execution runtime.
- **pandas**: Tabular data manipulation, time-series resampling, and window-based aggregations.
- **NumPy**: Linear algebra and vector calculations.
- **SciPy**: Statistical tests (e.g., box-cox transformation search).

### Time Series & Statistics
- **statsmodels**: Implementation of SARIMA modeling, classical decomposition, ADF tests, and Ljung-Box diagnostic checks.
- **Prophet**: Additive regression model for handling strong yearly seasonality.
- **XGBoost**: Gradient boosted trees used as a machine learning alternative for regression-based time-series forecasting.

### Machine Learning & Clustering
- **scikit-learn**: Isolation Forest estimator for multi-dimensional anomaly detection, K-Means for clustering, and PCA for visualization.

### Visualization & User Interface
- **Plotly (Express & Graphic Objects)**: Engine for generating interactive charts in the Streamlit interface.
- **Matplotlib & Seaborn**: Production of publication-quality static figures inside the analysis notebook.
- **Streamlit**: Multi-page dashboard framework.

---

## Installation & Setup

Ensure Python 3.9 or higher is installed. 

### 1. Clone the Repository
```bash
git clone https://github.com/vishaljaiswal14/sales-demand-intelligence-system.git
cd sales-demand-intelligence-system
```

### 2. Configure Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Place Data Asset
The raw transaction dataset is not checked into version control. Download the [Superstore Sales dataset](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) from Kaggle and place the raw file in the `data/raw/` directory as `train.csv`.

---

## Usage & Execution Workflow

### 1. Run the Narrative Analysis
To review or run the analytical steps, open the Jupyter environment:
```bash
jupyter notebook analysis.ipynb
```
Executing the notebook from top to bottom executes data audits, feature extraction, model evaluations, anomaly identification, and exports all figures to the `charts/` directory.

### 2. Generate Dashboard Artifacts
Before running the dashboard, serialize the pipeline results to disk. This runs the data aggregation, training, anomaly detection, and clustering pipelines, writing the final JSON and CSV outputs to `data/processed/dashboard/`.
```bash
python3 -m src.dashboard.artifacts
```
*Note: The dashboard app requires these artifacts to run; launching the dashboard without executing this step will cause data loading errors.*

### 3. Launch the Dashboard
Run the Streamlit server locally:
```bash
streamlit run app.py
```
This opens the multi-page dashboard in your default browser at `http://localhost:8501`.

---

## Dashboard & Visualizations

The dashboard contains distinct view pages matching specific operational goals:

- **Sales Forecasting View:** Displays model fits and quarterly predictions along with statistical confidence intervals.
  
  ![Forecasting Screenshot](docs/screenshots/forecasting.png)

- **Anomaly Detection View:** Displays anomalous weeks side-by-side from the rolling Z-score and Isolation Forest pipelines, with detailed order breakdowns.
  
  ![Anomaly Detection Screenshot](docs/screenshots/anomalies.png)

- **Demand Segmentation View:** Plots the K-Means clusters and lists custom stocking recommendations (e.g., JIT, Safety Stock) based on cluster profiles.
  
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

1. **Incorporate External Regressors:** Integrate local marketing campaign calendars and major regional holidays (e.g., Black Friday) as exogenous variables to improve holiday forecasts.
2. **Model Retraining and Drift Alerts:** Implement an automated retraining schedule (cron) that monitors Mean Absolute Percentage Error (MAPE) drift and flags when models require hyperparameter adjustment.
3. **Production Deployment:** Containerize the dashboard using Docker and deploy to a cloud instance (e.g., AWS ECS or GCP Cloud Run), using cached storage buckets for data artifacts.
4. **Near-Real-Time Anomalies:** Move the weekly batch anomaly detection script to a streaming or daily ingest pipeline to identify data entry errors or sudden order surges within 24 hours of occurrence.

---

## Data Sources & Methodology Notes

### Excluded Supplementary Datasets
During the exploratory and methodology planning phase, a supplementary Video Game Sales dataset from Kaggle was evaluated. This dataset was intentionally excluded from the final pipeline for the following reasons:
- **No Logical Join Keys:** The dataset shared no customer, temporal, or product attributes with the retail Superstore transactional data.
- **Arbitrary Merging:** Merging the two datasets would require generating artificial mapping keys, which would introduce synthetic noise into the time-series history rather than providing genuine analytical insights.

All details of this evaluation and the decision to maintain a single-source dataset are documented at the beginning of Task 5 in the analysis notebook.
