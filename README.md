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

<p align="center">
  <a href="https://sales-demand-intelligence-system-9qruggze9cgxvu6k6nimqw.streamlit.app/"><b>🚀 Deployed Live Dashboard</b></a>
  &nbsp;•&nbsp;
  <a href="reports/Executive_Business_Report.pdf"><b>📄 Executive Report (PDF)</b></a>
  &nbsp;•&nbsp;
  <a href="analysis.ipynb"><b>📓 Analysis Notebook</b></a>
  &nbsp;•&nbsp;
  <a href="https://github.com/vishaljaiswal14/sales-demand-intelligence-system"><b>⭐ GitHub Repository</b></a>
</p>

---

## Table of Contents

- [Executive Overview](#executive-overview)
- [Project Highlights](#project-highlights)
- [Project Scale](#project-scale)
- [Design Principles](#design-principles)
- [Key Features](#key-features)
- [Project Architecture & Engineering Decisions](#project-architecture--engineering-decisions)
- [Folder Structure](#folder-structure)
- [Technology Stack](#technology-stack)
- [Installation & Setup](#installation--setup)
- [Usage & Execution Workflow](#usage--execution-workflow)
- [Live Demo](#live-demo)
- [Dashboard Showcase](#dashboard-showcase)
- [Key Analytical Results](#key-analytical-results)
- [Engineering Decisions](#engineering-decisions)
- [Future Improvements](#future-improvements)
- [Data Sources & Methodology Notes](#data-sources--methodology-notes)

---

## Executive Overview

This project implements a demand planning pipeline using four years of historical Superstore retail sales data. To select the production model, the pipeline trains and backtests SARIMA, Prophet, and XGBoost models against a chronological hold-out set. Additionally, the system isolates irregular transaction weeks using rolling Z-scores and Isolation Forests, and clusters product categories with K-Means to suggest stocking rules. All analytics are served through an interactive Streamlit dashboard and summarized in a PDF business report.

![Dashboard Overview](docs/screenshots/overview.png)

---

## Project Highlights

* **Multi-Model Pipeline:** Evaluates SARIMA, Prophet, and XGBoost models against chronological hold-out splits.
* **Dual-Method Anomaly Detection:** Cross-checks statistical rolling Z-scores against a multi-dimensional Isolation Forest.
* **K-Means Demand Segmentation:** Groups products by revenue and volatility to assign distinct stocking rules.
* **Decoupled Architecture:** Serializes pipeline predictions to disk to enable sub-second dashboard loading.
* **Stakeholder Reporting:** Delivers findings in a PDF executive report formatted for non-technical leadership.
* **Full Reproducibility:** Entire modeling process runs sequentially from raw logs in a single Jupyter notebook.

---

## Project Scale

* **9,800** retail transactions across **4,922** unique customer orders spanning **4** years.
* **3** competing forecasting models compared under chronological hold-out splits.
* **2** anomaly detection methods (Z-score and Isolation Forest) auditing operations.
* **4** product demand clusters mapping inventory to custom stocking rules.
* **21** high-resolution figures generated for notebooks and reports.
* **7** interactive dashboard views served via Streamlit.
* **13-page** executive report summarizing pipeline findings for leadership.

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

* **Single Source of Truth (`src/`):** To prevent code duplication and runtime drift, all data loaders, model pipelines, and visualization styles live under `src/`. Both the research notebook and the dashboard builder import code from these shared modules. This ensures any bug fix or pipeline update is immediately reflected across all outputs.
* **Decoupled Training & Rendering:** Fitting models like Prophet or SARIMA on page load degrades user experience and introduces runtime failures. To avoid this, `src/dashboard/artifacts.py` runs modeling pipelines offline and serializes metrics, forecasts, and cluster profiles into static files. The Streamlit dashboard operates purely as a fast, read-only presentation layer.
* **Separation of Narrative & Interaction:** The Jupyter notebook (`analysis.ipynb`) acts as the auditable project log. It documents exploratory data analysis, stationarity checks, diagnostic plots, and methodology comparisons. The dashboard (`app.py`) is reserved exclusively for interactive stakeholder exploration.
* **Reproducibility & Maintainability:** The entire project can be rebuilt from scratch by running the notebook or the artifact builder. Placing core pipelines in testable python files makes the system maintainable and ready for future production containerization.

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

### Programming
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

### Analytics
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org/)

### Forecasting & Statistics
[![Statsmodels](https://img.shields.io/badge/Statsmodels-green?style=flat-square)](https://www.statsmodels.org/)
[![Prophet](https://img.shields.io/badge/Prophet-blue?style=flat-square)](https://facebook.github.io/prophet/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2F80ED?style=flat-square)](https://xgboost.readthedocs.io/)

### Machine Learning
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

### Visualization
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-gray?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-blue?style=flat-square)](https://seaborn.pydata.org/)

### Application Layer
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)

### Development
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)

---

## Installation & Setup

### Prerequisites
* Python 3.9 or higher.
* A Kaggle account (to download the dataset).

### Clone the Repository
```bash
git clone https://github.com/vishaljaiswal14/sales-demand-intelligence-system.git
cd sales-demand-intelligence-system
```

### Configure Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
```

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset Setup
1. Download the [Superstore Sales dataset](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) from Kaggle.
2. Place the downloaded `train.csv` file into the `data/raw/` directory.

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

## Live Demo

<p align="center">
  <a href="https://sales-demand-intelligence-system-9qruggze9cgxvu6k6nimqw.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Access%20Live%20Streamlit%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit App">
  </a>
</p>

This interactive Streamlit dashboard runs in the cloud. It loads the precompiled modeling, clustering, and anomaly metrics generated by the project pipeline, ensuring the dashboard matches the analytical outputs of the research notebook.

---

## Dashboard Showcase

### 1. Executive Sales Overview
An interactive review of operational metrics, historical sales trends, fulfillment times, and regional performance.
![Overview](docs/screenshots/overview.png)

### 2. Time-Series Forecasting
Side-by-side performance evaluation of SARIMA, Prophet, and XGBoost models against chronological hold-out test periods.
![Forecasting](docs/screenshots/forecasting.png)

### 3. Operational Anomaly Auditing
Weekly demand screening combining rolling statistical Z-scores and Isolation Forests, showing transaction-level logs for each outlier.
![Anomalies](docs/screenshots/anomalies.png)

### 4. Demand Segmentation & Inventory Policy
K-Means clustering categorizes product sub-categories into demand cohorts and suggests stocking rules (e.g., Just-In-Time vs. Safety Stock).
![Clustering](docs/screenshots/clustering.png)

---

## Key Analytical Results

| Target Area | Primary Metric / Finding | Operational Impact |
| :--- | :--- | :--- |
| **Scale** | 9,800 transactions, 4,922 unique orders, $2.26M revenue | Baseline business volume metrics. |
| **Growth** | +50.5% annual revenue (2018 vs. 2015), concentrated in 2017–2018 | High-growth operational demand context. |
| **Top Segment** | Technology category (36.6% of revenue); West region ($710K) | Focus area for high-value SKU management. |
| **Seasonality** | September, November, and December rank top-three in all years | Guides seasonal inventory buildup timing. |
| **Forecasting** | SARIMA(1,0,1)(0,1,1)₁₂ selected; rolling-origin backtest MAPE: 17.7% | Minimizes estimation error against Prophet (18.0%) and XGBoost (20.4%). |
| **Next Quarter** | Projected 2019 Q1: $50.1K (Jan), $35.7K (Feb), $69.7K (Mar) | Target demand levels for inventory planning. |
| **Anomalies** | 19 unusual weeks flagged (11 Z-score, 11 Isolation Forest, 3 by both) | Triggers audit of bulk orders and data entry. |
| **Segments** | 4 clusters (Revenue Backbone, Volatile, Consistent Low-Volume, Slow) | Dictates SKU-level stocking rules (Min-Max, JIT, Safety Stock). |

---

## Engineering Decisions

* **Why SARIMA was selected for production:** Though Prophet and XGBoost were evaluated, SARIMA(1,0,1)(0,1,1)₁₂ achieved the lowest error (17.7% MAPE) on rolling-origin backtests. SARIMA is mathematically structured to model the monthly seasonality and autoregressive features of Superstore sales without overfitting.
* **Why the dashboard loads precomputed artifacts:** Fitting statistical models during page load is computationally slow and causes Streamlit frontend lag. Serializing predictions and model coefficients to static JSON/CSV files ensures sub-second page loads and guarantees consistency with the notebook findings.
* **Why rolling Z-scores are paired with Isolation Forest:** Rolling Z-scores detect amplitude outliers (sudden spikes in revenue), while Isolation Forests detect multi-dimensional outliers (combinations of order frequency, category shifts, and region deviations), ensuring a more comprehensive operational audit.
* **Why the Video Game dataset was excluded:** The dataset shared no keys or attributes with the Superstore data. Merging them would have been forced and introduced artificial trends into the time-series models rather than providing real analytics insights.
* **Why a modular directory structure is used:** Moving core logic from notebook cells to python packages (`src/`) ensures code reusability and maintainability. This structure prevents logic duplication, facilitates testing, and simplifies potential future API packaging or Docker container deployments.

---

## Future Improvements

1. **Exogenous Variables:** Integrate regional marketing schedules and promotional events into Prophet and XGBoost to capture sudden demand spikes.
2. **Performance Monitoring & Retraining:** Establish a monitoring script to compute prediction drift (e.g., tracking MAPE degradation over time) and trigger pipeline runs when accuracy slips.
3. **Containerized Deployment:** Package the application using Docker for cloud hosting (AWS ECS or GCP Cloud Run), storing serialized artifacts in shared object storage.
4. **Automated Audits:** Transition the batch anomaly detection pipeline to run daily on incoming transaction streams, enabling near-real-time discovery of data entry errors or order spikes.

---

## Data Sources & Methodology Notes

### Excluded Supplementary Datasets
During the exploratory and planning phases, a supplementary Video Game Sales dataset was evaluated for cross-domain integration. It was excluded from the final model pipeline because it shares no logical join keys or temporal overlap with the Superstore transaction log. Merging the datasets would have been forced, introducing synthetic noise rather than adding explanatory power. The evaluation and rationale for using a single-source dataset are documented in the methodology discussion within [analysis.ipynb](file:///Users/vishaljaiswal/Desktop/SalesForecasting_VishalJaiswal/analysis.ipynb).
