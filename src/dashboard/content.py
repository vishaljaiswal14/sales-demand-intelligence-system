"""Curated narrative copy for the dashboard.

Presentation text only. Every figure quoted here was computed and verified in
``analysis.ipynb`` (Tasks 1-6); dynamic numbers are formatted in at render time from the
artifacts, and static ones are stated exactly as the notebook produced them.
"""

EXECUTIVE_SUMMARY = (
    "Across four years of Superstore trading (2015–2018), revenue grew "
    "<b>{growth}</b> to <b>{total}</b>, driven by a pronounced second-half seasonality — "
    "September, November, and December rank in each year's top three months in all four "
    "years. <b>Technology</b> leads categories at {cat_share} of revenue; <b>West</b> is "
    "the largest region while East is the most consistent grower (R² ≈ 0.997 on a "
    "straight-line trend). A SARIMA model — selected over Prophet and XGBoost on a "
    "rolling-origin backtest — projects the next quarter with calibrated confidence "
    "intervals, and a two-method anomaly screen flagged {anomalies} unusual sales weeks "
    "for review."
)

MODEL_BLURBS = {
    "SARIMA": (
        "Statistical state-space model — SARIMA(1,0,1)(0,1,1)₁₂, the classic airline "
        "structure. Orders were selected by AICc from an ACF/PACF-motivated grid; "
        "residuals pass the Ljung-Box white-noise test (p = {ljung_box:.2f}), so its 95% "
        "confidence intervals are statistically calibrated."
    ),
    "Prophet": (
        "Additive Bayesian decomposition (trend + Fourier seasonality). Configured for "
        "monthly data: weekly/daily seasonality off, yearly Fourier order capped at 4 to "
        "avoid overfitting 12 points per cycle. Its independent trend/seasonality "
        "estimates corroborate the classical decomposition from Task 2."
    ),
    "XGBoost": (
        "Gradient-boosted trees on engineered lag features (lags 1–3, 3-month rolling "
        "mean, month/quarter/season). Shallow, heavily regularised for a 45-row training "
        "set. Structural caveat: trees cannot extrapolate beyond the sales range seen in "
        "training and provide no native confidence interval."
    ),
}

RECOMMENDATION = (
    "<b>SARIMA is the production recommendation.</b> On the rolling-origin backtest (18 "
    "pooled forecasts across 6 origins) SARIMA and Prophet are statistically tied — "
    "SARIMA best on MAE and MAPE, Prophet marginally better RMSE — while XGBoost trails "
    "on every metric. The tie-break is trustworthiness at equal accuracy: SARIMA has "
    "validated white-noise residuals and calibrated confidence intervals from a "
    "transparent statistical model. The single-window hold-out below favours XGBoost, "
    "but that 3-point result did not survive the more reliable backtest — a caution "
    "against picking models on one lucky window."
)

CONFIDENCE_NOTES = {
    "SARIMA": ("success", "Confidence: 95% intervals shown are model-calibrated "
               "(Ljung-Box residual check passed)."),
    "Prophet": ("success", "Confidence: 95% intervals shown come from Prophet's Bayesian "
                "uncertainty simulation."),
    "XGBoost": ("warning", "Confidence: XGBoost provides no native prediction interval — "
                "point forecasts only. Its forecasts are also bounded by the training "
                "range, so it cannot project new highs."),
}

SEGMENT_INSIGHTS = [
    ("success", "<b>West</b> is the most credible growth call: +24.3% YoY forecast with "
     "the tightest confidence interval of all five segments (~125% of point forecast)."),
    ("warning", "<b>East</b>'s +195% headline growth rests on an unusually low Q1-2018 "
     "base and an interval ~200% as wide as the forecast itself — treat it as an upside "
     "scenario to monitor, not a committed number."),
    ("neutral", "<b>Furniture</b> is the laggard: +3.3% YoY with the widest relative "
     "interval (253%) — hold stocking steady pending a clearer trend."),
]

ANOMALY_METHOD_EXPLAINERS = {
    "Z-Score": (
        "Flags any week more than 2 standard deviations from its trailing 13-week "
        "average. Transparent and easy to act on; the 13-week window was chosen so the "
        "empirical flag rate (5.6%) matches the theoretical 4.55% two-sigma tail, and it "
        "recovers the September-2015 spike identified in decomposition. Blind spot: the "
        "first 13 weeks can never be scored."
    ),
    "Isolation Forest": (
        "Ensemble model isolating weeks that are unusual on sales level and "
        "week-over-week change jointly (contamination 5%, matched to the Z-score's "
        "implied rate; verified stable across 100–500 trees). Needs only one week of "
        "history, so it covers the early period Z-score cannot — but its percentage-"
        "change feature over-reacts to swings off the low 2015 base."
    ),
    "Agreement": (
        "Weeks flagged by <b>both</b> methods are the highest-confidence anomalies — all "
        "three are unambiguous broad-based demand surges. Single-method flags are "
        "lower-priority review items; the two methods disagree for structural reasons "
        "(burn-in vs. low-base sensitivity), not randomly."
    ),
}

CLUSTER_STRATEGIES = {
    "High Volume, Stable Demand": {
        "meaning": "The revenue backbone — eight sub-categories averaging ~$207K each "
                   "with moderate growth and the steadiest month-to-month demand.",
        "strategy": "Priority warehouse space, automated reorder-point replenishment, "
                    "and moderate safety stock sized to the low volatility. Stockouts "
                    "here are avoidable without holding excess buffer.",
        "tone": "indigo",
    },
    "Low Volume, Stable Demand": {
        "meaning": "Low-ticket, low-revenue lines (~$37K average, $50 avg order) with "
                   "the most predictable demand of any cluster.",
        "strategy": "Lean min-max inventory. Demand rarely deviates from its own "
                    "average, so tying up working capital in buffer stock has no payoff.",
        "tone": "slate",
    },
    "High-Value, Growing Demand": {
        "meaning": "Copiers and Machines: few transactions, each very large (~$1,931 "
                   "avg order), growing ~46% a year with elevated volatility.",
        "strategy": "Low-stock or make-to-order/drop-ship with pre-arranged supplier "
                    "lead times — holding expensive, infrequently-sold inventory ties up "
                    "capital that this demand pattern cannot absorb.",
        "tone": "green",
    },
    "Volatile, High-Growth Outlier": {
        "meaning": "Supplies alone: its +193% average growth traces to a boom-bust swing "
                   "($14.3K → $1.9K → $14.2K across 2015–2017), not steady demand — the "
                   "highest volatility in the portfolio (CV 2.03).",
        "strategy": "Short-cycle, demand-sensing replenishment with dedicated analyst "
                    "review. A static safety-stock policy would either stock out in a "
                    "collapse or sit on excess through a slow period.",
        "tone": "amber",
    },
}

METHODOLOGY_STEPS = [
    ("Data Foundation", "9,800 line items parsed with pinned day-first dates; quality "
     "audit (missing/duplicates/date integrity); time features engineered; daily, weekly "
     "and monthly series built loss-free (totals reconcile to the cent)."),
    ("Time Series Analysis", "Additive decomposition (seasonal strength 0.89 vs trend "
     "0.40); ADF test shows the level series is already stationary (p = 0.0003), and "
     "differencing diagnostics confirmed d=0 to avoid over-differencing."),
    ("Forecasting", "SARIMA, Prophet, and XGBoost compared leak-free on a chronological "
     "split, then stress-tested with a 6-origin rolling backtest; SARIMA recommended."),
    ("Segment Forecasts", "The production SARIMA spec reused across 3 categories and 2 "
     "regions (per-segment ACF checks + Ljung-Box adequacy); growth measured "
     "year-over-year to remove the Q1 seasonal trough."),
    ("Anomaly Detection", "Rolling Z-score (13-week window, 2σ) and Isolation Forest "
     "(contamination 5%) cross-compared; every flagged week backed by order-line "
     "evidence classifying broad-based vs concentrated demand."),
    ("Product Segmentation", "K-Means (k=4 via elbow + silhouette) on four engineered "
     "features; log transform applied only where it verifiably reduced skew; PCA used "
     "strictly for 2D visualization (91.6% variance retained)."),
]

TECH_STACK = [
    ("Python 3", "Core language"),
    ("pandas / NumPy", "Data engineering"),
    ("statsmodels", "SARIMA, decomposition, ADF, Ljung-Box"),
    ("Prophet", "Bayesian forecasting"),
    ("XGBoost", "ML forecasting"),
    ("scikit-learn", "Isolation Forest, K-Means, PCA, metrics"),
    ("Plotly", "Interactive dashboard charts"),
    ("Matplotlib / Seaborn", "Notebook figures"),
    ("Streamlit", "Dashboard framework"),
    ("Jupyter", "Analysis notebook"),
]
