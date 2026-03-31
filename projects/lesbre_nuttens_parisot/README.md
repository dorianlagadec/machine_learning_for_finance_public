# GDELT × Brent — Geopolitical Tension as a Predictive Signal for Oil Prices

Lesbre · Parisot · Nuttens

---

## Research Question

> Can a geopolitical tension score derived from GDELT news events predict the short-term direction of Brent crude oil prices?

The Middle East accounts for roughly a third of global oil production. Geopolitical instability in the region — conflicts, sanctions, diplomatic breakdowns — is a well-documented driver of oil price volatility. GDELT (Global Database of Events, Language, and Tone) encodes global media coverage of such events in near real-time, assigning each event a structured score (Goldstein Scale, average tone, conflict type). This project tests whether that signal carries predictive information beyond what price history alone provides.

**Task:** Binary classification — does Brent close higher 3 trading days from now?  
**Period:** January 2015 – December 2024  
**Asset:** Brent crude oil continuous futures (`BZ=F`)  
**Data source:** GDELT 1.0 (public, daily CSV files)

---

## Project Structure

```
projects/lesbre_parisot_nuttens/
│
├── README.md
│
├── data/
│   ├── raw/
│   │   ├── gdelt/              # Raw GDELT daily CSVs (gitignored)
│   │   └── brent/              # Raw Brent price data (gitignored)
│   └── processed/
│       ├── gdelt_features.parquet
│       └── final_dataset.parquet
│
├── src/
│   ├── data/
│   │   ├── download_gdelt.py   # GDELT download + Middle East filtering
│   │   ├── download_brent.py   # Brent prices via yfinance
│   │   └── build_dataset.py    # Merge GDELT features + Brent → final dataset
│   │
│   ├── features/
│   │   └── gdelt_features.py   # Daily feature aggregation from GDELT events
│   │
│   ├── models/
│   │   ├── benchmark.py        # Naive benchmark models
│   │   └── train.py            # Training pipeline, CV, evaluation
│   │
│   └── visualization/
│       └── plots.py            # All figures used in the presentation
│
└── notebooks/
    └── presentation.ipynb      # Full project walkthrough (HTML export)
```

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# From the repo root
uv sync
```

Or with pip:

```bash
pip install -e .
```

---

## Reproduce

Run the following scripts **in order** from `projects/lesbre_parisot_nuttens/`:

```bash
# 1. Download GDELT daily files (2015–2024) — runs several hours, incremental
python src/data/download_gdelt.py

# 2. Download Brent prices
python src/data/download_brent.py

# 3. Aggregate GDELT events into daily features
python src/features/gdelt_features.py

# 4. Merge GDELT features with Brent prices, build target variable
python src/data/build_dataset.py

# 5. Train models and evaluate against benchmarks
python src/models/train.py

# 6. Generate all figures
python src/visualization/plots.py
```

Then open and run `notebooks/presentation.ipynb` for the full analysis.

> **Note:** Steps 2–6 take a few minutes each. Step 1 downloads ~10 years of daily GDELT files and takes several hours. It is incremental — safe to interrupt and resume.

---

## Features

GDELT events are filtered to 9 Middle East countries (`IRN`, `IRQ`, `SAU`, `ISR`, `PSE`, `YEM`, `SYR`, `ARE`, `KWT`) and aggregated daily into the following features:

| Feature | Description |
|---|---|
| `n_events` | Total number of events involving the region |
| `n_conflict_events` | Events coded as conflict (CAMEO 18x, 19x, 20x) |
| `conflict_ratio` | Share of conflict events in daily total |
| `goldstein_mean` | Mean Goldstein stability score (-10 destabilizing → +10 stabilizing) |
| `goldstein_min` | Minimum Goldstein score (peak destabilization) |
| `avg_tone` | Mean media tone across all articles (negative = hostile framing) |
| `n_mentions` | Total media volume (sum of mentions across sources) |
| `n_articles` | Breadth of coverage (distinct articles) |
| `goldstein_7d_ma` | 7-day rolling mean of Goldstein score |
| `tension_spike` | Binary flag: sudden Goldstein drop below rolling mean − 1.5σ |
| `mentions_7d_ma` | 7-day rolling mean of media volume |
| `mentions_zscore` | Standardized media surge relative to rolling baseline |

**Target variable:**
```python
df["target"] = (df["brent_close"].shift(-3) > df["brent_close"]).astype(int)
```

---

## Models

| Model | Role |
|---|---|
| Majority class classifier | Naive benchmark — accuracy floor to beat |
| Rolling volatility signal | Financial benchmark — simple rule-based baseline |
| Logistic Regression | Interpretable linear model, coefficient analysis |
| Random Forest | Non-linear model, feature importance |

Cross-validation uses `TimeSeriesSplit` throughout — no standard KFold, which would introduce look-ahead bias.

---

## Key Design Choices & Limitations

**Look-ahead bias prevention:** The target is constructed with `shift(-3)` applied only to the price series, not to the features. `TimeSeriesSplit` ensures no future data leaks into training folds.

**Signal noise:** GDELT encodes media coverage of events, not the events themselves. High-profile crises are over-represented relative to quieter but equally impactful developments.

**Lag uncertainty:** J+3 is a modelling assumption. Liquid markets may price in geopolitical news within minutes, not days. The signal may capture medium-term sentiment drift rather than immediate reactions.

**Backtest limitations:** The cumulative return backtest assumes no transaction costs or slippage. Results should be interpreted as signal quality indicators, not trading performance.

---

## Notable Events in the Dataset

| Date | Event |
|---|---|
| 2015–2016 | Escalation of the Yemen civil war |
| Sep 2019 | Drone strikes on Saudi Aramco facilities |
| Jan 2020 | Assassination of General Soleimani |
| Apr 2019 | US ends Iran oil sanction waivers |
| Oct 2023 | Israel-Gaza conflict begins |

These events provide natural validation anchors: a well-calibrated signal should show measurable spikes around these dates.
