# 🏍️ M(L)GP — MotoGP Race & Sprint Prediction Pipeline

**M(L)GP** is an end-to-end **Data Engineering, Machine Learning, and Monte Carlo Simulation** pipeline that predicts **MotoGP Sprint** and **Grand Prix** race outcomes, delivering win, podium, and top-10 probabilities tailored to the exact stage of a race weekend.

## Highlights

- **Incremental MotoGP API Ingestion:** pulls official seasons, sessions, and classifications from the MotoGP PulseLive API while skipping already finished historical events.
- **Official Q1/Q2 Grid Reconstruction:** automatically rebuilds MotoGP's split qualifying starting grid (`Q2` + `Q1` offset) and parses race lap/total times.
- **100% Leak-Free Feature Engineering:** chronological sorting with `.shift(1)` rolling form, reliability (`DNF` ratios), circuit affinity, weather flags, and same weekend sprint pace.
- **Dual-Model ML Strategy (`RAC` & `SPR`):** separates finishing-pace regression (`HistGradientBoostingRegressor` on finishers) from crash/retirement risk (`HistGradientBoostingClassifier` on all starters), with native `NaN` handling.
- **10,000-Race Monte Carlo Simulator:** samples race-to-race pace variance ($\sigma$) and Bernoulli crash events across parallel simulated races to compute realistic `Win %`, `Podium %`, and `Top 10 %` odds.
- **Weekend State Auto-Detection & Backtesting:** adapts automatically to `PRE_QUALIFYING`, `POST_QUALIFYING`, and `POST_SPRINT` states, plus historical GP backtesting and an optional **Streamlit** dashboard.

## Overview

Predicting MotoGP races is notoriously noisy: a fast rider with aggressive race pace might also carry a high crash probability, while a consistent podium contender benefits whenever front runners DNF. A single deterministic regression score misses that interaction completely.

**M(L)GP** solves this in two decoupled stages. First, a strictly causal data pipeline transforms raw timing feeds into historical form, circuit affinity, and reliability features to train separate `RAC` and `SPR` Gradient Boosting models. Second, a vectorized Monte Carlo engine uses each rider's predicted pace ($\mu$), predicted crash risk ($P(\text{DNF})$), and historical validation dispersion ($\sigma$) to simulate **10,000 full race weekends** (default value) ranking all riders head-to-head inside every single simulated race.

## Usage

### Run the complete pipeline

Fetch the latest MotoGP data, rebuild features, and retrain both `RAC` and `SPR` models with a single command:

```bash
python src/complete_process.py --retrain
```

Generate Monte Carlo win/podium/top-10 probabilities for the upcoming Grand Prix (automatically detecting `PRE_QUALIFYING`, `POST_QUALIFYING`, or `POST_SPRINT`):

```bash
python src/simulation.py
```

Or run individual pipeline stages as needed:

```bash
python src/ingestion.py      # 1. Incremental API fetch -> data/raw/
python src/preprocessing.py  # 2. Cleaning & Q1/Q2 grid -> data/processed/cleaned_sessions.csv
python src/features.py       # 3. Leak-free features -> data/processed/dataset_ml.csv
python src/train.py          # 4. Train RAC & SPR bundles -> models/model_{rac,spr}.pkl
```

### Optional: Interactive Streamlit GUI

Explore historical datasets, tune model hyperparameters, edit starting grids in What-If scenarios, and run backtests visually:

```bash
streamlit run app.py
```

## Installation

**Requirements:** Python `3.10+`.

Install all required dependencies with `pip`:

```bash
pip install -r requirements.txt
```

*(Recommended: run inside a virtual environment such as `python -m venv .venv`.)*

---

### Author

Created by **Luca**, built to combine motorsport domain knowledge, leak free time series feature engineering, and probabilistic race simulation into a clean, reproducible Python pipeline.

## Roadmap

- **Production Full-History Training (`src/train.py --full`):** estimate `std_dev` via temporal validation first, then retrain `reg` and `clf` on 100% of historical seasons (including 2025–2026), wired to the UI retrain button.
- **DNF Classifier Enhancements:** compute leak-free circuit-specific retirement rates (`circuit_dnf_rate` with `.shift(1)`), add weekend form/risk signals (FP/PR-to-Qualifying delta or consecutive finish streak), and benchmark `ROC-AUC` & `Brier Score` improvements.
- **Free Practice Pace & Regressor Tuning:** extract race/qualifying pace signals from `FP` and `PR` sessions in `data/raw/results.csv`, and optimize `HistGradientBoostingRegressor` (`max_depth`, `min_samples_leaf`, `l2_regularization`) on 2025/2026 validation.
- **Probabilistic Backtesting & Forecast Archiving (`src/evaluate_history.py`):** evaluate `Brier Score` and Monte Carlo probability calibration across 2025–2026 races, and auto-save pre-race forecasts in `predictions/<date>_<circuit>_<session>.csv` for post-race comparison in the UI.
