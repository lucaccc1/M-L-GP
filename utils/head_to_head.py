"""Monte Carlo Head-to-Head simulation and telemetry comparison adapter (wraps src/simulation.py)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src import simulation as sim_mod
from utils.h2h_math import determine_metric_advantage, summarize_h2h_outcomes

FEATURE_SPECS: list[tuple[str, str, str, bool, str]] = [
    ("Simulation & Odds", "win_pct", "Win probability (%)", True, "{:.2f}%"),
    ("Simulation & Odds", "podium_pct", "Podium probability (%)", True, "{:.2f}%"),
    ("Simulation & Odds", "top10_pct", "Top 10 probability (%)", True, "{:.2f}%"),
    ("Simulation & Odds", "pred_pos", "Expected finish pos (Model)", False, "{:.2f}"),
    ("Simulation & Odds", "dnf_pct", "Predicted DNF risk (%)", False, "{:.1f}%"),
    ("Weekend Pace & Grid", "grid_position", "Starting grid", False, "P{:.0f}"),
    ("Weekend Pace & Grid", "fp_avg_pos", "Free Practice (FP) avg", False, "{:.2f}"),
    ("Weekend Pace & Grid", "pr_pos", "Practice (PR) position", False, "P{:.0f}"),
    ("Weekend Pace & Grid", "fp_vs_grid_delta", "FP vs Grid delta (FP - Grid)", False, "{:+.2f}"),
    ("Weekend Pace & Grid", "sprint_pos_same_weekend", "Sprint result (same GP)", False, "P{:.0f}"),
    ("Recent Rider Form", "rider_last3_avg", "Last 3 sessions avg finish", False, "{:.2f}"),
    ("Recent Rider Form", "rider_last5_avg", "Last 5 sessions avg finish", False, "{:.2f}"),
    ("Recent Rider Form", "rider_last3_avg_rac", "Last 3 RAC avg finish", False, "{:.2f}"),
    ("Recent Rider Form", "rider_last3_avg_spr", "Last 3 SPR avg finish", False, "{:.2f}"),
    ("Recent Rider Form", "rider_avg_gain", "Avg positions gained (last 5)", True, "{:+.2f}"),
    ("Circuit & Bike Affinity", "rider_circuit_avg", "Rider historical circuit avg", False, "{:.2f}"),
    ("Circuit & Bike Affinity", "constructor_circuit_avg", "Bike historical circuit avg", False, "{:.2f}"),
    ("Circuit & Bike Affinity", "constructor_last3_avg", "Constructor last 3 avg", False, "{:.2f}"),
    ("Circuit & Bike Affinity", "constructor_last5_avg", "Constructor last 5 avg", False, "{:.2f}"),
    ("Reliability & Risk", "dnf_ratio_last5", "Rider DNF rate (last 5)", False, "{:.1%}"),
    ("Reliability & Risk", "dnf_ratio_last10", "Rider DNF rate (last 10)", False, "{:.1%}"),
    ("Reliability & Risk", "clean_finish_streak", "Clean finish streak", True, "{:.0f} races"),
    ("Reliability & Risk", "circuit_dnf_rate", "Circuit avg DNF rate", False, "{:.1%}"),
]


def run_h2h_simulation(
    df_ml: pd.DataFrame,
    grid_df: pd.DataFrame,
    circuit_name: str,
    bundle: dict[str, Any],
    session_type: str = "RAC",
    is_wet: int = 0,
    n_sims: int = 10000,
    std_mult: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Run prediction + full Monte Carlo simulation preserving per-run ranks and crashes."""
    df_upcoming = sim_mod.build_upcoming_race(
        df_ml,
        grid_df,
        circuit_name,
        session_type=session_type,
        is_wet=is_wet,
    )
    df_pred = sim_mod.predict_session(df_upcoming, bundle)
    effective_std = float(bundle["std_dev"]) * float(std_mult)

    rng = np.random.default_rng(seed)
    n_riders = len(df_pred)
    mu = df_pred["pred_pos"].to_numpy(dtype=float)
    p_dnf = df_pred["dnf_prob"].to_numpy(dtype=float)

    scores = rng.normal(loc=mu, scale=effective_std, size=(n_sims, n_riders))
    crashes = rng.random((n_sims, n_riders)) < p_dnf
    scores = np.where(crashes, 999.0, scores)
    ranks = np.argsort(np.argsort(scores, axis=1), axis=1)

    sim_df = df_pred.copy()
    sim_df["win_pct"] = np.round((ranks == 0).mean(axis=0) * 100, 2)
    sim_df["podium_pct"] = np.round((ranks < 3).mean(axis=0) * 100, 2)
    sim_df["top10_pct"] = np.round((ranks < 10).mean(axis=0) * 100, 2)
    sim_df["dnf_pct"] = np.round(sim_df["dnf_prob"] * 100, 2)

    order_idx = np.argsort(-sim_df["win_pct"].to_numpy())
    sim_df = sim_df.iloc[order_idx].reset_index(drop=True)
    ranks = ranks[:, order_idx]
    crashes = crashes[:, order_idx]

    return {
        "sim_df": sim_df,
        "ranks": ranks,
        "crashes": crashes,
        "effective_std": effective_std,
    }


def compute_rider_pair_duel(
    sim_payload: dict[str, Any],
    rider_a: str,
    rider_b: str,
) -> dict[str, Any]:
    """Extract H2H Monte Carlo metrics for `(rider_a, rider_b)` from `run_h2h_simulation`."""
    sim_df: pd.DataFrame = sim_payload["sim_df"]
    ranks: np.ndarray = sim_payload["ranks"]
    crashes: np.ndarray = sim_payload["crashes"]

    idx_a_list = sim_df.index[sim_df["rider_name"] == rider_a].tolist()
    idx_b_list = sim_df.index[sim_df["rider_name"] == rider_b].tolist()
    if not idx_a_list or not idx_b_list:
        raise ValueError("Selected riders not found in simulation grid.")

    ia = idx_a_list[0]
    ib = idx_b_list[0]
    return summarize_h2h_outcomes(
        ranks[:, ia],
        crashes[:, ia],
        ranks[:, ib],
        crashes[:, ib],
        n_riders=len(sim_df),
        rider_a=rider_a,
        rider_b=rider_b,
    )


def _clean_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _fmt_value(val: float | None, fmt: str) -> str:
    if val is None:
        return "N/A"
    return fmt.format(val)


def build_feature_comparison_table(
    sim_df: pd.DataFrame,
    rider_a: str,
    rider_b: str,
) -> pd.DataFrame:
    """Build a structured side-by-side comparison DataFrame of all ML/telemetry features."""
    row_a = sim_df[sim_df["rider_name"] == rider_a].iloc[0]
    row_b = sim_df[sim_df["rider_name"] == rider_b].iloc[0]

    records: list[dict[str, Any]] = []
    for category, col, label, higher_is_better, fmt in FEATURE_SPECS:
        if col not in sim_df.columns:
            continue
        va = _clean_float(row_a.get(col))
        vb = _clean_float(row_b.get(col))
        edge = determine_metric_advantage(
            va,
            vb,
            higher_is_better=higher_is_better,
            rider_a=rider_a,
            rider_b=rider_b,
            decimals=4 if "%" in fmt and "{:.1%}" in fmt else 2,
        )
        records.append(
            {
                "Category": category,
                "Metric": label,
                rider_a: _fmt_value(va, fmt),
                rider_b: _fmt_value(vb, fmt),
                "Edge": edge,
            }
        )
    return pd.DataFrame(records)


def get_rider_recent_history(
    df_ml: pd.DataFrame,
    rider_name: str,
    session_type: str | None = None,
    limit: int = 10,
) -> pd.DataFrame:
    """Return recent chronological race results for a rider from dataset_ml.csv."""
    sub = df_ml[df_ml["rider_name"] == rider_name].copy()
    if session_type in ("RAC", "SPR"):
        sub = sub[sub["session_type"] == session_type]
    if sub.empty:
        return sub
    if "date" in sub.columns:
        sub = sub.sort_values(by=["date", "session_id"]).reset_index(drop=True)
    return sub.tail(limit).reset_index(drop=True)


def get_rider_circuit_history(
    df_ml: pd.DataFrame,
    rider_name: str,
    circuit_name: str,
    session_type: str | None = None,
) -> pd.DataFrame:
    """Return all historical results for a rider at the chosen circuit."""
    sub = df_ml[
        (df_ml["rider_name"] == rider_name)
        & df_ml["circuit_name"].astype(str).str.contains(circuit_name, case=False, regex=False, na=False)
    ].copy()
    if session_type in ("RAC", "SPR"):
        sub = sub[sub["session_type"] == session_type]
    if sub.empty:
        return sub
    if "date" in sub.columns:
        sub = sub.sort_values(by=["date", "session_id"], ascending=[False, False]).reset_index(drop=True)
    return sub
