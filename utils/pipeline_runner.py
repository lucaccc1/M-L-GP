"""Adapter layer that executes and wraps the user's untouched src/*.py pipeline modules."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score

from utils.config import AppConfig

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[float, str], None]
StepCallback = Callable[[str, int, int, str], None]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_src_module(name: str):
    """Import a module from the user's src/ folder dynamically."""
    import importlib

    return importlib.import_module(f"src.{name}")


def run_script_subprocess(
    script_rel_path: str,
    args: list[str] | None = None,
    log_cb: LogCallback | None = None,
) -> None:
    """Execute an original src/*.py script via subprocess and stream stdout/stderr live."""
    log = log_cb or print
    cmd = [sys.executable, "-u", script_rel_path, *(args or [])]
    log(f"$ {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        cleaned = line.rstrip()
        if cleaned:
            log(cleaned)

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Script '{script_rel_path}' exited with non-zero code {return_code}."
        )


def execute_ingestion(
    config: AppConfig,
    log_cb: LogCallback | None = None,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run src/ingestion.py as-is via subprocess with live stdout streaming."""
    log = log_cb or print
    if progress_cb:
        progress_cb(0.1, "Running src/ingestion.py (MotoGP PulseLive API)...")

    run_script_subprocess("src/ingestion.py", log_cb=log)

    if progress_cb:
        progress_cb(1.0, "src/ingestion.py finished.")

    return {
        "events_exists": config.events_csv.exists(),
        "sessions_exists": config.sessions_csv.exists(),
        "results_exists": config.results_csv.exists(),
    }


def execute_preprocessing(
    config: AppConfig,
    log_cb: LogCallback | None = None,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the functions from src/preprocessing.py with configurable start_year."""
    log = log_cb or print
    if not config.results_csv.exists() or not config.sessions_csv.exists():
        raise FileNotFoundError(
            "Missing data/raw/results.csv or data/raw/sessions.csv. Run Step 1 (src/ingestion.py) first."
        )

    prep = _import_src_module("preprocessing")

    if progress_cb:
        progress_cb(0.15, "Loading raw results & sessions (src.preprocessing.load_data)...")
    results, sessions = prep.load_data()
    log(f"Loaded {len(results):,} raw results and {len(sessions):,} sessions.")

    if progress_cb:
        progress_cb(0.4, f"Filtering sessions >= {config.start_year}...")
    filtered_sessions = prep.filter_sessions_year(
        results, sessions, start_year=config.start_year
    )

    if progress_cb:
        progress_cb(0.65, "Cleaning columns & reconstructing Q1/Q2 starting grid...")
    cleaned_sessions = prep.clean_columns(filtered_sessions)
    final_races_df = prep.starting_grid(cleaned_sessions)
    final_races_df["time_seconds"] = final_races_df["time"].apply(prep.parse_time)

    config.processed_path.mkdir(parents=True, exist_ok=True)
    final_races_df.to_csv(config.cleaned_sessions_csv, index=False)
    log(f"Saved {config.cleaned_sessions_csv} ({len(final_races_df):,} rows).")

    if progress_cb:
        progress_cb(1.0, "Preprocessing complete.")

    return {"output_rows": len(final_races_df)}


def execute_features(
    config: AppConfig,
    log_cb: LogCallback | None = None,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run all 12 feature engineering functions from src/features.py with live progress."""
    log = log_cb or print
    if not config.cleaned_sessions_csv.exists():
        raise FileNotFoundError(
            "Missing data/processed/cleaned_sessions.csv. Run Step 2 (src/preprocessing.py) first."
        )

    feat = _import_src_module("features")
    steps = [
        ("load_data", feat.load_data),
        ("order_by_date", feat.order_by_date),
        ("grid_delta", feat.grid_delta),
        ("rider_last3_avg", feat.rider_last3_avg),
        ("rider_last5_avg", feat.rider_last5_avg),
        ("rider_session_avg", feat.rider_session_avg),
        ("sprint_same_weekend", feat.sprint_same_weekend),
        ("weather", feat.weather),
        ("affidability", feat.affidability),
        ("grid_delta_gain", feat.grid_delta_gain),
        ("circuit_affinity", feat.circuit_affinity),
        ("constructor_rolling_avg", feat.constructor_rolling_avg),
        ("circuit_dnf_rate", feat.circuit_dnf_rate),
        ("clean_finish_streak", feat.clean_finish_streak),
        ("practice_features", feat.practice_features),
    ]

    df = None
    total = len(steps)
    for idx, (name, fn) in enumerate(steps):
        if progress_cb:
            progress_cb(idx / total, f"Running src.features.{name}()...")
        df = fn() if df is None else fn(df)
        log(f"src.features.{name}() completed.")

    config.processed_path.mkdir(parents=True, exist_ok=True)
    assert df is not None
    df.to_csv(config.dataset_ml_csv, index=False)
    log(f"Saved {config.dataset_ml_csv} ({len(df):,} rows x {df.shape[1]} cols).")

    if progress_cb:
        progress_cb(1.0, "Feature engineering complete.")

    return {"rows": len(df), "columns": int(df.shape[1])}


def _safe_evaluate_split(
    reg: Any,
    clf: Any,
    df_eval: pd.DataFrame,
    session_type: str,
    label: str,
    dnf_rank_penalty: float = 15.0,
    log_cb: LogCallback | None = None,
) -> dict[str, Any]:
    """Evaluate a trained bundle safely without crashing on empty splits or single-class DNF."""
    log = log_cb or print
    train_mod = _import_src_module("train")

    if df_eval.empty:
        log(f"[{label}] Split is empty — skipping evaluation.")
        return {
            "label": label,
            "rows": 0,
            "sessions": 0,
            "mae": None,
            "std_dev": None,
            "dnf_auc": None,
            "dnf_brier": None,
            "podium_hits": None,
            "predictions": [],
        }

    eval_fin = df_eval[df_eval["is_dnf"] == 0].reset_index(drop=True)
    if not eval_fin.empty:
        X_fin, y_fin, _ = train_mod.get_features_target(eval_fin, session_type)
        preds_fin = reg.predict(X_fin)
        mae = float(mean_absolute_error(y_fin, preds_fin))
        std_dev = float(np.std(y_fin - preds_fin))
    else:
        mae = None
        std_dev = None

    X_all, _, y_dnf = train_mod.get_features_target(df_eval, session_type)
    if hasattr(clf, "feature_names_in_"):
        X_clf = df_eval[list(clf.feature_names_in_)]
    elif hasattr(train_mod, "get_dnf_features"):
        X_clf, y_dnf = train_mod.get_dnf_features(df_eval)
    else:
        X_clf = X_all
    dnf_probs = clf.predict_proba(X_clf)[:, 1]
    auc = float(roc_auc_score(y_dnf, dnf_probs)) if y_dnf.nunique() > 1 else None
    brier = float(brier_score_loss(y_dnf, dnf_probs)) if len(y_dnf) > 0 else None

    eval_cols = [
        c
        for c in [
            "session_id",
            "season_year",
            "rider_name",
            "constructor_name",
            "grid_position",
            "position_cleaned",
            "is_dnf",
        ]
        if c in df_eval.columns
    ]
    eval_df = df_eval[eval_cols].copy()
    eval_df["pred_pos"] = np.clip(reg.predict(X_all), 1.0, 22.0)
    eval_df["dnf_prob"] = dnf_probs
    eval_df["pred_score"] = eval_df["pred_pos"] + dnf_probs * dnf_rank_penalty
    eval_df["pred_rank"] = eval_df.groupby("session_id")["pred_score"].rank(
        method="first"
    )

    podium_hits = [
        len(
            set(g[g["position_cleaned"] <= 3]["rider_name"])
            & set(g[g["pred_rank"] <= 3]["rider_name"])
        )
        for _, g in eval_df.groupby("session_id")
    ]
    avg_podium_hits = float(np.mean(podium_hits)) if podium_hits else None

    mae_str = f"{mae:.2f}" if mae is not None else "N/A"
    std_str = f"{std_dev:.2f}" if std_dev is not None else "N/A"
    auc_str = f"{auc:.3f}" if auc is not None else "N/A"
    brier_str = f"{brier:.3f}" if brier is not None else "N/A"
    pod_str = f"{avg_podium_hits:.2f}/3" if avg_podium_hits is not None else "N/A"
    log(
        f"[{label}] MAE (arrivati): {mae_str} | Std Dev: {std_str} | DNF AUC: {auc_str} | DNF Brier: {brier_str} | Podi: {pod_str}"
    )

    return {
        "label": label,
        "rows": int(len(df_eval)),
        "sessions": int(df_eval["session_id"].nunique()),
        "mae": round(mae, 4) if mae is not None else None,
        "std_dev": round(std_dev, 4) if std_dev is not None else None,
        "dnf_auc": round(auc, 4) if auc is not None else None,
        "dnf_brier": round(brier, 5) if brier is not None else None,
        "podium_hits": round(avg_podium_hits, 4)
        if avg_podium_hits is not None
        else None,
        "predictions": eval_df.to_dict(orient="records"),
    }


def _compute_importances(
    reg: Any,
    clf: Any,
    df_eval: pd.DataFrame,
    session_type: str,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    train_mod = _import_src_module("train")
    if df_eval.empty:
        return []

    eval_fin = df_eval[df_eval["is_dnf"] == 0].reset_index(drop=True)
    X_fin, y_fin, _ = train_mod.get_features_target(eval_fin, session_type)
    X_all, _, y_dnf = train_mod.get_features_target(df_eval, session_type)
    if hasattr(clf, "feature_names_in_"):
        X_clf = df_eval[list(clf.feature_names_in_)]
    elif hasattr(train_mod, "get_dnf_features"):
        X_clf, y_dnf = train_mod.get_dnf_features(df_eval)
    else:
        X_clf = X_all

    reg_imp = (
        permutation_importance(
            reg,
            X_fin,
            y_fin,
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=random_state,
        ).importances_mean
        if len(eval_fin) >= 10
        else np.zeros(X_fin.shape[1])
    )
    clf_imp = (
        permutation_importance(
            clf,
            X_clf,
            y_dnf,
            scoring="roc_auc",
            n_repeats=5,
            random_state=random_state,
        ).importances_mean
        if len(df_eval) >= 10 and y_dnf.nunique() > 1
        else np.zeros(X_clf.shape[1])
    )
    clf_imp_map = dict(zip(X_clf.columns, clf_imp))
    records = [
        {
            "feature": feat,
            "reg_importance": round(float(reg_imp[idx]), 5),
            "clf_importance": round(float(clf_imp_map.get(feat, 0.0)), 5),
        }
        for idx, feat in enumerate(X_all.columns)
    ]
    return sorted(records, key=lambda r: r["reg_importance"], reverse=True)


def execute_training(
    config: AppConfig,
    use_default_src_train: bool = True,
    full_retrain: bool | None = None,
    log_cb: LogCallback | None = None,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Train RAC/SPR models using src/train.py and export UI diagnostics JSON."""
    log = log_cb or print
    if not config.dataset_ml_csv.exists():
        raise FileNotFoundError(
            "Missing data/processed/dataset_ml.csv. Run Step 3 (src/features.py) first."
        )

    effective_full_retrain = (
        bool(getattr(config, "full_retrain", True))
        if full_retrain is None
        else bool(full_retrain)
    )

    train_mod = _import_src_module("train")
    sessions_to_train = config.train_sessions or ["RAC", "SPR"]
    report: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "sessions": {},
    }

    hyperparam_kwargs = dict(
        reg_max_iter=config.reg_max_iter,
        reg_learning_rate=config.reg_learning_rate,
        reg_max_depth=config.reg_max_depth,
        clf_max_iter=config.clf_max_iter,
        clf_learning_rate=config.clf_learning_rate,
        clf_max_depth=config.clf_max_depth,
        clf_min_samples_leaf=config.clf_min_samples_leaf,
        clf_l2_regularization=config.clf_l2_regularization,
        random_state=config.random_state,
    )

    total = max(len(sessions_to_train), 1)
    for idx, session_type in enumerate(sessions_to_train):
        base = idx / total
        if progress_cb:
            progress_cb(base + 0.05, f"Loading {session_type} dataset via src.train...")
        df = train_mod.load_dataset(session_type)
        train_df, val_df, test_df = train_mod.split_data(df)

        if train_df.empty:
            raise ValueError(f"Training split is empty for {session_type}.")

        years = sorted(df["season_year"].unique())
        val_year = years[-2] if len(years) >= 2 else years[-1]
        test_year = years[-1]

        log(
            f"--- {session_type} split -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,} ---"
        )
        if progress_cb:
            progress_cb(base + 0.3 / total, f"Fitting {session_type} models...")

        reg, clf = train_mod.train_models(train_df, session_type, **hyperparam_kwargs)

        val_metrics = _safe_evaluate_split(
            reg,
            clf,
            val_df,
            session_type,
            f"Validation {val_year}",
            dnf_rank_penalty=config.dnf_rank_penalty,
            log_cb=log,
        )
        test_metrics = _safe_evaluate_split(
            reg,
            clf,
            test_df,
            session_type,
            f"Test {test_year}",
            dnf_rank_penalty=config.dnf_rank_penalty,
            log_cb=log,
        )

        effective_std_dev = val_metrics["std_dev"]
        if effective_std_dev is None or effective_std_dev <= 0:
            train_fin = train_df[train_df["is_dnf"] == 0]
            X_tr, y_tr, _ = train_mod.get_features_target(train_fin, session_type)
            effective_std_dev = float(np.std(y_tr - reg.predict(X_tr)))

        if effective_full_retrain:
            log(f"[--full] Retraining {session_type} on 100% of data ({years[0]}–{years[-1]})...")
            reg, clf = train_mod.train_models(df, session_type, **hyperparam_kwargs)

        train_mod.save_bundle(reg, clf, effective_std_dev, session_type)
        log(f"Bundle saved via src.train.save_bundle -> models/model_{session_type.lower()}.pkl")

        imp_df = val_df if not val_df.empty else train_df
        importances = _compute_importances(
            reg, clf, imp_df, session_type, random_state=config.random_state
        )

        report["sessions"][session_type] = {
            "bundle_path": str(config.model_bundle_path(session_type)),
            "train_rows": int(len(train_df)),
            "full_retrain": effective_full_retrain,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "effective_std_dev": round(float(effective_std_dev), 4),
            "feature_importances": importances,
        }

    config.models_path.mkdir(parents=True, exist_ok=True)
    config.training_metrics_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Diagnostics report saved at {config.training_metrics_json}")

    if progress_cb:
        progress_cb(1.0, "Training complete.")
    return report


def evaluate_existing_bundles(
    config: AppConfig, log_cb: LogCallback | None = None
) -> dict[str, Any]:
    """Generate training_metrics.json from pre-existing models/*.pkl without retraining."""
    log = log_cb or print
    train_mod = _import_src_module("train")
    sim_mod = _import_src_module("simulation")

    report: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "sessions": {},
    }

    for session_type in ["RAC", "SPR"]:
        bundle_path = config.model_bundle_path(session_type)
        if not bundle_path.exists():
            continue
        bundle = sim_mod.load_bundle(session_type)
        reg, clf, std_dev = bundle["reg"], bundle["clf"], float(bundle["std_dev"])
        df = train_mod.load_dataset(session_type)
        train_df, val_df, test_df = train_mod.split_data(df)
        years = sorted(df["season_year"].unique())
        val_year = years[-2] if len(years) >= 2 else years[-1]
        test_year = years[-1]

        val_metrics = _safe_evaluate_split(
            reg, clf, val_df, session_type, f"Validation {val_year}", log_cb=log
        )
        test_metrics = _safe_evaluate_split(
            reg, clf, test_df, session_type, f"Test {test_year}", log_cb=log
        )
        imp_df = val_df if not val_df.empty else train_df
        importances = _compute_importances(reg, clf, imp_df, session_type)

        report["sessions"][session_type] = {
            "bundle_path": str(bundle_path),
            "train_rows": int(len(train_df)),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "effective_std_dev": round(std_dev, 4),
            "feature_importances": importances,
        }

    config.models_path.mkdir(parents=True, exist_ok=True)
    config.training_metrics_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def artifacts_need_bootstrap(config: AppConfig) -> bool:
    """Return True when data/raw/ exists locally but dataset_ml.csv or trained .pkl bundles are missing."""
    raw_ready = config.results_csv.exists() and config.sessions_csv.exists()
    if not raw_ready:
        return False
    required_sessions = config.train_sessions or ["RAC", "SPR"]
    missing_dataset = not config.dataset_ml_csv.exists()
    missing_models = any(
        not config.model_bundle_path(session_type).exists()
        for session_type in required_sessions
    )
    return missing_dataset or missing_models


def execute_complete_process(
    config: AppConfig,
    retrain: bool = False,
    raw_subprocess: bool = False,
    include_ingestion: bool = True,
    full_retrain: bool | None = None,
    log_cb: LogCallback | None = None,
    step_cb: StepCallback | None = None,
) -> dict[str, Any]:
    """Run full pipeline using either src/complete_process.py or step-by-step src modules."""
    log = log_cb or print
    if raw_subprocess and include_ingestion:
        args = ["--retrain"] if retrain else []
        if step_cb:
            step_cb("complete_process", 1, 1, "Running src/complete_process.py...")
        run_script_subprocess("src/complete_process.py", args=args, log_cb=log)
        if retrain and config.dataset_ml_csv.exists():
            evaluate_existing_bundles(config, log_cb=log)
        return {"mode": "src/complete_process.py", "retrain": retrain}

    steps: list[tuple[str, str, Callable[[], dict[str, Any]]]] = []
    if include_ingestion:
        steps.append(
            (
                "ingestion",
                "Step 1: src/ingestion.py",
                lambda: execute_ingestion(config, log_cb=log),
            )
        )
    steps.append(
        (
            "preprocessing",
            "Step 2: src.preprocessing",
            lambda: execute_preprocessing(config, log_cb=log),
        )
    )
    steps.append(
        (
            "features",
            "Step 3: src.features",
            lambda: execute_features(config, log_cb=log),
        )
    )
    if retrain:
        steps.append(
            (
                "training",
                "Step 4: src.train",
                lambda: execute_training(config, full_retrain=full_retrain, log_cb=log),
            )
        )

    results: dict[str, Any] = {}
    total = max(len(steps), 1)
    for idx, (res_key, desc, fn) in enumerate(steps, start=1):
        step_name = "train" if res_key == "training" else res_key
        if step_cb:
            step_cb(step_name, idx, total, desc)
        results[res_key] = fn()

    return results


def execute_local_rebuild(
    config: AppConfig,
    full_retrain: bool | None = True,
    log_cb: LogCallback | None = None,
    step_cb: StepCallback | None = None,
) -> dict[str, Any]:
    """Rebuild processed dataset and models locally from data/raw/ (Steps 2 -> 3 -> 4, no API)."""
    if not config.results_csv.exists() or not config.sessions_csv.exists():
        raise FileNotFoundError(
            "Missing data/raw/results.csv or data/raw/sessions.csv. Run Step 1 (src/ingestion.py) first."
        )
    return execute_complete_process(
        config=config,
        retrain=True,
        raw_subprocess=False,
        include_ingestion=False,
        full_retrain=full_retrain,
        log_cb=log_cb,
        step_cb=step_cb,
    )


def safe_detect_weekend_state(
    df_ml: pd.DataFrame, config: AppConfig
) -> dict[str, Any]:
    """Call src.simulation.detect_weekend_state with safe off-season fallback."""
    sim_mod = _import_src_module("simulation")
    try:
        return sim_mod.detect_weekend_state(df_ml)
    except IndexError:
        last_rac_sid = df_ml[df_ml["session_type"] == "RAC"]["session_id"].iloc[-1]
        roster = (
            df_ml[df_ml["session_id"] == last_rac_sid][
                ["rider_id", "rider_name", "constructor_name"]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        grid_df = sim_mod.estimate_grid_from_history(roster, df_ml)
        return {
            "state": "OFF_SEASON_FALLBACK",
            "circuit_name": str(df_ml["circuit_name"].iloc[-1]),
            "race_date": str(df_ml["date"].max())[:10],
            "is_wet": 0,
            "grid_df": grid_df,
            "sessions_to_predict": ["SPR", "RAC"],
        }


def list_completed_events(df_ml: pd.DataFrame) -> pd.DataFrame:
    """List completed GPs in dataset_ml.csv for Historical Backtest selection."""
    summary = (
        df_ml.groupby("event_id", as_index=False)
        .agg(
            season_year=("season_year", "first"),
            date=("date", "max"),
            circuit_name=("circuit_name", "first"),
            sessions=("session_type", lambda s: ", ".join(sorted(set(s)))),
        )
        .sort_values(by="date", ascending=False)
        .reset_index(drop=True)
    )
    summary["label"] = (
        summary["season_year"].astype(str)
        + " - "
        + summary["circuit_name"].astype(str)
        + " ("
        + summary["date"].astype(str).str[:10]
        + ")"
    )
    return summary
