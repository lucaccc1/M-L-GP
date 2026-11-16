"""Centralized Streamlit session state initialization and artifact health checks."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from utils.config import AppConfig


def init_session_state() -> None:
    """Initialize all persistent st.session_state keys in one predictable place."""
    if "pipeline_config" not in st.session_state:
        st.session_state["pipeline_config"] = AppConfig()

    if "pipeline_logs" not in st.session_state:
        st.session_state["pipeline_logs"] = []

    if "step_statuses" not in st.session_state:
        st.session_state["step_statuses"] = {
            "ingestion": {"status": "idle", "last_run": None, "summary": None},
            "preprocessing": {"status": "idle", "last_run": None, "summary": None},
            "features": {"status": "idle", "last_run": None, "summary": None},
            "train": {"status": "idle", "last_run": None, "summary": None},
            "simulation": {"status": "idle", "last_run": None, "summary": None},
        }

    if "last_sim_results" not in st.session_state:
        st.session_state["last_sim_results"] = {}

    if "last_backtest_results" not in st.session_state:
        st.session_state["last_backtest_results"] = {}


def get_config() -> AppConfig:
    init_session_state()
    return st.session_state["pipeline_config"]


def append_log(message: str, level: str = "INFO") -> None:
    init_session_state()
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["pipeline_logs"].append(f"[{ts}] [{level}] {message}")


def clear_logs() -> None:
    init_session_state()
    st.session_state["pipeline_logs"] = []


def update_step_status(
    step: str, status: str, summary: dict[str, Any] | None = None
) -> None:
    init_session_state()
    st.session_state["step_statuses"][step] = {
        "status": status,
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
    }


def _file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_kb": 0.0,
            "modified": "Missing",
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_kb": round(stat.st_size / 1024, 1),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
    }


def get_artifact_inventory(config: AppConfig) -> dict[str, dict[str, Any]]:
    """Inspect presence, size, and timestamp of all pipeline artifacts."""
    return {
        "raw_seasons": _file_info(config.seasons_csv),
        "raw_events": _file_info(config.events_csv),
        "raw_sessions": _file_info(config.sessions_csv),
        "raw_results": _file_info(config.results_csv),
        "cleaned_sessions": _file_info(config.cleaned_sessions_csv),
        "dataset_ml": _file_info(config.dataset_ml_csv),
        "model_rac": _file_info(config.model_bundle_path("RAC")),
        "model_spr": _file_info(config.model_bundle_path("SPR")),
        "training_metrics": _file_info(config.training_metrics_json),
    }
