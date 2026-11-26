"""Automatic local bootstrap from data/raw/ when processed/model artifacts are missing."""

from __future__ import annotations

import threading
from typing import Any

import streamlit as st

from utils.config import AppConfig
from utils.data_loaders import invalidate_all_caches
from utils.pipeline_runner import artifacts_need_bootstrap, execute_local_rebuild
from utils.state import append_log, init_session_state, update_step_status


@st.cache_resource(show_spinner=False)
def _get_bootstrap_lock() -> threading.Lock:
    """Process-wide lock preventing concurrent cold-start bootstrap races."""
    return threading.Lock()


def ensure_local_artifacts_bootstrapped(config: AppConfig) -> dict[str, Any] | None:
    """Rebuild dataset_ml.csv and model bundles from data/raw/ if missing (Steps 2 -> 3 -> 4)."""
    if not artifacts_need_bootstrap(config):
        return None

    init_session_state()
    lock = _get_bootstrap_lock()
    with lock:
        if not artifacts_need_bootstrap(config):
            invalidate_all_caches()
            return None

        try:
            with st.spinner(
                "Bootstrapping ML dataset and models from `data/raw/` (Steps 2 → 3 → 4)..."
            ):
                results = execute_local_rebuild(
                    config=config,
                    full_retrain=bool(getattr(config, "full_retrain", True)),
                    log_cb=lambda msg: append_log(msg, level="BOOTSTRAP"),
                )
        except Exception as exc:
            append_log(f"Auto-bootstrap failed: {exc}", level="ERROR")
            st.error(
                f"Automatic rebuild from `data/raw/` failed: {exc}",
                title="Bootstrap error",
                icon=":material/error:",
            )
            return None

        for step_key in ("preprocessing", "features", "train"):
            res_key = "training" if step_key == "train" else step_key
            if isinstance(results.get(res_key), dict):
                update_step_status(step_key, "success", results[res_key])

        invalidate_all_caches()
        append_log(
            "Auto-bootstrapped dataset_ml.csv and model bundles from data/raw/ (zero API calls).",
            level="BOOTSTRAP",
        )
        return results
