"""Page 1: Orchestrator running the user's untouched src/*.py scripts and functions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.bootstrap import ensure_local_artifacts_bootstrapped
from utils.config import AppConfig
from utils.data_loaders import invalidate_all_caches
from utils.pipeline_runner import (
    execute_complete_process,
    execute_features,
    execute_ingestion,
    execute_local_rebuild,
    execute_preprocessing,
    execute_training,
)
from utils.state import (
    append_log,
    clear_logs,
    get_artifact_inventory,
    get_config,
    init_session_state,
    update_step_status,
)

init_session_state()
config = get_config()
ensure_local_artifacts_bootstrapped(config)
inventory = get_artifact_inventory(config)

st.title(":material/account_tree: Pipeline orchestrator")
st.caption(
    "Run and monitor the end-to-end MotoGP pipeline from API ingestion and Q1/Q2 grid preprocessing to feature engineering and model training."
)

# --- Artifact readiness KPI row ---
raw_ready = inventory["raw_results"]["exists"] and inventory["raw_sessions"]["exists"]
prep_ready = inventory["cleaned_sessions"]["exists"]
feat_ready = inventory["dataset_ml"]["exists"]
models_ready = inventory["model_rac"]["exists"] and inventory["model_spr"]["exists"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Step 1 · Raw ingestion",
        "Ready" if raw_ready else "Missing",
        delta=f"{inventory['raw_results']['size_kb']} KB" if raw_ready else "src/ingestion.py",
        delta_color="normal" if raw_ready else "off",
        delta_arrow="off",
    )
with col2:
    st.metric(
        "Step 2 · Preprocessing",
        "Ready" if prep_ready else "Missing",
        delta=f"{inventory['cleaned_sessions']['size_kb']} KB" if prep_ready else "src/preprocessing.py",
        delta_color="normal" if prep_ready else "off",
        delta_arrow="off",
    )
with col3:
    st.metric(
        "Step 3 · Feature dataset",
        "Ready" if feat_ready else "Missing",
        delta=f"{inventory['dataset_ml']['size_kb']} KB" if feat_ready else "src/features.py",
        delta_color="normal" if feat_ready else "off",
        delta_arrow="off",
    )
with col4:
    st.metric(
        "Step 4 · Trained models",
        "RAC + SPR" if models_ready else "Missing",
        delta=inventory["model_rac"]["modified"] if models_ready else "src/train.py",
        delta_color="normal" if models_ready else "off",
        delta_arrow="off",
    )

st.divider()

st.markdown("#### Workspace view")
active_panel = st.segmented_control(
    "Workspace view",
    options=["Execution controls", "Pipeline configuration", "Artifact inventory"],
    default="Execution controls",
    label_visibility="collapsed",
    key="orchestrator_panel_view",
)


def _execute_step(step_key: str, label: str, fn) -> None:
    """Run a single pipeline step with live progress, status, and log capture."""
    progress_bar = st.progress(0.0, text=f"Initializing {label}...")
    with st.status(f"Running {label}...", expanded=True) as status_box:
        def _log(msg: str) -> None:
            append_log(msg, level=step_key.upper())
            st.write(msg)

        def _prog(frac: float, text: str) -> None:
            progress_bar.progress(min(max(frac, 0.0), 1.0), text=text)

        try:
            summary = fn(config, log_cb=_log, progress_cb=_prog)
            update_step_status(step_key, "success", summary)
            invalidate_all_caches()
            progress_bar.progress(1.0, text=f"{label} completed.")
            status_box.update(label=f"{label} completed", state="complete", expanded=False)
            st.success(
                f"{label} finished successfully.",
                title="Step complete",
                icon=":material/check_circle:",
            )
        except Exception as exc:
            append_log(f"ERROR in {label}: {exc}", level="ERROR")
            update_step_status(step_key, "error", {"error": str(exc)})
            status_box.update(label=f"{label} failed", state="error", expanded=True)
            st.error(f"{exc}", title=f"{label} failed", icon=":material/error:")


def _run_multi_step_pipeline(
    runner_fn,
    start_text: str,
    status_label: str,
    complete_label: str,
    fail_label: str,
    success_msg: str,
    success_title: str,
) -> None:
    """Execute a multi-step pipeline sequence with live progress, status box, and logs."""
    progress = st.progress(0.0, text=start_text)
    with st.status(status_label, expanded=True) as status_box:
        def _log(msg: str) -> None:
            append_log(msg, level="PIPELINE")
            st.write(msg)

        def _step(step_name: str, idx: int, n_total: int, desc: str) -> None:
            progress.progress((idx - 1) / max(n_total, 1), text=desc)
            update_step_status(step_name, "running")

        try:
            summary = runner_fn(_log, _step)
            for k, v in summary.items():
                if isinstance(v, dict):
                    mapped = "train" if k == "training" else k
                    update_step_status(mapped, "success", v)
            invalidate_all_caches()
            progress.progress(1.0, text=f"{complete_label}.")
            status_box.update(label=complete_label, state="complete", expanded=False)
            st.success(
                success_msg,
                title=success_title,
                icon=":material/check_circle:",
            )
        except Exception as exc:
            append_log(f"Pipeline halted: {exc}", level="ERROR")
            status_box.update(label=fail_label, state="error", expanded=True)
            st.error(f"{exc}", title="Pipeline execution error", icon=":material/error:")


if active_panel == "Execution controls":
    with st.container(border=True):
        st.subheader(":material/play_circle: End-to-end automation")

        e2e_c1, e2e_c2, e2e_c3 = st.columns([1.1, 1.1, 1], vertical_alignment="bottom")
        with e2e_c1:
            retrain_toggle = st.toggle(
                "Retrain RAC & SPR models (--retrain)",
                value=True,
                key="e2e_retrain_toggle",
            )
        with e2e_c2:
            raw_subproc_toggle = st.toggle(
                "Execute `src/complete_process.py` directly via subprocess",
                value=False,
                key="e2e_raw_subproc_toggle",
            )
        with e2e_c3:
            run_e2e = st.button(
                "Run full pipeline",
                type="primary",
                icon=":material/rocket_launch:",
                width="stretch",
                wrap=False,
                key="btn_run_e2e",
            )

        if run_e2e:
            _run_multi_step_pipeline(
                runner_fn=lambda log_cb, step_cb: execute_complete_process(
                    config=config,
                    retrain=retrain_toggle,
                    raw_subprocess=raw_subproc_toggle,
                    log_cb=log_cb,
                    step_cb=step_cb,
                ),
                start_text="Starting complete pipeline...",
                status_label="Executing end-to-end MotoGP pipeline...",
                complete_label="End-to-end pipeline completed",
                fail_label="End-to-end pipeline failed",
                success_msg="Complete pipeline finished cleanly.",
                success_title="Pipeline complete",
            )

    with st.container(border=True):
        st.subheader(":material/bolt: Local rebuild from data/raw/ (no API)")
        loc_c1, loc_c2 = st.columns([2.25, 1], vertical_alignment="bottom")
        with loc_c1:
            st.caption(
                "Rebuilds `cleaned_sessions.csv`, `dataset_ml.csv`, and trained `RAC` / `SPR` models directly from local `data/raw/` CSVs (Steps 2 → 3 → 4) with zero API calls."
            )
        with loc_c2:
            run_local_rebuild = st.button(
                "Rebuild without new API calls (Step 2 → 3 → 4)",
                type="primary",
                icon=":material/build_circle:",
                width="stretch",
                wrap=False,
                disabled=not raw_ready,
                key="btn_local_rebuild",
            )
        if not raw_ready:
            st.caption("Requires `data/raw/results.csv` and `sessions.csv` from Step 1.")

        if run_local_rebuild:
            _run_multi_step_pipeline(
                runner_fn=lambda log_cb, step_cb: execute_local_rebuild(
                    config=config,
                    log_cb=log_cb,
                    step_cb=step_cb,
                ),
                start_text="Starting local rebuild (Steps 2 → 3 → 4)...",
                status_label="Rebuilding dataset and models from data/raw/...",
                complete_label="Local rebuild completed",
                fail_label="Local rebuild failed",
                success_msg="Local rebuild (Steps 2 → 3 → 4) finished cleanly without API calls.",
                success_title="Rebuild complete",
            )

    st.subheader(":material/tune: Granular step execution")
    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("#### 1. `src/ingestion.py`")
            st.caption("Incremental download from PulseLive MotoGP API into `data/raw/`.")
            b1 = st.button(
                "Run ingestion",
                icon=":material/cloud_download:",
                width="stretch",
                key="btn_step_ingestion",
            )
            if b1:
                _execute_step("ingestion", "Step 1: src/ingestion.py", execute_ingestion)

        with st.container(border=True):
            st.markdown("#### 3. `src/features.py`")
            st.caption("Builds leak-free rolling, sprint, weather & circuit features.")
            b3 = st.button(
                "Run feature engineering",
                icon=":material/engineering:",
                width="stretch",
                disabled=not prep_ready,
                key="btn_step_features",
            )
            if not prep_ready:
                st.caption("Requires `data/processed/cleaned_sessions.csv` from Step 2.")
            if b3:
                _execute_step("features", "Step 3: src/features.py", execute_features)

    with c2:
        with st.container(border=True):
            st.markdown("#### 2. `src/preprocessing.py`")
            st.caption(
                f"Filters season >= `{config.start_year}` and reconstructs Q1/Q2 starting grids."
            )
            b2 = st.button(
                "Run preprocessing",
                icon=":material/cleaning_services:",
                width="stretch",
                disabled=not raw_ready,
                key="btn_step_preprocessing",
            )
            if not raw_ready:
                st.caption("Requires `data/raw/results.csv` and `sessions.csv` from Step 1.")
            if b2:
                _execute_step("preprocessing", "Step 2: src/preprocessing.py", execute_preprocessing)

        with st.container(border=True):
            st.markdown("#### 4. `src/train.py`")
            st.caption(
                f"Trains RAC & SPR models (Train <= `{config.train_end_year}`, Val `{config.val_year}`, Test `{config.test_year}`)."
            )
            b4 = st.button(
                "Run model training",
                icon=":material/model_training:",
                width="stretch",
                disabled=not feat_ready,
                key="btn_step_train",
            )
            if not feat_ready:
                st.caption("Requires `data/processed/dataset_ml.csv` from Step 3.")
            if b4:
                _execute_step("train", "Step 4: src/train.py", execute_training)

elif active_panel == "Pipeline configuration":
    with st.form("pipeline_config_form"):
        st.subheader(":material/settings: Pipeline runtime parameters")

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            start_year = st.number_input(
                "Preprocessing start year",
                min_value=2002,
                max_value=2030,
                value=config.start_year,
            )
        with f2:
            train_end_year = st.number_input(
                "Train end year (<=)",
                min_value=2018,
                max_value=2028,
                value=config.train_end_year,
            )
        with f3:
            val_year = st.number_input(
                "Validation year (==)",
                min_value=2019,
                max_value=2029,
                value=config.val_year,
            )
        with f4:
            test_year = st.number_input(
                "Test year (==)",
                min_value=2020,
                max_value=2030,
                value=config.test_year,
            )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            save_btn = st.form_submit_button(
                "Apply configuration",
                type="primary",
                icon=":material/save:",
                width="stretch",
            )
        with btn_col2:
            reset_btn = st.form_submit_button(
                "Reset defaults",
                icon=":material/restart_alt:",
                width="stretch",
            )

        if save_btn:
            cfg_dict = config.to_dict()
            cfg_dict.update(
                {
                    "start_year": int(start_year),
                    "train_end_year": int(train_end_year),
                    "val_year": int(val_year),
                    "test_year": int(test_year),
                }
            )
            st.session_state["pipeline_config"] = AppConfig.from_dict(cfg_dict)
            invalidate_all_caches()
            append_log("Updated runtime configuration from UI.")
            st.success("Configuration saved.", title="Saved", icon=":material/check_circle:")
            st.rerun()

        if reset_btn:
            st.session_state["pipeline_config"] = AppConfig()
            invalidate_all_caches()
            append_log("Reset runtime configuration to defaults.")
            st.info("Defaults restored.", title="Reset", icon=":material/info:")
            st.rerun()

else:
    st.subheader(":material/folder_open: Pipeline artifacts inventory")
    inv_rows = [
        {
            "Artifact": name,
            "Path": info["path"],
            "Status": "Available" if info["exists"] else "Missing",
            "Size (KB)": info["size_kb"],
            "Last modified": info["modified"],
        }
        for name, info in inventory.items()
    ]
    st.dataframe(
        pd.DataFrame(inv_rows),
        width="stretch",
        hide_index=True,
        alt="Inventory table of raw, processed, and model pipeline artifacts",
    )

st.divider()
log_h1, log_h2, log_h3 = st.columns([3, 1, 1], vertical_alignment="bottom")
with log_h1:
    st.subheader(":material/terminal: Execution logs")
with log_h2:
    logs_text = "\n".join(st.session_state["pipeline_logs"]) or "No execution logs recorded yet."
    st.download_button(
        "Download logs",
        data=logs_text.encode("utf-8"),
        file_name="motogp_pipeline_logs.txt",
        mime="text/plain",
        icon=":material/download:",
        width="stretch",
        disabled=not bool(st.session_state["pipeline_logs"]),
    )
with log_h3:
    if st.button("Clear logs", icon=":material/delete_sweep:", width="stretch", key="btn_clear_logs"):
        clear_logs()
        st.rerun()

st.code(logs_text, language="log")
