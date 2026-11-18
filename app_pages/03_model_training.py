"""Page 3: ML Model Training, Hyperparameter Tuning & Diagnostics (wrapping src/train.py)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.charts import (
    feature_importance_chart,
    model_comparison_chart,
    pred_vs_actual_chart,
)
from utils.config import AppConfig
from utils.data_loaders import (
    file_mtime,
    invalidate_all_caches,
    load_json_cached,
    to_csv_bytes,
    to_excel_bytes,
)
from utils.pipeline_runner import evaluate_existing_bundles, execute_training
from utils.state import append_log, get_config, init_session_state, update_step_status

init_session_state()
config = get_config()

st.title(":material/model_training: Model training & diagnostics")

dataset_exists = config.dataset_ml_csv.exists()
bundles_exist = (
    config.model_bundle_path("RAC").exists() or config.model_bundle_path("SPR").exists()
)

with st.form("model_training_form"):
    st.subheader(":material/tune: Training configuration")
    st.caption(
        "Hover over each `?` icon for parameter guidance, or open the configuration guide below."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        train_sessions = st.multiselect(
            "Sessions to train",
            options=["RAC", "SPR"],
            default=config.train_sessions,
            help="Select which session models to train: RAC (Sunday Grand Prix) and/or SPR (Saturday Sprint). Each saves a separate bundle in models/.",
        )
    with c2:
        train_end_year = st.number_input(
            "Train end year (<=)",
            min_value=2018,
            max_value=2028,
            value=config.train_end_year,
            help="Last season included in the training set (default: 2024). Keep it strictly earlier than Validation year to avoid temporal leakage.",
        )
    with c3:
        val_year = st.number_input(
            "Validation year (==)",
            min_value=2019,
            max_value=2029,
            value=config.val_year,
            help="Holdout season used to benchmark generalization and calibrate Residual σ for Monte Carlo simulations (default: 2025).",
        )
    with c4:
        test_year = st.number_input(
            "Test year (==)",
            min_value=2020,
            max_value=2030,
            value=config.test_year,
            help="Final unseen season evaluated after training (default: 2026). Must come after Validation year.",
        )

    r1, r2, r3, d1, d2, d3 = st.columns(6)
    with r1:
        reg_max_iter = st.number_input(
            "Reg max_iter",
            min_value=20,
            max_value=1000,
            value=config.reg_max_iter,
            step=25,
            help="Boosting trees for the finishing-position Regressor (default: 250). Higher captures finer pace patterns but can overfit if lr is high.",
        )
    with r2:
        reg_lr = st.number_input(
            "Reg lr",
            min_value=0.005,
            max_value=0.5,
            value=float(config.reg_learning_rate),
            step=0.01,
            format="%.3f",
            help="Learning rate for the position Regressor (default: 0.040). Lower values (0.02–0.04) smooth updates and pair well with higher max_iter.",
        )
    with r3:
        reg_depth = st.number_input(
            "Reg max_depth",
            min_value=2,
            max_value=12,
            value=config.reg_max_depth,
            help="Max tree depth for the position Regressor (default: 3). Higher models complex feature interactions; lower (3–4) prevents overfitting.",
        )
    with d1:
        clf_max_iter = st.number_input(
            "DNF max_iter",
            min_value=20,
            max_value=1000,
            value=config.clf_max_iter,
            step=25,
            help="Boosting trees for the DNF Classifier (default: 75). Adjust while monitoring Validation DNF ROC-AUC.",
        )
    with d2:
        clf_lr = st.number_input(
            "DNF lr",
            min_value=0.005,
            max_value=0.5,
            value=float(config.clf_learning_rate),
            step=0.01,
            format="%.3f",
            help="Learning rate for the DNF Classifier (default: 0.020). Smaller values keep crash probabilities better calibrated.",
        )
    with d3:
        clf_depth = st.number_input(
            "DNF max_depth",
            min_value=2,
            max_value=12,
            value=config.clf_max_depth,
            help="Max tree depth for the DNF Classifier (default: 2). Shallow trees (2–3) avoid memorizing rare one-off crashes.",
        )

    with st.expander(":material/menu_book: Parameter tuning guide", expanded=False):
        st.markdown(
            """
            - **Temporal split (`Train end year` < `Validation year` < `Test year`)**: Keeps training strictly chronological. `Validation year` is also used to measure `Residual σ`, which drives Monte Carlo variance in the Race Simulator.
            - **Position Regressor (`Reg max_iter`, `Reg lr`, `Reg max_depth`)**: Trained only on classified finishers (`is_dnf == 0`) to predict expected finishing position (`pred_pos`). Increase `max_iter` while lowering `lr` if `Val MAE` improves without hurting `Test MAE`; keep `max_depth` around `3–4` to prevent overfitting.
            - **DNF Classifier (`DNF max_iter`, `DNF lr`, `DNF max_depth`)**: Trained on all starters to estimate retirement risk (`dnf_prob`). Since MotoGP crashes are noisy, shallow trees (`max_depth` `2–3`) and low learning rates (`0.02–0.04`) usually yield higher `DNF ROC-AUC`.
            """
        )

    train_submit = st.form_submit_button(
        "Train & evaluate models",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
        disabled=not dataset_exists,
    )

if not dataset_exists:
    st.warning(
        f"Missing `{config.dataset_ml_csv}`. Run Steps 1–3 in **Pipeline orchestrator** first.",
        title="Training dataset unavailable",
        icon=":material/warning:",
    )

if train_submit:
    if not train_sessions:
        st.error("Select at least one session type (RAC or SPR).", title="Invalid selection")
    else:
        cfg_dict = config.to_dict()
        cfg_dict.update(
            {
                "train_sessions": train_sessions,
                "train_end_year": int(train_end_year),
                "val_year": int(val_year),
                "test_year": int(test_year),
                "reg_max_iter": int(reg_max_iter),
                "reg_learning_rate": float(reg_lr),
                "reg_max_depth": int(reg_depth),
                "clf_max_iter": int(clf_max_iter),
                "clf_learning_rate": float(clf_lr),
                "clf_max_depth": int(clf_depth),
            }
        )
        new_cfg = AppConfig.from_dict(cfg_dict)
        st.session_state["pipeline_config"] = new_cfg
        config = new_cfg

        prog = st.progress(0.0, text="Starting training...")
        with st.status("Training RAC & SPR models via `src/train.py`...", expanded=True) as status_box:
            def _tlog(msg: str) -> None:
                append_log(msg, level="TRAIN")
                st.write(msg)

            def _tprog(frac: float, text: str) -> None:
                prog.progress(min(max(frac, 0.0), 1.0), text=text)

            try:
                report = execute_training(
                    config=new_cfg,
                    log_cb=_tlog,
                    progress_cb=_tprog,
                )
                update_step_status("train", "success", {"sessions": list(report["sessions"].keys())})
                invalidate_all_caches()
                prog.progress(1.0, text="Training completed.")
                status_box.update(label="Model training complete", state="complete", expanded=False)
                st.success(
                    "Model bundles (`models/*.pkl`) and diagnostics updated.",
                    title="Training finished",
                    icon=":material/check_circle:",
                )
            except Exception as exc:
                append_log(f"Training failed: {exc}", level="ERROR")
                update_step_status("train", "error", {"error": str(exc)})
                status_box.update(label="Training failed", state="error", expanded=True)
                st.error(f"{exc}", title="Training error", icon=":material/error:")

st.divider()

metrics_mtime = file_mtime(config.training_metrics_json)
metrics_report = load_json_cached(str(config.training_metrics_json), metrics_mtime)

if (not metrics_report or not metrics_report.get("sessions")) and dataset_exists and bundles_exist:
    st.info(
        "Found pre-existing `.pkl` models in `models/`. Generate diagnostics without retraining to inspect their metrics and feature importances.",
        title="Existing models detected",
        icon=":material/info:",
    )
    if st.button(
        "Evaluate existing models in models/",
        type="primary",
        icon=":material/assessment:",
        key="btn_eval_existing_bundles",
    ):
        with st.spinner("Evaluating existing model bundles..."):
            metrics_report = evaluate_existing_bundles(config, log_cb=lambda m: append_log(m, "EVAL"))
            invalidate_all_caches()
            st.rerun()

if not metrics_report or not metrics_report.get("sessions"):
    st.info(
        "No training diagnostics available yet. Click **Train & evaluate models** above to build `model_rac.pkl` and `model_spr.pkl`.",
        title="No model metrics available",
        icon=":material/info:",
    )
    st.stop()

st.caption(f"Diagnostics timestamp: `{metrics_report.get('trained_at', 'N/A')[:19].replace('T', ' ')} UTC`")

def _benchmark_delta_style(
    val_num: float | None,
    test_num: float | None,
    higher_is_better: bool,
    decimals: int = 2,
) -> tuple[str, str]:
    """Return (delta_color, delta_arrow) comparing Test vs Val for st.metric."""
    if val_num is None or test_num is None:
        return "off", "off"
    v_disp = round(float(val_num), decimals)
    t_disp = round(float(test_num), decimals)
    if t_disp == v_disp:
        return "off", "off"
    went_up = t_disp > v_disp
    is_good = went_up if higher_is_better else not went_up
    color = "normal" if is_good else "inverse"
    arrow = "up" if went_up else "down"
    return color, arrow


avail_sessions = list(metrics_report["sessions"].keys())
for sess_key in avail_sessions:
    sdata = metrics_report["sessions"][sess_key]
    val_m = sdata.get("val_metrics", {})
    test_m = sdata.get("test_metrics", {})

    st.markdown(f"#### :material/leaderboard: {sess_key} model benchmarks")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        mae_v = val_m.get("mae")
        mae_t = test_m.get("mae")
        mae_color, mae_arrow = _benchmark_delta_style(
            mae_v, mae_t, higher_is_better=False, decimals=2
        )
        st.metric(
            f"{sess_key} Val MAE",
            f"{mae_v:.2f} pos" if mae_v is not None else "N/A",
            delta=f"Test: {mae_t:.2f}" if mae_t is not None else "No test data",
            delta_color=mae_color,
            delta_arrow=mae_arrow,
        )
    with m2:
        std_v = sdata.get("effective_std_dev")
        st.metric(
            f"{sess_key} Residual σ",
            f"{std_v:.2f}" if std_v is not None else "N/A",
            delta="Used in Monte Carlo",
            delta_color="off",
            delta_arrow="off",
        )
    with m3:
        auc_v = val_m.get("dnf_auc")
        auc_t = test_m.get("dnf_auc")
        auc_color, auc_arrow = _benchmark_delta_style(
            auc_v, auc_t, higher_is_better=True, decimals=3
        )
        st.metric(
            f"{sess_key} DNF ROC-AUC",
            f"{auc_v:.3f}" if auc_v is not None else "N/A",
            delta=f"Test: {auc_t:.3f}" if auc_t is not None else "No test data",
            delta_color=auc_color,
            delta_arrow=auc_arrow,
        )
    with m4:
        pod_v = val_m.get("podium_hits")
        pod_t = test_m.get("podium_hits")
        pod_color, pod_arrow = _benchmark_delta_style(
            pod_v, pod_t, higher_is_better=True, decimals=2
        )
        st.metric(
            f"{sess_key} Podium hits",
            f"{pod_v:.2f} / 3" if pod_v is not None else "N/A",
            delta=f"Test: {pod_t:.2f} / 3" if pod_t is not None else "No test data",
            delta_color=pod_color,
            delta_arrow=pod_arrow,
        )

st.divider()

st.markdown("#### Diagnostic view")
diag_view = st.segmented_control(
    "Diagnostic view",
    options=[
        "Split comparison",
        "Predicted vs actual",
        "Feature importances",
        "Evaluation table",
    ],
    default="Split comparison",
    label_visibility="collapsed",
    key="train_diag_view",
)

if diag_view == "Split comparison":
    st.altair_chart(
        model_comparison_chart(metrics_report),
        width="stretch",
        alt="Bar chart comparing MAE, DNF AUC, and Podium Hits across validation and test splits",
    )
else:
    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        chosen_sess = st.segmented_control(
            "Session model",
            options=avail_sessions,
            default=avail_sessions[0],
            key="diag_session_pick",
        )
    with ctrl2:
        chosen_split = st.segmented_control(
            "Evaluation split",
            options=["Validation", "Test"],
            default="Validation",
            key="diag_split_pick",
        )

    s_info = metrics_report["sessions"][chosen_sess]
    split_obj = s_info["val_metrics"] if chosen_split == "Validation" else s_info["test_metrics"]
    preds = split_obj.get("predictions", [])

    if diag_view == "Predicted vs actual":
        if not preds:
            st.info(f"No predictions in {chosen_split} split for {chosen_sess}.", title="Empty split")
        else:
            st.altair_chart(
                pred_vs_actual_chart(preds),
                width="stretch",
                alt="Scatter plot of predicted finishing position versus actual position",
            )
    elif diag_view == "Feature importances":
        imps = s_info.get("feature_importances", [])
        if not imps:
            st.info("Feature importances not available.", title="No importances")
        else:
            ic1, ic2 = st.columns(2)
            with ic1:
                st.altair_chart(
                    feature_importance_chart(imps, "reg_importance"),
                    width="stretch",
                    alt="Horizontal bar chart of regressor permutation feature importances",
                )
            with ic2:
                st.altair_chart(
                    feature_importance_chart(imps, "clf_importance"),
                    width="stretch",
                    alt="Horizontal bar chart of DNF classifier permutation feature importances",
                )
    else:
        if not preds:
            st.info(f"No evaluation records in {chosen_split} split for {chosen_sess}.", title="Empty split")
        else:
            df_preds = pd.DataFrame(preds)
            st.dataframe(
                df_preds,
                width="stretch",
                hide_index=True,
                column_config={
                    "pred_pos": st.column_config.NumberColumn("Predicted pos", format="%.2f"),
                    "dnf_prob": st.column_config.ProgressColumn("DNF prob", min_value=0.0, max_value=1.0, format="%.2f"),
                },
                alt="Evaluation predictions table comparing actual vs predicted positions",
            )
            e1, e2 = st.columns(2)
            with e1:
                st.download_button(
                    "Download predictions CSV",
                    data=to_csv_bytes(df_preds),
                    file_name=f"eval_{chosen_sess.lower()}_{chosen_split.lower()}.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    width="stretch",
                )
            with e2:
                st.download_button(
                    "Download predictions Excel",
                    data=to_excel_bytes(df_preds, f"{chosen_sess}_{chosen_split}"),
                    file_name=f"eval_{chosen_sess.lower()}_{chosen_split.lower()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    icon=":material/table_view:",
                    width="stretch",
                )
