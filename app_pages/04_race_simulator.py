"""Page 4: Weekend Race Predictor, What-If Grid Simulator & Historical Backtester (using src/simulation.py)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import simulation as sim_mod
from utils.charts import (
    backtest_comparison_chart,
    pace_vs_crash_risk_chart,
    simulation_odds_chart,
)
from utils.data_loaders import (
    file_mtime,
    load_csv_cached,
    load_model_bundle_cached,
    to_csv_bytes,
    to_excel_bytes,
)
from utils.bootstrap import ensure_local_artifacts_bootstrapped
from utils.pipeline_runner import list_completed_events, safe_detect_weekend_state
from utils.state import append_log, get_config, init_session_state, update_step_status
from utils.upcoming_gp_ui import render_upcoming_gp_controls

init_session_state()
config = get_config()
ensure_local_artifacts_bootstrapped(config)

st.title(":material/flag: Race simulator & backtest")
st.caption(
    "Simulate upcoming MotoGP sprint and grand prix outcomes across 10,000+ Monte Carlo runs with What-If grid overrides, or backtest past weekends against actual results."
)

ml_mtime = file_mtime(config.dataset_ml_csv)
df_ml = load_csv_cached(str(config.dataset_ml_csv), ml_mtime)

if df_ml.empty:
    st.warning(
        f"Missing `{config.dataset_ml_csv}`. Run Steps 1–3 in **Pipeline orchestrator** first.",
        title="ML dataset unavailable",
        icon=":material/warning:",
    )
    st.stop()

# Ensure chronological ordering expected by src/simulation.py
if "date" in df_ml.columns:
    df_ml = df_ml.sort_values(by=["date", "session_id"]).reset_index(drop=True)

rac_path = config.model_bundle_path("RAC")
spr_path = config.model_bundle_path("SPR")
rac_mtime = file_mtime(rac_path)
spr_mtime = file_mtime(spr_path)
rac_bundle = load_model_bundle_cached(str(rac_path), rac_mtime)
spr_bundle = load_model_bundle_cached(str(spr_path), spr_mtime)

if rac_bundle is None and spr_bundle is None:
    st.warning(
        "No trained model bundles found (`models/model_rac.pkl` / `models/model_spr.pkl`). Train models first.",
        title="Models unavailable",
        icon=":material/warning:",
    )
    st.stop()

st.markdown("#### Simulation mode")
mode = st.segmented_control(
    "Simulation mode",
    options=["Upcoming GP forecast", "Historical backtest"],
    default="Upcoming GP forecast",
    label_visibility="collapsed",
    key="sim_mode_selector",
)

if mode == "Upcoming GP forecast":
    try:
        weekend = safe_detect_weekend_state(df_ml, config)
    except Exception as exc:
        st.error(
            f"{exc}",
            title="Unable to detect upcoming GP state",
            icon=":material/error:",
        )
        st.stop()

    chosen_circuit, chosen_session, weather_opt, n_sims, std_mult = (
        render_upcoming_gp_controls(
            weekend,
            config,
            rac_bundle,
            spr_bundle,
            key_prefix="sim",
        )
    )

    bundle = rac_bundle if chosen_session == "RAC" else spr_bundle
    if bundle is None:
        st.error(
            f"Model bundle for {chosen_session} is not trained yet.",
            title="Missing model",
        )
        st.stop()

    edit_grid = st.toggle(
        "Enable What-If starting grid & sprint override editor",
        value=False,
        key="sim_enable_whatif",
    )

    grid_source_df = (
        weekend["grid_df"].copy().sort_values("grid_position").reset_index(drop=True)
    )
    if edit_grid:
        st.caption(
            "Edit any rider's `grid_position` (1–24) or `sprint_pos_same_weekend` below to test custom scenarios."
        )
        col_cfg = {
            "grid_position": st.column_config.NumberColumn(
                "Grid position", min_value=1, max_value=30, step=1
            )
        }
        if "sprint_pos_same_weekend" in grid_source_df.columns:
            col_cfg["sprint_pos_same_weekend"] = st.column_config.NumberColumn(
                "Sprint finish", min_value=1, max_value=30, step=1
            )
        if "fp_avg_pos" in grid_source_df.columns:
            col_cfg["fp_avg_pos"] = st.column_config.NumberColumn(
                "FP avg", min_value=1.0, max_value=30.0, step=0.1, format="%.2f"
            )
        if "pr_pos" in grid_source_df.columns:
            col_cfg["pr_pos"] = st.column_config.NumberColumn(
                "Practice (PR)", min_value=1, max_value=30, step=1
            )

        grid_source_df = st.data_editor(
            grid_source_df,
            width="stretch",
            hide_index=True,
            disabled=["rider_id", "rider_name", "constructor_name"],
            column_config=col_cfg,
            key="whatif_grid_editor",
            alt="Editable starting grid table for What-If race simulation",
        )

    is_wet_flag = 1 if weather_opt == "Wet / Rain" else 0
    bundle_mtime = rac_mtime if chosen_session == "RAC" else spr_mtime
    grid_snapshot = tuple(
        tuple(None if pd.isna(v) else v for v in row)
        for row in grid_source_df.itertuples(index=False, name=None)
    )
    cache_sig = (
        weekend["state"],
        chosen_circuit,
        chosen_session,
        is_wet_flag,
        int(n_sims),
        round(float(std_mult), 3),
        ml_mtime,
        bundle_mtime,
        grid_snapshot,
    )

    if st.session_state["last_sim_results"].get("sig") != cache_sig:
        with st.spinner(
            f"Running {n_sims:,} Monte Carlo simulations for {chosen_session}..."
        ):
            df_upcoming = sim_mod.build_upcoming_race(
                df_ml,
                grid_source_df,
                chosen_circuit,
                session_type=chosen_session,
                is_wet=is_wet_flag,
            )
            df_pred = sim_mod.predict_session(df_upcoming, bundle)
            effective_std = float(bundle["std_dev"]) * float(std_mult)
            sim_df = sim_mod.simulate_race(
                df_pred,
                effective_std,
                n_sims=int(n_sims),
            )
            sim_df = sim_df.reset_index(drop=True)
            sim_df.insert(0, "sim_rank", range(1, len(sim_df) + 1))

            st.session_state["last_sim_results"] = {
                "sig": cache_sig,
                "circuit": chosen_circuit,
                "session": chosen_session,
                "weather": weather_opt,
                "n_sims": int(n_sims),
                "std_dev": effective_std,
                "df": sim_df,
            }
            update_step_status(
                "simulation",
                "success",
                {"circuit": chosen_circuit, "session": chosen_session},
            )
            append_log(
                f"Simulated {chosen_session} at {chosen_circuit} ({weather_opt}, {n_sims:,} runs)."
            )

    sim_payload = st.session_state.get("last_sim_results", {})
    if sim_payload and not sim_payload["df"].empty:
        sim_df: pd.DataFrame = sim_payload["df"]
        fav = sim_df.iloc[0]
        top_podium = sim_df.sort_values("podium_pct", ascending=False).iloc[0]
        top_risk = sim_df.sort_values("dnf_prob", ascending=False).iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric(
                "Projected winner",
                str(fav["rider_name"]),
                delta=f"{fav['win_pct']:.1f}% win odds",
            )
        with k2:
            st.metric(
                "Highest podium probability",
                str(top_podium["rider_name"]),
                delta=f"{top_podium['podium_pct']:.1f}% podium",
            )
        with k3:
            st.metric(
                "Highest DNF crash risk",
                str(top_risk["rider_name"]),
                delta=f"{top_risk['dnf_prob'] * 100:.1f}% DNF",
                delta_color="inverse",
            )
        with k4:
            st.metric(
                "Simulation noise σ",
                f"{sim_payload['std_dev']:.2f} pos",
                delta=f"{sim_payload['n_sims']:,} runs · {sim_payload['weather']}",
                delta_color="off",
                delta_arrow="off",
            )

        session_kind = "sprint" if sim_payload["session"] == "SPR" else "race"
        st.markdown(
            f"#### :material/format_list_numbered: Projected {session_kind} classification — {sim_payload['session']}"
        )

        with st.container(key="podium_card_styles"):
            st.html(
                """
                <style>
                .st-key-podium_card_styles {
                    display: none !important;
                }
                .st-key-podium_card_1 {
                    background: linear-gradient(135deg, rgba(234, 179, 8, 0.25), rgba(234, 179, 8, 0.08)) !important;
                    border-color: rgba(234, 179, 8, 0.55) !important;
                }
                .st-key-podium_card_2 {
                    background: linear-gradient(135deg, rgba(203, 213, 225, 0.22), rgba(148, 163, 184, 0.07)) !important;
                    border-color: rgba(203, 213, 225, 0.50) !important;
                }
                .st-key-podium_card_3 {
                    background: linear-gradient(135deg, rgba(217, 119, 6, 0.24), rgba(180, 83, 9, 0.08)) !important;
                    border-color: rgba(217, 119, 6, 0.52) !important;
                }
                .st-key-podium_card_1 h3,
                .st-key-podium_card_2 h3,
                .st-key-podium_card_3 h3 {
                    margin: 0.1rem 0 0.25rem 0 !important;
                    padding: 0 !important;
                    font-size: 1.45rem !important;
                    font-weight: 700 !important;
                    line-height: 1.2 !important;
                }
                </style>
                """
            )

        def _grid_move_badge(rank: int, grid_val: object) -> str:
            if pd.isna(grid_val):
                return ""
            grid_pos = int(grid_val)
            gain = grid_pos - rank
            if gain > 0:
                return f":green[▲ +{gain}]"
            if gain < 0:
                return f":red[▼ {gain}]"
            return ":gray[—]"

        podium_df = sim_df.head(3)
        podium_cols = st.columns(len(podium_df))
        podium_badges = [
            ":orange[:material/trophy: P1 · WINNER]",
            ":gray[:material/military_tech: P2 · PODIUM]",
            ":red[:material/military_tech: P3 · PODIUM]",
        ]
        for idx, (_, row) in enumerate(podium_df.iterrows()):
            rank = int(row["sim_rank"])
            move_txt = _grid_move_badge(rank, row.get("grid_position"))
            grid_txt = (
                f"Grid P{int(row['grid_position'])}"
                if pd.notna(row.get("grid_position"))
                else "Grid N/A"
            )
            with podium_cols[idx]:
                with st.container(border=True, key=f"podium_card_{idx + 1}"):
                    st.markdown(f"**{podium_badges[idx]}**")
                    st.markdown(f"### {row['rider_name']}")
                    st.caption(f"{row['constructor_name']} · {grid_txt} {move_txt}")
                    st.markdown(
                        f"Win **{row['win_pct']:.1f}%** · Podium **{row['podium_pct']:.1f}%**"
                    )

        rest_df = sim_df.iloc[3:].reset_index(drop=True)
        if not rest_df.empty:
            n_cols = min(3, len(rest_df))
            chunk_size = (len(rest_df) + n_cols - 1) // n_cols
            with st.container(border=True):
                grid_cols = st.columns(n_cols)
                for col_idx in range(n_cols):
                    chunk = rest_df.iloc[col_idx * chunk_size : (col_idx + 1) * chunk_size]
                    lines = []
                    for _, row in chunk.iterrows():
                        rank = int(row["sim_rank"])
                        move_txt = _grid_move_badge(rank, row.get("grid_position"))
                        if rank <= 15:
                            lines.append(
                                f":primary-badge[**P{rank}**] **{row['rider_name']}** "
                                f":gray[({row['constructor_name']})] {move_txt}"
                            )
                        else:
                            lines.append(
                                f":gray-badge[P{rank}] :gray[{row['rider_name']} "
                                f"({row['constructor_name']})] {move_txt}"
                            )
                    with grid_cols[col_idx]:
                        st.markdown("\n\n".join(lines))
            st.caption(
                ":primary-badge[P1–P15] Inside Points · :gray-badge[P16+] Outside Top 15"
            )

        st.divider()
        st.markdown("#### Forecast visualization")
        chart_tab = st.segmented_control(
            "Forecast visualization",
            options=["Win / Podium / Top 10 odds", "Expected pace vs crash risk"],
            default="Win / Podium / Top 10 odds",
            label_visibility="collapsed",
            key="sim_chart_choice",
        )
        if chart_tab == "Win / Podium / Top 10 odds":
            st.altair_chart(
                simulation_odds_chart(sim_df, top_n=15),
                width="stretch",
                alt="Grouped bar chart of Win, Podium, and Top 10 probabilities per rider",
            )
        else:
            st.altair_chart(
                pace_vs_crash_risk_chart(sim_df),
                width="stretch",
                alt="Scatter plot of expected finishing position versus DNF crash risk",
            )

        st.subheader(
            f":material/table_rows: {sim_payload['session']} classification probabilities — {sim_payload['circuit']}"
        )
        display_sim = sim_df.copy()
        display_sim["dnf_pct"] = (display_sim["dnf_prob"] * 100).round(2)
        display_sim["pred_pos"] = display_sim["pred_pos"].round(2)

        st.dataframe(
            display_sim,
            width="stretch",
            hide_index=True,
            column_config={
                "sim_rank": st.column_config.NumberColumn("Proj rank", format="%d"),
                "grid_position": st.column_config.NumberColumn("Grid", format="%d"),
                "pred_pos": st.column_config.NumberColumn("Expected pos", format="%.2f"),
                "win_pct": st.column_config.ProgressColumn(
                    "Win %", min_value=0.0, max_value=100.0, format="%.1f%%"
                ),
                "podium_pct": st.column_config.ProgressColumn(
                    "Podium %", min_value=0.0, max_value=100.0, format="%.1f%%"
                ),
                "top10_pct": st.column_config.ProgressColumn(
                    "Top 10 %", min_value=0.0, max_value=100.0, format="%.1f%%"
                ),
                "dnf_pct": st.column_config.NumberColumn("DNF risk %", format="%.1f%%"),
            },
            alt="Monte Carlo race prediction probabilities table",
        )

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download forecast CSV",
                data=to_csv_bytes(display_sim),
                file_name=f"forecast_{sim_payload['session'].lower()}.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
            )
        with dl2:
            st.download_button(
                "Download forecast Excel",
                data=to_excel_bytes(display_sim, f"Forecast_{sim_payload['session']}"),
                file_name=f"forecast_{sim_payload['session'].lower()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                icon=":material/table_view:",
                width="stretch",
            )

else:
    events_list = list_completed_events(df_ml)
    if events_list.empty:
        st.info("No completed historical events found in dataset.", title="Empty dataset")
        st.stop()

    with st.container(border=True):
        st.markdown("#### :material/history: Historical GP backtesting")
        b1, b2, b3, b4 = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
        with b1:
            selected_label = st.selectbox(
                "Completed Grand Prix",
                options=events_list["label"].tolist(),
                index=0,
                key="bt_event_select",
            )
            chosen_row = events_list[events_list["label"] == selected_label].iloc[0]
            event_data = sim_mod.get_completed_event(df_ml, str(chosen_row["event_id"]))
        with b2:
            bt_sessions = [
                s
                for s in event_data["sessions_data"].keys()
                if (s == "RAC" and rac_bundle is not None)
                or (s == "SPR" and spr_bundle is not None)
            ]
            bt_sess = st.segmented_control(
                "Session",
                options=bt_sessions,
                default=bt_sessions[-1] if bt_sessions else None,
                key="bt_session_select",
            )
        with b3:
            bt_n_sims = st.select_slider(
                "Simulations",
                options=[1000, 5000, 10000, 25000],
                value=10000,
                key="bt_n_sims",
            )
        with b4:
            run_bt_btn = st.button(
                "Run backtest",
                type="primary",
                icon=":material/replay:",
                width="stretch",
                key="btn_run_backtest",
                disabled=not bool(bt_sess),
            )

    if run_bt_btn and bt_sess:
        bundle = rac_bundle if bt_sess == "RAC" else spr_bundle
        ev_sub = event_data["sessions_data"][bt_sess]
        with st.spinner(
            f"Backtesting {bt_sess} at {event_data['circuit_name']}..."
        ):
            df_pred = sim_mod.predict_session(ev_sub, bundle)
            bt_df = sim_mod.simulate_race(
                df_pred,
                float(bundle["std_dev"]),
                n_sims=int(bt_n_sims),
            )
            # Re-attach is_dnf from historical session slice for accurate metrics & coloring
            if "is_dnf" in ev_sub.columns:
                bt_df = bt_df.merge(
                    ev_sub[["rider_name", "is_dnf"]].drop_duplicates("rider_name"),
                    on="rider_name",
                    how="left",
                )
                bt_df["is_dnf"] = bt_df["is_dnf"].fillna(0).astype(int)
            else:
                bt_df["is_dnf"] = (bt_df["position_cleaned"] >= 25).astype(int)

            bt_df = bt_df.reset_index(drop=True)
            bt_df.insert(0, "sim_rank", range(1, len(bt_df) + 1))

            st.session_state["last_backtest_results"] = {
                "circuit": event_data["circuit_name"],
                "date": event_data["race_date"],
                "session": bt_sess,
                "df": bt_df,
            }
            append_log(
                f"Backtested {bt_sess} at {event_data['circuit_name']} ({event_data['race_date']})."
            )

    bt_payload = st.session_state.get("last_backtest_results", {})
    if bt_payload and not bt_payload["df"].empty:
        bt_df: pd.DataFrame = bt_payload["df"]
        actual_winner = bt_df.sort_values("position_cleaned").iloc[0]["rider_name"]
        pred_winner = bt_df.iloc[0]["rider_name"]
        actual_podium = set(bt_df[bt_df["position_cleaned"] <= 3]["rider_name"])
        pred_podium = set(bt_df.head(3)["rider_name"])
        podium_hits = len(actual_podium & pred_podium)

        fin_only = bt_df[bt_df["is_dnf"] == 0]
        event_mae = (
            float((fin_only["position_cleaned"] - fin_only["pred_pos"]).abs().mean())
            if not fin_only.empty
            else 0.0
        )

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            winner_hit = str(pred_winner).strip() == str(actual_winner).strip()
            st.metric(
                "Predicted vs actual winner",
                pred_winner,
                delta=f"Actual: {actual_winner}",
                delta_color="normal" if winner_hit else "inverse",
                delta_arrow="off",
            )
        with k2:
            st.metric(
                "Podium accuracy",
                f"{podium_hits} / 3",
                delta="Top 3 overlap",
                delta_color="off",
                delta_arrow="off",
            )
        with k3:
            st.metric(
                "Event MAE (finishers)",
                f"{event_mae:.2f} pos",
                delta=bt_payload["session"],
                delta_color="off",
                delta_arrow="off",
            )
        with k4:
            st.metric(
                "Grand Prix",
                bt_payload["circuit"],
                delta=bt_payload["date"],
                delta_color="off",
                delta_arrow="off",
            )

        st.altair_chart(
            backtest_comparison_chart(bt_df),
            width="stretch",
            alt="Scatter plot comparing simulated predicted positions against actual historical race results",
        )

        bt_display = bt_df.copy()
        bt_display["dnf_pct"] = (bt_display["dnf_prob"] * 100).round(2)
        bt_display["pred_pos"] = bt_display["pred_pos"].round(2)
        st.dataframe(
            bt_display,
            width="stretch",
            hide_index=True,
            column_config={
                "sim_rank": st.column_config.NumberColumn("Pred rank", format="%d"),
                "position_cleaned": st.column_config.NumberColumn("Actual finish", format="%d"),
                "grid_position": st.column_config.NumberColumn("Grid", format="%d"),
                "pred_pos": st.column_config.NumberColumn("Expected pos", format="%.2f"),
                "win_pct": st.column_config.ProgressColumn(
                    "Win %", min_value=0.0, max_value=100.0, format="%.1f%%"
                ),
                "podium_pct": st.column_config.ProgressColumn(
                    "Podium %", min_value=0.0, max_value=100.0, format="%.1f%%"
                ),
            },
            alt="Backtest comparison table between Monte Carlo probabilities and actual finish",
        )
