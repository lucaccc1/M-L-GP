"""Page 5: Upcoming GP Rider Head-to-Head (H2H) Duel & Telemetry Comparison."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.charts import (
    h2h_outcome_bar_chart,
    h2h_position_distribution_chart,
    h2h_recent_form_chart,
)
from utils.data_loaders import (
    file_mtime,
    load_csv_cached,
    load_model_bundle_cached,
    to_csv_bytes,
    to_excel_bytes,
)
from utils.head_to_head import (
    build_feature_comparison_table,
    compute_rider_pair_duel,
    get_rider_circuit_history,
    get_rider_recent_history,
    run_h2h_simulation,
)
from utils.pipeline_runner import safe_detect_weekend_state
from utils.state import get_config, init_session_state
from utils.upcoming_gp_ui import render_upcoming_gp_controls

init_session_state()
config = get_config()

st.title(":material/compare_arrows: Rider head-to-head")
st.caption(
    "Compare any two riders from the upcoming MotoGP weekend across 10,000+ Monte Carlo simulations, engineered pace/risk features, and historical circuit data."
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
        key_prefix="h2h",
    )
)

bundle = rac_bundle if chosen_session == "RAC" else spr_bundle
if bundle is None:
    st.error(f"Model bundle for {chosen_session} is not trained yet.", title="Missing model")
    st.stop()

grid_source_df = weekend["grid_df"].copy().sort_values("grid_position").reset_index(drop=True)
is_wet_flag = 1 if weather_opt == "Wet / Rain" else 0
bundle_mtime = rac_mtime if chosen_session == "RAC" else spr_mtime
grid_snapshot = tuple(
    (str(r["rider_name"]), int(r["grid_position"]))
    for _, r in grid_source_df.iterrows()
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

cached_h2h = st.session_state.get("h2h_sim_cache", {})
if cached_h2h.get("sig") != cache_sig:
    with st.spinner(
        f"Running {int(n_sims):,} Monte Carlo simulations for {chosen_session} H2H matrix..."
    ):
        sim_payload = run_h2h_simulation(
            df_ml=df_ml,
            grid_df=grid_source_df,
            circuit_name=chosen_circuit,
            bundle=bundle,
            session_type=chosen_session,
            is_wet=is_wet_flag,
            n_sims=int(n_sims),
            std_mult=float(std_mult),
        )
        st.session_state["h2h_sim_cache"] = {
            "sig": cache_sig,
            "payload": sim_payload,
        }
else:
    sim_payload = cached_h2h["payload"]

sim_df: pd.DataFrame = sim_payload["sim_df"]
if len(sim_df) < 2:
    st.warning("Need at least two riders on the grid to run a head-to-head comparison.")
    st.stop()

rider_names = sim_df["rider_name"].tolist()
rider_meta = {
    row["rider_name"]: f"{row['rider_name']} ({row['constructor_name']} · Grid P{int(row['grid_position'])})"
    for _, row in sim_df.iterrows()
}

with st.container(key="h2h_card_styles"):
    st.html(
        """
        <style>
        .st-key-h2h_card_styles {
            display: none !important;
        }
        .st-key-h2h_rider_a [data-testid="stWidgetLabel"] p,
        .st-key-h2h_rider_b [data-testid="stWidgetLabel"] p {
            font-size: 1.05rem !important;
            font-weight: 600 !important;
        }
        .st-key-h2h_card_a {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.22), rgba(37, 99, 235, 0.05)) !important;
            border-color: rgba(59, 130, 246, 0.55) !important;
        }
        .st-key-h2h_card_b {
            background: linear-gradient(135deg, rgba(217, 4, 41, 0.22), rgba(217, 4, 41, 0.05)) !important;
            border-color: rgba(239, 68, 68, 0.55) !important;
        }
        .st-key-h2h_card_a h5,
        .st-key-h2h_card_b h5 {
            margin: 0 !important;
            padding: 0 !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.03em !important;
        }
        .st-key-h2h_card_a h2,
        .st-key-h2h_card_b h2 {
            margin: 0.1rem 0 0.25rem 0 !important;
            padding: 0 !important;
            font-size: 1.55rem !important;
            font-weight: 700 !important;
        }
        .st-key-h2h_card_a [data-testid="stHeaderActionElements"],
        .st-key-h2h_card_b [data-testid="stHeaderActionElements"] {
            display: none !important;
        }
        </style>
        """
    )

sel1, sel2 = st.columns(2)
with sel1:
    rider_a = st.selectbox(
        "Rider A (Blue)",
        options=rider_names,
        index=0,
        format_func=lambda r: rider_meta.get(r, r),
        key="h2h_rider_a",
    )
with sel2:
    default_b_idx = 1 if rider_names[0] == rider_a else 0
    rider_b = st.selectbox(
        "Rider B (Red)",
        options=rider_names,
        index=default_b_idx,
        format_func=lambda r: rider_meta.get(r, r),
        key="h2h_rider_b",
    )

if rider_a == rider_b:
    st.info(
        "Select two different riders above to compute their head-to-head duel odds.",
        title="Same rider selected",
        icon=":material/info:",
    )
    st.stop()

duel = compute_rider_pair_duel(sim_payload, rider_a, rider_b)
row_a = sim_df[sim_df["rider_name"] == rider_a].iloc[0]
row_b = sim_df[sim_df["rider_name"] == rider_b].iloc[0]

card_a, card_b = st.columns(2)
with card_a:
    with st.container(border=True, key="h2h_card_a"):
        st.markdown("##### :blue[:material/sports_motorsports: RIDER A]")
        st.markdown(f"## {rider_a}")
        st.caption(
            f"{row_a['constructor_name']} · Grid P{int(row_a['grid_position'])} · Expected pos **{row_a['pred_pos']:.2f}**"
        )
        st.markdown(
            f"### Beats {rider_b}: **{duel['a_beats_b_total_pct']:.1f}%**"
        )
        st.markdown(
            f"Win **{row_a['win_pct']:.1f}%** · Podium **{row_a['podium_pct']:.1f}%** · Top 10 **{row_a['top10_pct']:.1f}%** · DNF **{row_a['dnf_pct']:.1f}%**"
        )

with card_b:
    with st.container(border=True, key="h2h_card_b"):
        st.markdown("##### :red[:material/sports_motorsports: RIDER B]")
        st.markdown(f"## {rider_b}")
        st.caption(
            f"{row_b['constructor_name']} · Grid P{int(row_b['grid_position'])} · Expected pos **{row_b['pred_pos']:.2f}**"
        )
        st.markdown(
            f"### Beats {rider_a}: **{duel['b_beats_a_total_pct']:.1f}%**"
        )
        st.markdown(
            f"Win **{row_b['win_pct']:.1f}%** · Podium **{row_b['podium_pct']:.1f}%** · Top 10 **{row_b['top10_pct']:.1f}%** · DNF **{row_b['dnf_pct']:.1f}%**"
        )

st.markdown("#### :material/insights: Monte Carlo duel summary")
k1, k2, k3, k4 = st.columns(4)

with k1:
    fav_name = rider_a if duel["a_beats_b_total_pct"] >= duel["b_beats_a_total_pct"] else rider_b
    fav_pct = max(duel["a_beats_b_total_pct"], duel["b_beats_a_total_pct"])
    dog_pct = min(duel["a_beats_b_total_pct"], duel["b_beats_a_total_pct"])
    st.metric(
        "Projected H2H favorite",
        fav_name,
        delta=f"{fav_pct:.1f}% vs {dog_pct:.1f}%",
        delta_color="off",
        delta_arrow="off",
    )

with k2:
    if duel["a_ahead_both_finish_pct"] is not None:
        pace_fav = rider_a if duel["a_ahead_both_finish_pct"] >= 50.0 else rider_b
        pace_pct = max(duel["a_ahead_both_finish_pct"], duel["b_ahead_both_finish_pct"])
        st.metric(
            "Pure pace H2H (both finish)",
            f"{pace_fav} ({pace_pct:.1f}%)",
            delta=f"Both finish {duel['both_finish_pct']:.1f}% of runs",
            delta_color="off",
            delta_arrow="off",
        )
    else:
        st.metric("Pure pace H2H (both finish)", "N/A")

with k3:
    gap = duel["mean_pos_gap_both_finish"]
    if gap is not None:
        leader = rider_a if gap >= 0 else rider_b
        st.metric(
            "Avg gap (when both finish)",
            f"{abs(gap):.2f} pos",
            delta=f"Edge to {leader}" if abs(gap) > 0.01 else "Dead even",
            delta_color="off",
            delta_arrow="off",
        )
    else:
        st.metric("Avg gap (when both finish)", "N/A")

with k4:
    crash_decided = round(duel["a_beats_b_dnf_pct"] + duel["b_beats_a_dnf_pct"], 1)
    st.metric(
        "Decided by rival DNF",
        f"{crash_decided:.1f}%",
        delta=f"Both DNF: {duel['both_dnf_pct']:.1f}%",
        delta_color="off",
        delta_arrow="off",
    )

st.altair_chart(
    h2h_outcome_bar_chart(duel["outcome_breakdown"], rider_a, rider_b),
    width="stretch",
    alt="100 percent stacked horizontal bar showing Monte Carlo head-to-head outcome breakdown",
)

st.markdown("#### :material/bar_chart: Finishing position distribution (Monte Carlo)")
st.caption(
    f"Probability across {duel['n_sims']:,} simulated {chosen_session} races · Median classified finish: "
    f"**{rider_a}** P{duel['median_finish_a'] or 'N/A'} vs **{rider_b}** P{duel['median_finish_b'] or 'N/A'}"
)
st.altair_chart(
    h2h_position_distribution_chart(duel["position_distribution"], rider_a, rider_b),
    width="stretch",
    alt="Grouped bar chart comparing simulated finishing position probabilities between the two riders",
)

st.divider()
st.markdown("#### :material/dataset: Full telemetry, pace & risk feature comparison")

feat_table = build_feature_comparison_table(sim_df, rider_a, rider_b)
categories = ["All", *feat_table["Category"].unique().tolist()]
cat_pick = st.segmented_control(
    "Feature category filter",
    options=categories,
    default="All",
    label_visibility="collapsed",
    key="h2h_cat_filter",
)
filtered_feats = (
    feat_table
    if cat_pick == "All" or cat_pick is None
    else feat_table[feat_table["Category"] == cat_pick]
)

st.dataframe(
    filtered_feats,
    width="stretch",
    hide_index=True,
    alt="Side-by-side comparison table of all model features and simulation probabilities",
)

dl1, dl2 = st.columns(2)
slug_a = rider_a.lower().replace(" ", "_")
slug_b = rider_b.lower().replace(" ", "_")
with dl1:
    st.download_button(
        "Download H2H comparison CSV",
        data=to_csv_bytes(feat_table),
        file_name=f"h2h_{chosen_session.lower()}_{slug_a}_vs_{slug_b}.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )
with dl2:
    st.download_button(
        "Download H2H comparison Excel",
        data=to_excel_bytes(feat_table, f"H2H_{chosen_session}"),
        file_name=f"h2h_{chosen_session.lower()}_{slug_a}_vs_{slug_b}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/table_view:",
        width="stretch",
    )

st.divider()
st.markdown("#### :material/history: Historical form & circuit record")

hist_tab_1, hist_tab_2 = st.tabs(["Recent form (last 10)", f"History at {chosen_circuit}"])
hist_cols = [
    c
    for c in [
        "date",
        "session_type",
        "circuit_name",
        "constructor_name",
        "grid_position",
        "position_cleaned",
        "is_dnf",
    ]
    if c in df_ml.columns
]

with hist_tab_1:
    form_sess_filter = st.segmented_control(
        "Session filter for recent form",
        options=["All (RAC + SPR)", "RAC only", "SPR only"],
        default="All (RAC + SPR)",
        key="h2h_form_sess_filter",
    )
    sess_arg = (
        "RAC"
        if form_sess_filter == "RAC only"
        else ("SPR" if form_sess_filter == "SPR only" else None)
    )
    rec_a = get_rider_recent_history(df_ml, rider_a, session_type=sess_arg, limit=10)
    rec_b = get_rider_recent_history(df_ml, rider_b, session_type=sess_arg, limit=10)

    if rec_a.empty and rec_b.empty:
        st.info("No recent historical records found for these riders.")
    else:
        st.altair_chart(
            h2h_recent_form_chart(rec_a, rec_b, rider_a, rider_b),
            width="stretch",
            alt="Line chart comparing recent MotoGP race finishes for both riders",
        )
        hc1, hc2 = st.columns(2)
        with hc1:
            st.markdown(f"**{rider_a} — recent results**")
            st.dataframe(
                rec_a[hist_cols].sort_values("date", ascending=False)
                if not rec_a.empty
                else rec_a,
                width="stretch",
                hide_index=True,
                alt=f"Recent historical results for {rider_a}",
            )
        with hc2:
            st.markdown(f"**{rider_b} — recent results**")
            st.dataframe(
                rec_b[hist_cols].sort_values("date", ascending=False)
                if not rec_b.empty
                else rec_b,
                width="stretch",
                hide_index=True,
                alt=f"Recent historical results for {rider_b}",
            )

with hist_tab_2:
    circ_sess_filter = st.segmented_control(
        "Session filter for circuit history",
        options=["All (RAC + SPR)", "RAC only", "SPR only"],
        default="All (RAC + SPR)",
        key="h2h_circ_sess_filter",
    )
    circ_sess_arg = (
        "RAC"
        if circ_sess_filter == "RAC only"
        else ("SPR" if circ_sess_filter == "SPR only" else None)
    )
    circ_a = get_rider_circuit_history(df_ml, rider_a, chosen_circuit, session_type=circ_sess_arg)
    circ_b = get_rider_circuit_history(df_ml, rider_b, chosen_circuit, session_type=circ_sess_arg)

    if circ_a.empty and circ_b.empty:
        st.info(f"No prior starts at {chosen_circuit} in dataset.")
    else:
        st.altair_chart(
            h2h_recent_form_chart(circ_a, circ_b, rider_a, rider_b, mode="circuit"),
            width="stretch",
            alt=f"Line chart comparing historical MotoGP race finishes at {chosen_circuit} for both riders",
        )
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"**{rider_a} at {chosen_circuit}**")
            if circ_a.empty:
                st.info(f"No prior starts at {chosen_circuit} in dataset.")
            else:
                st.dataframe(
                    circ_a[hist_cols],
                    width="stretch",
                    hide_index=True,
                    alt=f"Circuit history at {chosen_circuit} for {rider_a}",
                )
        with cc2:
            st.markdown(f"**{rider_b} at {chosen_circuit}**")
            if circ_b.empty:
                st.info(f"No prior starts at {chosen_circuit} in dataset.")
            else:
                st.dataframe(
                    circ_b[hist_cols],
                    width="stretch",
                    hide_index=True,
                    alt=f"Circuit history at {chosen_circuit} for {rider_b}",
                )
