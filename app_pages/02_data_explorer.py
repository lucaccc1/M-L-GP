"""Page 2: Interactive Data & Feature Engineering Explorer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.charts import (
    constructor_season_trend_chart,
    grid_vs_finish_chart,
    rider_pace_vs_dnf_chart,
)
from utils.data_loaders import (
    file_mtime,
    load_csv_cached,
    to_csv_bytes,
    to_excel_bytes,
)
from utils.bootstrap import ensure_local_artifacts_bootstrapped
from utils.state import get_config, init_session_state

init_session_state()
config = get_config()
ensure_local_artifacts_bootstrapped(config)

st.title(":material/analytics: Data & feature explorer")
st.caption(
    "Explore historical MotoGP sessions, engineered rolling features, and telemetry patterns."
)

st.markdown("#### Dataset source")
dataset_choice = st.segmented_control(
    "Dataset source",
    options=[
        "ML dataset (dataset_ml.csv)",
        "Cleaned sessions (cleaned_sessions.csv)",
        "Raw events (events.csv)",
    ],
    default="ML dataset (dataset_ml.csv)",
    label_visibility="collapsed",
    key="explorer_dataset_choice",
)

if dataset_choice == "ML dataset (dataset_ml.csv)":
    target_path = config.dataset_ml_csv
elif dataset_choice == "Cleaned sessions (cleaned_sessions.csv)":
    target_path = config.cleaned_sessions_csv
else:
    target_path = config.events_csv

mtime = file_mtime(target_path)
df = load_csv_cached(str(target_path), mtime)

if df.empty:
    st.warning(
        f"Artifact `{target_path}` is not available yet. Run the upstream steps in **Pipeline orchestrator** first.",
        title="Dataset not found",
        icon=":material/warning:",
    )
    st.stop()

# --- Raw Events View ---
if dataset_choice == "Raw events (events.csv)":
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total events", f"{len(df):,}")
    with c2:
        finished_cnt = int((df["status"] == "FINISHED").sum()) if "status" in df.columns else 0
        st.metric("Finished events", f"{finished_cnt:,}")
    with c3:
        upcoming_cnt = len(df) - finished_cnt
        st.metric("Upcoming / open events", f"{upcoming_cnt:,}")

    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        alt="Raw MotoGP events metadata table",
    )
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download CSV",
            data=to_csv_bytes(df),
            file_name="motogp_events.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )
    with dl2:
        st.download_button(
            "Download Excel",
            data=to_excel_bytes(df, "Events"),
            file_name="motogp_events.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            icon=":material/table_view:",
            width="stretch",
        )
    st.stop()

# --- Interactive Filters (applied outside @st.cache_data) ---
with st.container(border=True):
    st.markdown("#### :material/filter_alt: Interactive filters")
    f1, f2, f3, f4 = st.columns([1.2, 1, 1.4, 1.4])

    seasons_avail = sorted(df["season_year"].dropna().astype(int).unique().tolist())
    with f1:
        if len(seasons_avail) >= 2:
            year_range = st.slider(
                "Season range",
                min_value=min(seasons_avail),
                max_value=max(seasons_avail),
                value=(min(seasons_avail), max(seasons_avail)),
                key="exp_year_range",
            )
        else:
            only_yr = seasons_avail[0] if seasons_avail else 2025
            year_range = (only_yr, only_yr)
            st.caption(f"Season available: `{only_yr}`")

    with f2:
        sess_opts = sorted(df["session_type"].dropna().unique().tolist())
        selected_sessions = st.pills(
            "Session type",
            options=sess_opts,
            default=sess_opts,
            selection_mode="multi",
            key="exp_sessions",
        )

    with f3:
        const_opts = sorted(df["constructor_name"].dropna().unique().tolist())
        selected_consts = st.multiselect(
            "Constructors",
            options=const_opts,
            default=const_opts,
            key="exp_consts",
        )

    with f4:
        rider_query = st.text_input(
            "Filter rider name",
            placeholder="e.g. Bagnaia, Marquez, Martin...",
            key="exp_rider_query",
        )

mask = (df["season_year"] >= year_range[0]) & (df["season_year"] <= year_range[1])
if selected_sessions:
    mask &= df["session_type"].isin(selected_sessions)
if selected_consts:
    mask &= df["constructor_name"].isin(selected_consts)
if rider_query.strip():
    mask &= df["rider_name"].astype(str).str.contains(rider_query.strip(), case=False, na=False)

filtered = df[mask].copy()

if filtered.empty:
    st.info(
        "No rows match the current filter combination. Widen the season range or clear rider filters.",
        title="Empty filter result",
        icon=":material/info:",
    )
    st.stop()

# --- KPI Row ---
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Filtered entries", f"{len(filtered):,}")
with k2:
    st.metric("Unique riders", f"{filtered['rider_name'].nunique():,}")
with k3:
    n_circuits = (
        int(filtered["circuit_name"].nunique())
        if "circuit_name" in filtered.columns
        else int(filtered["event_id"].nunique())
    )
    st.metric("Circuits / events", f"{n_circuits:,}")
with k4:
    dnf_rate = float(filtered["is_dnf"].mean() * 100) if len(filtered) else 0.0
    st.metric("DNF rate", f"{dnf_rate:.1f}%")

st.divider()

# --- Chart View Switcher ---
st.markdown("#### Visual exploration")
chart_view = st.segmented_control(
    "Visual exploration",
    options=[
        "Constructor trends",
        "Rider pace vs DNF",
        "Grid vs finish heatmap",
    ],
    default="Constructor trends",
    label_visibility="collapsed",
    key="explorer_chart_view",
)

if chart_view == "Constructor trends":
    st.altair_chart(
        constructor_season_trend_chart(filtered),
        width="stretch",
        alt="Line chart of average finishing position per constructor across seasons",
    )
elif chart_view == "Rider pace vs DNF":
    min_starts = st.slider(
        "Minimum race starts per rider",
        min_value=1,
        max_value=50,
        value=5,
        key="exp_min_starts",
    )
    st.altair_chart(
        rider_pace_vs_dnf_chart(filtered, min_starts=min_starts),
        width="stretch",
        alt="Scatter plot of rider average finishing position versus DNF rate",
    )
else:
    st.altair_chart(
        grid_vs_finish_chart(filtered),
        width="stretch",
        alt="Heatmap of starting grid position versus race finishing position",
    )

st.divider()

# --- Interactive Data Table + Export ---
st.subheader(":material/table_chart: Filtered records")
display_cols = [
    c
    for c in [
        "season_year",
        "date",
        "circuit_name",
        "session_type",
        "rider_name",
        "constructor_name",
        "grid_position",
        "position_cleaned",
        "is_dnf",
        "is_wet",
        "grid_delta",
        "rider_last3_avg",
        "rider_last5_avg",
        "sprint_pos_same_weekend",
        "dnf_ratio_last5",
        "rider_circuit_avg",
        "constructor_last3_avg",
    ]
    if c in filtered.columns
]

show_all_cols = st.toggle("Show all dataset columns", value=False, key="exp_show_all_cols")
table_df = filtered if show_all_cols else filtered[display_cols]

st.dataframe(
    table_df.head(2000),
    width="stretch",
    hide_index=True,
    column_config={
        "season_year": st.column_config.NumberColumn("Season", format="%d"),
        "grid_position": st.column_config.NumberColumn("Grid", format="%d"),
        "position_cleaned": st.column_config.NumberColumn("Finish", format="%d"),
        "is_dnf": st.column_config.CheckboxColumn("DNF"),
        "rider_last3_avg": st.column_config.NumberColumn("Last 3 avg", format="%.2f"),
        "dnf_ratio_last5": st.column_config.ProgressColumn(
            "DNF ratio (5)", min_value=0.0, max_value=1.0, format="%.2f"
        ),
    },
    alt="Interactive MotoGP session and engineered features table",
)
if len(table_df) > 2000:
    st.caption(f"Showing first 2,000 of {len(table_df):,} filtered rows in preview. Exports include all filtered rows.")

exp1, exp2 = st.columns(2)
with exp1:
    st.download_button(
        "Download filtered CSV",
        data=to_csv_bytes(table_df),
        file_name="motogp_filtered_dataset.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )
with exp2:
    st.download_button(
        "Download filtered Excel",
        data=to_excel_bytes(table_df, "MotoGP_Data"),
        file_name="motogp_filtered_dataset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/table_view:",
        width="stretch",
    )
