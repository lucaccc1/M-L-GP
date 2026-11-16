"""MotoGP ML Pipeline & Race Predictor — Streamlit GUI entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.state import get_artifact_inventory, get_config, init_session_state  # noqa: E402

st.set_page_config(
    page_title="M(L)GP — MotoGP Race Predictor",
    page_icon=":material/sports_motorsports:",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
config = get_config()
inventory = get_artifact_inventory(config)

pages = [
    st.Page(
        "app_pages/04_race_simulator.py",
        title="Race simulator & backtest",
        icon=":material/flag:",
        default=True,
    ),
    st.Page(
        "app_pages/01_orchestrator.py",
        title="Pipeline orchestrator",
        icon=":material/account_tree:",
    ),
    st.Page(
        "app_pages/02_data_explorer.py",
        title="Data & feature explorer",
        icon=":material/analytics:",
    ),
    st.Page(
        "app_pages/03_model_training.py",
        title="Model training & diagnostics",
        icon=":material/model_training:",
    ),
]

navigation = st.navigation(pages, position="hidden")

with st.sidebar:
    st.title(":material/sports_motorsports: M(L)GP")
    st.caption(
        "Machine learning pipeline and Monte Carlo simulator for MotoGP Sprint and Grand Prix race predictions."
    )

    st.divider()
    for page in pages:
        st.page_link(page, width="stretch")

    st.divider()
    st.markdown("#### Artifact readiness")

    ready_checks = [
        (
            "Raw API data",
            inventory["raw_results"]["exists"] and inventory["raw_sessions"]["exists"],
        ),
        ("Cleaned sessions", inventory["cleaned_sessions"]["exists"]),
        ("ML dataset", inventory["dataset_ml"]["exists"]),
        ("RAC model", inventory["model_rac"]["exists"]),
        ("SPR model", inventory["model_spr"]["exists"]),
    ]

    for label, ok in ready_checks:
        icon = ":material/check_circle:" if ok else ":material/radio_button_unchecked:"
        status_txt = "Ready" if ok else "Pending"
        st.caption(f"{icon} **{label}** — {status_txt}")

navigation.run()
