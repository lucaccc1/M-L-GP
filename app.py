"""MotoGP ML Pipeline & Race Predictor — Streamlit GUI entrypoint."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if Path.cwd().resolve() != ROOT_DIR:
    os.chdir(ROOT_DIR)

from utils.bootstrap import ensure_local_artifacts_bootstrapped  # noqa: E402
from utils.state import get_config, init_session_state  # noqa: E402

st.set_page_config(
    page_title="M(L)GP — MotoGP Race Predictor",
    page_icon=":material/sports_motorsports:",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": "https://github.com/lucaccc1/M-L-GP",
        "Report a bug": "https://github.com/lucaccc1/M-L-GP/issues",
        "About": "**M(L)GP** — MotoGP ML Pipeline & Monte Carlo Race Predictor.\n\nRepository: https://github.com/lucaccc1/M-L-GP",
    },
)

init_session_state()
config = get_config()

pages = [
    st.Page(
        "app_pages/04_race_simulator.py",
        title="Race simulator",
        icon=":material/flag:",
        default=True,
    ),
    st.Page(
        "app_pages/05_rider_comparison.py",
        title="Rider H2H",
        icon=":material/compare_arrows:",
    ),
    st.Page(
        "app_pages/03_model_training.py",
        title="Model training",
        icon=":material/model_training:",
    ),
    st.Page(
        "app_pages/02_data_explorer.py",
        title="Data explorer",
        icon=":material/analytics:",
    ),
    st.Page(
        "app_pages/01_orchestrator.py",
        title="Orchestrator",
        icon=":material/account_tree:",
    ),
]

active_page = st.navigation(pages, position="hidden")

_CHROME_CSS = """
<style>
.st-key-app_chrome_styles {
    display: none !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"] {
    display: none !important;
}
[data-testid="stMainBlockContainer"],
.stMainBlockContainer {
    padding-top: 4.5rem !important;
}
.st-key-top_header_shell,
.st-key-top_header_shell[data-testid="stVerticalBlock"],
.st-key-top_header_shell > [data-testid="stVerticalBlock"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    align-items: flex-end !important;
    justify-content: flex-start !important;
    gap: 0.3rem !important;
    width: 100% !important;
    margin-bottom: 0.75rem !important;
}
.st-key-top_header_shell > .stElementContainer,
.st-key-top_header_shell > [data-testid="stLayoutWrapper"],
.st-key-top_header_shell > [data-testid="stVerticalBlock"] > .stElementContainer,
.st-key-top_header_shell > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"],
.st-key-top_brand_wrap,
[class*="st-key-nav_pill_"] {
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 0 0 auto !important;
}
[class*="st-key-nav_pill_"] {
    padding-bottom: 0.14rem !important;
}
.st-key-top_brand_wrap {
    margin-right: 0.85rem !important;
}
.st-key-top_brand_wrap [data-testid="stHeadingWithActionElements"],
.st-key-top_brand_wrap h1 {
    margin: 0 !important;
    padding: 0 !important;
    font-size: clamp(2.35rem, 3.4vw, 3.0rem) !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em !important;
    line-height: 1.05 !important;
    white-space: nowrap !important;
}
.st-key-top_brand_wrap [data-testid="stHeaderActionElements"] {
    display: none !important;
}
.st-key-top_brand_wrap h1 span[role="img"],
.st-key-top_brand_wrap h1 .material-symbols-rounded {
    font-size: 1em !important;
    line-height: 1 !important;
}
[class*="st-key-nav_pill_"] > div,
[class*="st-key-nav_pill_"] [data-testid="stPageLink"] {
    width: auto !important;
    min-width: 0 !important;
}
[class*="st-key-nav_pill_"] a[data-testid="stPageLink-NavLink"] {
    width: auto !important;
    min-height: 0 !important;
    height: auto !important;
    border-radius: 999px !important;
    padding: 0.22rem 0.62rem !important;
    margin: 0 !important;
    gap: 0.35rem !important;
    justify-content: center !important;
    white-space: nowrap !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    line-height: 1.25 !important;
    border: 1px solid transparent !important;
    transition: background-color 150ms ease, border-color 150ms ease, opacity 150ms ease !important;
}
[class*="st-key-nav_pill_"][class$="_idle"] a[data-testid="stPageLink-NavLink"] {
    opacity: 0.72 !important;
    background-color: transparent !important;
}
[class*="st-key-nav_pill_"][class$="_idle"] a[data-testid="stPageLink-NavLink"]:hover {
    opacity: 1 !important;
    background-color: rgba(148, 163, 184, 0.1) !important;
}
[class*="st-key-nav_pill_"][class$="_active"] a[data-testid="stPageLink-NavLink"] {
    opacity: 1 !important;
    font-weight: 600 !important;
    background-color: rgba(148, 163, 184, 0.14) !important;
    border-color: rgba(148, 163, 184, 0.3) !important;
}
</style>
"""

with st.container(key="app_chrome_styles"):
    st.html(_CHROME_CSS)

_container_params = inspect.signature(st.container).parameters
_page_link_params = inspect.signature(st.page_link).parameters

_shell_kwargs: dict[str, object] = {"key": "top_header_shell"}
if "horizontal" in _container_params:
    _shell_kwargs["horizontal"] = True
    _shell_kwargs["horizontal_alignment"] = "left"
    _shell_kwargs["vertical_alignment"] = "bottom"

_item_container_kwargs: dict[str, object] = {}
if "width" in _container_params:
    _item_container_kwargs["width"] = "content"

_page_link_kwargs: dict[str, object] = {}
if "width" in _page_link_params:
    _page_link_kwargs["width"] = "content"

with st.container(**_shell_kwargs):
    with st.container(key="top_brand_wrap", **_item_container_kwargs):
        st.title(":material/sports_motorsports: M(L)GP")

    for idx, page in enumerate(pages):
        is_active = page.title == active_page.title
        state_suffix = "active" if is_active else "idle"
        with st.container(key=f"nav_pill_{idx}_{state_suffix}", **_item_container_kwargs):
            st.page_link(page, **_page_link_kwargs)

ensure_local_artifacts_bootstrapped(config)
active_page.run()
