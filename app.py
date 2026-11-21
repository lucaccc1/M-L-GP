"""MotoGP ML Pipeline & Race Predictor — Streamlit GUI entrypoint."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from urllib.parse import quote

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
get_config()

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

GITHUB_URL = "https://github.com/lucaccc1/M-L-GP"
_GITHUB_SVG_RAW = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
    '<path fill="black" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
    "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53."
    "63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 "
    "0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 "
    "2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 "
    '3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>'
    "</svg>"
)
_GITHUB_MASK_URI = f"data:image/svg+xml,{quote(_GITHUB_SVG_RAW)}"

_CHROME_CSS = f"""
<style>
.st-key-app_chrome_styles {{
    display: none !important;
}}
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"] {{
    display: none !important;
}}
[data-testid="stMainBlockContainer"],
.stMainBlockContainer {{
    padding-top: 2.5rem !important;
}}
.st-key-topbar_github_btn {{
    position: fixed !important;
    top: 0.75rem;
    right: 3.75rem;
    width: auto !important;
    min-width: 2rem !important;
    height: 2rem !important;
    z-index: 999999 !important;
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
}}
.st-key-topbar_github_btn > div {{
    width: auto !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
}}
body:has([data-testid="stAppDeployButton"]) .st-key-topbar_github_btn {{
    right: 8.75rem;
}}
.mlgp-github-btn {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border-radius: 0.5rem;
    color: inherit;
    text-decoration: none;
    opacity: 0.85;
    transition: background-color 150ms ease, opacity 150ms ease;
}}
.mlgp-github-btn:hover {{
    background-color: rgba(128, 128, 128, 0.16);
    opacity: 1;
}}
.mlgp-github-icon {{
    display: inline-block;
    width: 1.2rem;
    height: 1.2rem;
    background-color: currentColor;
    -webkit-mask-image: url("{_GITHUB_MASK_URI}");
    mask-image: url("{_GITHUB_MASK_URI}");
    -webkit-mask-repeat: no-repeat;
    mask-repeat: no-repeat;
    -webkit-mask-position: center;
    mask-position: center;
    -webkit-mask-size: contain;
    mask-size: contain;
}}
.st-key-top_header_shell,
.st-key-top_header_shell[data-testid="stVerticalBlock"],
.st-key-top_header_shell > [data-testid="stVerticalBlock"] {{
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    align-items: flex-end !important;
    justify-content: flex-start !important;
    gap: 0.3rem !important;
    width: 100% !important;
    margin-bottom: 0.75rem !important;
}}
.st-key-top_header_shell > .stElementContainer,
.st-key-top_header_shell > [data-testid="stLayoutWrapper"],
.st-key-top_header_shell > [data-testid="stVerticalBlock"] > .stElementContainer,
.st-key-top_header_shell > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"],
.st-key-top_brand_wrap,
[class*="st-key-nav_pill_"] {{
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: 0 0 auto !important;
}}
[class*="st-key-nav_pill_"] {{
    padding-bottom: 0.14rem !important;
}}
.st-key-top_brand_wrap {{
    margin-right: 0.85rem !important;
}}
.st-key-top_brand_wrap [data-testid="stHeadingWithActionElements"],
.st-key-top_brand_wrap h1 {{
    margin: 0 !important;
    padding: 0 !important;
    font-size: clamp(2.35rem, 3.4vw, 3.0rem) !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em !important;
    line-height: 1.05 !important;
    white-space: nowrap !important;
}}
.st-key-top_brand_wrap [data-testid="stHeaderActionElements"] {{
    display: none !important;
}}
.st-key-top_brand_wrap h1 span[role="img"],
.st-key-top_brand_wrap h1 .material-symbols-rounded {{
    font-size: 1em !important;
    line-height: 1 !important;
}}
[class*="st-key-nav_pill_"] > div,
[class*="st-key-nav_pill_"] [data-testid="stPageLink"] {{
    width: auto !important;
    min-width: 0 !important;
}}
[class*="st-key-nav_pill_"] a[data-testid="stPageLink-NavLink"] {{
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
}}
[class*="st-key-nav_pill_"][class$="_idle"] a[data-testid="stPageLink-NavLink"] {{
    opacity: 0.72 !important;
    background-color: transparent !important;
}}
[class*="st-key-nav_pill_"][class$="_idle"] a[data-testid="stPageLink-NavLink"]:hover {{
    opacity: 1 !important;
    background-color: rgba(148, 163, 184, 0.1) !important;
}}
[class*="st-key-nav_pill_"][class$="_active"] a[data-testid="stPageLink-NavLink"] {{
    opacity: 1 !important;
    font-weight: 600 !important;
    background-color: rgba(148, 163, 184, 0.14) !important;
    border-color: rgba(148, 163, 184, 0.3) !important;
}}
</style>
"""

_CHROME_JS = """
<script>
(() => {
    const alignGithubBtn = () => {
        const wrap = document.querySelector('.st-key-topbar_github_btn');
        if (!wrap) return;
        const anchor =
            document.querySelector('[data-testid="stAppDeployButton"]') ||
            document.querySelector('[data-testid="stMainMenu"]') ||
            document.querySelector('[data-testid="stToolbar"]');
        if (!anchor) return;
        const r = anchor.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) {
            const btnSize = 32;
            const topPx = Math.round(r.top + (r.height - btnSize) / 2);
            const rightPx = Math.round(window.innerWidth - r.left + 8);
            wrap.style.setProperty("top", `${topPx}px`, "important");
            wrap.style.setProperty("right", `${rightPx}px`, "important");
        }
    };

    if (window.__mlgpObserver) window.__mlgpObserver.disconnect();
    if (window.__mlgpResizeObserver) window.__mlgpResizeObserver.disconnect();

    window.__mlgpObserver = new MutationObserver(alignGithubBtn);
    window.__mlgpObserver.observe(document.body, {
        childList: true,
        subtree: true,
    });

    window.__mlgpResizeObserver = new ResizeObserver(alignGithubBtn);
    const hdr = document.querySelector('[data-testid="stHeader"]');
    if (hdr) window.__mlgpResizeObserver.observe(hdr);
    window.addEventListener("resize", alignGithubBtn);

    alignGithubBtn();
    setTimeout(alignGithubBtn, 150);
    setTimeout(alignGithubBtn, 450);
})();
</script>
"""

with st.container(key="app_chrome_styles"):
    if "unsafe_allow_javascript" in inspect.signature(st.html).parameters:
        st.html(_CHROME_CSS + _CHROME_JS, unsafe_allow_javascript=True)
    else:
        st.html(_CHROME_CSS)

with st.container(key="topbar_github_btn"):
    st.html(
        f'<a class="mlgp-github-btn" href="{GITHUB_URL}" target="_blank" '
        f'rel="noopener noreferrer" title="GitHub: lucaccc1/M-L-GP" '
        f'aria-label="Open GitHub repository"><span class="mlgp-github-icon"></span></a>'
    )

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

active_page.run()
