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

from utils.state import get_artifact_inventory, get_config, init_session_state  # noqa: E402

st.set_page_config(
    page_title="M(L)GP — MotoGP Race Predictor",
    page_icon=":material/sports_motorsports:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/lucaccc1/M-L-GP",
        "Report a bug": "https://github.com/lucaccc1/M-L-GP/issues",
        "About": "**M(L)GP** — MotoGP ML Pipeline & Monte Carlo Race Predictor.\n\nRepository: https://github.com/lucaccc1/M-L-GP",
    },
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
        "app_pages/05_rider_comparison.py",
        title="Rider head-to-head",
        icon=":material/compare_arrows:",
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
.st-key-collapsed_brand_header {{
    display: none !important;
    width: 100% !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    margin-bottom: 0.75rem !important;
}}
body[data-mlgp-sidebar-collapsed="true"] .st-key-collapsed_brand_header {{
    display: flex !important;
}}
body[data-mlgp-sidebar-collapsed="false"] .st-key-collapsed_brand_header {{
    display: none !important;
}}
body:not([data-mlgp-sidebar-collapsed]):has([data-testid="stSidebar"][aria-expanded="false"]) .st-key-collapsed_brand_header,
body:not([data-mlgp-sidebar-collapsed]):has([data-testid="stExpandSidebarButton"]) .st-key-collapsed_brand_header {{
    display: flex !important;
}}
.st-key-collapsed_brand_header [data-testid="stHeadingWithActionElements"],
.st-key-collapsed_brand_header h1 {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    width: 100% !important;
    font-size: clamp(2.85rem, 5vw, 4.25rem) !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.1 !important;
    margin: 0 !important;
    padding: 0.15rem 0 0.35rem 0 !important;
}}
.st-key-collapsed_brand_header [data-testid="stHeaderActionElements"] {{
    display: none !important;
}}
.st-key-collapsed_brand_header h1 span[role="img"],
.st-key-collapsed_brand_header h1 .material-symbols-rounded {{
    font-size: 1em !important;
    line-height: 1 !important;
}}
</style>
"""

_CHROME_JS = """
<script>
(() => {
    const syncSidebarState = () => {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        let collapsed = false;
        if (sidebar) {
            const rect = sidebar.getBoundingClientRect();
            const style = window.getComputedStyle(sidebar);
            const aria = sidebar.getAttribute("aria-expanded");
            collapsed = (
                aria === "false" ||
                rect.width < 40 ||
                rect.right <= 10 ||
                style.display === "none" ||
                style.visibility === "hidden"
            );
        } else {
            collapsed = Boolean(document.querySelector('[data-testid="stExpandSidebarButton"]'));
        }
        document.body.dataset.mlgpSidebarCollapsed = collapsed ? "true" : "false";
    };

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

    const syncAll = () => {
        syncSidebarState();
        alignGithubBtn();
    };

    if (window.__mlgpObserver) window.__mlgpObserver.disconnect();
    if (window.__mlgpResizeObserver) window.__mlgpResizeObserver.disconnect();

    window.__mlgpObserver = new MutationObserver(syncAll);
    window.__mlgpObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ["aria-expanded", "style", "class", "data-collapsed"],
    });

    window.__mlgpResizeObserver = new ResizeObserver(syncAll);
    const sb = document.querySelector('[data-testid="stSidebar"]');
    if (sb) {
        window.__mlgpResizeObserver.observe(sb);
        sb.addEventListener("transitionend", syncAll);
    }
    const hdr = document.querySelector('[data-testid="stHeader"]');
    if (hdr) window.__mlgpResizeObserver.observe(hdr);
    window.addEventListener("resize", syncAll);

    syncAll();
    setTimeout(syncAll, 150);
    setTimeout(syncAll, 450);
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

with st.container(key="collapsed_brand_header"):
    st.title(":material/sports_motorsports: M(L)GP")

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
