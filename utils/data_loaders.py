"""Cached data/model loaders and CSV/Excel export helpers."""

from __future__ import annotations

import importlib
import io
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import pandas as pd
import streamlit as st


def file_mtime(path: str | Path) -> float:
    p = Path(path)
    return p.stat().st_mtime if p.exists() else -1.0


def reload_module_if_changed(module: ModuleType) -> ModuleType:
    """Reload a Python module in-place if its source file mtime changed on disk."""
    mod_file = getattr(module, "__file__", None)
    if not mod_file:
        return module
    mtime = file_mtime(mod_file)
    if mtime >= 0 and getattr(module, "_LOADED_MTIME", None) != mtime:
        importlib.reload(module)
        module._LOADED_MTIME = mtime  # type: ignore[attr-defined]
    return module


@st.cache_data(ttl=3600, max_entries=16, show_spinner=False)
def load_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    """Load a CSV file into a DataFrame, invalidated automatically via mtime."""
    if mtime < 0 or not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df["date"] = parsed.dt.tz_localize(None)
    return df


@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def load_json_cached(path: str, mtime: float) -> dict[str, Any]:
    """Load a JSON diagnostics file, invalidated automatically via mtime."""
    if mtime < 0 or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource(max_entries=6, show_spinner=False)
def load_model_bundle_cached(path: str, mtime: float) -> dict[str, Any] | None:
    """Load a serialized scikit-learn joblib bundle."""
    if mtime < 0 or not Path(path).exists():
        return None
    return joblib.load(path)


def invalidate_all_caches() -> None:
    """Clear data and model caches after any pipeline mutation."""
    load_csv_cached.clear()
    load_json_cached.clear()
    load_model_bundle_cached.clear()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _strip_timezones(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    for col in df_out.columns:
        if isinstance(df_out[col].dtype, pd.DatetimeTZDtype):
            df_out[col] = df_out[col].dt.tz_convert(None)
        elif df_out[col].dtype == object:
            df_out[col] = df_out[col].apply(
                lambda v: v.replace(tzinfo=None)
                if hasattr(v, "tzinfo") and v.tzinfo is not None
                else v
            )
    return df_out


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    df_export = _strip_timezones(df)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_export.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    buffer.seek(0)
    return buffer.getvalue()
