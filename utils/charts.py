"""Reusable Altair interactive charts for MotoGP telemetry, ML diagnostics, and simulations."""

from __future__ import annotations

import hashlib
import re

import altair as alt
import pandas as pd

CONSTRUCTOR_BRAND_COLORS: list[tuple[str, str]] = [
    ("ducati", "#E10600"),     # Rosso Ducati
    ("aprilia", "#9333EA"),    # Viola Aprilia
    ("ktm", "#FF7A00"),        # Arancione KTM
    ("yamaha", "#2563EB"),     # Blu Yamaha
    ("honda", "#E23D10"),      # Arancione scuro Honda (più scuro di KTM)
    ("suzuki", "#06B6D4"),     # Azzurro Suzuki Ecstar
    ("gasgas", "#BE123C"),     # Cremisi GASGAS
    ("kawasaki", "#16A34A"),   # Verde Kawasaki
    ("bmw", "#0EA5E9"),        # Celeste BMW Motorrad
    ("kalex", "#EAB308"),      # Oro/giallo Kalex
    ("suter", "#14B8A6"),      # Verde acqua Suter
    ("art", "#64748B"),        # Grigio ardesia ART
    ("ftr", "#84CC16"),        # Lime FTR
    ("cagiva", "#9F1239"),     # Rosso scuro Cagiva
    ("mv agusta", "#B91C1C"),  # Rosso classico MV Agusta
    ("bimota", "#10B981"),     # Smeraldo Bimota
    ("ilmor", "#6366F1"),      # Indaco Ilmor
]

_FALLBACK_CONSTRUCTOR_PALETTE = [
    "#0D9488",
    "#EC4899",
    "#65A30D",
    "#0284C7",
    "#A855F7",
    "#CA8A04",
    "#059669",
    "#475569",
]


def constructor_color_for_name(name: object) -> str:
    """Return a consistent MotoGP-coherent hex color for a constructor name."""
    cleaned = str(name or "").strip().lower()
    for brand, hex_color in CONSTRUCTOR_BRAND_COLORS:
        if re.search(rf"\b{re.escape(brand)}\b", cleaned):
            return hex_color
    digest = hashlib.md5(cleaned.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(_FALLBACK_CONSTRUCTOR_PALETTE)
    return _FALLBACK_CONSTRUCTOR_PALETTE[idx]


def constructor_color_scale(constructors: pd.Series) -> alt.Scale:
    """Build an explicit Altair domain/range Scale mapping each constructor to its color."""
    unique_names = sorted({str(c) for c in constructors.dropna().tolist()})
    if not unique_names:
        return alt.Scale(range=_FALLBACK_CONSTRUCTOR_PALETTE)
    return alt.Scale(
        domain=unique_names,
        range=[constructor_color_for_name(name) for name in unique_names],
    )


def constructor_season_trend_chart(df: pd.DataFrame) -> alt.Chart:
    """Line chart showing average finishing position by constructor across seasons."""
    agg = (
        df.groupby(["season_year", "constructor_name"], as_index=False)
        .agg(
            avg_position=("position_cleaned", "mean"),
            entries=("position_cleaned", "count"),
        )
        .sort_values("season_year")
    )
    agg["avg_position"] = agg["avg_position"].round(2)

    selection = alt.selection_point(fields=["constructor_name"], bind="legend")
    chart = (
        alt.Chart(agg)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("season_year:O", title="Season"),
            y=alt.Y(
                "avg_position:Q",
                title="Average finishing position (lower is better)",
                scale=alt.Scale(reverse=True, zero=False),
            ),
            color=alt.Color(
                "constructor_name:N",
                title="Constructor",
                scale=constructor_color_scale(agg["constructor_name"]),
            ),
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.2)),
            tooltip=[
                alt.Tooltip("season_year:O", title="Season"),
                alt.Tooltip("constructor_name:N", title="Constructor"),
                alt.Tooltip("avg_position:Q", title="Avg position", format=".2f"),
                alt.Tooltip("entries:Q", title="Race entries"),
            ],
        )
        .add_params(selection)
        .properties(height=360)
        .interactive()
    )
    return chart


def rider_pace_vs_dnf_chart(df: pd.DataFrame, min_starts: int = 5) -> alt.Chart:
    """Scatter chart comparing rider average finishing position vs DNF rate."""
    agg = (
        df.groupby(["rider_name", "constructor_name"], as_index=False)
        .agg(
            starts=("position_cleaned", "count"),
            avg_pos=("position_cleaned", "mean"),
            dnf_rate=("is_dnf", "mean"),
            wins=("position_cleaned", lambda s: int((s == 1).sum())),
        )
    )
    agg = agg[agg["starts"] >= min_starts].copy()
    agg["dnf_pct"] = (agg["dnf_rate"] * 100).round(1)
    agg["avg_pos"] = agg["avg_pos"].round(2)

    chart = (
        alt.Chart(agg)
        .mark_circle(opacity=0.85, stroke="#FFFFFF", strokeWidth=1)
        .encode(
            x=alt.X(
                "avg_pos:Q",
                title="Average finishing position",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y("dnf_pct:Q", title="DNF rate (%)", scale=alt.Scale(zero=True)),
            size=alt.Size("starts:Q", title="Starts", scale=alt.Scale(range=[80, 500])),
            color=alt.Color(
                "constructor_name:N",
                title="Constructor",
                scale=constructor_color_scale(agg["constructor_name"]),
            ),
            tooltip=[
                alt.Tooltip("rider_name:N", title="Rider"),
                alt.Tooltip("constructor_name:N", title="Constructor"),
                alt.Tooltip("starts:Q", title="Starts"),
                alt.Tooltip("wins:Q", title="Wins"),
                alt.Tooltip("avg_pos:Q", title="Avg position", format=".2f"),
                alt.Tooltip("dnf_pct:Q", title="DNF rate (%)", format=".1f"),
            ],
        )
        .properties(height=360)
        .interactive()
    )
    return chart


def grid_vs_finish_chart(df: pd.DataFrame) -> alt.Chart:
    """Heatmap of starting grid position vs final finishing position."""
    fin = df[df["is_dnf"] == 0].copy()
    agg = (
        fin.groupby(["grid_position", "position_cleaned"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    chart = (
        alt.Chart(agg)
        .mark_rect()
        .encode(
            x=alt.X("grid_position:O", title="Starting grid position"),
            y=alt.Y(
                "position_cleaned:O",
                title="Finishing position",
                sort="descending",
            ),
            color=alt.Color(
                "count:Q",
                title="Occurrences",
                scale=alt.Scale(scheme="reds"),
            ),
            tooltip=[
                alt.Tooltip("grid_position:O", title="Grid"),
                alt.Tooltip("position_cleaned:O", title="Finish"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(height=360)
    )
    return chart


def model_comparison_chart(metrics_report: dict) -> alt.Chart:
    """Grouped bar chart comparing MAE and DNF AUC across sessions and splits."""
    rows = []
    for session_type, sdata in metrics_report.get("sessions", {}).items():
        for split_key, split_label in [
            ("val_metrics", "Validation"),
            ("test_metrics", "Test"),
        ]:
            m = sdata.get(split_key, {})
            if m.get("mae") is not None:
                rows.append(
                    {
                        "session": session_type,
                        "split": split_label,
                        "MAE": m["mae"],
                        "DNF_AUC": m["dnf_auc"] if m.get("dnf_auc") is not None else 0.0,
                        "Podium_Hits": m["podium_hits"]
                        if m.get("podium_hits") is not None
                        else 0.0,
                    }
                )
    df_m = pd.DataFrame(rows)
    if df_m.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

    melted = df_m.melt(
        id_vars=["session", "split"],
        value_vars=["MAE", "DNF_AUC", "Podium_Hits"],
        var_name="metric",
        value_name="value",
    )
    chart = (
        alt.Chart(melted)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("split:N", title="Evaluation split"),
            y=alt.Y("value:Q", title="Metric value"),
            color=alt.Color(
                "session:N",
                title="Session",
                scale=alt.Scale(range=["#D90429", "#1D4ED8"]),
            ),
            xOffset="session:N",
            column=alt.Column("metric:N", title=None),
            tooltip=[
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("session:N", title="Session"),
                alt.Tooltip("split:N", title="Split"),
                alt.Tooltip("value:Q", title="Value", format=".3f"),
            ],
        )
        .properties(height=260, width=180)
    )
    return chart


def pred_vs_actual_chart(predictions: list[dict]) -> alt.Chart:
    """Scatter plot of predicted vs actual finishing position with ideal diagonal."""
    df_p = pd.DataFrame(predictions)
    if df_p.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_point()

    fin = df_p[df_p["is_dnf"] == 0].copy()
    fin["pred_pos"] = fin["pred_pos"].round(2)

    points = (
        alt.Chart(fin)
        .mark_circle(size=55, opacity=0.65)
        .encode(
            x=alt.X(
                "position_cleaned:Q",
                title="Actual finishing position",
                scale=alt.Scale(domain=[1, 22]),
            ),
            y=alt.Y(
                "pred_pos:Q",
                title="Predicted position",
                scale=alt.Scale(domain=[1, 22]),
            ),
            color=alt.Color(
                "constructor_name:N",
                title="Constructor",
                scale=constructor_color_scale(fin["constructor_name"]),
            ),
            tooltip=[
                alt.Tooltip("rider_name:N", title="Rider"),
                alt.Tooltip("constructor_name:N", title="Constructor"),
                alt.Tooltip("grid_position:Q", title="Grid"),
                alt.Tooltip("position_cleaned:Q", title="Actual finish"),
                alt.Tooltip("pred_pos:Q", title="Predicted", format=".2f"),
            ],
        )
    )
    diag = (
        alt.Chart(pd.DataFrame({"x": [1, 22], "y": [1, 22]}))
        .mark_line(strokeDash=[5, 4], color="#6B7280", strokeWidth=1.5)
        .encode(x="x:Q", y="y:Q")
    )
    return (points + diag).properties(height=340).interactive()


def feature_importance_chart(
    importances: list[dict], target: str = "reg_importance"
) -> alt.Chart:
    """Horizontal bar chart of permutation feature importances."""
    df_i = pd.DataFrame(importances)
    if df_i.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

    title = (
        "Regressor importance (MAE impact)"
        if target == "reg_importance"
        else "DNF classifier importance (AUC impact)"
    )
    color_hex = "#D90429" if target == "reg_importance" else "#1D4ED8"

    chart = (
        alt.Chart(df_i)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color=color_hex)
        .encode(
            x=alt.X(f"{target}:Q", title=title),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip(f"{target}:Q", title="Importance", format=".4f"),
            ],
        )
        .properties(height=360)
    )
    return chart


def simulation_odds_chart(sim_df: pd.DataFrame, top_n: int = 15) -> alt.Chart:
    """Grouped horizontal bar chart of Monte Carlo Win / Podium / Top-10 probabilities."""
    sub = sim_df.head(top_n).copy()
    melted = sub.melt(
        id_vars=["rider_name", "constructor_name", "grid_position", "pred_pos"],
        value_vars=["win_pct", "podium_pct", "top10_pct"],
        var_name="outcome",
        value_name="probability",
    )
    label_map = {
        "win_pct": "Win %",
        "podium_pct": "Podium %",
        "top10_pct": "Top 10 %",
    }
    melted["outcome_label"] = melted["outcome"].map(label_map)
    rider_order = sub["rider_name"].tolist()

    chart = (
        alt.Chart(melted)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("rider_name:N", sort=rider_order, title=None),
            x=alt.X("probability:Q", title="Probability (%)", scale=alt.Scale(domain=[0, 100])),
            yOffset="outcome_label:N",
            color=alt.Color(
                "outcome_label:N",
                title="Outcome",
                scale=alt.Scale(
                    domain=["Win %", "Podium %", "Top 10 %"],
                    range=["#D90429", "#EA580C", "#1D4ED8"],
                ),
            ),
            tooltip=[
                alt.Tooltip("rider_name:N", title="Rider"),
                alt.Tooltip("constructor_name:N", title="Constructor"),
                alt.Tooltip("grid_position:Q", title="Grid"),
                alt.Tooltip("outcome_label:N", title="Outcome"),
                alt.Tooltip("probability:Q", title="Probability (%)", format=".2f"),
            ],
        )
        .properties(height=max(320, len(sub) * 28))
    )
    return chart


def pace_vs_crash_risk_chart(
    sim_df: pd.DataFrame, theme_type: str | None = None
) -> alt.Chart:
    """Scatter chart plotting expected finishing position against DNF risk."""
    if theme_type is None:
        try:
            import streamlit as st

            theme_type = getattr(getattr(st.context, "theme", None), "type", "dark")
        except Exception:
            theme_type = "dark"

    is_dark = str(theme_type).lower() != "light"
    label_color = "#F3F4F6" if is_dark else "#0F172A"
    point_stroke = "#0E1117" if is_dark else "#FFFFFF"

    df_c = sim_df.copy()
    df_c["dnf_pct"] = (df_c["dnf_prob"] * 100).round(2)
    df_c["pred_pos_r"] = df_c["pred_pos"].round(2)

    points = (
        alt.Chart(df_c)
        .mark_circle(opacity=0.88, stroke=point_stroke, strokeWidth=1.2)
        .encode(
            x=alt.X(
                "pred_pos_r:Q",
                title="Expected finishing position (lower is faster)",
                scale=alt.Scale(domain=[1, 22]),
            ),
            y=alt.Y("dnf_pct:Q", title="DNF crash risk (%)", scale=alt.Scale(zero=True)),
            size=alt.Size(
                "podium_pct:Q",
                title="Podium %",
                scale=alt.Scale(range=[70, 520]),
            ),
            color=alt.Color(
                "constructor_name:N",
                title="Constructor",
                scale=constructor_color_scale(df_c["constructor_name"]),
            ),
            tooltip=[
                alt.Tooltip("rider_name:N", title="Rider"),
                alt.Tooltip("constructor_name:N", title="Constructor"),
                alt.Tooltip("grid_position:Q", title="Grid"),
                alt.Tooltip("pred_pos_r:Q", title="Expected pos", format=".2f"),
                alt.Tooltip("dnf_pct:Q", title="DNF risk (%)", format=".1f"),
                alt.Tooltip("win_pct:Q", title="Win %", format=".2f"),
                alt.Tooltip("podium_pct:Q", title="Podium %", format=".2f"),
            ],
        )
    )
    labels = (
        alt.Chart(df_c.head(10))
        .mark_text(
            align="left",
            dx=8,
            dy=-6,
            fontSize=11,
            fontWeight="bold",
            color=label_color,
        )
        .encode(
            x="pred_pos_r:Q",
            y="dnf_pct:Q",
            text="rider_name:N",
        )
    )
    return (points + labels).properties(height=380).interactive()


def backtest_comparison_chart(sim_df: pd.DataFrame) -> alt.Chart:
    """Compare simulated expected position vs actual race outcome in backtest mode."""
    df_b = sim_df.copy()
    df_b["pred_pos_r"] = df_b["pred_pos"].round(2)
    if "is_dnf" in df_b.columns:
        df_b["actual_status"] = df_b["is_dnf"].map({0: "Finished", 1: "DNF"}).fillna("Finished")
    else:
        df_b["actual_status"] = df_b["position_cleaned"].apply(
            lambda pos: "DNF" if pos >= 25 else "Finished"
        )

    points = (
        alt.Chart(df_b)
        .mark_circle(size=120, opacity=0.9)
        .encode(
            x=alt.X(
                "position_cleaned:Q",
                title="Actual finishing position",
                scale=alt.Scale(domain=[1, 25]),
            ),
            y=alt.Y(
                "pred_pos_r:Q",
                title="Predicted position",
                scale=alt.Scale(domain=[1, 22]),
            ),
            color=alt.Color(
                "actual_status:N",
                title="Race status",
                scale=alt.Scale(domain=["Finished", "DNF"], range=["#1D4ED8", "#D90429"]),
            ),
            tooltip=[
                alt.Tooltip("rider_name:N", title="Rider"),
                alt.Tooltip("grid_position:Q", title="Grid"),
                alt.Tooltip("position_cleaned:Q", title="Actual finish"),
                alt.Tooltip("pred_pos_r:Q", title="Predicted pos", format=".2f"),
                alt.Tooltip("podium_pct:Q", title="Podium %", format=".2f"),
                alt.Tooltip("actual_status:N", title="Status"),
            ],
        )
    )
    diag = (
        alt.Chart(pd.DataFrame({"x": [1, 22], "y": [1, 22]}))
        .mark_line(strokeDash=[5, 4], color="#6B7280", strokeWidth=1.5)
        .encode(x="x:Q", y="y:Q")
    )
    return (points + diag).properties(height=340).interactive()
