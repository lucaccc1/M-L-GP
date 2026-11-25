"""Shared Streamlit UI controls for the Upcoming GP simulation header."""

from __future__ import annotations

from typing import Any

import streamlit as st

from utils.config import AppConfig


def render_upcoming_gp_controls(
    weekend: dict[str, Any],
    config: AppConfig,
    rac_bundle: dict[str, Any] | None,
    spr_bundle: dict[str, Any] | None,
    key_prefix: str,
) -> tuple[str, str, str, int, float]:
    """Render the shared Upcoming GP header and simulation controls container."""
    chosen_circuit = str(weekend["circuit_name"])
    default_weather = "Wet / Rain" if weekend["is_wet"] == 1 else "Dry"

    with st.container(border=True):
        st.markdown(
            f"#### :material/radar: Upcoming GP: **{chosen_circuit}** ({weekend['race_date']}) · `{weekend['state']}`"
        )
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            sess_options = [
                s
                for s in weekend["sessions_to_predict"]
                if (s == "RAC" and rac_bundle is not None)
                or (s == "SPR" and spr_bundle is not None)
            ]
            if not sess_options:
                sess_options = ["RAC"] if rac_bundle is not None else ["SPR"]
            chosen_session = st.segmented_control(
                "Session",
                options=sess_options,
                default=sess_options[-1],
                key=f"{key_prefix}_session_select",
            )
            if chosen_session is None:
                chosen_session = sess_options[-1]

        with c2:
            weather_opt = st.segmented_control(
                "Conditions",
                options=["Dry", "Wet / Rain"],
                default=default_weather,
                key=f"{key_prefix}_weather_select",
            )
            if weather_opt is None:
                weather_opt = default_weather

        with c3:
            run_options = [1000, 2500, 5000, 10000, 25000, 50000]
            n_sims = st.select_slider(
                "Monte Carlo runs",
                options=run_options,
                value=config.n_sims if config.n_sims in run_options else 10000,
                key=f"{key_prefix}_n_sims",
            )

        with c4:
            std_mult = st.slider(
                "Dispersion σ mult",
                min_value=0.5,
                max_value=2.0,
                value=float(config.std_dev_multiplier),
                step=0.05,
                key=f"{key_prefix}_std_mult",
            )

    return chosen_circuit, str(chosen_session), str(weather_opt), int(n_sims), float(std_mult)
