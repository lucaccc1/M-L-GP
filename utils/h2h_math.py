"""Pure-Python statistical helpers for Monte Carlo Head-to-Head (H2H) rider comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import median
from typing import Any


def summarize_h2h_outcomes(
    ranks_a: Sequence[int],
    crashes_a: Sequence[bool],
    ranks_b: Sequence[int],
    crashes_b: Sequence[bool],
    n_riders: int,
    rider_a: str = "Rider A",
    rider_b: str = "Rider B",
) -> dict[str, Any]:
    """Compute exact mutually exclusive H2H duel probabilities and finish distributions.

    Expects 0-indexed `ranks_*` (0 = P1) and boolean `crashes_*` sequences of equal length.
    """
    n_sims = len(ranks_a)
    if not (len(crashes_a) == n_sims and len(ranks_b) == n_sims and len(crashes_b) == n_sims):
        raise ValueError("All H2H rank and crash sequences must have identical length.")
    if n_sims == 0:
        raise ValueError("Cannot compute H2H summary on zero simulations.")
    if n_riders <= 0:
        raise ValueError("n_riders must be positive.")

    a_pace = 0
    b_pace = 0
    a_via_dnf = 0
    b_via_dnf = 0
    both_dnf = 0
    diff_sum = 0
    fin_pos_a: list[int] = []
    fin_pos_b: list[int] = []

    counts_a = [0] * n_riders
    counts_b = [0] * n_riders
    dnf_count_a = 0
    dnf_count_b = 0

    for ra, ca, rb, cb in zip(ranks_a, crashes_a, ranks_b, crashes_b):
        ca_b = bool(ca)
        cb_b = bool(cb)
        ra_i = int(ra)
        rb_i = int(rb)

        if ca_b:
            dnf_count_a += 1
        else:
            if 0 <= ra_i < n_riders:
                counts_a[ra_i] += 1
            fin_pos_a.append(ra_i + 1)

        if cb_b:
            dnf_count_b += 1
        else:
            if 0 <= rb_i < n_riders:
                counts_b[rb_i] += 1
            fin_pos_b.append(rb_i + 1)

        if not ca_b and not cb_b:
            if ra_i < rb_i:
                a_pace += 1
            elif rb_i < ra_i:
                b_pace += 1
            diff_sum += rb_i - ra_i
        elif not ca_b and cb_b:
            a_via_dnf += 1
        elif ca_b and not cb_b:
            b_via_dnf += 1
        else:
            both_dnf += 1

    both_finish = a_pace + b_pace
    scale = 100.0 / n_sims

    a_pace_pct = round(a_pace * scale, 2)
    a_dnf_pct = round(a_via_dnf * scale, 2)
    b_pace_pct = round(b_pace * scale, 2)
    b_dnf_pct = round(b_via_dnf * scale, 2)
    both_dnf_pct = round(both_dnf * scale, 2)

    a_total_pct = round((a_pace + a_via_dnf) * scale, 2)
    b_total_pct = round((b_pace + b_via_dnf) * scale, 2)
    both_finish_pct = round(both_finish * scale, 2)

    a_ahead_both_finish_pct = (
        round((a_pace / both_finish) * 100.0, 2) if both_finish > 0 else None
    )
    b_ahead_both_finish_pct = (
        round((b_pace / both_finish) * 100.0, 2) if both_finish > 0 else None
    )
    mean_pos_gap = round(diff_sum / both_finish, 2) if both_finish > 0 else None

    med_a = round(float(median(fin_pos_a)), 1) if fin_pos_a else None
    med_b = round(float(median(fin_pos_b)), 1) if fin_pos_b else None

    outcome_breakdown = [
        {
            "segment": f"{rider_a} ahead (pace)",
            "side": rider_a,
            "order": 1,
            "pct": a_pace_pct,
            "count": a_pace,
        },
        {
            "segment": f"{rider_a} ahead ({rider_b} DNF)",
            "side": rider_a,
            "order": 2,
            "pct": a_dnf_pct,
            "count": a_via_dnf,
        },
        {
            "segment": "Both DNF",
            "side": "Both DNF",
            "order": 3,
            "pct": both_dnf_pct,
            "count": both_dnf,
        },
        {
            "segment": f"{rider_b} ahead ({rider_a} DNF)",
            "side": rider_b,
            "order": 4,
            "pct": b_dnf_pct,
            "count": b_via_dnf,
        },
        {
            "segment": f"{rider_b} ahead (pace)",
            "side": rider_b,
            "order": 5,
            "pct": b_pace_pct,
            "count": b_pace,
        },
    ]

    position_distribution: list[dict[str, Any]] = []
    for idx in range(n_riders):
        pos_label = f"P{idx + 1}"
        position_distribution.append(
            {
                "position_label": pos_label,
                "position_order": idx + 1,
                "rider": rider_a,
                "pct": round(counts_a[idx] * scale, 2),
            }
        )
        position_distribution.append(
            {
                "position_label": pos_label,
                "position_order": idx + 1,
                "rider": rider_b,
                "pct": round(counts_b[idx] * scale, 2),
            }
        )

    position_distribution.append(
        {
            "position_label": "DNF",
            "position_order": n_riders + 1,
            "rider": rider_a,
            "pct": round(dnf_count_a * scale, 2),
        }
    )
    position_distribution.append(
        {
            "position_label": "DNF",
            "position_order": n_riders + 1,
            "rider": rider_b,
            "pct": round(dnf_count_b * scale, 2),
        }
    )

    return {
        "n_sims": n_sims,
        "rider_a": rider_a,
        "rider_b": rider_b,
        "a_beats_b_total_pct": a_total_pct,
        "b_beats_a_total_pct": b_total_pct,
        "a_beats_b_pace_pct": a_pace_pct,
        "a_beats_b_dnf_pct": a_dnf_pct,
        "b_beats_a_pace_pct": b_pace_pct,
        "b_beats_a_dnf_pct": b_dnf_pct,
        "both_dnf_pct": both_dnf_pct,
        "both_finish_pct": both_finish_pct,
        "a_ahead_both_finish_pct": a_ahead_both_finish_pct,
        "b_ahead_both_finish_pct": b_ahead_both_finish_pct,
        "mean_pos_gap_both_finish": mean_pos_gap,
        "median_finish_a": med_a,
        "median_finish_b": med_b,
        "outcome_breakdown": outcome_breakdown,
        "position_distribution": position_distribution,
    }


def determine_metric_advantage(
    val_a: float | None,
    val_b: float | None,
    higher_is_better: bool,
    rider_a: str,
    rider_b: str,
    decimals: int = 2,
) -> str:
    """Return which rider holds the edge on a numeric telemetry/model feature."""
    if val_a is None or val_b is None:
        return "N/A"
    ra = round(float(val_a), decimals)
    rb = round(float(val_b), decimals)
    if ra == rb:
        return "Tie"
    a_wins = (ra > rb) if higher_is_better else (ra < rb)
    return rider_a if a_wins else rider_b
