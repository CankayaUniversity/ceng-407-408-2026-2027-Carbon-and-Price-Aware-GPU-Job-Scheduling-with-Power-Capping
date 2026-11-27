"""Transparent historical scheduling *demo*, not the team's final scheduler.

Rates are fixed within each hourly slot, and job energy is distributed uniformly
throughout the measured job duration. This is an approximation for a PoC.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd


def estimate_at(
    start: datetime,
    duration_seconds: float,
    energy_kwh: float,
    hourly: pd.DataFrame,
) -> Optional[dict]:
    """Integrate hourly price and average-grid carbon intensity over a job run."""
    if duration_seconds <= 0 or energy_kwh < 0:
        return None
    # Preindex market day by local start-of-hour timestamp.
    indexed = hourly.set_index("timestamp")
    end = start + timedelta(seconds=float(duration_seconds))
    cursor = start
    cost_tl = 0.0
    carbon_kg = 0.0
    while cursor < end:
        hour = pd.Timestamp(cursor).floor("h")
        if hour not in indexed.index:
            return None
        row = indexed.loc[hour]
        next_hour = (hour + pd.Timedelta(hours=1)).to_pydatetime()
        segment_end = min(end, next_hour)
        share = (segment_end - cursor).total_seconds() / float(duration_seconds)
        segment_energy_kwh = float(energy_kwh) * share
        # EPİAŞ PTF is TRY/MWh, hence divide by 1000 for TRY/kWh.
        cost_tl += segment_energy_kwh * float(row["ptf_try"]) / 1000.0
        carbon_kg += segment_energy_kwh * float(row["ci"])
        cursor = segment_end
    return {"start": start, "end": end, "cost_tl": cost_tl, "carbon_kg": carbon_kg}


def find_schedule(
    hourly: pd.DataFrame,
    arrival: datetime,
    deadline: datetime,
    duration_seconds: float,
    energy_kwh: float,
    objective: str,
) -> tuple[Optional[dict], Optional[dict], pd.DataFrame]:
    """Evaluate hourly start candidates between arrival and deadline.

    Returns (best, immediate_baseline, all_candidates). No jobs are executed.
    """
    baseline = (
        estimate_at(arrival, duration_seconds, energy_kwh, hourly)
        if arrival + timedelta(seconds=duration_seconds) <= deadline
        else None
    )
    valid = []
    for ts in hourly["timestamp"]:
        start = pd.Timestamp(ts).to_pydatetime()
        if start < arrival or start + timedelta(seconds=duration_seconds) > deadline:
            continue
        estimate = estimate_at(start, duration_seconds, energy_kwh, hourly)
        if estimate is not None:
            valid.append(estimate)
    if not valid:
        return None, baseline, pd.DataFrame()
    scores = pd.DataFrame(valid)
    if objective == "Lowest electricity cost":
        scores["score"] = scores["cost_tl"]
    elif objective == "Lowest estimated carbon":
        scores["score"] = scores["carbon_kg"]
    else:
        # Equally weighted *normalized* objectives; not an official algorithm.
        def normalize(series: pd.Series) -> pd.Series:
            span = series.max() - series.min()
            if span == 0:
                return pd.Series(0.0, index=series.index)
            return (series - series.min()) / span
        scores["score"] = (
            0.5 * normalize(scores["cost_tl"])
            + 0.5 * normalize(scores["carbon_kg"])
        )
    best = scores.loc[scores["score"].idxmin()].to_dict()
    return best, baseline, scores
