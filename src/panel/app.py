"""CO2RE prototype: historical EPİAŞ data + measured GPU job profiles."""
from __future__ import annotations

from datetime import datetime, time

import pandas as pd
import streamlit as st

from data_loader import (
    MARKET_FILE,
    PROFILES_FILE,
    load_job_profiles,
    load_market_data,
)
from simulator import find_schedule

st.set_page_config(page_title="CO₂RE | GPU Job Scheduling", layout="wide")
@st.cache_data(show_spinner="Reading EPİAŞ market data...")
def get_market() -> pd.DataFrame:
    return load_market_data()

@st.cache_data(show_spinner="Reading GPU measurement profiles...")
def get_profiles() -> pd.DataFrame:
    return load_job_profiles()

st.title("CO₂RE")
st.markdown("**Carbon- and Price-Aware GPU Job Scheduling with Power Capping**")
st.info(
    "**CENG 407 prototype.** This dashboard reads existing historical EPİAŞ data "
    "and measured GPU job profiles. It does not control GPUs, run jobs, "
    "or connect to live market data. The scheduling page is a simple demonstration, "
    "not the team's final scheduling algorithm."
)

try:
    market = get_market()
    profiles = get_profiles()
except (FileNotFoundError, ValueError, ImportError) as exc:
    st.error(f"Required data could not be loaded: {exc}")
    st.code(f"Expected market file: {MARKET_FILE}\nExpected GPU profiles: {PROFILES_FILE}")
    st.stop()

market_start = market["timestamp"].iloc[0].date()
market_end = market["timestamp"].iloc[-1].date()
job_names = sorted(profiles["job"].unique().tolist())
st.sidebar.header("Dataset information")
st.sidebar.caption(f"Market data: {market_start} → {market_end}")
st.sidebar.caption(f"Measured job types: {len(job_names)}")
st.sidebar.caption("Times shown are Türkiye local time (Europe/Istanbul).")

market_tab, gpu_tab, scheduler_tab = st.tabs(
    ["Market data", "GPU profiles", "Scheduling demo"]
)

with market_tab:
    st.subheader("Electricity market overview")
    selected_date = st.date_input(
        "Choose a historical date",
        value=market_end,
        min_value=market_start,
        max_value=market_end,
        key="market_date",
    )
    day = market.loc[market["timestamp"].dt.date == selected_date].copy()
    if day.empty:
        st.warning("No observations for this date. Choose another date.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Hourly records", str(len(day)))
        c2.metric("Lowest price (TRY/MWh)", f"{day['ptf_try'].min():,.2f}")
        c3.metric("Lowest avg. CO₂ (kg/kWh)", f"{day['ci'].min():.3f}")
        st.markdown("#### Day-ahead market price")
        st.line_chart(day.set_index("timestamp")["ptf_try"], x_label="Local hour", y_label="TRY/MWh")
        st.markdown("#### Estimated average grid carbon intensity")
        st.line_chart(day.set_index("timestamp")["ci"], x_label="Local hour", y_label="kg CO₂/kWh")
        st.caption(
            "Carbon intensity is an estimated *average operational* grid indicator, "
            "not marginal emissions or verified avoided CO₂. The market dataset is historical."
        )
        with st.expander("View hourly source data"):
            st.dataframe(
                day[["timestamp", "ptf_try", "ci"]].rename(
                    columns={"ptf_try": "Price (TRY/MWh)", "ci": "Avg CO₂ (kg/kWh)"}
                ),
                use_container_width=True,
                hide_index=True,
            )

with gpu_tab:
    st.subheader("Measured GPU job profiles")
    job = st.selectbox("Select a measured workload", job_names, key="profile_job")
    subset = profiles.loc[profiles["job"] == job].copy()
    subset = subset.sort_values("duration_s", ascending=False)
    if "job_description" in subset.columns:
        st.caption(str(subset["job_description"].iloc[0]))
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Tested settings", str(len(subset)))
    metric_b.metric("Shortest measured runtime", f"{subset['duration_s'].min():.1f} s")
    metric_c.metric("Lowest measured wall energy", f"{subset['energy_wh'].min():.3f} Wh")
    duration_col, energy_col = st.columns(2)
    with duration_col:
        st.markdown("#### Job duration (seconds)")
        st.bar_chart(subset.set_index("setting_label")["duration_s"], x_label="GPU setting", y_label="Seconds")
    with energy_col:
        st.markdown("#### Wall energy (Wh)")
        st.bar_chart(subset.set_index("setting_label")["energy_wh"], x_label="GPU setting", y_label="Wh")
    display_cols = [
        "setting_label", "n_runs", "duration_s", "energy_wh", "energy_kwh", "avg_wall_w"
    ]
    st.dataframe(
        subset[[c for c in display_cols if c in subset.columns]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "These are measurements from test/shelly-plug-test/results/job_profile.csv. "
        "The laptop experiments used GPU clock locking (MHz), not a configurable watt limit. "
        "Measurements apply to these fixed workload sizes and this test machine."
    )

with scheduler_tab:
    st.subheader("Historical scheduling demonstration")
    st.caption(
        "Select one already-measured workload and GPU setting. Candidate starts are "
        "on the hour, subject to the deadline. No GPU work is executed."
    )
    left, right = st.columns(2)
    with left:
        demo_date = st.date_input(
            "Historical date", value=market_end, min_value=market_start,
            max_value=market_end, key="sim_date"
        )
        demo_job = st.selectbox("Workload", job_names, key="sim_job")
        job_rows = profiles.loc[profiles["job"] == demo_job].copy()
        labels = job_rows["setting_label"].tolist()
        demo_setting = st.selectbox("Measured GPU setting", labels, key="sim_setting")
        selected_profile = job_rows.loc[job_rows["setting_label"] == demo_setting].iloc[0]
    with right:
        arrival_time = st.time_input("Job becomes available", value=time(9, 0))
        deadline_time = st.time_input("Deadline (same day)", value=time(20, 0))
        objective = st.selectbox(
            "Demo strategy",
            ["Lowest electricity cost", "Lowest estimated carbon", "Balanced (50/50)"],
        )

    duration_s = float(selected_profile["duration_s"])
    energy_kwh = float(selected_profile["energy_kwh"])
    st.caption(
        f"Measured job: **{duration_s:.1f} seconds**, "
        f"**{energy_kwh * 1000:.3f} Wh** from the wall meter "
        f"({int(selected_profile['n_runs']) if pd.notna(selected_profile.get('n_runs')) else '?'} runs)."
    )

    sim_day = market.loc[market["timestamp"].dt.date == demo_date].copy()
    arrival = datetime.combine(demo_date, arrival_time)
    deadline = datetime.combine(demo_date, deadline_time)
    if deadline <= arrival:
        st.warning("Deadline must be later than the job arrival time.")
    elif sim_day.empty:
        st.warning("There are no market records for this date.")
    else:
        best, baseline, candidates = find_schedule(
            sim_day, arrival, deadline, duration_s, energy_kwh, objective
        )
        if best is None:
            st.warning(
                "No eligible hourly start fits this deadline and available price/CO₂ data. "
                "Try an earlier arrival or a later deadline."
            )
        else:
            st.success(f"Demo recommendation: start at **{best['start']:%H:%M}**, "
                       f"finish around **{best['end']:%H:%M:%S}**.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Estimated electricity cost", f"{best['cost_tl']:.5f} TRY")
            m2.metric("Est. average-grid CO₂ indicator", f"{best['carbon_kg'] * 1000:.2f} g")
            m3.metric("Feasible hourly starts", str(len(candidates)))
            if baseline is not None:
                st.markdown("#### Comparison with running immediately")
                comparison = pd.DataFrame([
                    {"Plan": "Immediately", "Start": baseline["start"].strftime("%H:%M"),
                     "Est. cost (TRY)": baseline["cost_tl"],
                     "CO₂ indicator (g)": baseline["carbon_kg"] * 1000},
                    {"Plan": "Scheduled (demo)", "Start": best["start"].strftime("%H:%M"),
                     "Est. cost (TRY)": best["cost_tl"],
                     "CO₂ indicator (g)": best["carbon_kg"] * 1000},
                ])
                st.dataframe(comparison, use_container_width=True, hide_index=True)
            with st.expander("All feasible hourly start options"):
                preview = candidates[["start", "end", "cost_tl", "carbon_kg"]].copy()
                preview["start"] = preview["start"].dt.strftime("%H:%M")
                preview["end"] = preview["end"].dt.strftime("%H:%M:%S")
                preview["carbon_kg"] = preview["carbon_kg"] * 1000
                preview = preview.rename(columns={
                    "start": "Start", "end": "Finish", "cost_tl": "Est. cost (TRY)",
                    "carbon_kg": "CO₂ indicator (g)"
                })
                st.dataframe(preview, hide_index=True, use_container_width=True)
    st.warning(
        "**Demo assumptions:** historical rather than future market data; "
        "energy and runtime taken from a fixed measured workload; energy assumed "
        "uniformly distributed over the job runtime; simple hourly brute-force search. "
        "No energy consumption or CO₂ reduction has been physically achieved by this screen. "
        "The final scheduler and predictions must be integrated and evaluated separately."
    )

st.divider()
st.caption("CO₂RE · Çankaya University · CENG 407–408 · Prototype dashboard")
