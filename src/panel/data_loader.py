"""Load and validate existing CO2RE experiment datasets (no API keys needed)."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKET_FILE = REPO_ROOT / "data" / "processed" / "hourly.parquet"
PROFILES_FILE = (
    REPO_ROOT / "test" / "shelly-plug-test" / "results" / "job_profile.csv"
)


def load_market_data(path: Path = MARKET_FILE) -> pd.DataFrame:
    """Return market data with naive Europe/Istanbul-local hourly timestamps.

    The original Parquet timestamps use Europe/Istanbul; stripping the timezone
    here is only for local historical visualization, not for scheduling UTC jobs.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Market data not found: {path}")
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        if df.index.name != "timestamp":
            raise ValueError("Market file has no timestamp index or column")
        df = df.reset_index()
    required = {"timestamp", "ptf_try", "ci"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Market dataset is missing columns: {sorted(missing)}")
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    df["timestamp"] = ts
    for col in ("ptf_try", "ci"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "ptf_try", "ci"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if df.empty:
        raise ValueError("Market dataset has no usable records")
    return df


def load_job_profiles(path: Path = PROFILES_FILE) -> pd.DataFrame:
    """Return wall-energy profiles measured in shelly-plug-test."""
    if not path.is_file():
        raise FileNotFoundError(f"Job profiles not found: {path}")
    df = pd.read_csv(path)
    required = {"job", "setting", "duration_s", "energy_kwh", "energy_wh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"GPU profile file is missing columns: {sorted(missing)}")
    df = df.copy()
    for col in ("duration_s", "energy_kwh", "energy_wh", "clock_mhz", "n_runs"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["job", "setting", "duration_s", "energy_kwh"])
    df = df.loc[(df["duration_s"] > 0) & (df["energy_kwh"] >= 0)].copy()
    if df.empty:
        raise ValueError("GPU profile file has no usable records")
    df["setting_label"] = df.apply(
        lambda r: (
            f"{str(r['setting'])} ({int(r['clock_mhz'])} MHz)"
            if "clock_mhz" in r.index and pd.notna(r["clock_mhz"])
            else str(r["setting"])
        ),
        axis=1,
    )
    return df
