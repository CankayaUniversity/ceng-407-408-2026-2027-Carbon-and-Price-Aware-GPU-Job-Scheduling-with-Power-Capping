"""Download hourly EPİAŞ data for a date range and save each dataset as Parquet."""

import time
from datetime import datetime, timezone

import pandas as pd
from eptr2 import EPTR2

from config import START, END, CALLS, RAW_DIR, REQUEST_DELAY_S, RATE_LIMIT_WAIT_S
eptr = EPTR2(use_dotenv=True, recycle_tgt=True)


def call_with_retry(call, start, end):
    """EPİAŞ allows a limited number of requests per minute (HTTP 429 when exceeded)."""
    while True:
        try:
            df = eptr.call(call, start_date=start, end_date=end)
            time.sleep(REQUEST_DELAY_S)
            return df
        except Exception as e:
            if "429" not in str(e):
                raise
            print(f"rate limit reached, waiting {RATE_LIMIT_WAIT_S} s ...")
            time.sleep(RATE_LIMIT_WAIT_S)


def fetch_call(call, months):
    """Fetch one dataset month by month and return it as a single DataFrame."""
    # one request per month, to stay within the API's date-range limits
    parts = []
    for m in months:
        parts.append(call_with_retry(call, m.start_time.strftime("%Y-%m-%d"),
                                     m.end_time.strftime("%Y-%m-%d")))
        print(f"{call} {m}: {len(parts[-1])} rows")
    
    return pd.concat(parts, ignore_index=True)


def save_call(call, df, expected_hours):
    """Stamp the fetch time and write the dataset to RAW_DIR as Parquet."""
    df["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df.to_parquet(RAW_DIR / f"{call}.parquet", index=False)
    print(f"{call}: {len(df)} / {expected_hours} hours saved\n")


def main():
    months = pd.period_range(START, END, freq="M")
    expected_hours = sum(m.days_in_month for m in months) * 24
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for call in CALLS:
        df = fetch_call(call, months)
        save_call(call, df, expected_hours)


if __name__ == "__main__":
    main()
