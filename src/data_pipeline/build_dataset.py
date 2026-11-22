"""
Build the hourly price + carbon table from the raw EPİAŞ files.
Method (see notebooks/EPIAS_eda.ipynb)
"""

import pandas as pd

from config import RAW_DIR, PROCESSED_DIR, HOURLY_PATH, EMISSION_FACTORS

RT_GEN_META = {"date", "hour", "total", "importExport", "fetched_at"}
REPORT_COLUMNS = ["ptf_try", "gen_licensed_mwh", "gen_unlicensed_mwh", "ci"]


def load(name):
    """Read a raw file and index it by a tz-aware timestamp (start of the hour)."""
    df = pd.read_parquet(RAW_DIR / f"{name}.parquet")
    df["timestamp"] = pd.to_datetime(df["date"]).dt.tz_convert("Europe/Istanbul")
    dupes = df["timestamp"].duplicated().sum()
    assert dupes == 0, f"{name}: {dupes} duplicate timestamps"
    return df.set_index("timestamp")


def check_emission_factors(gen):
    """Every generation column must have an emission factor (and vice versa)."""
    sources = set(gen.columns) - RT_GEN_META
    assert sources == set(EMISSION_FACTORS), sources ^ set(EMISSION_FACTORS)


def align_hourly(gen, *others):
    """Reindex all frames to a full hourly index over the rt-gen period; missing hours become NaN."""
    index = pd.date_range(gen.index.min(), gen.index.max(), freq="h", name="timestamp")
    return index, [df.reindex(index) for df in (gen, *others)]


def compute_generation(gen, ul):
    """Return licensed generation, unlicensed generation and emissions (tCO2) per hour."""
    src = gen[list(EMISSION_FACTORS)]
    emission_t = (src * pd.Series(EMISSION_FACTORS)).sum(axis=1, min_count=1)
    gen_licensed = src.sum(axis=1, min_count=1)
    gen_unlicensed = ul["toplam"]
    return gen_licensed, gen_unlicensed, emission_t


def build_hourly(ptf, gen, ul, index):
    """Combine price and carbon intensity into one hourly table."""
    gen_licensed, gen_unlicensed, emission_t = compute_generation(gen, ul)
    return pd.DataFrame({
        "ptf_try": ptf["price"],
        "ptf_eur": ptf["priceEur"],
        "gen_licensed_mwh": gen_licensed,
        "gen_unlicensed_mwh": gen_unlicensed,
        "ci": emission_t / (gen_licensed + gen_unlicensed),   # NaN if ren-ul-gen missing
        "ci_licensed_only": emission_t / gen_licensed,
        "unlicensed_available": gen_unlicensed.notna(),
    }, index=index)


def save(hourly):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(HOURLY_PATH)


def report(hourly, gen):

    print(f"period: {hourly.index.min()} -> {hourly.index.max()}  ({len(hourly)} hours)")
    for col in REPORT_COLUMNS:
        missing = hourly[col].isna().sum()
        print(f"{col:<20} missing {missing:>5} hours")
    print(f"ci available for {hourly['ci'].notna().mean():.2%} of hours (target > 95%)")

    # info: Ulutaş and Kurt report up to 23 MWh/h mismatch in `total`
    mismatch = (gen["total"] - hourly["gen_licensed_mwh"] - gen["importExport"]).abs().max()
    print(f"max |total - sources - importExport| = {mismatch:.2f} MWh")


def main():
    ptf, gen, ul = load("ptf"), load("rt-gen"), load("ren-ul-gen")
    check_emission_factors(gen)
    
    index, (gen, ptf, ul) = align_hourly(gen, ptf, ul)
    hourly = build_hourly(ptf, gen, ul, index)
    
    save(hourly)
    report(hourly, gen)


if __name__ == "__main__":
    main()
