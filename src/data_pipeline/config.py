"""Shared settings for the EPİAŞ data pipeline."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
HOURLY_PATH = PROCESSED_DIR / "hourly.parquet"

# Download range
START = "2021-01"   # first month
END = "2026-09"     # last month; ren-ul-gen is published with a 1-2 month delay
CALLS = ["ptf", "rt-gen", "ren-ul-gen"]

# EPİAŞ rate limit
REQUEST_DELAY_S = 1        
RATE_LIMIT_WAIT_S = 60  


# tCO2/MWh (= kgCO2/kWh)
# Fossil: T.C. ETKB, "Türkiye Elektrik Üretimi ve Elektrik Tüketim Noktası
#   Emisyon Faktörleri (2023)", 2025.
# Geothermal: M. Akın, T. Orucu, T. Fridriksson, "Characterizing the Declining
#   CO2 Emissions from Turkish Geothermal Power Plants", Stanford Geothermal
#   Workshop, 2020.
EMISSION_FACTORS = {
    "lignite": 1.122, "asphaltiteCoal": 1.078, "blackCoal": 1.045,
    "importCoal": 0.800, "naturalGas": 0.400, "lng": 0.400,
    "fueloil": 1.240, "naphta": 1.240, "geothermal": 0.580,
    "dammedHydro": 0.0, "river": 0.0, "wind": 0.0, "sun": 0.0,
    "biomass": 0.0, "wasteheat": 0.0,
}