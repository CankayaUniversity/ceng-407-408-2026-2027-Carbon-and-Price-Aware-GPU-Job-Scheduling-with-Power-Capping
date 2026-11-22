# Data pipeline

Builds an hourly table of electricity price and grid carbon intensity for Türkiye from EPİAŞ Transparency Platform data.

Current range: 2021-01 to 2026-09

## Run

Requires `pip install -r requirements.txt` and EPİAŞ credentials in `.env`.

```bash
python src/data_pipeline/fetch_data.py      # 1. download raw data 
python src/data_pipeline/build_dataset.py   # 2. build the table   -> data/processed/hourly.parquet
```

The date range is set in `config.py` (`START`, `END`). `END` must be at least ~2 months ago, because unlicensed generation (`ren-ul-gen`) is published with a delay. `build_dataset.py` prints a report (missing hours, share of hours with carbon intensity).

## Updating the data

The file is always `data/processed/hourly.parquet`; versions are tracked with git, not file names.

1. Move `END` forward in `config.py`, then run `fetch_data.py` and `build_dataset.py`.
2. Check the report, and check that values in the old range did not change (`rt-gen` can be revised later).
3. Update "Current range" above, commit, then `git tag data-YYYY-MM` and `git push origin data-YYYY-MM`.

Experiments must not use the whole table: fix a date range in the experiment's own config, and state which data tag the results were produced with.

## Output: `data/processed/hourly.parquet`

| Column | Unit | Description |
|---|---|---|
| `timestamp` (index) | Europe/Istanbul | Start of the hour; `08:00` covers 08:00–09:00 |
| `ptf_try`, `ptf_eur` | TL/MWh, EUR/MWh | Day-ahead market clearing price (PTF) |
| `gen_licensed_mwh` | MWh | Sum of `rt-gen` source columns (excludes imports/exports) |
| `gen_unlicensed_mwh` | MWh | Unlicensed generation (`ren-ul-gen`), mostly solar |
| `ci` | kgCO₂/kWh | Carbon intensity, unlicensed generation included. **Use this one.** |
| `ci_licensed_only` | kgCO₂/kWh | Same without unlicensed generation (comparison only) |
| `unlicensed_available` | bool | Whether unlicensed data exists for that hour |

## Method

`ci = Σ (generation_i × emission_factor_i) / (licensed + unlicensed generation)`, following Ulutaş and Kurt (2026).

- Direct (operational) CO₂ only; no life-cycle emissions. Renewables, biomass and all unlicensed sources = 0.
- Geothermal = 0.58 tCO₂/MWh (measured Turkish value); Ulutaş and Kurt use 0.
- Unlicensed generation is settlement-based grid injection, a lower bound, so `ci` may be slightly overstated.
- `ci` is an average, not a marginal, intensity.

## References

- Ulutaş, T. & Kurt, Ü. (2026). *Electricity*, 7(4), 123. https://doi.org/10.3390/electricity7040123
- T.C. ETKB (2025). "Türkiye Elektrik Üretimi ve Elektrik Tüketim Noktası Emisyon Faktörleri (2023)". Fossil emission factors.
- Akın, M., Orucu, T. & Fridriksson, T. (2020). "Characterizing the Declining CO2 Emissions from Turkish Geothermal Power Plants". Stanford Geothermal Workshop. Geothermal factor (0.58).