# CO₂RE Streamlit Panel — CENG 407 Prototype

This is a **historical-data demonstration dashboard**, not the final CO₂RE scheduler.

The dashboard displays existing EPİAŞ electricity market data and measured GPU job profiles. It also provides a basic historical scheduling demonstration.

It does not execute GPU workloads, control GPU settings, or connect to live electricity market data.

## Run Locally

Run the following commands **from the repository root** using the VS Code terminal.

### Windows (PowerShell)

Create a virtual environment (only required once):

```powershell
python -m venv .venv
```

Install the panel dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r src/panel/requirements.txt
```

Start the Streamlit dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/panel/app.py
```

Open the following address in your browser:

http://localhost:8501

Press `Ctrl+C` in the terminal to stop the application.

### Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r src/panel/requirements.txt
.venv/bin/python -m streamlit run src/panel/app.py
```

## Project Files

The panel consists of the following files:

- `app.py` — Streamlit interface, charts, data tables, and scheduling demonstration controls.
- `data_loader.py` — Reads and validates existing EPİAŞ market data and GPU measurement profiles.
- `simulator.py` — Evaluates simple historical scheduling options. This is **not the final scheduling algorithm**.
- `requirements.txt` — Python dependencies required by the dashboard.
- `README.md` — Setup instructions, component descriptions, and current limitations.

## Dashboard Features

### 1. Electricity Market Overview

Displays historical electricity market information, including:

- Hourly day-ahead electricity prices.
- Estimated average grid carbon intensity.
- Interactive date selection.
- Electricity price and carbon intensity charts.
- Hourly source data tables.

### 2. GPU Job Profiles

Displays existing GPU measurement results, including:

- Measured GPU workloads.
- Different GPU clock settings.
- Job execution duration.
- Wall energy consumption.
- Comparison charts and measurement tables.

### 3. Historical Scheduling Demonstration

Provides a basic scheduling demonstration using historical market data and measured GPU job profiles.

Users can select:

- A historical date.
- A measured GPU workload.
- A GPU clock setting.
- Job availability time.
- Job deadline.
- A demonstration strategy.

The available demonstration strategies are:

- Lowest electricity cost.
- Lowest estimated carbon.
- Balanced cost and carbon.

The dashboard evaluates eligible hourly start times and presents an estimated schedule.

**Important:** This is a simplified demonstration. The team's final runtime prediction model and scheduling algorithm are not yet integrated.

## Required Data

The dashboard uses the following files from the existing repository:

### Electricity Market Data

`data/processed/hourly.parquet`

Contains historical EPİAŞ electricity market information, including:

- `timestamp` — Hourly timestamp.
- `ptf_try` — Electricity market price (TRY/MWh).
- `ci` — Estimated average grid carbon intensity (kgCO₂/kWh).

The current dataset covers January 2021 through September 2026.

### GPU Measurement Data

`test/shelly-plug-test/results/job_profile.csv`

Contains measured GPU workload information, including:

- Job identifiers.
- GPU clock settings.
- Execution duration.
- Wall energy consumption.
- Measurement metadata.

These files are read by the dashboard without modification.

An NVIDIA GPU, Shelly smart plug, or EPİAŞ API credentials are **not required** to run the historical dashboard.

Internet access is only necessary when installing Python dependencies that are not already available locally.

## Current Limitations

- The electricity market data is historical, not live or forecast data.
- Carbon intensity is an estimated **average operational grid indicator**, not marginal emissions or verified avoided CO₂.
- Electricity cost is estimated using EPİAŞ day-ahead market prices (TRY/MWh), not retail electricity tariffs.
- The scheduling demonstration evaluates hourly candidate start times.
- Measured energy consumption is assumed to be distributed uniformly throughout each job's execution.
- The demonstration uses measured durations for fixed workloads rather than predicting arbitrary new workloads.
- The dashboard does not execute jobs or modify GPU settings.
- The existing measurements use GPU clock locking (MHz), **not configurable watt-based power limits**.
- The team's final runtime prediction model and deadline-aware scheduler are not yet integrated.

This dashboard serves as an initial CENG 407 prototype and a foundation for future system integration.

## Contributing

Follow the repository's root [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

Development must take place in a feature branch rather than directly on `main`.

Before submitting changes:

- Review the modified files.
- Test the dashboard locally.
- Ensure that no credentials or unnecessary files are included.
- Open a pull request for teammate review.
- Do not merge without approval.

From the repository root, after reviewing the changes:

```bash
git status
git add .gitignore src/panel
git diff --cached
git commit -m "feat(panel): add initial Streamlit dashboard"
git push -u origin feat/panel-skeleton
```

The local `.venv/` directory and temporary launcher scripts must **not** be committed.

The pull request should clearly describe the dashboard as an **initial prototype**, not a completed scheduling system.