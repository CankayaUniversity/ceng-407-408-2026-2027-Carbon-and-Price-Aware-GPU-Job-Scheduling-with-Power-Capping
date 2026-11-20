# CO₂RE – Carbon- and Price-Aware GPU Job Scheduling with Power Capping

Çankaya University · Computer Engineering · CENG 407–408 · 2026–2027

This repository contains the measurement scripts, raw measurements and first results of the CO₂RE capstone project. The project studies how deferrable GPU jobs can be scheduled using hourly electricity price and carbon-intensity data from Türkiye's grid, choosing a start time and a GPU power setting for each job while meeting its deadline.

> **Status:** Term 1 (CENG 407). The repository currently covers wall-power measurement, GPU clock sweeps and data preparation. Forecasting and scheduling code will be added later.

## Hardware and setup

All measurements so far were made on **one laptop**, not on a desktop PC as written in the project proposal.

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB |
| Operating system | Windows; NVIDIA driver 610.91 |
| Wall meter | Shelly Plug M Gen3 smart plug, read once per second over its local HTTP API |
| GPU telemetry | NVML (`nvidia-ml-py`), once per second |
| GPU setting | Clock lock with `nvidia-smi -lgc` (the power-limit option `-pl` is not available on this GPU) |
| Language model | Qwen3.5-9B, Q4_K_M GGUF, served with llama.cpp b11960 |

**Battery.** Because a laptop battery can store or release energy, the plug could miss or add energy. In all plug validation tests (`data/raw/plug_validation/`), the battery state was logged every second and stayed `Idle` for the whole test. The sweep scripts do not log the battery state.

**Idle power.** The idle baseline of the whole computer is about 26.9 W (median of three 10-minute blocks; `data/raw/idle/`). The browser was closed during measurements, because a browser using the GPU kept idle power near 41 W.

## Repository structure

```
scripts/                 measurement and analysis scripts
data/raw/plug_validation plug readings during the first validation tests
data/raw/sweeps          one CSV per run (1 s samples), runs.csv, sweep plans, llama-bench logs
data/raw/idle            idle baseline log and summary
data/tasks/              task files created by prepare_data.py from the CFPB archive
results/                 summary tables and plots computed from data/raw
```

## Scripts

| Script | Purpose |
|---|---|
| `plug_check.py` | Logs plug power (and battery state on Windows) for 5 minutes to validate the plug |
| `idle_baseline.py` | Measures the idle power of the computer in several blocks |
| `measure_run.py` | Runs one job (a fixed number of matrix multiplications, or an external command) with idle phases before and after, and logs plug and GPU readings every second |
| `sweep.py` | Runs a shuffled, repeated sweep over GPU clock settings, with a control run before the sweep |
| `analyze_sweep.py` | Recomputes energy and duration for every valid run in a plan and writes summary tables and a plot |
| `plot_comparison.py` | Compares the matrix job and the language-model job in one figure |
| `prepare_data.py` | Builds the classification, summarization and synthetic-data task files from the CFPB complaint archive |

The plug IP address is required on every script that reads the plug:

```bash
python plug_check.py --plug-ip <PLUG_IP>
python idle_baseline.py --plug-ip <PLUG_IP>
python measure_run.py --plug-ip <PLUG_IP> --label test --work 1990
python sweep.py --plug-ip <PLUG_IP> --plan llm_plan.csv --clocks 900 1200 1500 1800 2100 full --command "<llama-bench command>"
```

`sweep.py` must be run from the folder that contains `measure_run.py`, in a PowerShell window opened as administrator (needed for the clock lock). New runs are written to the current folder.

To recompute the results from the raw data:

```bash
python scripts/analyze_sweep.py --folder data/raw/sweeps --plan matmul_plan.csv --prefix matmul --out-dir results
python scripts/analyze_sweep.py --folder data/raw/sweeps --plan llm_plan.csv --prefix llm --out-dir results --job-name "Qwen3.5-9B, 27 requests (1024 tokens read + 256 tokens written)"
cd results && python ../scripts/plot_comparison.py
```

## First results

Two jobs were run at fixed GPU clocks and at unlocked full speed, three times each in shuffled order. The same amount of work was done in every run. Values are means of three runs, relative to full speed.

**Language-model job** (Qwen3.5-9B, 27 requests × (1024 prompt + 256 generated tokens), `llama-bench`):

| GPU clock | Duration | Wall energy | GPU energy (NVML) |
|---|---|---|---|
| 900 MHz | +43% | −12% | −35% |
| 1200 MHz | +14% | −25% | −41% |
| 1500 MHz | +7% | −25% | −37% |
| 1800 MHz | +5% | −19% | −29% |
| 2100 MHz | −1% | −15% | −21% |
| full speed | 0 | 0 | 0 |

**Matrix job** (1990 × 6144×6144 fp32 matrix multiplications, compute-bound):

| GPU clock | Duration | Wall energy | GPU energy (NVML) |
|---|---|---|---|
| 900 MHz | +117% | +17% | −25% |
| 1200 MHz | +62% | −3% | −28% |
| 1500 MHz | +29% | −8% | −22% |
| 1800 MHz | +8% | −4% | −8% |
| full speed | 0 | 0 | 0 |

Two observations so far:
- Savings measured by NVML are always larger than savings measured at the wall, because the rest of the computer keeps drawing power while the job takes longer.
- The language-model job tolerates lower clocks much better than the compute-bound matrix job. For the matrix job, the lowest clock uses more wall energy than full speed.

**Limitations:** one computer, one model, Windows; `llama-bench` uses random tokens, so the job structure is realistic but the content is not; the duration includes model loading (a few seconds).

## Data notes

- **Run IDs** keep the labels used during the measurements: `tam` = full speed, `isinma` = warm-up, `kontrol` = control run, `deneme` = trial. Raw file names contain these IDs and are not renamed.
- `runs.csv` lists every run, including early tests, warm-ups and control runs. Only runs listed as `done` in a plan file (`matmul_plan.csv`, `llm_plan.csv`) are used in the results.
- The first `f1800_r3` run of the matrix sweep (`run_pc-1_20261028_160846_f1800_r3.csv`) was disturbed by a paused console window and was repeated. It is kept in the raw data but is not in the plan file, so it is not used.
- `analyze_sweep.py` also drops any run with a logging gap longer than 5 s.
- `job_profile.csv` is the per-job summary used as input for the scheduler.

## Task data

The task files in `data/tasks/` come from the **CFPB Consumer Complaint Database** narrative archive (FOIA Electronic Reading Room), file `CCDB_Export_16_March_2026.zip` (22 MB; SHA-256 `0d337dac973d1fdf874c1d9cd5909502bb29a71334543572f0b1aedcd5a58d24`). The ZIP is not stored in this repository; the task files can be recreated from it with:

```bash
python scripts/prepare_data.py --zip CCDB_Export_16_March_2026.zip --out data/tasks
```

Word limits were chosen from the length distribution of all narratives (10th percentile 50 words, median 158, 75th percentile 270, 99th percentile 984): classification uses complaints of 50–269 words, summarization 270–1,000 words. With seed 42 the script produces the same files every time; `manifest.json` records the parameters and the SHA-256 of the source ZIP and of every output file.

| File | Rows | Use |
|---|---|---|
| `classify_test.jsonl` | 480 | routing (8 products × 60) |
| `classify_dev.jsonl` | 80 | prompt development only (8 × 10) |
| `summarize_test.jsonl` | 120 | case summaries |
| `summarize_dev.jsonl` | 10 | prompt development only |
| `synth_spec.jsonl` | 32 | synthetic complaints (4 most frequent issues per product) |
| `tr_subset.jsonl` | 50 | Turkish side measurement (to be translated) |

## Team

See the project wiki.
