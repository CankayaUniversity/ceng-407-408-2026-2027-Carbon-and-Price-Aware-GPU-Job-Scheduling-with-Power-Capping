# LLM Estimation Test

Can an LLM predict how long a GPU job will take and how much energy it will use, before the job runs? This folder tests that question against wall-power measurements.

> **Relation to `test/shelly-plug-test`:** the measurement method is the same (Shelly plug + NVML + clock lock), but it was run independently on a **second laptop (RTX 3060)**. The new parts here are the LLM estimation test, the observer-effect finding, and per-second battery logging.

## Hardware and setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB, driver 610.88 |
| Operating system | Windows 11 |
| Wall meter | Shelly Plug M Gen3, read once per second over its local HTTP API |
| GPU telemetry | NVML, once per second (power, energy counter, clock, temperature) |
| Battery | Windows WMI, once per second; charge limit 80% |
| GPU setting | Clock lock with `nvidia-smi -lgc` (`-pl` is not supported on this GPU) |
| Job | Fixed number of 4096×4096 fp16 matrix multiplications (PyTorch, CUDA) |
| LLM (predictor) | Gemini 3.5 Flash Lite, via Google's cloud API (free tier); not run locally |

## Measurement rules

The first idle readings were about 70 W; the real idle power was about 28 W. Three effects were found and removed:

| Effect | Size | Fix |
|---|---|---|
| Battery was charging | +40 W | Charge limit 80%; every sample logs `bat_charging`, must be 0 |
| **Observer effect:** polling NVML wakes the sleeping GPU | +32 W (27.9 → 59.6 W, A/B test) | Idle is measured with `--no-gpu` |
| System went to Modern Standby | logging froze for 18 min | Recorder blocks sleep while running |

Idle is measured again in every session: 28.1 W on 18 Nov, 15.1 W on 22 Nov (cause not identified; display settings are now logged in `meta.json`).

**Net energy** = wall energy during the job − idle power × duration.

## Test 1: GPU clock → duration and energy (18 Nov)

20,000 multiplications, one run per setting:

| GPU clock | Duration | Wall power | GPU power | Net energy |
|---|---|---|---|---|
| default (~1660 MHz) | 115 s | 127 W | 84 W | 3.16 Wh |
| 1200 MHz | 160 s | 97 W | 50 W | **3.07 Wh** |
| 900 MHz | 213 s | 87 W | 40 W | 3.46 Wh |

- Duration scales with 1/clock: `duration ≈ 115.3 s × (work / 20000) × (1660 / MHz)`.
- Net energy is lowest near 1200 MHz; at 900 MHz the job runs so long that energy rises again. This matches the matrix-job result in `shelly-plug-test`.

## Test 2: LLM prediction vs measurement

**Method**

1. **19 Nov:** the LLM was asked to predict duration, wall power and net energy for two jobs that had **not been measured**. It was not asked to run them. Each question was asked 3 times at 3 information levels:
   - **A**: job description and GPU name only
   - **B**: + hardware facts, idle power, energy definitions
   - **C**: + the Test 1 measurements
2. **22 Nov:** both jobs were run and measured.
3. Predictions were compared with the measurements. The formula from Test 1 (no LLM) was added as a baseline.

**Measured**

| Job | Duration | Wall power | Net energy |
|---|---|---|---|
| 30,000 mult. @ 1500 MHz | 191.7 s | 116.8 W | 5.41 Wh |
| 10,000 mult. @ default | 57.3 s | 126.5 W | 1.77 Wh |

**Mean absolute error of the duration prediction**

| Method | 30,000 @ 1500 MHz | 10,000 @ default |
|---|---|---|
| Formula (no LLM) | **0.2%** | **0.6%** |
| LLM, level A | 76% | 75% |
| LLM, level B | 31% | 23% |
| LLM, level C | 17.5% | 0.6% |

**Observations**

- Without information, the LLM predicts the job about 4× too fast.
- More information helps (A → B → C), but the LLM struggles with a setting it has not seen (1500 MHz).
- In 1 of 6 level-B answers, the energy was miscalculated by a factor of 10.
- A simple formula fitted to measurements beats every LLM level.
- **Proposal:** let the LLM turn the job description into parameters, and let a measurement-based formula compute duration and energy.

### Prompt (English translation)

The prompt is built from blocks; level A uses blocks 3–4, level B adds block 1, level C adds blocks 1 and 2.

1. **Guideline (B, C):** Hardware and measurement context: laptop, NVIDIA GeForce RTX 3060 Laptop GPU (6 GB), Windows 11. GPU power limit: 70 W by default, up to ~85 W with Dynamic Boost. GPU core clock under load in default mode is about 1650 MHz; with a clock lock (e.g. 1200 MHz) the GPU runs at that fixed frequency, the VRAM clock does not change. Measurement: the laptop's wall power (plug meter); idle wall power is 28.1 W. Net energy = wall energy during the job − 28.1 W × duration. Physical frame: energy = power × time; for compute-bound jobs duration is roughly inversely proportional to frequency; power rises with frequency and voltage.
2. **Measured examples (C):** "Measured results of the same kind of job on this machine:" followed by the three Test 1 runs (iterations, clock, duration, mean wall power, net energy).
3. **Job:** "GPU: NVIDIA GeForce RTX 3060 Laptop GPU (laptop). Job: *N* 4096×4096 fp16 matrix multiplications with PyTorch (torch.matmul, CUDA). Setting: GPU core clock locked to *X* MHz (nvidia-smi -lgc)" or "default GPU clock (no lock)".
4. **Instruction:** "Do NOT run this job. Only predict how long it would take on this GPU, the mean wall power and the net energy. If information is missing, make reasonable assumptions and state them in the reasoning."

## Job profile (scheduler input)

The amount of work is given in the job's own unit (here: number of matrix multiplications), because duration and energy change with the GPU setting while the work does not. `results/job_profile.csv` lists duration and energy per job and GPU setting, with the same columns as `test/shelly-plug-test/results/job_profile.csv`, plus per-1000-unit columns for scaling to other job sizes.

Scaling check at default clock, per 1,000 multiplications:

| | 10,000 mult. (22 Nov) | 20,000 mult. (18 Nov) | Difference |
|---|---|---|---|
| Duration | 5.73 s | 5.77 s | 0.6% |
| Wall energy (gross) | 0.201 Wh | 0.203 Wh | 1% |
| Net energy (above idle) | 0.177 Wh | 0.158 Wh | 12% |

Duration and gross energy scale linearly with the amount of work, even across measurement days. Net energy does not, because idle power changed between the two days. For scheduling, gross energy may therefore be the more stable quantity.

## Limitations

- One LLM, two test jobs, one measurement per setting.
- Net energy depends on the idle reference. The formula's 11–13% net-energy error comes from idle changing between sessions (28.1 → 15.1 W); its duration and power errors are below 1%.
- Clock locking is an indirect power limit, not a watt limit.

## Structure

```
scripts/                          measurement, analysis and LLM scripts
data/reference_measurements.json  reference runs given to the LLM at level C
data/raw/                         one folder per recorder run, job log, LLM prediction logs, notes
results/                          summary tables computed from data/raw
```

## Scripts

| Script | Purpose |
|---|---|
| `recorder.py` | Logs wall power, GPU and battery every second; `--no-gpu` for idle |
| `matmul_job.py` | Runs a fixed number of matrix multiplications and logs start/end time |
| `analyze_job.py` | Computes duration and energy for one job from a recorder run |
| `build_summary.py` | Builds `results/measurement_summary.csv` from all runs; picks the same-day idle automatically |
| `llm_predict.py` | Asks the LLM for predictions (Gemini or Claude) and logs prompt + answer |
| `compare_predictions.py` | Compares predictions and the formula with measurements → `results/prediction_comparison.csv` |
| `job_profile.py` | Builds the job profile for the scheduler → `results/job_profile.csv` |

Run all commands from this folder:

```bash
# idle, then a job (start the recorder, then the job ~10 s later)
python scripts/recorder.py --ip <PLUG_IP> --label idle_nogpu --no-gpu --duration 200
python scripts/recorder.py --ip <PLUG_IP> --label matmul_default --duration 180
python scripts/matmul_job.py --iters 20000 --tag default

# LLM predictions (needs GEMINI_API_KEY)
python scripts/llm_predict.py --model gemini-3.5-flash-lite --iters 30000 --mhz 1500 --repeats 3

# rebuild the result tables
python scripts/build_summary.py
python scripts/compare_predictions.py
python scripts/job_profile.py
```

The clock lock (`nvidia-smi -lgc 1500,1500`, reset with `nvidia-smi -rgc`) needs an administrator PowerShell. Measurements and clock changes are started by a person, not by an agent (see `CONTRIBUTING.md`).

## Data notes

- Each run folder in `data/raw/` holds `samples.csv` (1 s samples), `events.jsonl` (errors, missed samples) and `meta.json` (settings and summary).
- `data/raw/notes.json` lists which runs are invalid and why. Invalid runs are kept but not used.
- **LLM prompts are in Turkish.** `llm_predictions.jsonl` stores the full prompt of every prediction exactly as sent, and the LLM's answer (also in Turkish). `llm_predict.py` keeps the prompt text and answer schema unchanged so repeat runs use the same experiment. The answer fields are `sure_s` (duration, s), `duvar_gucu_w` (wall power, W), `net_enerji_wh` (net energy, Wh) and `gerekce` (reasoning). An English translation is in *Prompt (English translation)* above.
- `llm_predictions_trial.jsonl` holds trial requests, including one with a faulty prompt (GPU name missing); these are not used in the results.
- The 18 and 22 Nov measurements were started by an AI coding agent on the user's request, before the rule in `CONTRIBUTING.md` existed. Later measurements are started by a person.
- **Cleanup before merging:** the machine name, plug address, plug MAC and GPU UUID were removed from run folder names, `meta.json` and `events.jsonl` (the repository is public), and Turkish free-text notes, one event message and the record keys of the LLM logs were translated to English. No measured value, prompt or LLM answer was changed; `samples.csv` files are byte-identical and all result tables were recomputed and checked to be identical. `recorder.py` no longer writes the removed fields.
