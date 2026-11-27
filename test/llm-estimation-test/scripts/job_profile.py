"""Job profile: duration and energy per job and GPU setting (scheduler input).

The amount of work is given in the job's own unit (here: number of matrix
multiplications), because duration and energy change with the GPU setting
while the work does not.

Columns match test/shelly-plug-test/results/job_profile.csv, so the scheduler
can read both files the same way. Extra columns:
  work_unit, work_amount   unit and amount of work
  *_per_1k                 duration/energy per 1000 work units (for scaling)
  idle_w                   idle power used for above_idle_wh
  source_runs              recorder run(s) the row is based on

Input : results/measurement_summary.csv  (build_summary.py)
Output: results/job_profile.csv

Example:
  python scripts/job_profile.py
"""

import argparse
import csv
from collections import defaultdict
import statistics as st

COLUMNS = [
    "job", "setting", "clock_mhz", "n_runs", "duration_s", "duration_sd_s",
    "duration_h", "energy_wh", "energy_sd_wh", "energy_kwh", "above_idle_wh",
    "avg_wall_w", "max_gpu_temp_c", "job_description",
    "work_unit", "work_amount", "duration_s_per_1k", "energy_wh_per_1k",
    "above_idle_wh_per_1k", "idle_w", "source_runs",
]


def sd(v):
  return round(st.stdev(v), 3) if len(v) > 1 else ""


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--measurements", default="results/measurement_summary.csv")
  p.add_argument("--out", default="results/job_profile.csv")
  a = p.parse_args()

  with open(a.measurements, encoding="utf-8") as f:
    jobs = [r for r in csv.DictReader(f)
            if r["kind"] == "job" and r["valid"] == "True"]

  # Same amount of work + same setting = repeats; summarized together.
  groups = defaultdict(list)
  for r in jobs:
    groups[(int(r["iters"]), int(r["mhz"]))].append(r)

  rows = []
  for (iters, mhz), rr in sorted(groups.items(),
                                 key=lambda x: (-x[0][0], x[0][1] or 9e9)):
    dur = [float(r["duration_s"]) for r in rr]
    gross = [float(r["gross_wh"]) for r in rr]
    net = [float(r["net_wh"]) for r in rr]
    k = iters / 1000
    rows.append({
        "job": f"matmul4096fp16_{iters}",
        "setting": "full" if mhz == 0 else f"lgc_{mhz}",
        "clock_mhz": round(st.mean(float(r["gpu_clk_mean"]) for r in rr)),
        "n_runs": len(rr),
        "duration_s": round(st.mean(dur), 1), "duration_sd_s": sd(dur),
        "duration_h": round(st.mean(dur) / 3600, 5),
        "energy_wh": round(st.mean(gross), 3), "energy_sd_wh": sd(gross),
        "energy_kwh": round(st.mean(gross) / 1000, 6),
        "above_idle_wh": round(st.mean(net), 3),
        "avg_wall_w": round(st.mean(float(r["wall_w_mean"]) for r in rr), 1),
        "max_gpu_temp_c": max(int(float(r["gpu_temp_max"])) for r in rr),
        "job_description": f"{iters} x 4096x4096 fp16 matrix multiplications "
                           "(PyTorch, synthetic, compute-bound), RTX 3060 Laptop",
        "work_unit": "matmul_4096_fp16", "work_amount": iters,
        "duration_s_per_1k": round(st.mean(dur) / k, 3),
        "energy_wh_per_1k": round(st.mean(gross) / k, 4),
        "above_idle_wh_per_1k": round(st.mean(net) / k, 4),
        "idle_w": " ".join(sorted({r["idle_ref_w"] for r in rr})),
        "source_runs": " ".join(r["run_id"] for r in rr),
    })

  with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLUMNS)
    w.writeheader()
    w.writerows(rows)
  print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
  main()
