"""Builds one summary table (CSV) from all recorder runs in data/raw/.

Each recorder run becomes one row:
  - idle runs : mean/median wall power (the first seconds are dropped)
  - job runs  : duration, power, gross/net/GPU energy (same as analyze_job)

Net energy uses the MEAN of the latest VALID idle run taken on the same day
before the job (energy = mean power x duration; the median is only reported
as a noise-robust check). The idle run used is written in the row.

Validity and reasons are read from data/raw/notes.json (hand-written).

Example:
  python scripts/build_summary.py        -> results/measurement_summary.csv
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics as st

from analyze_job import load_samples, num, ts, window_analysis

COLUMNS = [
    "run_id", "start_utc", "kind", "label", "valid", "reason",
    "job_tag", "iters", "mhz", "duration_s", "tflops",
    "wall_w_mean", "wall_w_median", "wall_w_p10", "wall_w_p90",
    "gpu_w_mean", "gpu_temp_mean", "gpu_temp_max", "gpu_clk_mean",
    "gpu_clk_reasons", "idle_ref_w", "idle_ref_run",
    "gross_wh", "net_wh", "gpu_wh", "net_wh_per_1000_iters",
    "charging_samples", "discharging_samples", "gaps_s", "comment",
]


def mhz_of(tag):
  """lgc1200 -> 1200, default -> 0 (default clock)."""
  m = re.search(r"(\d{3,4})", tag)
  return int(m.group(1)) if m else 0


def idle_row(meta, rows, note):
  skip = note.get("idle_skip_s", 0)
  w = sorted(num(r["wall_w"]) for r in rows
             if num(r["wall_w"]) is not None
             and float(r["t_mono_s"]) >= skip)
  s = meta.get("summary", {})
  return {
      "kind": "idle", "duration_s": s.get("duration_s"),
      "wall_w_mean": round(st.mean(w), 2) if w else None,
      "wall_w_median": round(st.median(w), 2) if w else None,
      "wall_w_p10": w[int(len(w) * 0.1)] if w else None,
      "wall_w_p90": w[int(len(w) * 0.9)] if w else None,
      "charging_samples": s.get("bat_charging_samples"),
      "discharging_samples": s.get("bat_discharging_samples"),
      "gaps_s": sum(s.get("wall_gaps_excluded_s") or []),
      "comment": f"first {skip} s dropped" if skip else "",
  }


def job_row(rows, job, idle):
  idle_w = idle["wall_w_mean"] if idle else 0.0
  a = window_analysis(rows, job, idle_w)
  return {
      "kind": "job", "job_tag": job["tag"], "iters": job["iters"],
      "mhz": mhz_of(job["tag"]), "duration_s": a["duration_s"],
      "tflops": a["tflops"],
      "wall_w_mean": a["wall_w_mean_median"][0],
      "wall_w_median": a["wall_w_mean_median"][1],
      "gpu_w_mean": a["gpu_w_mean_median"][0],
      "gpu_temp_mean": a["gpu_temp_c_mean_median"][0],
      "gpu_temp_max": a["gpu_temp_c_max"],
      "gpu_clk_mean": a["gpu_clk_mhz_mean_median"][0],
      "gpu_clk_reasons": " ".join(a["gpu_clk_reasons"]),
      "idle_ref_w": idle_w if idle else None,
      "idle_ref_run": idle["run_id"] if idle else "NONE (net = gross)",
      "gross_wh": a["wall_wh_gross"], "net_wh": a["wall_wh_net"],
      "gpu_wh": a["gpu_wh"], "net_wh_per_1000_iters": a["net_wh_per_1000_iters"],
      "charging_samples": a["bat_charging_samples"],
      "discharging_samples": a["bat_discharging_samples"],
  }


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--raw", default="data/raw")
  p.add_argument("--out", default="results/measurement_summary.csv")
  a = p.parse_args()

  with open(f"{a.raw}/notes.json", encoding="utf-8") as f:
    notes = json.load(f)
  with open(f"{a.raw}/job_log.jsonl", encoding="utf-8") as f:
    jobs = [json.loads(l) for l in f if l.strip()]

  out, last_idle = [], None
  for meta_path in sorted(glob.glob(f"{a.raw}/*/meta.json")):
    run_dir = os.path.dirname(meta_path)
    run_id = os.path.basename(run_dir)
    with open(meta_path, encoding="utf-8") as f:
      meta = json.load(f)
    note = notes.get(run_id, {})
    rows = load_samples(run_dir)
    t0 = ts(meta["start_utc"])
    t1 = ts(meta.get("end_utc") or meta["start_utc"])
    base = {"run_id": run_id, "start_utc": meta["start_utc"],
            "label": meta["label"], "valid": note.get("valid", True),
            "reason": note.get("reason", "")}

    # Did a job start inside this recorder run?
    job = next((j for j in jobs if t0 <= ts(j["start_utc"]) <= t1), None)
    if job is None:
      row = {**base, **idle_row(meta, rows, note)}
      if row["valid"]:
        last_idle = row
    else:
      same_day = (last_idle and last_idle["start_utc"][:10]
                  == meta["start_utc"][:10])
      row = {**base, **job_row(rows, job, last_idle if same_day else None)}
    out.append(row)

  os.makedirs(os.path.dirname(a.out), exist_ok=True)
  with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLUMNS)
    w.writeheader()
    for s in out:
      w.writerow({k: s.get(k, "") for k in COLUMNS})
  print(f"{len(out)} rows -> {a.out}")


if __name__ == "__main__":
  main()
