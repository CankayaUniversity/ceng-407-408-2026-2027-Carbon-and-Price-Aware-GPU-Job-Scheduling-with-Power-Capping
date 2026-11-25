"""Computes the energy of one job from the job window in a recorder run.

The job's start_utc/end_utc (job_log.jsonl) select rows of samples.csv.
  Gross energy = integral of wall power over the job window
  Net energy   = gross - idle_w * duration   (the job's own share)

Example:
  python scripts/analyze_job.py --run data/raw/<run_id> --tag default --idle-w 15.1

The same calculation is used by build_summary.py (window_analysis).
"""

import argparse
import csv
from datetime import datetime
import json
import statistics as st


def ts(s):
  return datetime.fromisoformat(s).timestamp()


def num(v):
  return None if v in (None, "") else float(v)


def load_samples(run_dir):
  with open(f"{run_dir}/samples.csv", encoding="utf-8") as f:
    return list(csv.DictReader(f))


def window_analysis(rows, job, idle_w):
  """rows: samples.csv rows; job: job_log record; idle_w: W."""
  t0, t1 = ts(job["start_utc"]), ts(job["end_utc"])
  win = [r for r in rows if t0 <= ts(r["t_utc"]) <= t1]
  pts = [(ts(r["t_utc"]), num(r["wall_w"])) for r in win
         if num(r["wall_w"]) is not None]

  # Trapezoidal integral; window edges are closed with the nearest sample.
  wh = sum((p0 + p1) / 2 * (b - a) for (a, p0), (b, p1) in zip(pts, pts[1:]))
  wh += pts[0][1] * (pts[0][0] - t0) + pts[-1][1] * (t1 - pts[-1][0])
  wh /= 3600
  dur = t1 - t0

  def stat(key):
    v = [num(r[key]) for r in win if num(r[key]) is not None]
    return (round(st.mean(v), 1), round(st.median(v), 1)) if v else (None, None)

  gpu_e = [num(r["gpu_energy_j"]) for r in win if num(r["gpu_energy_j"])]
  reasons = sorted({r["gpu_clk_reasons"] for r in win if r["gpu_clk_reasons"]})
  out = {
      "tag": job["tag"], "iters": job["iters"], "duration_s": round(dur, 1),
      "tflops": job["tflops"], "n_samples": len(win),
      "wall_w_mean_median": stat("wall_w"),
      "gpu_w_mean_median": stat("gpu_w_from_energy"),
      "gpu_temp_c_mean_median": stat("gpu_temp_c"),
      "gpu_temp_c_max": max((num(r["gpu_temp_c"]) for r in win
                             if num(r["gpu_temp_c"]) is not None),
                            default=None),
      "gpu_clk_mhz_mean_median": stat("gpu_clk_mhz"),
      "gpu_clk_reasons": reasons,
      "wall_wh_gross": round(wh, 3),
      "wall_wh_net": round(wh - idle_w * dur / 3600, 3),
      "gpu_wh": round((gpu_e[-1] - gpu_e[0]) / 3600, 3) if gpu_e else None,
      "bat_charging_samples": sum(1 for r in win if r["bat_charging"] == "1"),
      "bat_discharging_samples": sum(1 for r in win
                                     if r["bat_discharging"] == "1"),
  }
  out["net_wh_per_1000_iters"] = round(
      out["wall_wh_net"] / job["iters"] * 1000, 4)
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--run", required=True, help="Recorder run folder")
  p.add_argument("--tag", required=True, help="Tag in the job log")
  p.add_argument("--log", default="data/raw/job_log.jsonl")
  p.add_argument("--idle-w", type=float, required=True,
                 help="Idle wall power measured in the same session (W)")
  a = p.parse_args()
  with open(a.log, encoding="utf-8") as f:
    job = [j for j in map(json.loads, f) if j["tag"] == a.tag][-1]
  out = window_analysis(load_samples(a.run), job, a.idle_w)
  print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
