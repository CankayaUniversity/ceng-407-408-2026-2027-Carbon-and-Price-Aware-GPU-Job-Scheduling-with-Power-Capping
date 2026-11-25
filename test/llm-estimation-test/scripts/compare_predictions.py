"""Compares LLM predictions and a simple formula with the measurements.

Input:
  data/raw/llm_predictions.jsonl     LLM predictions (llm_predict.py)
  results/measurement_summary.csv    measurements (build_summary.py)
  data/reference_measurements.json   reference runs the formula is fitted to
Output:
  results/prediction_comparison.csv  one row per prediction, percent errors
  screen: mean absolute percent error (MAPE) per test and method

Formula (baseline without an LLM), from the reference runs:
  duration = T_ref x (iters / iters_ref) x (f_ref / f)
  power and net energy per iteration = linear interpolation between clocks
The effective default clock is DEFAULT_CLOCK_MHZ (18 Nov run, ~1660 MHz).

Example:
  python scripts/compare_predictions.py --model gemini-3.5-flash-lite
"""

import argparse
import csv
import json
import os
import statistics as st

DEFAULT_CLOCK_MHZ = 1660.0

# Answer fields of the (Turkish) prediction schema -> English names.
ANSWER_FIELDS = {"duration_s": "sure_s", "wall_w": "duvar_gucu_w",
                 "net_wh": "net_enerji_wh"}


def normalize(rec):
  """Records from 19 Nov use Turkish keys; newer ones use English keys."""
  return {"level": rec.get("level", rec.get("seviye")),
          "prediction": rec.get("prediction", rec.get("tahmin")),
          "model": rec["model"], "iters": rec["iters"], "mhz": rec["mhz"]}


def interpolate(points, f):
  """points: [(mhz, value)], linear inter-/extrapolation by clock."""
  p = sorted(points)
  for (f0, v0), (f1, v1) in zip(p, p[1:]):
    if f0 <= f <= f1:
      return v0 + (v1 - v0) * (f - f0) / (f1 - f0)
  (f0, v0), (f1, v1) = (p[0], p[1]) if f < p[0][0] else (p[-2], p[-1])
  return v0 + (v1 - v0) * (f - f0) / (f1 - f0)


def formula(ref, iters, mhz):
  f = mhz or DEFAULT_CLOCK_MHZ
  by_clock = [((r["mhz"] or DEFAULT_CLOCK_MHZ), r) for r in ref]
  default = next(r for r in ref if r["mhz"] == 0)
  duration = (default["duration_s"] * iters / default["iters"]
              * DEFAULT_CLOCK_MHZ / f)
  w = interpolate([(fc, r["wall_w"]) for fc, r in by_clock], f)
  e_iter = interpolate([(fc, r["net_wh"] / r["iters"]) for fc, r in by_clock],
                       f)
  return {"duration_s": duration, "wall_w": w, "net_wh": e_iter * iters}


def error(pred, actual):
  return None if pred is None else (pred - actual) / actual * 100


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--model", default="gemini-3.5-flash-lite")
  p.add_argument("--predictions", default="data/raw/llm_predictions.jsonl")
  p.add_argument("--measurements", default="results/measurement_summary.csv")
  p.add_argument("--reference", default="data/reference_measurements.json")
  p.add_argument("--out", default="results/prediction_comparison.csv")
  a = p.parse_args()

  with open(a.reference, encoding="utf-8") as f:
    ref = json.load(f)
  with open(a.measurements, encoding="utf-8") as f:
    measured = [r for r in csv.DictReader(f)
                if r["kind"] == "job" and r["valid"] == "True"]
  with open(a.predictions, encoding="utf-8") as f:
    preds = [r for r in map(normalize, map(json.loads, f))
             if r["prediction"] and r["model"].startswith(a.model)]

  # Reference runs are not tests (the LLM was given their answers).
  ref_keys = {(r["iters"], r["mhz"]) for r in ref}
  tests = sorted({(t["iters"], t["mhz"]) for t in preds} - ref_keys)

  rows = []
  for iters, mhz in tests:
    g = [r for r in measured if int(r["iters"]) == iters and int(r["mhz"]) == mhz]
    if not g:
      print(f"! {iters} @ {mhz}: no measurement, skipped")
      continue
    g = g[-1]
    actual = {"duration_s": float(g["duration_s"]),
              "wall_w": float(g["wall_w_mean"]),
              "net_wh": float(g["net_wh"])}
    candidates = [("formula", formula(ref, iters, mhz))] + [
        (t["level"], {k: t["prediction"][v] for k, v in ANSWER_FIELDS.items()})
        for t in preds if (t["iters"], t["mhz"]) == (iters, mhz)]
    for method, t in candidates:
      rows.append({
          "test": f"{iters} @ {'default' if mhz == 0 else f'{mhz} MHz'}",
          "method": method, "measured_run": g["run_id"],
          "idle_ref_w": g["idle_ref_w"],
          **{f"{k}_pred": round(t[k], 3) for k in actual},
          **{f"{k}_actual": actual[k] for k in actual},
          **{f"{k}_error_pct": round(error(t[k], actual[k]), 1)
             for k in actual},
      })

  os.makedirs(os.path.dirname(a.out), exist_ok=True)
  with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

  print(f"{len(rows)} rows -> {a.out}\n")
  print("Mean absolute percent error (MAPE), [best .. worst] repeat")
  for test in dict.fromkeys(s["test"] for s in rows):
    print(f"\n{test}")
    for m in ["formula", "A", "B", "C"]:
      ss = [s for s in rows if s["test"] == test and s["method"] == m]
      if not ss:
        continue
      parts = []
      for k, name in [("duration_s", "duration"), ("wall_w", "power"),
                      ("net_wh", "net Wh")]:
        h = [abs(s[f"{k}_error_pct"]) for s in ss]
        parts.append(f"{name} {st.mean(h):6.1f}% [{min(h):.0f}..{max(h):.0f}]")
      print(f"  {m:8s} n={len(ss)}  " + " | ".join(parts))


if __name__ == "__main__":
  main()
