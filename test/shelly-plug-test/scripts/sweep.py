import argparse
import csv
import os
import random
import subprocess
import sys
import time
from datetime import datetime

DEFAULT_PLAN = "sweep_plan.csv"
CONTROL_WORK = 1990
CONTROL_HISTORY = "control_history.csv"
PLAN_FIELDS = ["order", "label", "clock", "repeat", "status", "run_id"]


def nvidia_smi(*args):
    result = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr).strip()


def lock_clock(clock):
    if clock == "full":
        return nvidia_smi("-rgc")
    return nvidia_smi("-lgc", f"{clock},{clock}")


def last_run_row():
    if not os.path.exists("runs.csv"):
        return None
    with open("runs.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def run_one(label, clock, work, plug_ip, command=None):
    code, output = lock_clock(clock)
    if code != 0 or "not supported" in output.lower():
        raise RuntimeError(f"Could not apply clock setting ({clock}): {output}")
    print(f"\n========== {label}  (clock: {clock}) ==========")
    try:
        load = ["--command", command] if command else ["--work", str(work)]
        result = subprocess.run([sys.executable, "measure_run.py", "--label", label, "--plug-ip", plug_ip, *load])
    finally:
        nvidia_smi("-rgc")
    if result.returncode != 0:
        raise RuntimeError(f"Run {label} failed (code {result.returncode})")

    row = last_run_row()
    if row is None or row["label"] != label:
        raise RuntimeError(f"{label}: row not found in runs.csv")
    if clock != "full" and row["avg_clock_mhz"]:
        deviation = abs(float(row["avg_clock_mhz"]) - int(clock)) / int(clock)
        if deviation > 0.05:
            print(f"WARNING: {label} average clock {row['avg_clock_mhz']} MHz, target {clock} MHz. "
                  f"The lock may not have held.")
    return row


def control_run(reference, tolerance, plug_ip):
    row = run_one(f"control_{datetime.now():%m%d_%H%M}", "full", CONTROL_WORK, plug_ip)
    energy = float(row["wall_energy_wh"])
    deviation = 100 * (energy / reference - 1)
    passed = abs(deviation) <= tolerance
    new_file = not os.path.exists(CONTROL_HISTORY)
    with open(CONTROL_HISTORY, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["run_id", "start_utc", "duration_s", "idle_w", "wall_energy_wh",
                             "reference_wh", "deviation_pct", "passed"])
        writer.writerow([row["run_id"], row["start_utc"], row["duration_s"], row["idle_w"], energy,
                         reference, round(deviation, 2), passed])
    print(f"\n*** CONTROL: {energy:.3f} Wh, reference {reference:.2f} Wh, deviation {deviation:+.1f}% "
          f"(limit ±{tolerance:g}%) -> {'PASSED' if passed else 'FAILED'} ***")
    return passed


def write_plan(path, plan):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(plan)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plug-ip", required=True, help="IP address of the Shelly plug on the local network")
    parser.add_argument("--work", type=int, default=1990)
    parser.add_argument("--command", default=None,
                        help="external command to run instead of the matmul job (e.g. llama-bench); "
                             "the control run still uses the matmul job")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--clocks", nargs="+", default=["900", "1200", "1500", "1800", "full"])
    parser.add_argument("--seed", type=int, default=42, help="fixed seed for the shuffled order")
    parser.add_argument("--reference", type=float, default=4.74, help="expected energy of the control run (Wh)")
    parser.add_argument("--tolerance", type=float, default=5.0, help="accepted deviation (percent)")
    parser.add_argument("--control-only", action="store_true", help="only do the control run")
    parser.add_argument("--plan", default=DEFAULT_PLAN, help="plan file; use a new name for each new sweep")
    parser.add_argument("--force", action="store_true", help="continue the sweep even if the control run fails")
    return parser.parse_args()


def check_admin():
    code, output = nvidia_smi("-lgc", "1200,1200")
    nvidia_smi("-rgc")
    if code != 0 or "not supported" in output.lower() or "insufficient" in output.lower():
        sys.exit(f"Could not lock the clock. Did you open PowerShell as ADMINISTRATOR?\n{output}")


def load_or_create_plan(path, clocks, repeats, seed):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            plan = list(csv.DictReader(f))
        done = sum(p["status"] == "done" for p in plan)
        print(f"Existing plan found: {done}/{len(plan)} runs done, continuing.")
        return plan
    runs = [(c, r) for c in clocks for r in range(1, repeats + 1)]
    random.Random(seed).shuffle(runs)
    return [{"order": i + 1, "label": f"f{c}_r{r}", "clock": c, "repeat": r, "status": "pending", "run_id": ""}
            for i, (c, r) in enumerate(runs)]


def run_plan(args, plan, remaining):
    start = time.monotonic()
    try:
        print("\n### Control + warm-up run (not part of the results) ###")
        if not control_run(args.reference, args.tolerance, args.plug_ip) and not args.force:
            print("Control run failed; sweep not started. Find the cause and run again "
                  "(or use --force to continue anyway).")
            return
        if args.command:
            print("\n### External command warm-up run (loads the model file; not part of the results) ###")
            run_one(f"warmup_command_{datetime.now():%H%M}", "full", args.work, args.plug_ip, args.command)

        for i, p in enumerate(remaining, 1):
            print(f"\n### {i}/{len(remaining)}  elapsed {(time.monotonic() - start) / 60:.0f} min ###")
            row = run_one(p["label"], p["clock"], args.work, args.plug_ip, args.command)
            p["status"], p["run_id"] = "done", row["run_id"]
            write_plan(args.plan, plan)
            print(f"   -> {p['label']}: {float(row['duration_s']):.0f} s, {float(row['wall_energy_wh']):.3f} Wh")
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue where you left off.")
    finally:
        nvidia_smi("-rgc")
        print("Clock lock removed.")
    print(f"\nTotal time: {(time.monotonic() - start) / 60:.0f} min. Results: runs.csv, plan: {args.plan}")


def main():
    args = parse_args()
    if not os.path.exists("measure_run.py"):
        sys.exit("measure_run.py is not in this folder.")
    check_admin()

    if args.control_only:
        try:
            control_run(args.reference, args.tolerance, args.plug_ip)
        finally:
            nvidia_smi("-rgc")
        return

    plan = load_or_create_plan(args.plan, args.clocks, args.repeats, args.seed)
    write_plan(args.plan, plan)
    remaining = [p for p in plan if p["status"] != "done"]
    job = f"external command: {args.command}" if args.command else f"{args.work} matmuls"
    print(f"Total {len(plan)} runs, {len(remaining)} remaining. Job: {job}")
    print("Order:", ", ".join(p["label"] for p in remaining))
    run_plan(args, plan, remaining)


if __name__ == "__main__":
    main()
