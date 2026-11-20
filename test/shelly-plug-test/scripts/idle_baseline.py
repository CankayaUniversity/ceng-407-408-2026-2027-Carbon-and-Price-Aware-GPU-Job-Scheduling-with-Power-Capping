import argparse
import csv
import os
import statistics
import time
from datetime import datetime, timezone

import requests

DEVICE = "pc-1"
SETTLE_S = 120
SUMMARY_FILE = "idle_baseline_summary.csv"


def read_plug(url):
    try:
        status = requests.get(url, timeout=0.8).json()
        return status["apower"]
    except (requests.RequestException, ValueError, KeyError):
        return None


def connect_gpu():
    try:
        import pynvml
    except ImportError:
        print("pynvml not found; only the plug will be logged.")
        return None, None
    try:
        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError as exc:
        print("GPU cannot be read; only the plug will be logged:", exc)
        return None, None


def read_gpu(pynvml, handle):
    if pynvml is None:
        return None, None
    try:
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, pynvml.nvmlDeviceGetTemperature(handle, 0)
    except pynvml.NVMLError:
        return None, None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plug-ip", required=True, help="IP address of the Shelly plug on the local network")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--minutes", type=int, default=10, help="length of each block (minutes)")
    parser.add_argument("--gap", type=int, default=2, help="wait between blocks (minutes)")
    return parser.parse_args()


def measure_block(writer, block, minutes, start_utc, file_name, pynvml, handle, url):
    t_start = time.monotonic()
    walls, gpus, missing = [], [], 0
    while time.monotonic() - t_start < minutes * 60:
        t0 = time.monotonic()
        wall_w = read_plug(url)
        gpu_w, temp = read_gpu(pynvml, handle)
        settled = (t0 - t_start) >= SETTLE_S
        writer.writerow([datetime.now(timezone.utc).isoformat(), DEVICE, block, settled, wall_w,
                         None if gpu_w is None else round(gpu_w, 2), temp])
        if wall_w is None:
            missing += 1
        elif settled:
            walls.append(wall_w)
            if gpu_w is not None:
                gpus.append(gpu_w)
        elapsed = int(t0 - t_start)
        if elapsed % 60 == 0:
            gpu_text = f"   gpu {gpu_w:.1f} W" if gpu_w else ""
            print(f"   {elapsed // 60:2d}. min   plug {wall_w if wall_w is not None else '---'} W{gpu_text}")
        time.sleep(max(0.0, 1.0 - (time.monotonic() - t0)))
    if not walls:
        return None
    return {
        "start_utc": start_utc.isoformat(), "block": block, "n_s": len(walls), "missing_s": missing,
        "wall_mean_w": round(statistics.mean(walls), 2),
        "wall_median_w": round(statistics.median(walls), 2),
        "wall_sd_w": round(statistics.stdev(walls), 2) if len(walls) > 1 else 0,
        "wall_min_w": min(walls), "wall_max_w": max(walls),
        "gpu_mean_w": round(statistics.mean(gpus), 2) if gpus else "",
        "file": file_name,
    }


def append_summary(rows):
    new_file = not os.path.exists(SUMMARY_FILE)
    with open(SUMMARY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, file_name):
    medians = [r["wall_median_w"] for r in rows]
    print("--- SUMMARY ---")
    for r in rows:
        print(f"Block {r['block']}: median {r['wall_median_w']} W   mean {r['wall_mean_w']} W   "
              f"range {r['wall_min_w']}–{r['wall_max_w']} W")
    print(f"\nIDLE BASELINE (median of block medians): {statistics.median(medians):.1f} W")
    if len(medians) > 1:
        print(f"Difference between blocks: {max(medians) - min(medians):.1f} W")
    print(f"Log: {file_name}, summary: {SUMMARY_FILE}")


def main():
    args = parse_args()
    url = f"http://{args.plug_ip}/rpc/Switch.GetStatus?id=0"
    if read_plug(url) is None:
        print(f"Could not reach the plug ({args.plug_ip}).")
        return
    pynvml, handle = connect_gpu()

    start_utc = datetime.now(timezone.utc)
    file_name = f"idle_baseline_{start_utc:%Y%m%d_%H%M%S}.csv"
    total_min = args.blocks * args.minutes + (args.blocks - 1) * args.gap
    print(f"Idle baseline: {args.blocks} blocks x {args.minutes} min, gaps {args.gap} min. Total ~{total_min} min. "
          f"Expected end: {time.strftime('%H:%M', time.localtime(time.time() + total_min * 60))}")
    print("Do not touch the computer.\n")

    rows = []
    with open(file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ts_utc", "device", "block", "settled", "wall_w", "gpu_power_w", "gpu_temp_c"])
        try:
            for block in range(1, args.blocks + 1):
                if block > 1:
                    print(f"   gap: {args.gap} min")
                    time.sleep(args.gap * 60)
                print(f"--- Block {block}/{args.blocks} started: {time.strftime('%H:%M:%S')} ---")
                row = measure_block(writer, block, args.minutes, start_utc, file_name, pynvml, handle, url)
                f.flush()
                if row:
                    rows.append(row)
                    print(f"   Block {block}: mean {row['wall_mean_w']} W, median {row['wall_median_w']} W, "
                          f"sd {row['wall_sd_w']} W ({row['n_s']} s after settling)\n")
        except KeyboardInterrupt:
            print("\nstopped")

    if rows:
        append_summary(rows)
        print_summary(rows, file_name)


if __name__ == "__main__":
    main()
