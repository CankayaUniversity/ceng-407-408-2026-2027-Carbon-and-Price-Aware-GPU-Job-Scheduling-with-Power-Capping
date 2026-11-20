import argparse
import csv
import math
import multiprocessing as mp
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone

import requests

DEVICE = "pc-1"
TEMP_LIMIT_C = 92

PHASE, TIME, WALL, GPU, TEMP, CLOCK, WORK = range(7)


def _cpu_burn(end_time):
    x = 0
    while time.time() < end_time:
        x = (x * 1103515245 + 12345) % 2147483647


class RunState:
    def __init__(self):
        self.phase = "cool"
        self.work_done = 0
        self.last_temp_c = 0
        self.samples = []
        self.gpu_name = "?"
        self.power_limit_w = 0.0


def read_plug(url):
    try:
        status = requests.get(url, timeout=0.8).json()
        return status["apower"], status["aenergy"]["total"]
    except (requests.RequestException, ValueError, KeyError):
        return None, None


def logger(file_name, label, url, stop, state):
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    state.gpu_name = name.decode() if isinstance(name, bytes) else name
    state.power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
    print(f"GPU: {state.gpu_name}   power limit: {state.power_limit_w:.0f} W")

    with open(file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ts_utc", "device", "label", "phase", "wall_w", "wall_energy_wh",
                         "gpu_power_w", "gpu_util_pct", "gpu_temp_c", "gpu_clock_mhz", "work_done"])
        while not stop.is_set():
            t0 = time.monotonic()
            wall_w, wall_energy = read_plug(url) if url else (None, None)
            try:
                gpu_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            except pynvml.NVMLError as exc:
                print("GPU read error:", exc)
                gpu_w = util = temp = clock = None
            if temp is not None:
                state.last_temp_c = temp
            writer.writerow([datetime.now(timezone.utc).isoformat(), DEVICE, label, state.phase,
                             wall_w, wall_energy, None if gpu_w is None else round(gpu_w, 2),
                             util, temp, clock, state.work_done])
            f.flush()
            state.samples.append((state.phase, time.monotonic(), wall_w, gpu_w, temp, clock, state.work_done))
            wall_text = f"{wall_w:6.1f} W" if wall_w is not None else "  ---   "
            gpu_text = f"{gpu_w:6.1f} W" if gpu_w is not None else "  ---  "
            print(f"[{state.phase:5}] plug {wall_text}   gpu {gpu_text}   {temp} °C   {clock} MHz   "
                  f"work {state.work_done}")
            time.sleep(max(0.0, 1.0 - (time.monotonic() - t0)))
    pynvml.nvmlShutdown()


def gpu_load(state, duration, target_work, fp16):
    import cupy as cp
    dtype = cp.float16 if fp16 else cp.float32
    n = 8192 if fp16 else 6144
    a = cp.random.random((n, n), dtype=cp.float32).astype(dtype)
    b = cp.random.random((n, n), dtype=cp.float32).astype(dtype)
    c = a @ b
    cp.cuda.Device().synchronize()

    state.phase = "load"
    start = time.monotonic()
    while True:
        if target_work is not None and state.work_done >= target_work:
            break
        if target_work is None and time.monotonic() - start >= duration:
            break
        if state.last_temp_c >= TEMP_LIMIT_C:
            print(f"GPU reached {TEMP_LIMIT_C} °C, load stopped.")
            break
        for _ in range(10):
            c = a @ b
        cp.cuda.Device().synchronize()
        state.work_done += 10
    elapsed = time.monotonic() - start
    del a, b, c
    cp.get_default_memory_pool().free_all_blocks()
    return elapsed, n


def run_command(state, command, log_name):
    state.phase = "load"
    start = time.monotonic()
    with open(log_name, "w", encoding="utf-8") as log:
        log.write(f"# command: {command}\n")
        log.flush()
        result = subprocess.run(command, shell=True, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - start
    if result.returncode == 0:
        state.work_done = 1
    else:
        print(f"WARNING: command exited with code {result.returncode}, see {log_name}")
    return elapsed


def energy_wh(samples, phase, column):
    energy, duration, last = 0.0, 0.0, None
    for s1, s2 in zip(samples, samples[1:]):
        if s1[column] is not None:
            last = s1[column]
        if s1[PHASE] == phase and last is not None:
            dt = s2[TIME] - s1[TIME]
            energy += last * dt
            duration += dt
    return energy / 3600, duration


def mean_of(samples, phase, column):
    values = [s[column] for s in samples if s[PHASE] == phase and s[column] is not None]
    return sum(values) / len(values) if values else float("nan")


def median_of(samples, phase, column):
    values = [s[column] for s in samples if s[PHASE] == phase and s[column] is not None]
    return statistics.median(values) if values else float("nan")


def round_or_blank(x):
    return "" if math.isnan(x) else round(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="test", help="name of this run: full, 1200mhz ...")
    parser.add_argument("--duration", type=int, default=120, help="fixed-time load (s)")
    parser.add_argument("--work", type=int, default=None, help="fixed-work load: number of matmuls")
    parser.add_argument("--idle", type=int, default=30, help="idle time before and after the load (s)")
    parser.add_argument("--cool-below", type=int, default=50,
                        help="wait until the GPU is below this temperature (0 = no wait)")
    parser.add_argument("--cpu", action="store_true", help="also put the CPU under full load")
    parser.add_argument("--fp16", action="store_true")
    plug = parser.add_mutually_exclusive_group(required=True)
    plug.add_argument("--plug-ip", help="IP address of the Shelly plug on the local network")
    plug.add_argument("--no-plug", action="store_true", help="run without reading the Shelly plug")
    parser.add_argument("--command", default=None, help="external command to run as the load (in quotes)")
    return parser.parse_args()


def run_phases(args, state, run_id):
    processes, elapsed, n, start_utc, end_utc = [], None, None, None, None
    try:
        if args.cool_below:
            while state.last_temp_c >= args.cool_below:
                print(f"   GPU {state.last_temp_c} °C, waiting until it is below {args.cool_below} °C...")
                time.sleep(5)
        state.phase = "idle1"
        print(f"\n--- {args.idle} s idle ---")
        time.sleep(args.idle)

        if args.command:
            kind = "external command"
        else:
            kind = f"{args.work} matmuls" if args.work else f"{args.duration} s"
        print(f"\n--- LOAD ({kind}, {'GPU + CPU' if args.cpu else 'GPU'}) starting: {time.strftime('%H:%M:%S')} ---")
        state.phase = "init"
        start_utc = datetime.now(timezone.utc)
        if args.cpu:
            cpu_s = args.duration if (args.work is None and not args.command) else 24 * 3600
            end_time = time.time() + cpu_s
            processes = [mp.Process(target=_cpu_burn, args=(end_time,))
                         for _ in range(max(1, mp.cpu_count() - 1))]
            for p in processes:
                p.start()
        if args.command:
            elapsed = run_command(state, args.command, f"command_{run_id}.log")
        else:
            elapsed, n = gpu_load(state, args.duration, args.work, args.fp16)
        for p in processes:
            p.terminate()
        end_utc = datetime.now(timezone.utc)

        state.phase = "idle2"
        print(f"\n--- load finished: {time.strftime('%H:%M:%S')}, {args.idle} s idle ---")
        time.sleep(args.idle)
    except KeyboardInterrupt:
        print("\nstopped")
        for p in processes:
            p.terminate()
    return elapsed, n, start_utc, end_utc


def compute_results(samples, url):
    idle_w = median_of(samples, "idle1", WALL)
    wall_wh, load_s = energy_wh(samples, "load", WALL)
    gpu_wh, _ = energy_wh(samples, "load", GPU)
    above_idle_wh = wall_wh - idle_w * load_s / 3600 if url else float("nan")
    max_temp = max((s[TEMP] for s in samples if s[TEMP] is not None), default=None)
    return {"idle_w": idle_w, "wall_wh": wall_wh, "gpu_wh": gpu_wh,
            "above_idle_wh": above_idle_wh, "max_temp_c": max_temp}


def print_summary(args, state, file_name, run_id, elapsed, n, r):
    samples, work = state.samples, state.work_done
    print("\n--- SUMMARY ---")
    print(f"file             : {file_name}")
    print(f"GPU              : {state.gpu_name}   power limit {state.power_limit_w:.0f} W")
    print(f"idle (before, median): plug {r['idle_w']:.1f} W   gpu {mean_of(samples, 'idle1', GPU):.1f} W")
    if not elapsed:
        return
    if args.command:
        print(f"load duration    : {elapsed:.1f} s   (external command, {'ok' if work else 'FAILED'}; "
              f"output: command_{run_id}.log)")
    else:
        print(f"load duration    : {elapsed:.1f} s   work: {work} matmuls ({n}x{n})   "
              f"rate: {work / elapsed:.2f} matmul/s")
    print(f"mean under load  : plug {mean_of(samples, 'load', WALL):.1f} W   gpu {mean_of(samples, 'load', GPU):.1f} W   "
          f"{mean_of(samples, 'load', CLOCK):.0f} MHz   max {r['max_temp_c'] or 0} °C")
    print(f"load energy      : plug {r['wall_wh']:.3f} Wh   (above idle {r['above_idle_wh']:.3f} Wh)   "
          f"gpu {r['gpu_wh']:.3f} Wh")
    if work and not args.command:
        print(f"per 1000 matmuls : plug {1000 * r['wall_wh'] / work:.3f} Wh   "
              f"above idle {1000 * r['above_idle_wh'] / work:.3f} Wh   gpu {1000 * r['gpu_wh'] / work:.3f} Wh")


def append_run_row(args, state, file_name, run_id, elapsed, start_utc, end_utc, r):
    new_file = not os.path.exists("runs.csv")
    mode = "command" if args.command else ("work" if args.work else "duration")
    with open("runs.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["run_id", "device", "label", "mode", "cpu_load", "start_utc", "end_utc",
                             "duration_s", "work_done", "idle_w", "wall_energy_wh", "above_idle_wh",
                             "gpu_energy_wh", "avg_clock_mhz", "max_temp_c", "log_file"])
        writer.writerow([run_id, DEVICE, args.label, mode, args.cpu,
                         start_utc.isoformat(), end_utc.isoformat() if end_utc else "", round(elapsed, 1),
                         state.work_done, round(r["idle_w"], 2), round(r["wall_wh"], 4),
                         round(r["above_idle_wh"], 4), round(r["gpu_wh"], 4),
                         round_or_blank(mean_of(state.samples, "load", CLOCK)),
                         "" if r["max_temp_c"] is None else r["max_temp_c"], file_name])
    print("Added one row to runs.csv.")


def main():
    args = parse_args()
    url = None if args.no_plug else f"http://{args.plug_ip}/rpc/Switch.GetStatus?id=0"
    if url and read_plug(url)[0] is None:
        print(f"Could not reach the plug ({args.plug_ip}). Check the IP or run with --no-plug.")
        return

    run_id = f"{DEVICE}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{args.label}"
    file_name = f"run_{run_id}.csv"
    state = RunState()
    stop = threading.Event()
    thread = threading.Thread(target=logger, args=(file_name, args.label, url, stop, state), daemon=True)
    thread.start()
    time.sleep(1.5)
    try:
        elapsed, n, start_utc, end_utc = run_phases(args, state, run_id)
    finally:
        stop.set()
        thread.join(timeout=3)

    r = compute_results(state.samples, url)
    print_summary(args, state, file_name, run_id, elapsed, n, r)
    if elapsed:
        append_run_row(args, state, file_name, run_id, elapsed, start_utc, end_utc, r)


if __name__ == "__main__":
    main()
