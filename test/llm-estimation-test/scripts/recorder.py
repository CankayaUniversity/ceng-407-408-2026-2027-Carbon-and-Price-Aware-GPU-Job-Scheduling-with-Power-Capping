"""Measurement recorder: wall power (Shelly), GPU (NVML) and battery.

Each run creates its own folder:
  <outdir>/<run_id>/samples.csv  -> one row per sample
  <outdir>/<run_id>/events.jsonl -> errors, missed samples
  <outdir>/<run_id>/meta.json    -> GPU, plug model, settings, summary

Sources are independent: if the plug, GPU or battery cannot be read, its
columns stay empty, logging continues and the error goes to events.jsonl.

Privacy (the repository is public): the machine name, plug address, plug MAC
and GPU UUID are never written. The plug address is logged as "<plug>".

Examples:
  python scripts/recorder.py --ip <PLUG> --label idle_nogpu --no-gpu --duration 200
  python scripts/recorder.py --label no_plug_test --duration 10    (without plug)
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
import platform
import subprocess
import sys
import threading
import time
import warnings

import psutil
import requests

with warnings.catch_warnings():
  warnings.simplefilter("ignore", FutureWarning)
  import pynvml

PLUG_PLACEHOLDER = "<plug>"


def utc_now():
  return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_args():
  p = argparse.ArgumentParser(description="Measurement recorder")
  p.add_argument("--ip", default=None,
                 help="Plug IP or hostname. Without it, wall power is not "
                      "measured.")
  p.add_argument("--label", required=True,
                 help="Run label, e.g. idle_nogpu, matmul_1200")
  p.add_argument("--interval", type=float, default=1.0,
                 help="Sampling interval (s)")
  p.add_argument("--duration", type=float, default=0,
                 help="Duration (s). 0 = until Ctrl+C")
  p.add_argument("--outdir", default="data/raw", help="Output root folder")
  p.add_argument("--gpu-index", type=int, default=0)
  p.add_argument("--no-gpu", action="store_true",
                 help="Do not query the GPU (NVML) at all. Use for idle runs: "
                      "polling wakes the sleeping GPU and inflates idle power.")
  p.add_argument("--note", default="",
                 help="Free-text note: power mode, workload, conditions, ...")
  return p.parse_args()


class EventLog:
  """Writes errors and events as JSON lines and counts them per source.

  Any text in `redact` (e.g. the plug address) is replaced by "<plug>".
  """

  def __init__(self, path, redact=()):
    self.f = open(path, "a", encoding="utf-8", buffering=1)
    self.counts = {}
    self.lock = threading.Lock()
    self.redact = [r for r in redact if r]

  def log(self, kind, source, msg, **extra):
    msg = str(msg)
    for r in self.redact:
      msg = msg.replace(r, PLUG_PLACEHOLDER)
    with self.lock:
      key = f"{kind}:{source}"
      self.counts[key] = self.counts.get(key, 0) + 1
      rec = {"t_utc": utc_now(), "kind": kind, "source": source, "msg": msg,
             **extra}
      self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")

  def close(self):
    self.f.close()


# --------------------------------------------------------------------------
# Shelly plug (Gen3, RPC API)
# --------------------------------------------------------------------------
class Shelly:

  # Device-info fields that do not identify the device.
  INFO_KEEP = ("model", "gen", "fw_id", "ver", "app")

  def __init__(self, ip, timeout, events):
    self.base = f"http://{ip}/rpc"
    self.timeout = timeout
    self.events = events
    self.session = requests.Session()

  def device_info(self):
    try:
      r = self.session.get(f"{self.base}/Shelly.GetDeviceInfo",
                           timeout=self.timeout)
      r.raise_for_status()
      d = r.json()
      return {k: d[k] for k in self.INFO_KEEP if k in d}
    except Exception as e:  # for meta only; logging still starts
      self.events.log("error", "shelly", f"GetDeviceInfo: {e!r}")
      return None

  def read(self):
    t0 = time.monotonic()
    out = {"wall_w": None, "wall_v": None, "wall_a": None,
           "wall_energy_wh": None, "shelly_latency_ms": None, "shelly_ok": 0}
    try:
      r = self.session.get(f"{self.base}/Switch.GetStatus?id=0",
                           timeout=self.timeout)
      r.raise_for_status()
      d = r.json()
      # Write None, not 0, for a missing field: missing data must not look
      # like a real 0 W reading.
      out["wall_w"] = d.get("apower")
      out["wall_v"] = d.get("voltage")
      out["wall_a"] = d.get("current")
      out["wall_energy_wh"] = (d.get("aenergy") or {}).get("total")
      if out["wall_w"] is None:
        self.events.log("error", "shelly", "no apower in response",
                        keys=list(d.keys()))
      else:
        out["shelly_ok"] = 1
    except Exception as e:
      self.events.log("error", "shelly", repr(e))
    out["shelly_latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
    return out

  def close(self):
    self.session.close()


# --------------------------------------------------------------------------
# GPU (NVML)
# --------------------------------------------------------------------------
class Gpu:

  def __init__(self, index, events, enabled=True):
    self.events = events
    self.ok = False
    self.prev_energy = None  # (t_mono, mJ)
    self.reasons_fn = None
    if not enabled:
      # On an Optimus laptop every NVML query wakes the sleeping GPU to P0 and
      # adds ~30 W at the wall (measured: 27.9 W -> 59.6 W). Off for idle runs.
      return
    try:
      pynvml.nvmlInit()
      self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
      self.ok = True
    except Exception as e:
      events.log("error", "gpu", f"nvmlInit: {e!r}")
    self.reasons_fn = getattr(
        pynvml, "nvmlDeviceGetCurrentClocksEventReasons",
        getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None))

  @staticmethod
  def _s(v):
    return v.decode("utf-8") if isinstance(v, bytes) else v

  def _try(self, name, fn):
    try:
      return fn()
    except Exception as e:
      self.events.log("error", "gpu", f"{name}: {e!r}")
      return None

  def meta(self):
    if not self.ok:
      return None
    h, t = self.h, self._try
    m = {
        "name": t("name", lambda: self._s(pynvml.nvmlDeviceGetName(h))),
        "driver": t("driver",
                    lambda: self._s(pynvml.nvmlSystemGetDriverVersion())),
        "default_limit_w": t("default_limit", lambda: pynvml.
                             nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000),
        "enforced_limit_w": t("enforced_limit", lambda: pynvml.
                              nvmlDeviceGetEnforcedPowerLimit(h) / 1000),
    }
    c = t("limit_constraints",
          lambda: pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h))
    if c:
      m["min_limit_w"], m["max_limit_w"] = c[0] / 1000, c[1] / 1000
    # Usually "Not Supported" on laptops: the limit cannot be set with -pl.
    try:
      m["settable_limit_w"] = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000
    except Exception as e:
      m["settable_limit_w"] = f"ERR {e}"
    return m

  def read(self, t_mono):
    keys = ["gpu_w_sensor", "gpu_w_from_energy", "gpu_energy_j",
            "gpu_limit_w", "gpu_temp_c", "gpu_util_pct", "gpu_mem_util_pct",
            "gpu_clk_mhz", "gpu_mem_clk_mhz", "gpu_pstate",
            "gpu_clk_reasons", "gpu_ok"]
    out = dict.fromkeys(keys)
    out["gpu_ok"] = 0
    if not self.ok:
      return out
    h, t = self.h, self._try
    w = t("power", lambda: pynvml.nvmlDeviceGetPowerUsage(h))
    out["gpu_w_sensor"] = None if w is None else w / 1000
    e = t("energy", lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
    if e is not None:
      out["gpu_energy_j"] = e / 1000
      # The instant sensor can stick or return junk on this machine; mean
      # power from the energy counter is an independent check.
      if self.prev_energy and t_mono > self.prev_energy[0]:
        out["gpu_w_from_energy"] = round(
            (e - self.prev_energy[1]) / 1000 / (t_mono - self.prev_energy[0]),
            3)
      self.prev_energy = (t_mono, e)
    lim = t("enforced_limit", lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(h))
    out["gpu_limit_w"] = None if lim is None else lim / 1000
    out["gpu_temp_c"] = t("temp", lambda: pynvml.nvmlDeviceGetTemperature(
        h, pynvml.NVML_TEMPERATURE_GPU))
    u = t("util", lambda: pynvml.nvmlDeviceGetUtilizationRates(h))
    if u is not None:
      out["gpu_util_pct"], out["gpu_mem_util_pct"] = u.gpu, u.memory
    out["gpu_clk_mhz"] = t("clk", lambda: pynvml.nvmlDeviceGetClockInfo(
        h, pynvml.NVML_CLOCK_GRAPHICS))
    out["gpu_mem_clk_mhz"] = t("memclk", lambda: pynvml.nvmlDeviceGetClockInfo(
        h, pynvml.NVML_CLOCK_MEM))
    out["gpu_pstate"] = t("pstate",
                          lambda: pynvml.nvmlDeviceGetPerformanceState(h))
    if self.reasons_fn:
      r = t("reasons", lambda: self.reasons_fn(h))
      out["gpu_clk_reasons"] = None if r is None else hex(r)
    out["gpu_ok"] = int(w is not None or e is not None)
    return out

  def close(self):
    if self.ok:
      try:
        pynvml.nvmlShutdown()
      except Exception:
        pass


# --------------------------------------------------------------------------
# Battery (Windows WMI root\wmi BatteryStatus)
# --------------------------------------------------------------------------
_PS_BATTERY = r"""
$ErrorActionPreference = 'Stop'
while ($true) {
  try {
    $b = Get-CimInstance -Namespace root\wmi -ClassName BatteryStatus |
         Select-Object -First 1
    $o = @{ ok = $true; online = $b.PowerOnline; charging = $b.Charging;
            discharging = $b.Discharging; charge_mw = $b.ChargeRate;
            discharge_mw = $b.DischargeRate;
            remaining_mwh = $b.RemainingCapacity; voltage_mv = $b.Voltage }
  } catch { $o = @{ ok = $false; err = $_.Exception.Message } }
  [Console]::Out.WriteLine(($o | ConvertTo-Json -Compress))
  [Console]::Out.Flush()
  Start-Sleep -Milliseconds __MS__
}
"""


class Battery:
  """The WMI query takes ~100 ms, so it runs in a separate, persistent
  PowerShell process; the main loop only takes the latest value and its age."""

  def __init__(self, interval, events):
    self.events = events
    self.latest = None  # (t_mono, dict)
    self.proc = None
    if sys.platform != "win32":
      return
    try:
      self.proc = subprocess.Popen(
          ["powershell", "-NoProfile", "-NonInteractive", "-Command",
           _PS_BATTERY.replace("__MS__", str(int(interval * 1000)))],
          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
      threading.Thread(target=self._reader, daemon=True).start()
    except Exception as e:
      events.log("error", "battery", f"could not start PowerShell: {e!r}")

  def _reader(self):
    for line in self.proc.stdout:
      try:
        d = json.loads(line)
      except ValueError:
        continue
      if not d.get("ok"):
        self.events.log("error", "battery", d.get("err"))
      self.latest = (time.monotonic(), d)

  def static_meta(self):
    m = {}
    try:
      b = psutil.sensors_battery()
      m["psutil_percent"] = None if b is None else b.percent
      m["psutil_plugged"] = None if b is None else b.power_plugged
    except Exception as e:
      m["psutil_err"] = repr(e)
    if sys.platform == "win32":
      m["full_charged_mwh"] = _ps_value(
          "(Get-CimInstance -Namespace root\\wmi -ClassName "
          "BatteryFullChargedCapacity | Select-Object -First 1)"
          ".FullChargedCapacity")
      m["design_mwh"] = _ps_value(
          "(Get-CimInstance -Namespace root\\wmi -ClassName "
          "BatteryStaticData | Select-Object -First 1).DesignedCapacity")
    return m

  def read(self, t_mono):
    keys = ["bat_online", "bat_charging", "bat_discharging", "bat_charge_mw",
            "bat_discharge_mw", "bat_remaining_mwh", "bat_age_s"]
    out = dict.fromkeys(keys)
    if self.latest is None:
      return out
    t, d = self.latest
    if d.get("ok"):
      for k in ("online", "charging", "discharging"):
        v = d.get(k)
        out[f"bat_{k}"] = None if v is None else int(bool(v))
      out["bat_charge_mw"] = d.get("charge_mw")
      out["bat_discharge_mw"] = d.get("discharge_mw")
      out["bat_remaining_mwh"] = d.get("remaining_mwh")
    out["bat_age_s"] = round(t_mono - t, 2)
    return out

  def close(self):
    if self.proc:
      self.proc.terminate()


def _ps_value(expr):
  try:
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                        "-Command", expr], capture_output=True, text=True,
                       timeout=15)
    return r.stdout.strip() or None
  except Exception as e:
    return f"ERR {e!r}"


def environment_info():
  """Settings that affect idle power. Idle was 28.1 W on 18 Nov and 15.1 W on
  22 Nov; the settings were not logged then, so the gap is unexplained."""
  if sys.platform != "win32":
    return None
  return {
      "brightness_pct": _ps_value(
          "(Get-CimInstance -Namespace root\\wmi -ClassName "
          "WmiMonitorBrightness).CurrentBrightness"),
      "power_scheme": _ps_value("powercfg /getactivescheme"),
      "power_mode_overlay": _ps_value(
          "(Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\"
          "Power\\User\\PowerSchemes').ActiveOverlayAcPowerScheme"),
      "refresh_hz": _ps_value(
          "(Get-CimInstance Win32_VideoController | Where-Object "
          "CurrentRefreshRate | Select-Object -First 1).CurrentRefreshRate"),
  }


def time_sync_status():
  """Clock sync status (query only, no admin needed)."""
  if sys.platform != "win32":
    return None
  try:
    r = subprocess.run(["w32tm", "/query", "/status"], capture_output=True,
                       text=True, timeout=10)
    return r.stdout.strip() or r.stderr.strip()
  except Exception as e:
    return f"ERR {e!r}"


# --------------------------------------------------------------------------
# Summary / energy
# --------------------------------------------------------------------------
def summarize(rows, interval):
  """Wall energy with two independent methods:
  1) difference of the plug counter (aenergy.total) -> main value
  2) trapezoidal integral of apower -> check; gaps longer than 3*interval
     are left out of the integral and reported.
  """
  s = {"n_samples": len(rows)}
  if not rows:
    return s
  s["duration_s"] = round(rows[-1]["t_mono_s"] - rows[0]["t_mono_s"], 3)

  def first_last(key):
    v = [(r["t_mono_s"], r[key]) for r in rows if r[key] is not None]
    return (v[0], v[-1]) if len(v) >= 2 else None

  fl = first_last("wall_energy_wh")
  s["wall_wh_counter"] = None if not fl else round(fl[1][1] - fl[0][1], 4)

  pts = [(r["t_mono_s"], r["wall_w"]) for r in rows if r["wall_w"] is not None]
  wh, covered, gaps = 0.0, 0.0, []
  for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
    dt = t1 - t0
    if dt > 3 * interval:
      gaps.append(round(dt, 2))
      continue
    wh += (p0 + p1) / 2 * dt / 3600
    covered += dt
  s["wall_wh_trapz"] = round(wh, 4) if len(pts) >= 2 else None
  s["wall_trapz_covered_s"] = round(covered, 2)
  s["wall_gaps_excluded_s"] = gaps
  s["wall_samples_ok"] = len(pts)
  if pts:
    w = [p for _, p in pts]
    s["wall_w_mean"] = round(sum(w) / len(w), 2)
    s["wall_w_max"] = max(w)

  fl = first_last("gpu_energy_j")
  s["gpu_wh_counter"] = None if not fl else round(
      (fl[1][1] - fl[0][1]) / 3600, 4)

  fl = first_last("bat_remaining_mwh")
  s["bat_delta_mwh"] = None if not fl else fl[1][1] - fl[0][1]
  dis = [r["bat_discharge_mw"] for r in rows if r["bat_discharge_mw"]]
  s["bat_discharging_samples"] = sum(1 for r in rows if r["bat_discharging"])
  s["bat_discharge_mw_max"] = max(dis) if dis else 0
  s["bat_charging_samples"] = sum(1 for r in rows if r["bat_charging"])

  late = [r["slot_late_ms"] for r in rows]
  s["slot_late_ms_max"] = max(late)
  s["slot_late_ms_mean"] = round(sum(late) / len(late), 2)
  return s


# --------------------------------------------------------------------------
def keep_awake(on):
  """Blocks sleep and display-off while recording (Windows).

  Sleep freezes logging, and a dark display lowers wall power by ~7 W. Keeps
  every run under the same condition (display on). Ends with the process.
  """
  if sys.platform != "win32":
    return
  import ctypes
  es_continuous, es_system, es_display = 0x80000000, 0x1, 0x2
  flags = es_continuous | (es_system | es_display if on else 0)
  ctypes.windll.kernel32.SetThreadExecutionState(flags)


def main():
  args = parse_args()
  keep_awake(True)
  start_utc = utc_now()
  run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + f"_{args.label}")
  run_dir = os.path.join(args.outdir, run_id)
  os.makedirs(run_dir, exist_ok=False)  # never append to an existing run

  events = EventLog(os.path.join(run_dir, "events.jsonl"), redact=[args.ip])
  shelly = (Shelly(args.ip, min(0.8 * args.interval, 1.5), events)
            if args.ip else None)
  gpu = Gpu(args.gpu_index, events, enabled=not args.no_gpu)
  battery = Battery(args.interval, events)
  psutil.cpu_percent(interval=None)  # the first call always returns 0

  logged_args = {**vars(args), "ip": PLUG_PLACEHOLDER if args.ip else None}
  meta = {
      "run_id": run_id, "label": args.label, "note": args.note,
      "start_utc": start_utc, "args": logged_args,
      "os": platform.platform(), "python": sys.version,
      "cpu_logical": psutil.cpu_count(),
      "gpu": gpu.meta(),
      "shelly": shelly.device_info() if shelly else "not used",
      "battery": battery.static_meta(),
      "time_sync_start": time_sync_status(),
      "environment": environment_info(),
  }
  meta_path = os.path.join(run_dir, "meta.json")

  def write_meta():
    with open(meta_path, "w", encoding="utf-8") as f:
      json.dump(meta, f, ensure_ascii=False, indent=2)

  write_meta()  # keep the start info even if the run is interrupted

  cols = (["seq", "t_utc", "t_mono_s", "slot_late_ms", "wall_w", "wall_v",
           "wall_a", "wall_energy_wh", "shelly_latency_ms", "shelly_ok"]
          + ["gpu_w_sensor", "gpu_w_from_energy", "gpu_energy_j",
             "gpu_limit_w", "gpu_temp_c", "gpu_util_pct", "gpu_mem_util_pct",
             "gpu_clk_mhz", "gpu_mem_clk_mhz", "gpu_pstate",
             "gpu_clk_reasons", "gpu_ok"]
          + ["bat_online", "bat_charging", "bat_discharging", "bat_charge_mw",
             "bat_discharge_mw", "bat_remaining_mwh", "bat_age_s", "cpu_pct"])

  print(f"[*] Run: {run_dir}")
  print(f"[*] Plug: {'yes' if args.ip else 'NO'} | GPU: "
        f"{(meta['gpu'] or {}).get('name', 'NO')} | Stop: Ctrl+C\n")

  # Short wait so the battery reader produces its first value
  for _ in range(30):
    if battery.latest or battery.proc is None:
      break
    time.sleep(0.1)

  rows, seq, missed = [], 0, 0
  stop_reason = "duration"
  t_start = time.monotonic()
  k = 0
  try:
    with open(os.path.join(run_dir, "samples.csv"), "w", newline="",
              encoding="utf-8", buffering=1) as f:
      w = csv.DictWriter(f, fieldnames=cols)
      w.writeheader()
      while True:
        # Fixed schedule: target = start + k*interval (no drift)
        target = t_start + k * args.interval
        now = time.monotonic()
        if now < target:
          time.sleep(target - now)
        elif now - target >= args.interval:
          skip = int((now - target) // args.interval)
          missed += skip
          events.log("missed", "loop", f"{skip} sample slots missed",
                     late_s=round(now - target, 3))
          k += skip
          target = t_start + k * args.interval
        if args.duration and target - t_start >= args.duration:
          break

        t_mono = time.monotonic()
        row = {"seq": seq, "t_utc": utc_now(),
               "t_mono_s": round(t_mono - t_start, 4),
               "slot_late_ms": round((t_mono - target) * 1000, 1)}
        if shelly:
          row.update(shelly.read())
        else:
          row.update({c: None for c in cols[4:10]})
        row.update(gpu.read(time.monotonic()))
        row.update(battery.read(time.monotonic()))
        row["cpu_pct"] = psutil.cpu_percent(interval=None)

        w.writerow(row)
        rows.append(row)
        seq += 1
        k += 1
        print(f"[{row['t_utc'][11:23]}] Wall {row['wall_w']} W | "
              f"GPU sensor {row['gpu_w_sensor']} W / energy "
              f"{row['gpu_w_from_energy']} W | {row['gpu_util_pct']}% | "
              f"Bat {row['bat_remaining_mwh']} mWh discharge "
              f"{row['bat_discharge_mw']} mW")
  except KeyboardInterrupt:
    stop_reason = "ctrl+c"
  except Exception as e:
    stop_reason = f"error: {e!r}"
    events.log("fatal", "main", repr(e))
    raise
  finally:
    meta["end_utc"] = utc_now()
    meta["stop_reason"] = stop_reason
    meta["missed_slots"] = missed
    meta["event_counts"] = events.counts
    meta["time_sync_end"] = time_sync_status()
    meta["summary"] = summarize(rows, args.interval)
    write_meta()
    keep_awake(False)
    battery.close()
    gpu.close()
    if shelly:
      shelly.close()
    events.close()
    print(f"\n[*] Done ({stop_reason}). Summary:")
    print(json.dumps(meta["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
