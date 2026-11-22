import argparse
import csv
import time
from datetime import datetime, timezone
import requests

DEVICE = "pc-1"
DURATION_S = 300
BATTERY_THRESHOLD_MW = 500


def connect_battery():
    try:
        import wmi
    except ImportError:
        print("wmi package not found, battery will not be logged.")
        return None
    connection = wmi.WMI(namespace="root\\wmi")
    try:
        read_battery(connection)
    except (wmi.x_wmi, IndexError) as exc:
        print("Battery could not be read, it will not be logged:", exc)
        return None
    return connection


def read_battery(connection):
    battery = connection.BatteryStatus()[0]
    if battery.Charging:
        state = "Charging"
    elif battery.Discharging:
        state = "Discharging"
    else:
        state = "Idle"
    return state, float(battery.ChargeRate or 0), float(battery.DischargeRate or 0)


def read_plug(url):
    status = requests.get(url, timeout=2).json()
    return status["apower"], status["voltage"], status["current"], status["aenergy"]["total"]


def log_loop(file_name, battery, url):
    result = {"rows": 0, "errors": 0, "powers": [], "times": [],
              "energy_start": None, "energy_end": None,
              "battery_active_s": 0, "battery_in_wh": 0.0, "battery_out_wh": 0.0}
    end_time = time.monotonic() + DURATION_S

    with open(file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ts_utc", "device", "apower_w", "voltage_v", "current_a", "aenergy_wh",
                         "batt_status", "batt_charge_mw", "batt_discharge_mw"])
        try:
            while time.monotonic() < end_time:
                t0 = time.monotonic()
                try:
                    power, voltage, current, energy = read_plug(url)
                except (requests.RequestException, ValueError, KeyError) as exc:
                    result["errors"] += 1
                    print("read error:", exc)
                else:
                    state = charge_mw = discharge_mw = None
                    if battery:
                        state, charge_mw, discharge_mw = read_battery(battery)
                        if charge_mw > BATTERY_THRESHOLD_MW or discharge_mw > BATTERY_THRESHOLD_MW:
                            result["battery_active_s"] += 1
                        result["battery_in_wh"] += charge_mw / 1000 / 3600
                        result["battery_out_wh"] += discharge_mw / 1000 / 3600

                    writer.writerow([datetime.now(timezone.utc).isoformat(), DEVICE,
                                     power, voltage, current, energy, state, charge_mw, discharge_mw])
                    f.flush()
                    result["rows"] += 1
                    result["powers"].append(power)
                    result["times"].append(time.monotonic())
                    if result["energy_start"] is None:
                        result["energy_start"] = energy
                    result["energy_end"] = energy

                    if battery:
                        print(f"{power:8.1f} W   battery: {state}  charge {charge_mw} mW  discharge {discharge_mw} mW")
                    else:
                        print(f"{power:8.1f} W")
                time.sleep(max(0.0, 1.0 - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            print("\nstopped")
    return result


def print_summary(file_name, result, battery):
    powers, times = result["powers"], result["times"]
    print("\n--- SUMMARY ---")
    print(f"file                : {file_name}")
    print(f"rows read           : {result['rows']}   (errors: {result['errors']})")
    if powers:
        print(f"power (W)           : min {min(powers):.1f}   mean {sum(powers) / len(powers):.1f}   max {max(powers):.1f}")
        print(f"energy, plug meter  : {result['energy_end'] - result['energy_start']:.2f} Wh")
        energy_wh = sum(p * (t2 - t1) for p, t1, t2 in zip(powers, times, times[1:])) / 3600
        gaps = sum(1 for t1, t2 in zip(times, times[1:]) if t2 - t1 > 1.5)
        print(f"energy, power x time: {energy_wh:.2f} Wh")
        print(f"data gaps           : {gaps} intervals longer than 1.5 s")
    if battery:
        print(f"battery active      : {result['battery_active_s']} s   "
              f"(into battery {result['battery_in_wh']:.2f} Wh, out of battery {result['battery_out_wh']:.2f} Wh)")
        if result["battery_active_s"] == 0:
            print("The battery stayed idle during the test: all energy came from the plug.")
        else:
            print("Warning: the battery was active in some seconds; evaluate that part separately.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plug-ip", required=True, help="IP address of the Shelly plug on the local network")
    args = parser.parse_args()
    url = f"http://{args.plug_ip}/rpc/Switch.GetStatus?id=0"
    battery = connect_battery()
    file_name = f"test_{DEVICE}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.csv"
    result = log_loop(file_name, battery, url)
    print_summary(file_name, result, battery)


if __name__ == "__main__":
    main()
