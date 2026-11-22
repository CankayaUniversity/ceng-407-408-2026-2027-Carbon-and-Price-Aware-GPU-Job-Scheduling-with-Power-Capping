import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

GAP_LIMIT_S = 5

INK, MUTED, GRID, MARK = "#1f1f1f", "#6b6b6b", "#e6e6e6", "#2b6cb0"


def read_run(path):
    d = pd.read_csv(path, parse_dates=["ts_utc"])
    d["dt"] = d.ts_utc.diff().shift(-1).dt.total_seconds()
    d["wall_i"] = d.wall_w.ffill()
    load = d[d.phase == "load"]
    return {
        "duration_s": load.dt.sum(),
        "wall_energy_wh": (load.wall_i * load.dt).sum() / 3600,
        "gpu_energy_wh": (load.gpu_power_w * load.dt).sum() / 3600,
        "mean_wall_w": load.wall_w.mean(),
        "idle_w": d[d.phase == "idle1"].wall_w.median(),
        "max_temp_c": load.gpu_temp_c.max(),
        "mean_clock_mhz": load.gpu_clock_mhz.mean(),
        "max_gap_s": load.dt.iloc[:-1].max(),
        "work_done": d.work_done.max(),
    }


def load_runs(folder, plan_file):
    plan = pd.read_csv(f"{folder}/{plan_file}", dtype=str)
    plan = plan[plan.status == "done"]
    rows = []
    for _, p in plan.iterrows():
        row = read_run(f"{folder}/run_{p.run_id}.csv")
        row.update(label=p["label"], clock=p["clock"], repeat=int(p["repeat"]), order=int(p["order"]))
        rows.append(row)
    return pd.DataFrame(rows)


def drop_invalid(runs):
    invalid = runs[runs.max_gap_s > GAP_LIMIT_S]
    if len(invalid):
        print("Invalid (logging gap):", ", ".join(invalid.label))
    return runs[runs.max_gap_s <= GAP_LIMIT_S].copy()


def add_energy_columns(runs):
    idle_w = runs.idle_w.median()
    longest_s = runs.groupby("clock").duration_s.mean().max()
    runs["above_idle_wh"] = runs.wall_energy_wh - idle_w * runs.duration_s / 3600
    runs["equal_window_wh"] = runs.wall_energy_wh + idle_w * (longest_s - runs.duration_s) / 3600
    runs["mhz"] = runs.mean_clock_mhz.where(runs.clock == "full", pd.to_numeric(runs.clock, errors="coerce"))
    return runs.sort_values(["mhz", "repeat"]), idle_w, longest_s


def summarize(runs):
    summary = runs.groupby("clock").agg(
        mhz=("mhz", "mean"), n=("label", "size"),
        duration_s=("duration_s", "mean"), duration_sd_s=("duration_s", "std"),
        wall_energy_wh=("wall_energy_wh", "mean"), wall_energy_sd_wh=("wall_energy_wh", "std"),
        above_idle_wh=("above_idle_wh", "mean"), equal_window_wh=("equal_window_wh", "mean"),
        gpu_energy_wh=("gpu_energy_wh", "mean"), mean_wall_w=("mean_wall_w", "mean"), max_temp_c=("max_temp_c", "max"),
    ).sort_values("mhz")
    ref = summary.loc["full"]
    for col in ["wall_energy_wh", "above_idle_wh", "equal_window_wh", "gpu_energy_wh"]:
        summary[col + "_diff_pct"] = 100 * (summary[col] / ref[col] - 1)
    summary["duration_ratio"] = summary.duration_s / ref.duration_s
    return summary


def plot(runs, summary, path, job_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    for ax, col, sd, title, unit in [
        (axes[0], "wall_energy_wh", "wall_energy_sd_wh", "Wall energy for the same job", "Wh"),
        (axes[1], "duration_s", "duration_sd_s", "Duration of the same job", "seconds"),
    ]:
        ax.scatter(runs.mhz, runs[col], s=22, color=MARK, alpha=0.35, linewidths=0, zorder=2)
        ax.errorbar(summary.mhz, summary[col], yerr=summary[sd], color=MARK, lw=2, marker="o", ms=7,
                    capsize=4, zorder=3, markeredgecolor="white", markeredgewidth=1.5)
        for _, r in summary.iterrows():
            ax.annotate(f"{r[col]:.2f}" if unit == "Wh" else f"{r[col]:.0f}",
                        (r.mhz, r[col]), textcoords="offset points", xytext=(9, -3),
                        ha="left", fontsize=8, color=INK)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("GPU clock (MHz)  ·  rightmost point: unlocked full speed", fontsize=8, color=MUTED)
        ax.set_ylabel(unit, fontsize=9, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.margins(y=0.18)
    job_name = job_name or f"{int(runs['work_done'].max())} matrix multiplications"
    fig.text(0.01, 0.01, f"RTX 4060 Laptop · {job_name} · {int(summary.n.max())} runs per clock "
             f"(light points), line = mean ± sd", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".")
    parser.add_argument("--plan", default="sweep_plan.csv")
    parser.add_argument("--prefix", default="sweep", help="prefix of the output files")
    parser.add_argument("--out-dir", default=None, help="folder for the output files (default: --folder)")
    parser.add_argument("--job-name", default=None, help="job description under the figure")
    return parser.parse_args()


def main():
    args = parse_args()
    out = f"{args.out_dir or args.folder}/{args.prefix}"
    runs = drop_invalid(load_runs(args.folder, args.plan))
    runs, idle_w, longest_s = add_energy_columns(runs)
    runs.round(4).to_csv(f"{out}_runs.csv", index=False)
    summary = summarize(runs)
    summary.round(3).to_csv(f"{out}_summary.csv")

    pd.set_option("display.width", 200)
    print(f"Idle power (median): {idle_w:.1f} W   equal window: {longest_s:.0f} s\n")
    print(summary[["n", "duration_s", "duration_ratio", "wall_energy_wh", "wall_energy_sd_wh",
                   "wall_energy_wh_diff_pct", "above_idle_wh_diff_pct", "equal_window_wh_diff_pct",
                   "gpu_energy_wh_diff_pct", "max_temp_c"]].round(2).to_string())
    plot(runs, summary, f"{out}_plot.png", args.job_name)
    print(f"\nSaved: {args.prefix}_runs.csv, {args.prefix}_summary.csv, {args.prefix}_plot.png")


if __name__ == "__main__":
    main()
