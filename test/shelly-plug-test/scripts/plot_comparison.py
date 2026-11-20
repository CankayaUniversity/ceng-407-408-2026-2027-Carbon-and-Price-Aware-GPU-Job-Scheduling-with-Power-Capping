import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
SERIES = [("matmul_summary.csv", "Matrix job (compute-bound)", "#2b6cb0", "o"),
          ("llm_summary.csv", "Language model job (memory-bound)", "#c05621", "s")]
OUTPUT = "comparison_plot.png"


def draw_series(axes, path, name, color, marker):
    summary = pd.read_csv(path).sort_values("mhz")
    duration_diff = 100 * (summary.duration_ratio - 1)
    for ax, values in [(axes[0], summary["wall_energy_wh_diff_pct"]), (axes[1], duration_diff)]:
        ax.plot(summary.mhz, values, color=color, lw=2, marker=marker, ms=6.5, markeredgecolor="white",
                markeredgewidth=1.3, label=name, zorder=3)
        above = marker == "s"
        for x, v in zip(summary.mhz, values):
            ax.annotate(f"{v:+.0f}%", (x, v), textcoords="offset points", xytext=(-6, 7) if above else (6, -13),
                        ha="right" if above else "left", fontsize=7.5, color=INK)


def style_axis(ax, title):
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlabel("GPU clock (MHz)  ·  rightmost points: unlocked full speed", fontsize=8, color=MUTED)
    ax.set_ylabel("% difference", fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.margins(x=0.08, y=0.15)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    for path, name, color, marker in SERIES:
        draw_series(axes, path, name, color, marker)
    style_axis(axes[0], "Wall energy vs. full speed")
    style_axis(axes[1], "Job duration vs. full speed")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, "RTX 4060 Laptop · 3 runs per clock, mean · same amount of work", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(OUTPUT)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
