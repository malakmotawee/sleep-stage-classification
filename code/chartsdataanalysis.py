import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

PALETTE = {
    "primary":   "#6C5CE7",   # purple
    "secondary": "#00B8A9",   # teal
    "accent":    "#FF6B6B",   # coral
    "warm":      "#FDCB6E",   # amber
    "dark":      "#2D3436",   # near-black
    "muted":     "#B2BEC3",   # grey
    "light":     "#DFE6E9",
    "bg":        "#F7F9FC",
}

STAGE_COLORS = {
    "Wake": "#FF6B6B",
    "N1":   "#FDCB6E",
    "N2":   "#74B9FF",
    "N3":   "#0984E3",
    "N4":   "#6C5CE7",
    "REM":  "#00B8A9",
    "Unscored": "#B2BEC3",
    "NREM": "#0984E3",
    "Sleep": "#00B8A9",
}

OUT = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}")

def chart_dataset_summary():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.suptitle("Dataset Summary", fontsize=22, fontweight="bold", y=0.98)

    cards = [
        ("31",          "Subjects",            PALETTE["primary"]),
        ("26,773",      "Total Epochs",        PALETTE["secondary"]),
        ("254,425",     "HR Readings",         PALETTE["accent"]),
        ("51.8M",       "Accel. Samples",      PALETTE["warm"]),
        ("≈223 h",      "Labelled Time",       PALETTE["primary"]),
        ("30 s",        "Epoch Length",        PALETTE["secondary"]),
    ]

    for i, (value, label, color) in enumerate(cards):
        row = i // 3
        col = i % 3
        x = col * 4 + 0.3
        y = 2.6 - row * 2.4
        box = FancyBboxPatch((x, y), 3.4, 1.9,
                             boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor=color, edgecolor="none", alpha=0.95)
        ax.add_patch(box)
        ax.text(x + 1.7, y + 1.15, value, ha="center", va="center",
                fontsize=26, fontweight="bold", color="white")
        ax.text(x + 1.7, y + 0.45, label, ha="center", va="center",
                fontsize=13, color="white", alpha=0.95)

    save(fig, "01_dataset_summary.png")


def chart_heart_rate():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle("Heart Rate Data", fontsize=20, fontweight="bold", y=1.02)

    ax.barh([0], [185], left=[30], color=PALETTE["accent"], alpha=0.85,
            height=0.45, label="Normal range")
    ax.barh([0], [35], left=[180], color=PALETTE["dark"], alpha=0.7,
            height=0.45, label="Outliers (~0.1%)")

    ax.scatter([30, 215], [0, 0], color="white", s=120, zorder=5,
               edgecolors=PALETTE["dark"], linewidths=2)
    ax.text(30, 0.45, "30 bpm\n(min)", ha="center", fontsize=11, fontweight="bold")
    ax.text(215, 0.45, "215 bpm\n(max)", ha="center", fontsize=11, fontweight="bold")
    ax.text(180, -0.45, "180 bpm\n(outlier threshold)", ha="center",
            fontsize=10, color=PALETTE["dark"])

    ax.set_xlim(0, 240)
    ax.set_ylim(-0.8, 0.9)
    ax.set_yticks([])
    ax.set_xlabel("Heart Rate (bpm)")
    ax.set_title("PPG-derived heart rate · 2,379–18,015 records per subject · interpolated to 1 Hz",
                 fontsize=12, fontweight="normal", pad=12)
    ax.legend(loc="lower right", frameon=False)
    ax.spines["left"].set_visible(False)

    save(fig, "02_heart_rate.png")

def chart_accelerometer():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Accelerometer (50 Hz)", fontsize=20, fontweight="bold", y=1.02)

    axes_lbl = ["X (lateral)", "Y (vertical)", "Z (longitudinal)"]
    means    = [-0.098, -0.054, -0.218]
    stds     = [ 0.347,  0.453,  0.784]
    mins     = [-5.514, -10.553, -16.041]
    maxs     = [ 7.173,  7.198,   8.423]
    colors   = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]

    x = np.arange(3)
    for i, (m, s, c) in enumerate(zip(means, stds, colors)):
        ax1.bar(i, 2 * s, bottom=m - s, color=c, alpha=0.30,
                width=0.55, edgecolor=c, linewidth=1.2)
        ax1.errorbar(i, m, yerr=s, fmt="none", ecolor=c, capsize=12,
                     capthick=2.2, elinewidth=2.2, zorder=4)
        ax1.scatter(i, m, color=c, s=200, zorder=5,
                    edgecolors="white", linewidths=2.5)
        ax1.text(i, m + s + 0.07, f"σ = {s:.3f}", ha="center", fontsize=10,
                 color=c, fontweight="bold")
        ax1.text(i, m - s - 0.13, f"μ = {m:.3f}", ha="center", fontsize=10,
                 color=c, fontweight="bold")

    ax1.axhline(0, color=PALETTE["muted"], linewidth=0.8, linestyle="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(axes_lbl)
    ax1.set_ylabel("Acceleration (g)")
    ax1.set_title("Mean ± Std per axis", fontsize=13, pad=14)
    ax1.set_xlim(-0.6, 2.6)
    top    = max(m + s for m, s in zip(means, stds)) + 0.25
    bottom = min(m - s for m, s in zip(means, stds)) - 0.30
    ax1.set_ylim(bottom, top)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="white",
               markerfacecolor=PALETTE["dark"], markeredgecolor="white",
               markeredgewidth=2, markersize=11, label="Mean (μ)"),
        Line2D([0], [0], color=PALETTE["dark"], linewidth=2.2,
               marker="_", markersize=14, markeredgewidth=2.2,
               label="±1 Std Dev (σ)"),
    ]
    ax1.legend(handles=legend_handles, frameon=True, loc="lower left",
               fontsize=10, facecolor="white", edgecolor=PALETTE["light"],
               framealpha=0.95)

    for i, (lo, hi) in enumerate(zip(mins, maxs)):
        ax2.plot([lo, hi], [i, i], color=colors[i], linewidth=8, solid_capstyle="round")
        ax2.scatter([lo, hi], [i, i], color=colors[i], s=120, zorder=5,
                    edgecolors="white", linewidths=2)
        ax2.text(lo - 0.6, i, f"{lo:.2f}", ha="right", va="center",
                 fontsize=10, fontweight="bold")
        ax2.text(hi + 0.6, i, f"{hi:.2f}", ha="left", va="center",
                 fontsize=10, fontweight="bold")

    ax2.set_yticks(range(3))
    ax2.set_yticklabels(axes_lbl)
    ax2.set_xlabel("Acceleration range (g)")
    ax2.set_title("Min → Max range per axis", fontsize=13)
    ax2.set_xlim(-20, 12)
    ax2.invert_yaxis()
    ax2.spines["left"].set_visible(False)

    save(fig, "03_accelerometer.png")


def chart_stage_codes():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                                    gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(wspace=0.35, bottom=0.18)
    fig.suptitle("PSG Sleep Stage Distribution", fontsize=20, fontweight="bold", y=0.98)

    stages   = ["Unscored", "Wake", "N1", "N2", "N3", "N4", "REM"]
    epochs   = [438, 2429, 1821, 12954, 3329, 356, 5884]
    pcts     = [1.6, 8.9, 6.7, 47.6, 12.2, 1.3, 21.6]
    colors   = [STAGE_COLORS[s] for s in stages]

    def _autopct(p):
        return f"{p:.1f}%" if p >= 5 else ""

    wedges, _, autotexts = ax1.pie(
        epochs, colors=colors, startangle=90,
        autopct=_autopct, pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        textprops=dict(color="white", fontsize=11, fontweight="bold"),
    )
    ax1.text(0, 0.1, "27,211", ha="center", va="center",
             fontsize=26, fontweight="bold", color=PALETTE["dark"])
    ax1.text(0, -0.18, "total epochs", ha="center", va="center",
             fontsize=11, color=PALETTE["muted"])
    ax1.set_title("Share of total epochs", fontsize=13, pad=14)

    legend_labels = [f"{s} ({p}%)" for s, p in zip(stages, pcts)]
    fig.legend(wedges, legend_labels,
               loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=7, frameon=False, fontsize=10, handlelength=1.4,
               columnspacing=1.4)

    order = np.argsort(epochs)[::-1]
    ax2.barh(range(len(stages)), [epochs[i] for i in order],
             color=[colors[i] for i in order], alpha=0.9)
    ax2.set_yticks(range(len(stages)))
    ax2.set_yticklabels([stages[i] for i in order])
    ax2.invert_yaxis()
    ax2.set_xlabel("Epoch count")
    ax2.set_title("Epoch counts (sorted)", fontsize=13, pad=14)
    for i, idx in enumerate(order):
        ax2.text(epochs[idx] + 200, i, f"{epochs[idx]:,}", va="center",
                 fontsize=10, fontweight="bold", color=PALETTE["dark"])
    ax2.set_xlim(0, max(epochs) * 1.2)

    fig.text(0.5, -0.02,
             "Unscored (−1) and N4 epochs are excluded before modelling",
             ha="center", fontsize=10, style="italic", color=PALETTE["muted"])

    save(fig, "04_stage_codes.png")

def chart_features():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.suptitle("Derived Feature Matrix — 6 features × 26,773 epochs",
                 fontsize=19, fontweight="bold", y=0.99)

    features = [
        ("activity_mean", "Accelerometer", "Mean activity in ±5-min window",
         PALETTE["primary"]),
        ("activity_std",  "Accelerometer", "Std dev of activity in ±5-min window",
         PALETTE["primary"]),
        ("hr_mean",       "Heart rate",    "Mean HR in ±5-min window (bpm)",
         PALETTE["accent"]),
        ("hr_local_std",  "Heart rate",    "Std dev of HR in ±5-min window",
         PALETTE["accent"]),
        ("hr_trend",      "Heart rate",    "Linear slope of HR (bpm/s)",
         PALETTE["accent"]),
        ("clock_proxy",   "Timestamp",     "cos(2π·t / 28,800) — circadian proxy",
         PALETTE["secondary"]),
    ]

    for i, (name, src, desc, color) in enumerate(features):
        row = i // 2
        col = i % 2
        x = col * 6 + 0.2
        y = 4.4 - row * 1.7

        ax.add_patch(FancyBboxPatch((x, y), 0.18, 1.3,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    facecolor=color, edgecolor="none"))
        ax.add_patch(FancyBboxPatch((x + 0.25, y), 5.4, 1.3,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=PALETTE["bg"], edgecolor=PALETTE["light"]))
        ax.text(x + 0.45, y + 0.95, name, fontsize=14, fontweight="bold",
                color=PALETTE["dark"])
        ax.text(x + 0.45, y + 0.55, src, fontsize=10, color=color, fontweight="bold")
        ax.text(x + 0.45, y + 0.20, desc, fontsize=10, color=PALETTE["dark"])

    ax.text(6, 0.2,
            "Per-subject StandardScaler applied · mean ≈ 0, std ≈ 1",
            ha="center", fontsize=10, style="italic", color=PALETTE["muted"])

    save(fig, "05_derived_features.png")


def chart_binary():
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Binary Target — Sleep vs Wake",
                 fontsize=19, fontweight="bold", y=0.98)

    labels = ["Sleep", "Wake"]
    sizes  = [24344, 2429]
    pcts   = [90.9, 9.1]
    colors = [STAGE_COLORS["Sleep"], STAGE_COLORS["Wake"]]

    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.4, edgecolor="white", linewidth=3))
    ax.text(0, 0.15, "≈ 10 : 1", ha="center", va="center",
            fontsize=28, fontweight="bold", color=PALETTE["dark"])
    ax.text(0, -0.18, "sleep / wake imbalance", ha="center", va="center",
            fontsize=11, color=PALETTE["muted"])

    ax.legend(wedges,
              [f"{l} — {n:,} epochs ({p}%)" for l, n, p in zip(labels, sizes, pcts)],
              loc="lower center", bbox_to_anchor=(0.5, -0.05),
              frameon=False, fontsize=11, ncol=1)

    save(fig, "06_binary_target.png")


def chart_three_class():
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("3-Class Target — Wake / NREM / REM",
                 fontsize=19, fontweight="bold", y=0.98)

    labels = ["NREM", "REM", "Wake"]
    sizes  = [18460, 5884, 2429]
    pcts   = [68.9, 22.0, 9.1]
    colors = [STAGE_COLORS["NREM"], STAGE_COLORS["REM"], STAGE_COLORS["Wake"]]

    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.4, edgecolor="white", linewidth=3))
    ax.text(0, 0.1, "26,773", ha="center", va="center",
            fontsize=24, fontweight="bold", color=PALETTE["dark"])
    ax.text(0, -0.18, "epochs total", ha="center", va="center",
            fontsize=11, color=PALETTE["muted"])

    ax.legend(wedges,
              [f"{l} — {n:,} ({p}%)" for l, n, p in zip(labels, sizes, pcts)],
              loc="lower center", bbox_to_anchor=(0.5, -0.05),
              frameon=False, fontsize=11)

    fig.text(0.5, 0.02, "NREM = N1 + N2 + N3", ha="center",
             fontsize=10, style="italic", color=PALETTE["muted"])

    save(fig, "07_three_class_target.png")


def chart_data_quality():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.suptitle("Data Quality Overview", fontsize=20, fontweight="bold", y=0.99)

    streams = [
        ("Heart Rate",  "0",   "Missing values",     "Irregular → 1 Hz",   "~0.1% > 180 bpm",   PALETTE["accent"]),
        ("Motion",      "0",   "Missing values",     "Uniform 50 Hz",      "—",                 PALETTE["primary"]),
        ("Labels",      "0",   "Missing values",     "Fixed 30-s epochs",  "438 unscored + 356 N4 excluded", PALETTE["secondary"]),
    ]

    for i, (title, missing, msub, sampling, notes, color) in enumerate(streams):
        x = i * 4 + 0.2
        ax.add_patch(FancyBboxPatch((x, 0.3), 3.6, 4.1,
                                    boxstyle="round,pad=0.05,rounding_size=0.15",
                                    facecolor="white", edgecolor=color, linewidth=2.5))
        ax.add_patch(FancyBboxPatch((x, 3.6), 3.6, 0.8,
                                    boxstyle="round,pad=0.05,rounding_size=0.15",
                                    facecolor=color, edgecolor="none"))
        ax.text(x + 1.8, 3.95, title, ha="center", va="center",
                fontsize=15, fontweight="bold", color="white")

        ax.text(x + 1.8, 2.85, missing, ha="center", va="center",
                fontsize=42, fontweight="bold", color=color)
        ax.text(x + 1.8, 2.25, msub, ha="center", va="center",
                fontsize=10, color=PALETTE["muted"])

        ax.text(x + 0.2, 1.55, "Sampling:", fontsize=10, fontweight="bold",
                color=PALETTE["dark"])
        ax.text(x + 0.2, 1.25, sampling, fontsize=10, color=PALETTE["dark"])
        ax.text(x + 0.2, 0.85, "Notes:", fontsize=10, fontweight="bold",
                color=PALETTE["dark"])
        ax.text(x + 0.2, 0.55, notes, fontsize=9, color=PALETTE["dark"])

    save(fig, "08_data_quality.png")


def chart_temporal():
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Temporal Context — Recording Duration per Subject",
                 fontsize=18, fontweight="bold", y=1.00)

    lo, mean, hi = 3.48, 7.20, 8.18

    ax.barh([0], [hi - lo], left=[lo], height=0.35,
            color=PALETTE["secondary"], alpha=0.35)
    ax.barh([0], [0.4], left=[mean - 0.2], height=0.35,
            color=PALETTE["primary"])

    for x, lbl, sub in [(lo, f"{lo} h", "min"),
                        (mean, f"{mean} h", "mean"),
                        (hi, f"{hi} h", "max")]:
        ax.scatter([x], [0], color="white", s=180, zorder=5,
                   edgecolors=PALETTE["dark"], linewidths=2)
        ax.text(x, 0.45, lbl, ha="center", fontsize=13, fontweight="bold",
                color=PALETTE["dark"])
        ax.text(x, -0.45, sub, ha="center", fontsize=10, color=PALETTE["muted"])

    ax.set_xlim(2.5, 9)
    ax.set_ylim(-1, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Recording duration (hours)")
    ax.set_title("Single overnight session per subject · 31 subjects · ≈223 h total",
                 fontsize=12, fontweight="normal", pad=10)
    ax.spines["left"].set_visible(False)

    save(fig, "09_temporal_context.png")


def main():
    print("Generating charts → ./charts/")
    chart_dataset_summary()
    chart_heart_rate()
    chart_accelerometer()
    chart_stage_codes()
    chart_features()
    chart_binary()
    chart_three_class()
    chart_data_quality()
    chart_temporal()
    print("Done.")


if __name__ == "__main__":
    main()
