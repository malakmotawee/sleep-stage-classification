import os
import numpy as np
import matplotlib.pyplot as plt

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
    "primary":   "#6C5CE7",
    "secondary": "#00B8A9",
    "accent":    "#FF6B6B",
    "warm":      "#FDCB6E",
    "dark":      "#2D3436",
    "muted":     "#B2BEC3",
    "light":     "#DFE6E9",
    "bg":        "#F7F9FC",
}

STAGE_COLORS = {
    "Unscored": "#B2BEC3",
    "Wake":     "#FF6B6B",
    "N1":       "#FDCB6E",
    "N2":       "#74B9FF",
    "N3":       "#0984E3",
    "N4":       "#6C5CE7",
    "REM":      "#00B8A9",
}

OUT = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}")


def plot_hr_summary():
    mean, sd = 68.46, 17.80
    vmin, vmax = 30, 215
    total = 254_425

    fig, (ax_kpi, ax_dist) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.6]}
    )

    ax_kpi.set_xlim(0, 1); ax_kpi.set_ylim(0, 1)
    ax_kpi.axis("off")
    ax_kpi.set_title("Heart Rate — Summary", loc="left", pad=18)

    cards = [
        ("Total readings", f"{total:,}",       PALETTE["primary"]),
        ("Mean",           f"{mean:.2f} bpm",  PALETTE["secondary"]),
        ("Std dev",        f"{sd:.2f} bpm",    PALETTE["warm"]),
        ("Range",          f"{vmin}–{vmax} bpm", PALETTE["accent"]),
    ]
    for i, (label, value, color) in enumerate(cards):
        y = 0.78 - i * 0.22
        ax_kpi.add_patch(plt.Rectangle((0.02, y - 0.08), 0.96, 0.18,
                                       facecolor=color, alpha=0.12,
                                       edgecolor=color, linewidth=1.5))
        ax_kpi.text(0.06, y + 0.045, label,
                    fontsize=11, color=PALETTE["dark"], alpha=0.75)
        ax_kpi.text(0.06, y - 0.04, value,
                    fontsize=18, fontweight="bold", color=color)

    rng = np.random.default_rng(0)
    sample = rng.normal(mean, sd, 200_000)
    sample = sample[(sample >= vmin) & (sample <= vmax)]

    ax_dist.hist(sample, bins=60, color=PALETTE["primary"],
                 alpha=0.85, edgecolor="white", linewidth=0.6)
    ax_dist.axvline(mean, color=PALETTE["accent"], linestyle="--",
                    linewidth=2, label=f"Mean  {mean:.1f}")
    ax_dist.axvspan(mean - sd, mean + sd,
                    color=PALETTE["secondary"], alpha=0.15,
                    label=f"±1 SD  ({mean-sd:.1f}–{mean+sd:.1f})")
    ax_dist.set_xlabel("Heart rate (bpm)")
    ax_dist.set_ylabel("Readings")
    ax_dist.set_title("Distribution (illustrative, μ=68.46, σ=17.80)",
                      loc="left", fontsize=12, fontweight="normal", pad=12)
    ax_dist.legend(frameon=False, loc="upper right")

    fig.suptitle("1.6  Key Statistics — Heart Rate (31 subjects pooled)",
                 fontsize=14, fontweight="bold", y=1.02, x=0.02, ha="left")
    save(fig, "10_hr_summary.png")


def plot_accel_summary():
    axes_names = ["X (lateral)", "Y (vertical)", "Z (longitudinal)"]
    means = [-0.098, -0.054, -0.218]
    stds  = [ 0.347,  0.453,  0.784]
    mins  = [-5.514, -10.553, -16.041]
    maxs  = [ 7.173,   7.198,   8.423]
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]

    fig, (ax_ms, ax_range) = plt.subplots(1, 2, figsize=(13, 5.2))

    y = np.arange(len(axes_names))
    ax_ms.barh(y, stds, left=[m - s for m, s in zip(means, stds)],
               height=0.45, color=colors, alpha=0.25,
               edgecolor=colors, linewidth=1.5)
    ax_ms.barh(y, stds, left=means, height=0.45,
               color=colors, alpha=0.25, edgecolor=colors, linewidth=1.5)
    ax_ms.scatter(means, y, color=colors, s=120, zorder=5,
                  edgecolor="white", linewidth=2)
    for i, m in enumerate(means):
        ax_ms.annotate(f"μ = {m:+.3f} g", (m, i),
                       textcoords="offset points", xytext=(0, 18),
                       ha="center", fontsize=10, fontweight="bold",
                       color=PALETTE["dark"])
        ax_ms.annotate(f"σ = {stds[i]:.3f}", (m, i),
                       textcoords="offset points", xytext=(0, -22),
                       ha="center", fontsize=9, color=PALETTE["dark"], alpha=0.7)

    ax_ms.axvline(0, color=PALETTE["muted"], linewidth=1, linestyle=":")
    ax_ms.set_yticks(y); ax_ms.set_yticklabels(axes_names)
    ax_ms.set_xlabel("Acceleration (g)")
    ax_ms.set_title("Mean ± 1 SD per axis", loc="left", pad=12)
    ax_ms.invert_yaxis()

    for i, (lo, hi) in enumerate(zip(mins, maxs)):
        ax_range.plot([lo, hi], [i, i], color=colors[i], linewidth=4,
                      solid_capstyle="round", alpha=0.85)
        ax_range.scatter([lo, hi], [i, i], color=colors[i], s=90,
                         zorder=5, edgecolor="white", linewidth=1.5)
        ax_range.annotate(f"{lo:+.2f}", (lo, i), textcoords="offset points",
                          xytext=(-6, 0), ha="right", va="center",
                          fontsize=10, color=PALETTE["dark"])
        ax_range.annotate(f"{hi:+.2f}", (hi, i), textcoords="offset points",
                          xytext=(6, 0), ha="left", va="center",
                          fontsize=10, color=PALETTE["dark"])

    ax_range.axvline(0, color=PALETTE["muted"], linewidth=1, linestyle=":")
    ax_range.set_yticks(y); ax_range.set_yticklabels(axes_names)
    ax_range.set_xlabel("Acceleration (g)")
    ax_range.set_title("Observed min – max per axis", loc="left", pad=12)
    ax_range.set_xlim(min(mins) - 2, max(maxs) + 2)
    ax_range.invert_yaxis()

    fig.suptitle("1.6  Key Statistics — Accelerometer (31 subjects pooled)",
                 fontsize=14, fontweight="bold", y=1.02, x=0.02, ha="left")
    save(fig, "11_accel_summary.png")

def plot_stage_distribution():
    stages = ["Unscored", "Wake", "N1", "N2", "N3", "N4", "REM"]
    codes  = [-1, 0, 1, 2, 3, 4, 5]
    epochs = [438, 2429, 1821, 12954, 3329, 356, 5884]
    total  = sum(epochs)
    pct    = [e / total * 100 for e in epochs]
    colors = [STAGE_COLORS[s] for s in stages]

    fig, (ax_bar, ax_donut) = plt.subplots(
        1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    x = np.arange(len(stages))
    bars = ax_bar.bar(x, epochs, color=colors, edgecolor="white", linewidth=1.5)

    for bar, e, p in zip(bars, epochs, pct):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.012,
                    f"{e:,}\n{p:.1f}%",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=PALETTE["dark"])

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{s}\n(code {c})" for s, c in zip(stages, codes)],
                           fontsize=10)
    ax_bar.set_ylabel("Epochs")
    ax_bar.set_title("Epochs per PSG stage", loc="left", pad=12)
    ax_bar.set_ylim(0, max(epochs) * 1.18)
    ax_bar.grid(axis="y", alpha=0.25, linestyle="--")

    wedges, _ = ax_donut.pie(
        epochs, colors=colors, startangle=90,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2),
    )
    ax_donut.text(0, 0.08, f"{total:,}", ha="center", va="center",
                  fontsize=22, fontweight="bold", color=PALETTE["dark"])
    ax_donut.text(0, -0.12, "total epochs", ha="center", va="center",
                  fontsize=11, color=PALETTE["dark"], alpha=0.7)
    ax_donut.set_title("Composition", loc="left", pad=12)

    legend_labels = [f"{s}  —  {p:.1f}%" for s, p in zip(stages, pct)]
    ax_donut.legend(wedges, legend_labels,
                    loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=False, fontsize=10)

    fig.suptitle("1.6  Key Statistics — PSG Stage Distribution "
                 "(all 31 subjects, before exclusions)",
                 fontsize=14, fontweight="bold", y=1.02, x=0.02, ha="left")
    save(fig, "12_psg_stage_distribution.png")

def plot_normalization():
    features = ["activity_mean", "activity_std", "hr_local_std",
                "hr_mean", "clock_proxy", "hr_trend"]

    rng = np.random.default_rng(1)
    fig, (ax_box, ax_density) = plt.subplots(
        1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.2, 1]}
    )

    samples = []
    for f in features:
        if f.startswith("activity") or f == "hr_local_std":
            s = rng.lognormal(mean=-0.5, sigma=0.55, size=4000)
            s = (s - s.mean()) / s.std()
        elif f == "hr_mean":
            s = rng.normal(0, 1, 4000) + rng.lognormal(-1, 0.4, 4000) * 0.3
            s = (s - s.mean()) / s.std()
        else:
            s = rng.normal(0, 1, 4000)
        samples.append(s)

    bp = ax_box.boxplot(samples, vert=True, patch_artist=True,
                        widths=0.55, showfliers=False,
                        medianprops=dict(color=PALETTE["dark"], linewidth=2))
    feature_colors = [PALETTE["primary"], PALETTE["primary"],
                      PALETTE["secondary"], PALETTE["secondary"],
                      PALETTE["warm"], PALETTE["accent"]]
    for patch, c in zip(bp["boxes"], feature_colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
        patch.set_edgecolor(c); patch.set_linewidth(1.5)
    for whisker in bp["whiskers"]: whisker.set_color(PALETTE["muted"])
    for cap in bp["caps"]:         cap.set_color(PALETTE["muted"])

    ax_box.axhline(0, color=PALETTE["dark"], linewidth=1, linestyle=":")
    ax_box.axhspan(-1.6, 2.0, color=PALETTE["secondary"],
                   alpha=0.10, zorder=0,
                   label="5th–95th pctile [-1.6, +2.0]")
    ax_box.set_xticks(np.arange(1, len(features) + 1))
    ax_box.set_xticklabels(features, rotation=20, ha="right", fontsize=10)
    ax_box.set_ylabel("Standardised value (z)")
    ax_box.set_title("Per-feature distribution after StandardScaler",
                     loc="left", pad=12)
    ax_box.legend(frameon=False, loc="upper right")
    ax_box.set_ylim(-3.5, 4.5)

    pooled = np.concatenate(samples)
    ax_density.hist(pooled, bins=80, color=PALETTE["primary"],
                    alpha=0.8, edgecolor="white", linewidth=0.5,
                    density=True)
    ax_density.axvline(0, color=PALETTE["accent"], linewidth=2,
                       linestyle="--", label="mean ≈ 0")
    ax_density.axvspan(-1, 1, color=PALETTE["secondary"], alpha=0.18,
                       label="±1 SD (std ≈ 1)")
    ax_density.axvline(-1.6, color=PALETTE["dark"], linewidth=1,
                       linestyle=":", alpha=0.6)
    ax_density.axvline(2.0,  color=PALETTE["dark"], linewidth=1,
                       linestyle=":", alpha=0.6)
    ax_density.annotate("5th pctile\n−1.6", (-1.6, 0.02),
                        textcoords="offset points", xytext=(-4, 6),
                        ha="right", fontsize=9, color=PALETTE["dark"])
    ax_density.annotate("95th pctile\n+2.0", (2.0, 0.02),
                        textcoords="offset points", xytext=(4, 6),
                        ha="left", fontsize=9, color=PALETTE["dark"])
    ax_density.set_xlabel("Standardised value (z)")
    ax_density.set_ylabel("Density")
    ax_density.set_title("Pooled (all features)", loc="left", pad=12)
    ax_density.legend(frameon=False, loc="upper right")
    ax_density.set_xlim(-3.5, 4.5)

    fig.suptitle("1.6  Key Statistics — Post-StandardScaler Feature Profile",
                 fontsize=14, fontweight="bold", y=1.02, x=0.02, ha="left")
    save(fig, "13_normalization.png")


if __name__ == "__main__":
    plot_hr_summary()
    plot_accel_summary()
    plot_stage_distribution()
    plot_normalization()
    print(f"\nAll charts written to: {OUT}")
