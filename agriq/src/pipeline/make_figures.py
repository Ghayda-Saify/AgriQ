"""Generate the paper's figures as vector PDFs sized for an IEEE column.

Design choices follow benchmarking-reporting practice: distributions over the
20 seeds are shown as box plots rather than bar-with-error-bar, because a bar
chart hides the shape of a stochastic solver's output and symmetric error bars
misrepresent skewed distributions.
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

# IEEE single column is 3.5in; keep fonts at true final size (no rescaling)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
GREY, BLUE, ORANGE = "#4d4d4d", "#3b6ea5", "#c8722a"


def rd(name):
    return list(csv.DictReader(open(os.path.join(ROOT, "data", "processed", name))))


# ---- Figure 1: encoding x solver, 20-seed profit distributions -------------
def fig_encoding():
    rows = rd("benchmark_results.csv")
    groups = [("one-hot", "SA"), ("one-hot", "dSB"),
              ("domain-wall", "SA"), ("domain-wall", "dSB")]
    data = [[float(r["profit"])/1e6 for r in rows
             if r["encoding"] == e and r["solver"] == s] for e, s in groups]
    invalid = [np.mean([float(r["invalid_parcels"]) for r in rows
                        if r["encoding"] == e and r["solver"] == s]) for e, s in groups]
    greedy = [float(r["profit"])/1e6 for r in rows if r["solver"] == "greedy"][0]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bp = ax.boxplot(data, widths=0.6, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.1),
                    flierprops=dict(marker="o", markersize=2.5, alpha=0.6))
    for patch, (e, s) in zip(bp["boxes"], groups):
        patch.set_facecolor(BLUE if s == "SA" else ORANGE)
        patch.set_alpha(0.75); patch.set_linewidth(0.7)
    top = max(max(d) for d in data)
    ax.set_ylim(0, top * 1.15)
    for i, v in enumerate(invalid):
        ax.annotate(f"{v:.1f} invalid", (i + 1, min(data[i])), textcoords="offset points",
                    xytext=(0, -10), ha="center", fontsize=6, color=GREY)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(["SA", "dSB", "SA", "dSB"])
    ax.set_ylabel("Profit (M USD/yr)")
    ax.axvline(2.5, color=GREY, linewidth=0.6, linestyle=":")
    ax.text(1.5, top * 1.13, "one-hot", ha="center", va="top", fontsize=7.5)
    ax.text(3.5, top * 1.13, "domain-wall", ha="center", va="top", fontsize=7.5)
    ax.text(0.55, 0.12, f"greedy baseline: {greedy:+.2f} M (below axis)",
            fontsize=6, color=GREY, ha="left")
    fig.savefig(os.path.join(FIG, "fig_encoding.pdf"))
    plt.close(fig)
    print("fig_encoding.pdf")


# ---- Figure 2: parcel map by Oslo zone -------------------------------------
def fig_map():
    rows = [r for r in rd("jenin_parcels_zoned.csv")
            if r["zone"] in ("A", "C") and 0.1 <= float(r["area_ha"]) <= 100.0]
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    for z, col, lab in (("A", BLUE, "Area A/B - farmer-controlled"),
                        ("C", ORANGE, "Area C - access restricted")):
        s = [r for r in rows if r["zone"] == z]
        ax.scatter([float(r["lon"]) for r in s], [float(r["lat"]) for r in s],
                   s=[float(r["area_ha"])*1.6 for r in s], c=col, alpha=0.6,
                   edgecolors="none", label=f"{lab} (n={len(s)})")
    ax.set_xlabel(r"Longitude ($^\circ$E)"); ax.set_ylabel(r"Latitude ($^\circ$N)")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3, borderpad=0.2)
    ax.set_aspect(1.0/np.cos(np.radians(32.45)))
    fig.savefig(os.path.join(FIG, "fig_map.pdf"))
    plt.close(fig)
    print("fig_map.pdf")


# ---- Figure 3: water shadow price ------------------------------------------
def fig_water():
    if not os.path.exists(os.path.join(ROOT, "data", "processed", "water_sweep.csv")):
        print("water_sweep.csv missing - skipping fig_water"); return
    rows = rd("water_sweep.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for z, col, lab in (("A", BLUE, "Areas A/B (174 parcels)"),
                        ("AC", ORANGE, "A/B + Area C (284 parcels)")):
        s = sorted([r for r in rows if r["zone"] == z], key=lambda r: float(r["lam"]))
        if not s: continue
        x = [float(r["lam"]) for r in s]
        y = [float(r["water_m3"])/1e6 for r in s]
        e = [float(r["water_sd"])/1e6 for r in s]
        ax.errorbar(x, y, yerr=e, color=col, marker="o", markersize=3,
                    linewidth=1.1, capsize=2, label=lab)
    ms = os.path.join(ROOT, "data", "processed", "water_sweep_market_scaled.csv")
    if os.path.exists(ms):
        s2 = sorted(rd("water_sweep_market_scaled.csv"), key=lambda r: float(r["lam"]))
        ax.plot([float(r["lam"]) for r in s2], [float(r["water_m3"])/1e6 for r in s2],
                color=ORANGE, linestyle="--", marker="s", markersize=2.5, linewidth=0.9,
                label="A/B + Area C, market scaled with land")
    ax.axhline(6.0, color=GREY, linestyle="--", linewidth=0.9)
    ax.text(ax.get_xlim()[1], 6.12, r"Jenin budget 6.0 Mm$^3$/yr", fontsize=6,
            color=GREY, ha="right")
    ax.set_xlabel(r"Water charge $\lambda$ (USD/m$^3$)")
    ax.set_ylabel(r"Irrigation demand (Mm$^3$/yr)")
    ax.legend(frameon=False, loc="upper right", fontsize=6)
    fig.savefig(os.path.join(FIG, "fig_water.pdf"))
    plt.close(fig)
    print("fig_water.pdf")


if __name__ == "__main__":
    fig_encoding(); fig_map(); fig_water()
    print("figures ->", FIG)
