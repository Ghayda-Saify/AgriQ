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


# ---- Figure 2: study parcel map -------------------------------------
def fig_map():
    rows = [r for r in rd("jenin_parcels_zoned.csv")
            if r["zone"] == "A" and 0.1 <= float(r["area_ha"]) <= 100.0]
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    for z, col, lab in (("A", BLUE, "study parcels"),):
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
    for z, col, lab in (("A", BLUE, "study parcels (174)"),):
        s = sorted([r for r in rows if r["zone"] == z], key=lambda r: float(r["lam"]))
        if not s: continue
        x = [float(r["lam"]) for r in s]
        y = [float(r["water_m3"])/1e6 for r in s]
        e = [float(r["water_sd"])/1e6 for r in s]
        ax.errorbar(x, y, yerr=e, color=col, marker="o", markersize=3,
                    linewidth=1.1, capsize=2, label=lab)
    # The licensed agricultural allocation, not the 6.0 Mm3 total that includes
    # domestic supply. Demand never approaches it at any price swept, which is
    # the point of the figure: pricing cannot reach this budget.
    ax.axhline(1.0, color=ORANGE, linestyle="--", linewidth=1.0)
    ax.text(ax.get_xlim()[1], 1.15,
            r"licensed agricultural budget 1.0 Mm$^3$/yr (PCBS)", fontsize=6,
            color=ORANGE, ha="right")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xticks([0, 0.5, 1, 2, 3, 8, 20])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, None)
    ax.set_xlabel(r"Water charge $\lambda$ (USD/m$^3$)")
    ax.set_ylabel(r"Irrigation demand (Mm$^3$/yr)")
    ax.legend(frameon=False, loc="upper right", fontsize=6)
    fig.savefig(os.path.join(FIG, "fig_water.pdf"))
    plt.close(fig)
    print("fig_water.pdf")


def fig_mwall():
    """Profit against the domain-wall ordering penalty, both solvers, with the
    one-hot references. This is the figure that carries the paper's argument:
    one-hot dSB sits far below everything, domain-wall lifts it past the
    one-hot reference, and the apparent sign of the annealing effect depends
    entirely on where M_wall is set."""
    f = os.path.join(ROOT, "data", "processed", "mwall_sweep.csv")
    if not os.path.exists(f):
        print("mwall_sweep.csv missing - skipping fig_mwall"); return
    rows = rd("mwall_sweep.csv")
    bench = rd("benchmark_results.csv")

    def ref(enc, s):
        v = [float(r["profit"]) for r in bench
             if r["encoding"] == enc and r["solver"] == s]
        return np.mean(v) / 1e6

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    mws = sorted({float(r["m_wall"]) for r in rows})
    for s, col, mk in (("SA", BLUE, "o"), ("dSB", ORANGE, "s")):
        mu, sd = [], []
        for mw in mws:
            v = np.array([float(r["profit"]) for r in rows
                          if float(r["m_wall"]) == mw and r["solver"] == s]) / 1e6
            mu.append(v.mean()); sd.append(v.std())
        ax.errorbar(mws, mu, yerr=sd, color=col, marker=mk, markersize=3.5,
                    linewidth=1.2, capsize=2, label=f"domain-wall, {s}")
    for s, col, ls in (("SA", BLUE, ":"), ("dSB", ORANGE, "-.")):
        ax.axhline(ref("one-hot", s), color=col, linestyle=ls, linewidth=0.9)
        ax.text(mws[0], ref("one-hot", s) + 0.12, f"one-hot {s}",
                fontsize=6, color=col, ha="left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(mws)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(r"Domain-wall ordering penalty $M_{\mathrm{wall}}$")
    ax.set_ylabel("Profit (M USD/yr)")
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    fig.savefig(os.path.join(FIG, "fig_mwall.pdf"))
    plt.close(fig)
    print("fig_mwall.pdf")


if __name__ == "__main__":
    fig_encoding(); fig_map(); fig_water(); fig_mwall()
    print("figures ->", FIG)
