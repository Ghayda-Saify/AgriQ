"""Sensitivity of the Area C counterfactual to market size.

area_c.py holds the market fixed at the size implied by Areas A/B, the
conservative case: Area C land adds supply but no buyers. The opposite bound
lets the market grow in proportion to land (market_scale = land ratio), as if
Area C access came with proportionate market access. The true value of access
lies between the two, and the gap between them measures how much the
counterfactual depends on the unobserved market size.
Both bounds hold the full-access plan to the 6.0 Mm3/yr water budget.
"""
import csv, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "optimizer"))
sys.path.insert(0, os.path.join(ROOT, "src", "pipeline"))
from instance import build, load_parcels
from solvers import sa, decode
from water_price import crossing, sweeps_from_benchmark

N_SEEDS = 10
LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

sweeps = sweeps_from_benchmark()
scale = sum(float(p["area_ha"]) for p in load_parcels("AC")) / sum(float(p["area_ha"]) for p in load_parcels("A"))
print(f"market scaled with land: x{scale:.3f}")
rows = []
for lam in LAMBDAS:
    q = build("AC", lam=lam, market_scale=scale)
    w, p = [], []
    for seed in range(N_SEEDS):
        b, _ = sa(q.Q, n_sweeps=sweeps, seed=seed)
        ev = q.evaluate(decode(q, b))
        w.append(ev["water"]); p.append(ev["profit"])
    rows.append({"zone": "AC_scaled", "lam": lam, "water_m3": float(np.mean(w)), "water_sd": float(np.std(w)),
                 "profit_usd": float(np.mean(p)), "profit_sd": float(np.std(p))})
    print(f"  lambda {lam:4.2f}: water {np.mean(w)/1e6:5.2f} Mm3, profit ${np.mean(p)/1e6:.2f}M")

lam_star, prof, binds = crossing(rows)
base = {r["regime"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data", "processed", "area_c.csv")))}
ab = float(base["Areas A/B"]["profit_budget"])
fixed = float(base["A/B + Area C"]["profit_budget"])
print(f"\nwithin the water budget:")
print(f"  market fixed        : full access ${fixed/1e6:.2f}M -> forgone ${(fixed-ab)/1e6:.2f}M/yr")
print(f"  market scales x{scale:.2f}: full access ${prof/1e6:.2f}M (shadow price ${lam_star:.2f}/m3) "
      f"-> forgone ${(prof-ab)/1e6:.2f}M/yr")

with open(os.path.join(ROOT, "data", "processed", "water_sweep_market_scaled.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
