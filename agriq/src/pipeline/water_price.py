"""Water shadow price: sweep the irrigation charge lambda and find where
irrigation demand meets Jenin's agricultural water budget (6.0 Mm3/yr).

Water is PRICED in the objective (src/optimizer/qubo.py), so the lambda at
which demand crosses the budget is the shadow price of irrigation water.

Both land regimes face the same market (demand defined from Areas A/B) and
the same suitability thresholds -- see instance.build(demand_zone=...).
The budget-feasible profit read off this sweep is what the Area C
counterfactual (area_c.py) reports.
"""
import csv, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "optimizer"))
sys.path.insert(0, os.path.join(ROOT, "src", "pipeline"))
from instance import build
from solvers import sa, decode

BUDGET = 6.0e6
N_SEEDS = 10
LAMBDAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 1.50, 2.00, 3.00]


def sweeps_from_benchmark():
    """Use the matched SA budget calibrated in benchmark.py (one-hot SA)."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "processed", "benchmark_results.csv"))))
    its = {int(r["iterations"]) for r in rows if r["encoding"] == "one-hot" and r["solver"] == "SA"}
    if len(its) != 1:
        raise RuntimeError(f"cannot read a single SA sweep count from benchmark_results.csv: {its}")
    return its.pop()


def crossing(rows):
    """(lambda, profit) where mean water use meets BUDGET, by linear interpolation.
    If demand is already within budget at lambda=0, the budget does not bind:
    shadow price 0 and the lambda=0 profit is budget-feasible."""
    rows = sorted(rows, key=lambda r: r["lam"])
    if rows[0]["water_m3"] <= BUDGET:
        return 0.0, rows[0]["profit_usd"], False
    for a, b in zip(rows, rows[1:]):
        if a["water_m3"] > BUDGET >= b["water_m3"]:
            t = (a["water_m3"] - BUDGET) / (a["water_m3"] - b["water_m3"])
            return (a["lam"] + t * (b["lam"] - a["lam"]),
                    a["profit_usd"] + t * (b["profit_usd"] - a["profit_usd"]), True)
    raise RuntimeError("budget not reached within the lambda grid -- extend LAMBDAS")


if __name__ == "__main__":
    sweeps = sweeps_from_benchmark()
    print(f"SA sweeps (matched budget from benchmark): {sweeps}, seeds per lambda: {N_SEEDS}")
    allrows = []
    for zone, label in (("A", "Areas A/B (174 parcels)"), ("AC", "A/B + Area C (284 parcels)")):
        print(f"\n{label}\n{'lambda':>7} {'water Mm3':>10} {'sd':>6} {'profit $M':>10} {'sd':>6}")
        rows = []
        for lam in LAMBDAS:
            q = build(zone, lam=lam)
            w, p = [], []
            for seed in range(N_SEEDS):
                b, _ = sa(q.Q, n_sweeps=sweeps, seed=seed)
                ev = q.evaluate(decode(q, b))
                w.append(ev["water"]); p.append(ev["profit"])
            r = {"zone": zone, "lam": lam, "water_m3": float(np.mean(w)), "water_sd": float(np.std(w)),
                 "profit_usd": float(np.mean(p)), "profit_sd": float(np.std(p))}
            rows.append(r)
            print(f"{lam:>7.2f} {r['water_m3']/1e6:>10.2f} {r['water_sd']/1e6:>6.2f} "
                  f"{r['profit_usd']/1e6:>10.2f} {r['profit_sd']/1e6:>6.2f}")
        lam_star, prof, binds = crossing(rows)
        print(f"  -> {'binding' if binds else 'NOT binding'}: shadow price ${lam_star:.2f}/m3, "
              f"budget-feasible profit ${prof/1e6:.2f}M")
        allrows += rows

    out = os.path.join(ROOT, "data", "processed", "water_sweep.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)
    print(f"\nwrote {out}")
