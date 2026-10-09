"""Price robustness: does the recommended plan survive price uncertainty?

A first attempt measured agreement as the share of parcels given the SAME crop
as the baseline plan. That metric turned out to be meaningless here: with an
inverse-demand objective the optimiser spreads planting until marginal revenue
equalises across crops, so many permutations of "which parcel grows what"
achieve nearly identical profit. Parcel identity is therefore degenerate, and
even two baseline runs that differ only in random seed disagree at parcel
level.

We therefore report:
  * a SEED CONTROL -- baseline vs baseline under different seeds -- which
    establishes the noise floor any scenario must be judged against;
  * CROP-MIX similarity (1 - total-variation distance between crop area
    shares), which is the decision-relevant quantity: how much land goes to
    each crop, not which particular parcel grows it.

Production cost is held at its baseline calibration throughout (see
instance.build), so a price shock is not silently cancelled by a cost shock.
"""
import csv, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "optimizer"))
sys.path.insert(0, os.path.join(ROOT, "src", "pipeline"))
from instance import build, fao_prices, CROPS
from solvers import sa, decode
from water_price import sweeps_from_benchmark

# 10 seeds left the mean's standard error (~0.16 Mm3) larger than the
# demand change across the budget crossing (~0.11 Mm3), so neither the
# shadow price nor any price scenario was separable from solver noise.
# 50 seeds halves the standard error.
N_SEEDS = 50
SWEEPS = sweeps_from_benchmark()   # matched SA budget from benchmark.py
PERISHABLE = {"tomatoes", "cucumbers", "onions", "eggplants", "potatoes", "oranges"}


def mix(q, assign):
    """Share of TOTAL AREA by crop -- the decision-relevant summary."""
    m = np.zeros(len(CROPS))
    for i, j in enumerate(assign):
        m[j] += q.area[i]
    return m / m.sum()


def similarity(m1, m2):
    """1 - total variation distance, in percent. 100% = identical crop mix."""
    return (1.0 - 0.5 * np.abs(m1 - m2).sum()) * 100.0


def solve(prices, seeds=range(N_SEEDS)):
    q = build("A", prices=prices)
    best, best_p, profits, mixes = None, -np.inf, [], []
    for seed in seeds:
        b, _ = sa(q.Q, n_sweeps=SWEEPS, seed=seed)
        a = decode(q, b)
        ev = q.evaluate(a)
        profits.append(ev["profit"]); mixes.append(mix(q, a))
        if ev["profit"] > best_p:
            best_p, best = ev["profit"], a
    return q, best, np.array(profits), np.array(mixes)


if __name__ == "__main__":
    base = fao_prices()
    fc = {r["crop"]: r for r in csv.DictReader(
        open(os.path.join(ROOT, "data", "processed", "price_forecast_clean.csv")))}

    qb, ab, pb, mb = solve(None)
    base_mix = mb.mean(0)

    # --- noise floor: same prices, disjoint seed sets ----------------------
    _, a1, p1, m1 = solve(None, seeds=range(100, 100 + N_SEEDS))
    floor_mix = similarity(base_mix, m1.mean(0))
    floor_parcel = (ab == a1).mean() * 100
    within = [similarity(mb[i], mb[j]) for i in range(len(mb)) for j in range(i+1, len(mb))]
    print("SEED CONTROL (identical prices, different seeds)")
    print(f"  crop-mix similarity     : {floor_mix:5.1f}%   <- noise floor for the table below")
    print(f"  seed-to-seed mix spread : {np.mean(within):5.1f}% (mean pairwise)")
    print(f"  parcel-level agreement  : {floor_parcel:5.1f}%   <- degenerate, not informative\n")

    print(f"{'scenario':<22}{'profit':>9}{'sd':>7}{'mix sim':>9}{'parcel':>8}   top crops by area")
    print("-" * 92)
    print(f"{'baseline 2019-23':<22}{pb.mean()/1e6:>8.2f}M{pb.std()/1e6:>6.2f}M"
          f"{'--':>9}{'--':>8}   " +
          ", ".join(f"{CROPS[j]} {base_mix[j]*100:.0f}%"
                    for j in np.argsort(-base_mix)[:4]))

    scen = [("LSTM forecast", {c: float(fc[c]["price_lstm_usd_t"]) for c in fc}),
            ("persistence (2023)", {c: float(fc[c]["price_persistence_2023_usd_t"]) for c in fc})]
    for d in (-30, -15, 15, 30):
        scen.append((f"uniform {d:+d}%", {c: base[c]*(1+d/100) for c in base if c != "fallow"}))
    for d in (-30, 30):
        scen.append((f"perishables {d:+d}%", {c: base[c]*(1+d/100) for c in PERISHABLE}))

    rows = [{"scenario": "baseline", "profit_usd": pb.mean(), "profit_sd": pb.std(),
             "mix_similarity_pct": 100.0, "parcel_agreement_pct": 100.0}]
    rows.append({"scenario": "SEED CONTROL", "profit_usd": p1.mean(), "profit_sd": p1.std(),
                 "mix_similarity_pct": floor_mix, "parcel_agreement_pct": floor_parcel})
    for label, pr in scen:
        q, a, p, m = solve(pr)
        sim = similarity(base_mix, m.mean(0))
        par = (a == ab).mean() * 100
        top = ", ".join(f"{CROPS[j]} {m.mean(0)[j]*100:.0f}%" for j in np.argsort(-m.mean(0))[:4])
        print(f"{label:<22}{p.mean()/1e6:>8.2f}M{p.std()/1e6:>6.2f}M{sim:>8.1f}%{par:>7.1f}%   {top}")
        rows.append({"scenario": label, "profit_usd": p.mean(), "profit_sd": p.std(),
                     "mix_similarity_pct": sim, "parcel_agreement_pct": par})

    out = os.path.join(ROOT, "data", "processed", "price_robustness.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
