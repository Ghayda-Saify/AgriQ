"""20-seed benchmark: {SA, dSB} x {one-hot, domain-wall}, one wall-clock budget.

Fairness conditions, each fixing a flaw in an earlier version:
  * ONE compute budget for all four configurations (the one-hot dSB runtime at
    DSB_STEPS). dSB steps and SA sweeps are both calibrated to it on each
    matrix. Previously the budget was matched only within an encoding, so the
    smaller domain-wall matrix got ~34% less time and the encoding comparison
    was confounded with compute.
  * Both encodings encode the SAME problem. Domain-wall drops only the
    one-crop-per-parcel penalty and keeps the suitability penalty (the earlier
    build("A", M=0) silently removed both).
  * Violations are counted in the same unit for both encodings: parcels whose
    binary block is not a valid assignment.
  * Seeds of different solvers are independent draws, not pairs, so the
    SA-vs-dSB test is Mann-Whitney U rather than Wilcoxon signed-rank.
Greedy is the trivial baseline and is written to the results file.
"""
import csv, os, sys, time
import numpy as np
from scipy.stats import mannwhitneyu

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "optimizer"))
sys.path.insert(0, os.path.join(ROOT, "src", "pipeline"))
from instance import build
from solvers import dsb, sa, greedy, decode
from domain_wall import to_domain_wall, decode_dw

N_SEEDS = 20
DSB_STEPS, DSB_DT, DSB_RESTARTS = 1000, 0.25, 1
M_WALL = 1.0
WATER_BUDGET = 6.0e6          # m3/yr, Jenin agricultural wells


def onehot_bad_parcels(b, P, C):
    return sum(1 for i in range(P) if b[i*C:(i+1)*C].sum() != 1)


def dw_bad_parcels(d, P, C):
    """Parcels whose domain-wall chain is not monotone non-increasing."""
    bad = 0
    for i in range(P):
        seg = d[i*(C-1):(i+1)*(C-1)]
        if np.any(np.diff(seg) > 0):
            bad += 1
    return bad


def profit_free_disposal(q, assign):
    """Profit when output above the revenue-maximising quantity D_j is left
    unsold. With price(Q) = p - beta*Q and beta = p/(2 D_j), marginal revenue
    is zero at Q = D_j and the price itself turns NEGATIVE beyond Q = 2 D_j.
    Farmers would not pay to sell, so this is the fair way to score a plan
    that overshoots the market (e.g. the greedy baseline)."""
    prod = np.zeros(q.C); cost = 0.0
    for i, j in enumerate(assign):
        prod[j] += q.prod[i, j]; cost += q.cost[j] * q.area[i]
    sold = np.minimum(prod, q.demand)
    return float((q.price * sold - q.beta * sold ** 2).sum() - cost)


def med_time(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    q = build("A")
    q.water_budget = WATER_BUDGET
    P, C = q.P, q.C
    q_nooh = build("A", M_onehot=0.0)           # suitability penalty retained
    Q_dw = to_domain_wall(q_nooh.Q, P, C, M_wall=M_WALL)
    print(f"instance: {P} parcels, {q.area.sum():,.0f} ha | one-hot {q.Q.shape[0]} vars, "
          f"domain-wall {Q_dw.shape[0]} vars")

    budget = med_time(lambda: dsb(q.Q, n_steps=DSB_STEPS, dt=DSB_DT, seed=0,
                                  n_restarts=DSB_RESTARTS))
    print(f"wall-clock budget per run: {budget:.2f}s (one-hot dSB, {DSB_STEPS} steps)")

    encs = (("one-hot", q.Q, lambda b: decode(q, b), lambda b: onehot_bad_parcels(b, P, C)),
            ("domain-wall", Q_dw, lambda b: decode_dw(b, P, C), lambda b: dw_bad_parcels(b, P, C)))
    rows = []
    for enc, Q, dec, viol in encs:
        # Iterative calibration at full length: run, measure, rescale, repeat.
        # Short-run extrapolation proved unreliable on this machine (fixed
        # overhead and BLAS/thermal noise), so the budget is hit directly.
        def calibrate(run, n):
            for _ in range(4):
                t = med_time(lambda: run(n), reps=2)
                if abs(t / budget - 1) < 0.05:
                    break
                n = max(1, int(round(n * budget / t)))
            return n
        steps = calibrate(lambda n: dsb(Q, n_steps=n, dt=DSB_DT, seed=0, n_restarts=1), DSB_STEPS)
        sweeps = calibrate(lambda n: sa(Q, n_sweeps=n, seed=0), 100)
        print(f"[{enc}] dSB {steps} steps, SA {sweeps} sweeps")
        for solver in ("SA", "dSB"):
            for seed in range(N_SEEDS):
                t0 = time.perf_counter()
                if solver == "SA":
                    b, _ = sa(Q, n_sweeps=sweeps, seed=seed)
                else:
                    b, _ = dsb(Q, n_steps=steps, dt=DSB_DT, seed=seed, n_restarts=DSB_RESTARTS)
                el = time.perf_counter() - t0
                a_ = dec(b)
                ev = q.evaluate(a_)
                rows.append({"encoding": enc, "solver": solver, "seed": seed,
                             "profit": ev["profit"], "profit_free_disposal": profit_free_disposal(q, a_), "objective": ev["objective"],
                             "water": ev["water"], "unsuitable": ev["unsuitable"],
                             "invalid_parcels": viol(b), "seconds": el,
                             "iterations": sweeps if solver == "SA" else steps})
            v = [r for r in rows if r["encoding"] == enc and r["solver"] == solver]
            pr = np.array([r["profit"] for r in v])
            print(f"  {solver:<4} profit ${pr.mean()/1e6:.2f}M +/- {pr.std()/1e6:.2f}M | "
                  f"invalid {np.mean([r['invalid_parcels'] for r in v]):5.1f} | "
                  f"unsuitable {np.mean([r['unsuitable'] for r in v]):5.1f} | "
                  f"water {np.mean([r['water'] for r in v])/1e6:.2f} Mm3 | "
                  f"{np.mean([r['seconds'] for r in v]):.2f}s")

    t0 = time.perf_counter()
    ga = greedy(q)
    g = q.evaluate(ga)
    g_fd = profit_free_disposal(q, ga)
    rows.append({"encoding": "none", "solver": "greedy", "seed": 0, "profit": g["profit"],
                 "profit_free_disposal": g_fd,
                 "objective": g["objective"], "water": g["water"],
                 "unsuitable": g["unsuitable"], "invalid_parcels": 0,
                 "seconds": time.perf_counter() - t0, "iterations": 0})
    print(f"\ngreedy: profit ${g['profit']/1e6:.2f}M (model), ${g_fd/1e6:.2f}M (free disposal), "
          f"water {g['water']/1e6:.2f} Mm3")

    def prof(e, s):
        return np.array([r["profit"] for r in rows if r["encoding"] == e and r["solver"] == s])

    print()
    for enc in ("one-hot", "domain-wall"):
        a, b = prof(enc, "SA"), prof(enc, "dSB")
        p = mannwhitneyu(a, b).pvalue
        print(f"{enc:<12} SA vs dSB: {(a.mean()/b.mean()-1)*100:+.1f}%  Mann-Whitney p={p:.3g}")
    for s in ("SA", "dSB"):
        oh, dw = prof("one-hot", s), prof("domain-wall", s)
        p = mannwhitneyu(oh, dw).pvalue
        print(f"{s:<4} one-hot -> domain-wall: {(dw.mean()/oh.mean()-1)*100:+.1f}%  Mann-Whitney p={p:.3g}")
    for e in ("one-hot", "domain-wall"):
        for s in ("SA", "dSB"):
            fd = np.mean([r["profit_free_disposal"] for r in rows if r["encoding"] == e and r["solver"] == s])
            print(f"{e:<12}{s:<4} free-disposal profit ${fd/1e6:.2f}M vs greedy ${g_fd/1e6:.2f}M")

    out = os.path.join(ROOT, "data", "processed", "benchmark_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
