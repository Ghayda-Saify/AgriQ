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
from solvers import dsb, sa, greedy, decode, repair_water
from domain_wall import to_domain_wall, decode_dw

N_SEEDS = 20
DSB_STEPS, DSB_DT, DSB_RESTARTS = 1000, 0.25, 1
# Fixed wall-clock budget per run, in seconds. Previously this was derived by
# timing one-hot dSB at DSB_STEPS on the host, which made the budget -- and so
# every reported profit -- depend on machine state: the same code gave 2.51 s
# and 0.79 s on the same machine under different BLAS/thermal conditions. A
# stated constant is reproducible; step and sweep counts are calibrated to it.
BUDGET_S = 2.5
M_WALL = 1.0
WATER_BUDGET = 1.0e6
# m3/yr. PCBS, "Palestinian Water Wells and Annual Pumping Quantity by
# Governorate and Type of Use, 2018": Jenin agriculture 1.0 Mm3, domestic
# 5.0 Mm3, total 6.0 Mm3. An earlier version used 6.0 as the AGRICULTURAL
# budget, but 6.0 is the total including domestic supply -- six times too
# large. The figure covers licensed wells only, so it excludes springs,
# Mekorot purchases, treated-wastewater reuse and unlicensed abstraction; it
# is a lower bound on agricultural water and is reported as such.


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

    budget = BUDGET_S
    ref = med_time(lambda: dsb(q.Q, n_steps=DSB_STEPS, dt=DSB_DT, seed=0,
                               n_restarts=DSB_RESTARTS))
    print(f"wall-clock budget per run: {budget:.2f}s (fixed) | "
          f"one-hot dSB at {DSB_STEPS} steps takes {ref:.2f}s on this host")

    encs = (("one-hot", q.Q, lambda b: decode(q, b), lambda b: onehot_bad_parcels(b, P, C)),
            ("domain-wall", Q_dw, lambda b: decode_dw(b, P, C), lambda b: dw_bad_parcels(b, P, C)))
    # Calibrating once and trusting it does not work on this host: the same
    # 1,000-step dSB run measured 3.10s, ~2.50s and 1.11s inside a single
    # session. Calibration converges, then the seed runs land somewhere else.
    # So each arm is calibrated, run, and CHECKED against its own achieved
    # median; if it missed, the achieved median rescales the count and the arm
    # is re-run. Convergence comes from measuring the thing being reported
    # rather than a proxy measured earlier.
    MAX_ATTEMPTS = 4

    def run_arm(Q, dec, viol, enc, solver, n):
        """One arm at count n. Returns (records, achieved median seconds)."""
        recs = []
        for seed in range(N_SEEDS):
            t0 = time.perf_counter()
            if solver == "SA":
                b, _ = sa(Q, n_sweeps=n, seed=seed)
            else:
                b, _ = dsb(Q, n_steps=n, dt=DSB_DT, seed=seed, n_restarts=DSB_RESTARTS)
            el = time.perf_counter() - t0
            a_ = dec(b)
            ev = q.evaluate(a_)
            a_rep, w_rep = repair_water(q, a_, WATER_BUDGET)
            ev_rep = q.evaluate(a_rep)
            recs.append({"encoding": enc, "solver": solver, "seed": seed,
                         "profit": ev["profit"],
                         "profit_free_disposal": profit_free_disposal(q, a_),
                         "objective": ev["objective"], "water": ev["water"],
                         "unsuitable": ev["unsuitable"], "invalid_parcels": viol(b),
                         "seconds": el, "iterations": n,
                         "profit_budget": ev_rep["profit"], "water_budget": w_rep,
                         "cultivated_budget": int((a_rep != 0).sum())})
        return recs, float(np.median([r["seconds"] for r in recs]))

    rows = []
    for enc, Q, dec, viol in encs:
        def calibrate(run, n):
            for _ in range(4):
                t = med_time(lambda: run(n), reps=2)
                if abs(t / budget - 1) < 0.05:
                    break
                n = max(1, int(round(n * budget / t)))
            return n
        steps = calibrate(lambda n: dsb(Q, n_steps=n, dt=DSB_DT, seed=0, n_restarts=1), DSB_STEPS)
        sweeps = calibrate(lambda n: sa(Q, n_sweeps=n, seed=0), 100)
        print(f"[{enc}] dSB {steps} steps, SA {sweeps} sweeps (pre-calibration)")
        for solver in ("SA", "dSB"):
            n = sweeps if solver == "SA" else steps
            for attempt in range(1, MAX_ATTEMPTS + 1):
                recs, med = run_arm(Q, dec, viol, enc, solver, n)
                if abs(med / budget - 1) <= 0.12 or attempt == MAX_ATTEMPTS:
                    if attempt > 1:
                        print(f"       {solver} settled at n={n} after {attempt} "
                              f"attempts ({med:.2f}s)")
                    break
                n_new = max(1, int(round(n * budget / med)))
                print(f"       {solver} n={n} achieved {med:.2f}s "
                      f"({med/budget-1:+.0%}) -> retry at n={n_new}")
                n = n_new
            rows.extend(recs)
            if solver == "SA":
                sweeps = n
            else:
                steps = n
        for solver in ("SA", "dSB"):
            v = [r for r in rows if r["encoding"] == enc and r["solver"] == solver]
            pr = np.array([r["profit"] for r in v])
            pb = np.array([r["profit_budget"] for r in v])
            print(f"  {solver:<4} profit ${pr.mean()/1e6:.2f}M +/- {pr.std()/1e6:.2f}M | "
                  f"invalid {np.mean([r['invalid_parcels'] for r in v]):5.1f} | "
                  f"unsuit {np.mean([r['unsuitable'] for r in v]):4.1f} | "
                  f"water {np.mean([r['water'] for r in v])/1e6:.2f} Mm3 | "
                  f"budget-feasible ${pb.mean()/1e6:.2f}M +/-{pb.std()/1e6:.2f}M on "
                  f"{np.mean([r['cultivated_budget'] for r in v]):.0f} parcels | "
                  f"{np.mean([r['seconds'] for r in v]):.2f}s")

    # The whole comparison is only meaningful if all four arms actually consumed
    # the same wall-clock time. Calibration can converge and then be invalidated
    # by machine load arriving between calibration and the seed runs, which once
    # gave one-hot 3.2x the compute of domain-wall and silently turned the
    # encoding effect into a compute effect. Verify the achieved times and refuse
    # to write results that are not comparable.
    TOL = 0.15
    achieved = {(e, s): float(np.median([r["seconds"] for r in rows
                                         if r["encoding"] == e and r["solver"] == s]))
                for e in ("one-hot", "domain-wall") for s in ("SA", "dSB")}
    print("\nachieved median wall-clock per run (budget "
          f"{budget:.2f}s, tolerance {TOL:.0%}):")
    for k, v in achieved.items():
        print(f"  {k[0]:<12}{k[1]:<4} {v:5.2f}s  ({v/budget - 1:+.0%})")
    off = {k: v for k, v in achieved.items() if abs(v / budget - 1) > TOL}
    if off:
        raise RuntimeError(
            "wall-clock budget not met, so the arms are not comparable and no "
            f"results were written: {off}. Rerun on an otherwise idle machine.")

    t0 = time.perf_counter()
    ga = greedy(q)
    g = q.evaluate(ga)
    g_fd = profit_free_disposal(q, ga)
    ga_rep, gw_rep = repair_water(q, ga, WATER_BUDGET)
    g_rep = q.evaluate(ga_rep)
    rows.append({"encoding": "none", "solver": "greedy", "seed": 0, "profit": g["profit"],
                 "profit_free_disposal": g_fd,
                 "objective": g["objective"], "water": g["water"],
                 "unsuitable": g["unsuitable"], "invalid_parcels": 0,
                 "seconds": time.perf_counter() - t0, "iterations": 0,
                 "profit_budget": g_rep["profit"], "water_budget": gw_rep,
                 "cultivated_budget": int((ga_rep != 0).sum())})
    print(f"\ngreedy: profit ${g['profit']/1e6:.2f}M (model), ${g_fd/1e6:.2f}M (free disposal), "
          f"water {g['water']/1e6:.2f} Mm3 | budget-feasible ${g_rep['profit']/1e6:.2f}M on "
          f"{int((ga_rep != 0).sum())} parcels")

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

    print(f"\n--- held to the {WATER_BUDGET/1e6:.1f} Mm3 PCBS agricultural budget ---")

    def profb(e, s):
        return np.array([r["profit_budget"] for r in rows
                         if r["encoding"] == e and r["solver"] == s])

    for s in ("SA", "dSB"):
        oh, dw = profb("one-hot", s), profb("domain-wall", s)
        print(f"{s:<4} one-hot -> domain-wall: {(dw.mean()/oh.mean()-1)*100:+.1f}%  "
              f"Mann-Whitney p={mannwhitneyu(oh, dw).pvalue:.3g}")
    for enc in ("one-hot", "domain-wall"):
        a, b2 = profb(enc, "SA"), profb(enc, "dSB")
        print(f"{enc:<12} SA vs dSB: {(a.mean()/b2.mean()-1)*100:+.1f}%  "
              f"Mann-Whitney p={mannwhitneyu(a, b2).pvalue:.3g}")
    print(f"greedy budget-feasible: ${g_rep['profit']/1e6:.2f}M")

    out = os.path.join(ROOT, "data", "processed", "benchmark_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
