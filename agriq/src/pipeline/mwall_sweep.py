"""Is the SA domain-wall penalty an encoding effect or a penalty-weight artefact?

The 2x2 benchmark compares one-hot at penalty weight M=0.5 against domain-wall
whose ordering penalty was fixed at M_wall=1.0 and never tuned. That is a
confound: the replacement penalty carries twice the weight of the penalty it
replaces, so "domain-wall costs SA 8.9%" could instead read "M_wall=1.0 is too
high for single-bit moves".

This sweeps M_wall and reports SA and dSB profit at each value, at the same
wall-clock budget and seed count as the benchmark. Two possible outcomes, and
the paper's claim differs between them:

  * SA's deficit persists across M_wall  -> the interaction is structural, and
    the encoding claim stands as written.
  * SA's deficit vanishes at some M_wall -> the claim must narrow to
    "domain-wall needs per-solver penalty tuning; one-hot does not".

The matrix dimension is unchanged by M_wall (1,740 either way), so the step and
sweep counts calibrated for the benchmark's domain-wall arm are reused, and the
achieved wall-clock is verified per arm.
"""
import csv
import os
import sys
import time

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "optimizer"))
sys.path.insert(0, os.path.join(ROOT, "src", "pipeline"))
from instance import build
from solvers import sa, dsb, decode, repair_water
from domain_wall import to_domain_wall, decode_dw
from benchmark import (dw_bad_parcels, onehot_bad_parcels, BUDGET_S, DSB_DT,
                       WATER_BUDGET)

M_WALLS = [0.25, 0.5, 1.0, 2.0, 4.0]
N_SEEDS = 20
TOL = 0.20          # looser than the benchmark's 0.15: this is a within-encoding
                    # comparison, so a few percent of drift cannot flip a sign


def med_time(fn, reps=2):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def calibrate(run, n):
    """Hit BUDGET_S directly. Reusing the benchmark's stored counts is wrong:
    they were calibrated under that run's machine load and gave 0.6-0.7s here."""
    for _ in range(5):
        t = med_time(lambda: run(n))
        if abs(t / BUDGET_S - 1) < 0.05:
            break
        n = max(1, int(round(n * BUDGET_S / t)))
    return n


def main():
    q = build("A")
    P, C = q.P, q.C
    q_nooh = build("A", M_onehot=0.0)        # suitability penalty retained
    Qd1 = to_domain_wall(q_nooh.Q, P, C, M_wall=1.0)
    print(f"instance: {P} parcels | domain-wall {P*(C-1)} vars")

    # Calibrate on this host. M_wall changes coefficients, not dimension, so one
    # calibration on the domain-wall matrix serves every M_wall value.
    cal = {"dSB": calibrate(lambda n: dsb(Qd1, n_steps=n, dt=DSB_DT, seed=0,
                                          n_restarts=1), 1000),
           "SA": calibrate(lambda n: sa(Qd1, n_sweeps=n, seed=0), 100)}
    cal_oh = {"dSB": calibrate(lambda n: dsb(q.Q, n_steps=n, dt=DSB_DT, seed=0,
                                             n_restarts=1), 1000),
              "SA": calibrate(lambda n: sa(q.Q, n_sweeps=n, seed=0), 100)}
    print(f"calibrated to {BUDGET_S:.2f}s: domain-wall dSB {cal['dSB']} steps / "
          f"SA {cal['SA']} sweeps | one-hot dSB {cal_oh['dSB']} / SA {cal_oh['SA']}")
    print(f"{N_SEEDS} seeds per arm\n")

    # one-hot reference, so the table shows whether SA's deficit closes
    ref, ref_w = {}, {}
    for solver in ("SA", "dSB"):
        pr, wt = [], []
        for seed in range(N_SEEDS):
            if solver == "SA":
                b, _ = sa(q.Q, n_sweeps=cal_oh["SA"], seed=seed)
            else:
                b, _ = dsb(q.Q, n_steps=cal_oh["dSB"], dt=DSB_DT, seed=seed, n_restarts=1)
            ev = q.evaluate(decode(q, b))
            pr.append(ev["profit"]); wt.append(ev["water"])
        ref[solver] = np.array(pr); ref_w[solver] = float(np.mean(wt))
    print(f"one-hot reference: SA ${ref['SA'].mean()/1e6:.2f}M "
          f"(water {ref_w['SA']/1e6:.2f} Mm3), dSB ${ref['dSB'].mean()/1e6:.2f}M "
          f"(water {ref_w['dSB']/1e6:.2f} Mm3)\n")

    rows, times = [], []
    print(f"{'M_wall':>7} {'solver':<5} {'profit $M':>13} {'invalid':>8} {'unsuit':>7} "
          f"{'water':>6} {'vs one-hot':>11} {'p':>9} {'budget$M':>8} {'sec':>5}")
    print("-" * 96)
    for mw in M_WALLS:
        Qd = to_domain_wall(q_nooh.Q, P, C, M_wall=mw)
        for solver in ("SA", "dSB"):
            pr, inv, uns, wat, el, pb = [], [], [], [], [], []
            for seed in range(N_SEEDS):
                t0 = time.perf_counter()
                if solver == "SA":
                    b, _ = sa(Qd, n_sweeps=cal["SA"], seed=seed)
                else:
                    b, _ = dsb(Qd, n_steps=cal["dSB"], dt=DSB_DT, seed=seed, n_restarts=1)
                el.append(time.perf_counter() - t0)
                a = decode_dw(b, P, C)
                ev = q.evaluate(a)
                pr.append(ev["profit"]); uns.append(ev["unsuitable"])
                wat.append(ev["water"]); inv.append(dw_bad_parcels(b, P, C))
                # The unconstrained profit can be bought with water nobody has,
                # so also report the plan held to the PCBS agricultural budget.
                a_rep, _w = repair_water(q, a, WATER_BUDGET)
                pb.append(q.evaluate(a_rep)["profit"])
            pr = np.array(pr); pb = np.array(pb)
            delta = (pr.mean() / ref[solver].mean() - 1) * 100
            p = mannwhitneyu(pr, ref[solver], alternative="two-sided")[1]
            med = float(np.median(el))
            times.append((mw, solver, med))
            print(f"{mw:>7.2f} {solver:<5} {pr.mean()/1e6:>7.2f} +/-{pr.std()/1e6:4.2f} "
                  f"{np.mean(inv):>8.1f} {np.mean(uns):>7.1f} "
                  f"{np.mean(wat)/1e6:>6.2f} {delta:>+10.1f}% {p:>9.2g} "
                  f"{pb.mean()/1e6:>7.2f} {med:>5.2f}")
            for seed, v, i2, u2, w2, v2 in zip(range(N_SEEDS), pr, inv, uns, wat, pb):
                rows.append({"m_wall": mw, "solver": solver, "seed": seed,
                             "profit": float(v), "invalid_parcels": int(i2),
                             "unsuitable": int(u2), "water": float(w2),
                             "profit_budget": float(v2)})

    off = [(mw, s, t) for mw, s, t in times if abs(t / BUDGET_S - 1) > TOL]
    if off:
        raise RuntimeError(f"budget not met, arms not comparable: {off}")
    print(f"\nall arms within {TOL:.0%} of the {BUDGET_S:.2f}s budget")

    out = os.path.join(ROOT, "data", "processed", "mwall_sweep.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
