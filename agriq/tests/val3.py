import itertools, numpy as np
from qubo2 import CropQUBO2

def greedy_slack(q, x, usage):
    """Optimal slack for a binary expansion: greedy on descending coefficients."""
    off, cs = q.sw
    target = max(0.0, q.water_budget - usage)
    for b, c in enumerate(cs):
        if c <= target + 1e-12:
            x[off + b] = 1.0; target -= c
        else:
            x[off + b] = 0.0
    return x

def energy_of(q, assign):
    x = np.zeros(q.n); usage = 0.0
    for i, j in enumerate(assign):
        if j >= 0:
            x[q.k(i, j)] = 1.0; usage += q.wuse[i, j]
    greedy_slack(q, x, usage)
    return q.energy(x)

def build(seed, P, C, bits, Mw, tight=0.60):
    rng = np.random.default_rng(seed)
    area = rng.uniform(0.5, 8.0, P)
    yld = np.zeros((P, C)); yld[:, 1:] = rng.uniform(1.0, 20.0, (P, C-1))
    price = np.zeros(C); price[1:] = rng.uniform(300, 700, C-1)
    cost = np.zeros(C); cost[1:] = rng.uniform(200, 2500, C-1)
    suit = np.ones((P, C)); suit[:, 1:] = rng.uniform(0.3, 1.0, (P, C-1))
    water = np.zeros(C); water[1:] = rng.uniform(2000, 6000, C-1)
    prod = area[:, None] * yld
    demand = np.full(C, 1e12); demand[1:] = prod[:, 1:].sum(0)*rng.uniform(.35,.75,C-1)
    wb = (area[:, None]*water[None, :])[:, 1:].mean(1).sum()*tight
    return CropQUBO2(area, yld, price, cost, suit, water, demand, wb,
                     water_bits=bits, M_water=Mw)

def test(seed, P, C, bits, Mw):
    q = build(seed, P, C, bits, Mw)
    truth, ta = -np.inf, None
    for a in itertools.product(range(C), repeat=P):
        ev = q.evaluate(a)
        if ev["feasible"] and ev["net_revenue"] > truth: truth, ta = ev["net_revenue"], a
    if ta is None: return None
    be, ba = np.inf, None
    for a in itertools.product(range(C), repeat=P):
        e = energy_of(q, a)
        if e < be: be, ba = e, a
    ev = q.evaluate(ba)
    return ev["feasible"] and abs(ev["net_revenue"] - truth) < 1e-6

print(f"{'bits':>5} {'M_water':>9} | {'5x3':>7} {'6x4':>7} {'7x3':>7}")
print("-" * 44)
for bits in (8, 10, 12, 14):
    for Mw in (1e2, 1e3, 1e4, 1e5, 1e6):
        out = []
        for P, C, n in [(5,3,24), (6,4,14), (7,3,12)]:
            r = [test(s,P,C,bits,Mw) for s in range(n)]
            r = [x for x in r if x is not None]
            out.append(f"{sum(r)}/{len(r)}")
        print(f"{bits:>5} {Mw:>9.0e} | {out[0]:>7} {out[1]:>7} {out[2]:>7}")
