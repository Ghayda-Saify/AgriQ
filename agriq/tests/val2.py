import itertools, numpy as np
from qubo2 import CropQUBO2

def best_e(q, assign):
    x = np.zeros(q.n)
    for i, j in enumerate(assign):
        if j >= 0: x[q.k(i, j)] = 1.0
    off, cs = q.sw
    if not cs: return q.energy(x)
    be = np.inf
    for bits in itertools.product([0, 1], repeat=len(cs)):
        for b, v in enumerate(bits): x[off + b] = v
        e = q.energy(x)
        if e < be: be = e
    return be

def build(seed, P, C, bits):
    rng = np.random.default_rng(seed)
    area = rng.uniform(0.5, 8.0, P)
    yld = np.zeros((P, C)); yld[:, 1:] = rng.uniform(1.0, 20.0, (P, C-1))
    price = np.zeros(C); price[1:] = rng.uniform(300, 700, C-1)
    cost = np.zeros(C); cost[1:] = rng.uniform(200, 2500, C-1)
    suit = np.ones((P, C)); suit[:, 1:] = rng.uniform(0.3, 1.0, (P, C-1))
    water = np.zeros(C); water[1:] = rng.uniform(2000, 6000, C-1)
    prod = area[:, None] * yld
    demand = np.full(C, 1e12)
    demand[1:] = prod[:, 1:].sum(0) * rng.uniform(.35, .75, C-1)
    wb = (area[:, None] * water[None, :])[:, 1:].mean(1).sum() * 0.60
    return CropQUBO2(area, yld, price, cost, suit, water, demand, wb, water_bits=bits)

def test(seed, P=5, C=3, bits=8):
    q = build(seed, P, C, bits)
    truth, ta = -np.inf, None
    for a in itertools.product(range(C), repeat=P):
        ev = q.evaluate(a)
        if ev["feasible"] and ev["net_revenue"] > truth:
            truth, ta = ev["net_revenue"], a
    if ta is None: return None
    be_, ba = np.inf, None
    for a in itertools.product(range(C), repeat=P):
        e = best_e(q, a)
        if e < be_: be_, ba = e, a
    ev = q.evaluate(ba)
    return ev["feasible"] and abs(ev["net_revenue"] - truth) < 1e-6

for P, C, bits, n in [(5,3,8,24), (6,4,7,12), (7,3,7,10)]:
    r = [test(s,P,C,bits) for s in range(n)]
    r = [x for x in r if x is not None]
    tag = "PASS" if sum(r)==len(r) else "FAIL"
    print(f"{P} parcels x (fallow+{C-1} crops), {bits}-bit water slack: "
          f"{sum(r)}/{len(r)} exact  [{tag}]")
