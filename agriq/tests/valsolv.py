import itertools, numpy as np
from qubo2 import CropQUBO2
from solvers import dsb, sa, greedy, decode
from val3 import build, energy_of

rows=[]
for seed in range(12):
    q=build(seed,5,3,14,1e4)
    truth,ta=-np.inf,None
    for a in itertools.product(range(3),repeat=5):
        ev=q.evaluate(a)
        if ev["feasible"] and ev["net_revenue"]>truth: truth,ta=ev["net_revenue"],a
    if ta is None: continue
    xb,_=dsb(q.Q,n_steps=1500,seed=seed); a_sb=decode(q,xb)
    xa,_=sa(q.Q,n_sweeps=300,seed=seed);  a_sa=decode(q,xa)
    a_gr=greedy(q)
    r=dict(seed=seed,truth=truth,
           sb=q.evaluate(a_sb),sa=q.evaluate(a_sa),gr=q.evaluate(list(a_gr)))
    rows.append(r)

def summ(k,name):
    ok=[r for r in rows if r[k]["feasible"]]
    opt=[r for r in rows if r[k]["feasible"] and abs(r[k]["net_revenue"]-r["truth"])<1e-6]
    gaps=[(r["truth"]-r[k]["net_revenue"])/abs(r["truth"])*100 for r in rows if r[k]["feasible"]]
    print(f"  {name:8s} feasible {len(ok)}/{len(rows)}   optimal {len(opt)}/{len(rows)}   "
          f"mean gap {np.mean(gaps) if gaps else float('nan'):6.2f}%")
print(f"5 parcels x (fallow+2), {len(rows)} instances with non-empty feasible space:")
summ("sb","dSB"); summ("sa","SA"); summ("gr","Greedy")
