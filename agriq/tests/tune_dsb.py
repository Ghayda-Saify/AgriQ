import itertools, numpy as np
from solvers import dsb, decode
from val4 import build, brute

print(f"{'dt':>5} {'steps':>6} {'restarts':>9} | {'opt':>6} {'gap%':>7}")
print("-"*40)
best=None
for dt in (0.05,0.1,0.25,0.5,1.0):
    for steps in (500,2000):
        for nr in (1,5):
            opt=0; gaps=[]; n=0
            for seed in range(16):
                q=build(seed,5,3,M=0.5)
                truth,_=brute(q)
                if truth==-np.inf: continue
                n+=1
                a=decode(q,dsb(q.Q,n_steps=steps,dt=dt,seed=seed,n_restarts=nr)[0])
                g=(truth-q.evaluate(a)["objective"])/abs(truth)*100
                gaps.append(g)
                if abs(g)<1e-6: opt+=1
            mg=float(np.mean(gaps))
            print(f"{dt:>5} {steps:>6} {nr:>9} | {opt:>3}/{n} {mg:>7.2f}")
            if best is None or (opt,-mg)>(best[0],-best[1]): best=(opt,mg,dt,steps,nr)
print(f"\nbest: dt={best[2]} steps={best[3]} restarts={best[4]} -> {best[0]}/16, gap {best[1]:.2f}%")
