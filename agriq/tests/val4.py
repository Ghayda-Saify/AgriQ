import itertools, numpy as np
from qubo3 import CropQUBO3
from solvers import dsb, sa, greedy, decode

def build(seed,P,C,lam=0.15,M=2.0):
    rng=np.random.default_rng(seed)
    area=rng.uniform(0.5,8.0,P)
    yld=np.zeros((P,C)); yld[:,1:]=rng.uniform(1.0,20.0,(P,C-1))
    price=np.zeros(C); price[1:]=rng.uniform(300,700,C-1)
    cost=np.zeros(C); cost[1:]=rng.uniform(200,2500,C-1)
    suit=np.ones((P,C)); suit[:,1:]=rng.uniform(0.3,1.0,(P,C-1))
    water=np.zeros(C); water[1:]=rng.uniform(2000,6000,C-1)
    prod=area[:,None]*yld
    demand=np.full(C,1e12); demand[1:]=prod[:,1:].sum(0)*rng.uniform(.35,.75,C-1)
    return CropQUBO3(area,yld,price,cost,suit,water,demand,lam=lam,
                     M_onehot=M,M_suit=M)

def brute(q):
    best,ba=-np.inf,None
    for a in itertools.product(range(q.C),repeat=q.P):
        ev=q.evaluate(a)
        if ev["feasible"] and ev["objective"]>best: best,ba=ev["objective"],a
    return best,ba

for M in (0.5,1.0,2.0,4.0,8.0):
    sb_opt=sa_opt=gr_opt=0; sb_g=[]; sa_g=[]; gr_g=[]; n=0
    for seed in range(16):
        q=build(seed,5,3,M=M)
        truth,_=brute(q)
        if truth==-np.inf: continue
        n+=1
        for name,assign in (("sb",decode(q,dsb(q.Q,n_steps=1200,seed=seed)[0])),
                            ("sa",decode(q,sa(q.Q,n_sweeps=300,seed=seed)[0])),
                            ("gr",greedy(q) if hasattr(q,'water_budget') else None)):
            pass
        a_sb=decode(q,dsb(q.Q,n_steps=1200,seed=seed)[0])
        a_sa=decode(q,sa(q.Q,n_sweeps=300,seed=seed)[0])
        m=q.area[:,None]*(q.price*q.yld-q.cost)-q.lam*q.wuse
        m=np.where(q.suit>=q.suit_min,m,-np.inf); m[:,0]=0.0
        a_gr=np.argmax(m,axis=1)
        for tag,a,lst in (("sb",a_sb,sb_g),("sa",a_sa,sa_g),("gr",a_gr,gr_g)):
            ev=q.evaluate(a); g=(truth-ev["objective"])/abs(truth)*100; lst.append(g)
            if abs(g)<1e-6:
                if tag=="sb": sb_opt+=1
                elif tag=="sa": sa_opt+=1
                else: gr_opt+=1
    print(f"M={M:>4}: dSB {sb_opt:2d}/{n} (gap {np.mean(sb_g):5.2f}%) | "
          f"SA {sa_opt:2d}/{n} (gap {np.mean(sa_g):5.2f}%) | "
          f"Greedy {gr_opt:2d}/{n} (gap {np.mean(gr_g):5.2f}%)")
