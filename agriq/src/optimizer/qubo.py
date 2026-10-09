"""AgriQ QUBO v3 -- fully priced formulation. No slack variables.

Both capacity limits enter the OBJECTIVE rather than as penalty constraints:

  * demand : linear inverse-demand curve, revenue_j(Q) = p_j Q - beta_j Q^2,
             beta_j = p_j/(2 D_j). Overproduction depresses price.
  * water  : charged at lambda $/m^3. Sweeping lambda until total usage meets
             the governorate budget recovers the SHADOW PRICE of irrigation
             water -- an economic result, not a tuning artefact.

This leaves exactly one penalty term (one-crop-per-parcel), which can be sized
directly against the objective range. The v2 formulation needed M_water ~ 1e4
for exactness, which drowned the profit signal (magnitude ~1) and made the
problem unsolvable by any heuristic. Pricing removes that pathology entirely.

Crop 0 is FALLOW.
"""
from __future__ import annotations
import numpy as np


class CropQUBO3:
    def __init__(self, area, yld, price, cost, suit, water_per_ha, demand,
                 lam=0.15, suit_min=0.6, M_onehot=None, M_suit=None):
        self.area=np.asarray(area,float); self.yld=np.asarray(yld,float)
        self.price=np.asarray(price,float); self.cost=np.asarray(cost,float)
        self.suit=np.asarray(suit,float); self.water=np.asarray(water_per_ha,float)
        self.demand=np.asarray(demand,float); self.lam=float(lam)
        self.suit_min=suit_min
        self.P,self.C=self.yld.shape
        self.prod=self.area[:,None]*self.yld
        self.wuse=self.area[:,None]*self.water[None,:]
        self.beta=np.where(self.demand>0,self.price/(2*np.maximum(self.demand,1e-9)),0.0)

        # linear part of net margin (before price depression)
        self.lin=(self.price*self.prod - self.cost*self.area[:,None]
                  - self.lam*self.wuse)
        self.pmax=float(np.abs(self.lin).max()) or 1.0
        self.n=self.P*self.C

        if M_onehot is None: M_onehot=2.0
        if M_suit is None:   M_suit=2.0
        self.M_onehot=M_onehot

        Q=np.zeros((self.n,self.n))
        for i in range(self.P):
            for j in range(self.C):
                k=self.k(i,j)
                Q[k,k]-=(self.lin[i,j]-self.beta[j]*self.prod[i,j]**2)/self.pmax
                for i2 in range(i+1,self.P):
                    Q[k,self.k(i2,j)]+=2*self.beta[j]*self.prod[i,j]*self.prod[i2,j]/self.pmax
        for i in range(self.P):
            for j in range(self.C):
                Q[self.k(i,j),self.k(i,j)]-=M_onehot
                for j2 in range(j+1,self.C):
                    Q[self.k(i,j),self.k(i,j2)]+=2*M_onehot
                if self.suit[i,j]<suit_min: Q[self.k(i,j),self.k(i,j)]+=M_suit
        self.Q=Q

    def k(self,i,j): return i*self.C+j
    def energy(self,x): x=np.asarray(x,float); return float(x@self.Q@x)

    def objective(self,assign):
        """True objective in USD: market-cleared revenue - cost - water charge."""
        prod=np.zeros(self.C); tot=0.0; wat=0.0
        for i,j in enumerate(assign):
            if j<0: continue
            prod[j]+=self.prod[i,j]
            tot-=self.cost[j]*self.area[i]
            wat+=self.wuse[i,j]
        tot+=float((self.price*prod-self.beta*prod**2).sum())
        return tot-self.lam*wat, wat

    def evaluate(self,assign):
        obj,wat=self.objective(assign)
        uns=sum(1 for i,j in enumerate(assign) if j>=0 and self.suit[i,j]<self.suit_min)
        # gross farm profit excluding the internal water charge (what a farmer sees)
        prod=np.zeros(self.C); gross=0.0
        for i,j in enumerate(assign):
            if j<0: continue
            prod[j]+=self.prod[i,j]; gross-=self.cost[j]*self.area[i]
        gross+=float((self.price*prod-self.beta*prod**2).sum())
        return {"objective":obj,"profit":gross,"water":wat,"unsuitable":uns,
                "feasible":uns==0}
