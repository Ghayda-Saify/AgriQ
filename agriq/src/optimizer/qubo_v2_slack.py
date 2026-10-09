"""AgriQ QUBO -- market-clearing formulation.

Demand enters as a linear inverse-demand curve, not a hard cap:
    revenue_j(Q) = p_j*Q - beta_j*Q^2 ,  beta_j = p_j/(2*D_j)
Marginal revenue reaches zero exactly at Q = D_j, so overproduction is
self-limiting through price. This is natively quadratic: no slack variables,
no penalty tuning, and none of the quantisation pathology that slack-encoded
hard caps suffer from.

Crop index 0 is FALLOW (zero yield/cost/water, always suitable): with a hard
"exactly one crop per parcel" constraint and a binding water budget, the
problem is infeasible unless land is allowed to rest.

Only the water budget stays a hard constraint (slack-encoded, single).
"""
from __future__ import annotations
import numpy as np


def _slack(bound, bits=8):
    if bound <= 0:
        return []
    return [bound * (2.0 ** -(k + 1)) for k in range(bits)]


class CropQUBO2:
    def __init__(self, area, yld, price, cost, suit, water_per_ha, demand,
                 water_budget, suit_min=0.6, M_onehot=8.0, M_suit=4.0,
                 M_water=8.0, water_bits=10):
        self.area = np.asarray(area, float)
        self.yld = np.asarray(yld, float)
        self.price = np.asarray(price, float)
        self.cost = np.asarray(cost, float)
        self.suit = np.asarray(suit, float)
        self.water = np.asarray(water_per_ha, float)
        self.demand = np.asarray(demand, float)
        self.water_budget = float(water_budget)
        self.suit_min = suit_min
        self.P, self.C = self.yld.shape
        self.prod = self.area[:, None] * self.yld
        self.wuse = self.area[:, None] * self.water[None, :]
        self.beta = np.where(self.demand > 0,
                             self.price / (2 * np.maximum(self.demand, 1e-9)), 0.0)

        self.n_x = self.P * self.C
        self.sw = (self.n_x, _slack(water_budget, water_bits))
        self.n = self.n_x + len(self.sw[1])
        self.pmax = float(np.abs(self.area[:, None] *
                                 (self.price * self.yld - self.cost)).max()) or 1.0

        Q = np.zeros((self.n, self.n))
        for i in range(self.P):
            for j in range(self.C):
                k = self.k(i, j)
                lin = self.price[j] * self.prod[i, j] - self.cost[j] * self.area[i]
                Q[k, k] -= (lin - self.beta[j] * self.prod[i, j] ** 2) / self.pmax
                for i2 in range(i + 1, self.P):
                    Q[k, self.k(i2, j)] += (2 * self.beta[j] * self.prod[i, j]
                                            * self.prod[i2, j]) / self.pmax
        for i in range(self.P):
            for j in range(self.C):
                Q[self.k(i, j), self.k(i, j)] -= M_onehot
                for j2 in range(j + 1, self.C):
                    Q[self.k(i, j), self.k(i, j2)] += 2 * M_onehot
        for i in range(self.P):
            for j in range(self.C):
                if self.suit[i, j] < suit_min:
                    Q[self.k(i, j), self.k(i, j)] += M_suit
        off, cs = self.sw
        terms = [(self.k(i, j), self.wuse[i, j])
                 for i in range(self.P) for j in range(self.C)]
        terms += [(off + b, c) for b, c in enumerate(cs)]
        s = self.water_budget ** 2
        for v, a in terms:
            Q[v, v] += M_water * (a * a - 2 * a * self.water_budget) / s
        for p in range(len(terms)):
            for q_ in range(p + 1, len(terms)):
                Q[terms[p][0], terms[q_][0]] += (2 * M_water * terms[p][1]
                                                 * terms[q_][1]) / s
        self.Q = Q

    def k(self, i, j):
        return i * self.C + j

    def energy(self, x):
        x = np.asarray(x, float)
        return float(x @ self.Q @ x)

    def net_revenue(self, assign):
        prod = np.zeros(self.C); tot = 0.0
        for i, j in enumerate(assign):
            if j < 0: continue
            prod[j] += self.prod[i, j]
            tot -= self.cost[j] * self.area[i]
        tot += float((self.price * prod - self.beta * prod ** 2).sum())
        return tot

    def evaluate(self, assign):
        wat = sum(self.wuse[i, j] for i, j in enumerate(assign) if j >= 0)
        uns = sum(1 for i, j in enumerate(assign)
                  if j >= 0 and self.suit[i, j] < self.suit_min)
        return {"net_revenue": self.net_revenue(assign), "water": wat,
                "over_water": max(wat - self.water_budget, 0.0),
                "unsuitable": uns,
                "feasible": wat <= self.water_budget + 1e-6 and uns == 0}
