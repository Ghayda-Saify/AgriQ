"""Solvers for the AgriQ QUBO: discrete Simulated Bifurcation, Simulated
Annealing, and a Greedy baseline.

dSB follows Goto et al., Science Advances 7, eabe7953 (2021):
  y <- y + [-(a0 - p(t)) x + c0 (J sign(x) + h)] dt
  x <- x + a0 y dt
  inelastic walls: |x|>1  =>  x = sign(x), y = 0
with the QUBO mapped to Ising via x_i = (s_i + 1)/2.

SA uses O(n) incremental delta-energy (NOT a full x'Qx per flip) so that any
runtime comparison against dSB reflects the algorithms rather than the
implementations.
"""
from __future__ import annotations
import numpy as np, time


def qubo_to_ising(Q):
    Qs = (Q + Q.T) / 2.0
    A = Qs / 2.0
    np.fill_diagonal(A, 0.0)
    b = 0.5 * Qs.sum(axis=1)
    return A, b


def spins_to_binary(s):
    return ((s + 1) // 2).astype(np.int8)


# --------------------------------------------------------------------------
def dsb(Q, n_steps=1500, dt=0.5, a0=1.0, seed=0, n_restarts=1):
    """Discrete Simulated Bifurcation. Returns (best_binary, best_energy)."""
    A, b = qubo_to_ising(Q)
    N = Q.shape[0]
    off = A[~np.eye(N, dtype=bool)]
    sigma = off.std() if off.size and off.std() > 0 else 1.0
    c0 = 0.5 * a0 / (np.sqrt(N) * sigma)
    rng = np.random.default_rng(seed)
    best_x, best_e = None, np.inf

    for r in range(n_restarts):
        x = 0.1 * (rng.random(N) * 2 - 1)
        y = 0.1 * (rng.random(N) * 2 - 1)
        for t in range(n_steps):
            p = a0 * t / n_steps                      # pump 0 -> a0
            sx = np.sign(x); sx[sx == 0] = 1.0
            y += (-(a0 - p) * x - c0 * (A @ sx + b)) * dt
            x += a0 * y * dt
            wall = np.abs(x) > 1.0
            if wall.any():
                x[wall] = np.sign(x[wall]); y[wall] = 0.0
            if t % 50 == 0 or t == n_steps - 1:       # track best seen
                xb = (np.sign(x) > 0).astype(float)
                e = float(xb @ Q @ xb)
                if e < best_e: best_e, best_x = e, xb.copy()
        xb = (np.sign(x) > 0).astype(float)
        e = float(xb @ Q @ xb)
        if e < best_e: best_e, best_x = e, xb.copy()
    return best_x.astype(int), best_e


# --------------------------------------------------------------------------
def sa(Q, n_sweeps=200, T0=None, T1=None, seed=0):
    """Simulated annealing with O(n) incremental delta-energy."""
    Qs = (Q + Q.T) / 2.0
    N = Q.shape[0]
    d = np.diag(Qs).copy()
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, N).astype(float)
    h = Qs @ x
    e = float(x @ Qs @ x)
    best_x, best_e = x.copy(), e

    if T0 is None:
        scale = np.abs(Qs).sum(axis=1).mean()
        T0 = max(scale * 0.5, 1e-9)
    if T1 is None:
        T1 = T0 * 1e-4
    n_steps = n_sweeps * N
    cool = (T1 / T0) ** (1.0 / max(n_steps, 1))
    T = T0
    for it in range(n_steps):
        k = int(rng.integers(0, N))
        v = x[k]; dv = 1.0 - 2.0 * v
        delta = dv * (d[k] + 2.0 * (h[k] - d[k] * v))
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            x[k] = 1.0 - v
            h += dv * Qs[:, k]
            e += delta
            if e < best_e: best_e, best_x = e, x.copy()
        T *= cool
    return best_x.astype(int), float(best_e)


# --------------------------------------------------------------------------
def repair_water(q, assign, budget):
    """Fallow parcels until total irrigation fits `budget`, worst first.

    Applied identically to EVERY arm. Previously only greedy was held to the
    budget while the QUBO solvers ran unconstrained, which both made the
    greedy comparison unfair and let the solvers buy profit with water nobody
    has: plans used 5.97-7.50 Mm3 against a licensed agricultural abstraction
    of 1.0 Mm3 for Jenin (PCBS 2018).

    Parcels are dropped in ascending order of margin PER CUBIC METRE, which is
    the correct order when water is the scarce input -- dropping by absolute
    margin would shed efficient low-water parcels before thirsty ones.
    """
    assign = np.asarray(assign).copy()
    margin = q.area[:, None] * (q.price * q.yld - q.cost)
    water = float(sum(q.wuse[i, assign[i]] for i in range(q.P)))
    if water <= budget:
        return assign, water
    eff = []
    for i in range(q.P):
        j = assign[i]
        w = q.wuse[i, j]
        eff.append(margin[i, j] / w if (j != 0 and w > 0) else -np.inf)
    for i in np.argsort(eff):          # ascending: least profit per m3 goes first
        if water <= budget:
            break
        if assign[i] != 0:
            water -= q.wuse[i, assign[i]]
            assign[i] = 0
    return assign, water


def greedy(q):
    """Per-parcel best gross margin, ignoring market coupling. Unconstrained;
    the caller applies repair_water so every arm is treated identically."""
    margin = q.area[:, None] * (q.price * q.yld - q.cost)
    margin = np.where(q.suit >= q.suit_min, margin, -np.inf)
    margin[:, 0] = 0.0                       # fallow always available
    return np.argmax(margin, axis=1)


def decode(q, binary):
    """Binary vector -> per-parcel crop index (argmax within each parcel block)."""
    assign = np.zeros(q.P, dtype=int)
    for i in range(q.P):
        seg = binary[i * q.C:(i + 1) * q.C]
        assign[i] = int(np.argmax(seg)) if seg.sum() > 0 else 0
    return assign


def timed(fn, *a, **kw):
    t = time.perf_counter(); out = fn(*a, **kw)
    return out, time.perf_counter() - t
