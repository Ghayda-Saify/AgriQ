"""Domain-wall encoding (Chancellor, Quantum Sci. Technol. 4, 045004 (2019)).

One-hot over C crops is replaced by C-1 binaries d_1..d_{C-1} per parcel,
required to be monotonically non-increasing (d_0=1, d_C=0 fixed). The single
descent -- the 'domain wall' -- selects the crop:

    x_{i,j} = d_{i,j} - d_{i,j+1}

Any monotone d is automatically one-hot in x, so the all-to-all one-hot
penalty disappears. What remains is a CHAIN of 2-body ordering terms, which
Ising-type solvers handle far better than dense penalty blocks.
"""
from __future__ import annotations
import numpy as np


def build_transform(P, C):
    """x = A d + b, with d of length P*(C-1)."""
    nx, nd = P * C, P * (C - 1)
    A = np.zeros((nx, nd)); b = np.zeros(nx)
    for i in range(P):
        for j in range(C):
            xi = i * C + j
            if j == 0:
                b[xi] = 1.0
                A[xi, i * (C - 1) + 0] = -1.0
            elif j == C - 1:
                A[xi, i * (C - 1) + (C - 2)] = 1.0
            else:
                A[xi, i * (C - 1) + (j - 1)] = 1.0
                A[xi, i * (C - 1) + j] = -1.0
    return A, b


def to_domain_wall(Qx, P, C, M_wall=1.0):
    """Transform an x-space QUBO (built WITHOUT a one-hot penalty) to d-space."""
    A, b = build_transform(P, C)
    Qs = (Qx + Qx.T) / 2.0
    Qd = A.T @ Qs @ A
    lin = 2.0 * (b @ Qs @ A)
    Qd[np.diag_indices_from(Qd)] += lin
    # ordering penalty: forbid d_j=0, d_{j+1}=1  ->  M*(d_{j+1} - d_j d_{j+1})
    for i in range(P):
        for j in range(C - 2):
            a = i * (C - 1) + j
            c = i * (C - 1) + j + 1
            Qd[c, c] += M_wall
            Qd[a, c] += -M_wall / 2.0
            Qd[c, a] += -M_wall / 2.0
    return Qd


def decode_dw(d, P, C):
    """d-vector -> per-parcel crop index. Reads the wall position."""
    assign = np.zeros(P, dtype=int)
    for i in range(P):
        seg = d[i * (C - 1):(i + 1) * (C - 1)]
        j = 0
        while j < C - 1 and seg[j] >= 0.5:
            j += 1
        assign[i] = j
    return assign


def wall_violations(d, P, C):
    v = 0
    for i in range(P):
        seg = d[i * (C - 1):(i + 1) * (C - 1)]
        for j in range(C - 2):
            if seg[j] < 0.5 <= seg[j + 1]:
                v += 1
    return v
