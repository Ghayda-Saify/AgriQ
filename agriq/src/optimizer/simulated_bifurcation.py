"""Simulated bifurcation solver (quantum-inspired) for QUBO."""

from __future__ import annotations

import numpy as np


def simulated_bifurcation(
    Q: np.ndarray,
    max_iter: int = 1000,
    dt: float = 0.1,
    pressure: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """
    Solve QUBO via simulated bifurcation dynamics.

    Continuous spins are driven toward ±1; final signs give binary solution.
    """
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    x = rng.normal(0, 0.1, size=n)
    y = rng.normal(0, 0.1, size=n)

    for step in range(max_iter):
        p = pressure * (step + 1) / max_iter
        force = -(Q @ np.sign(x)) - p * x
        y += dt * force
        x += dt * y

    binary = (np.sign(x) >= 0).astype(int)
    energy = float(binary @ Q @ binary)
    return binary, energy
