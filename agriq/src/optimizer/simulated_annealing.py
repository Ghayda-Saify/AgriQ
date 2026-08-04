"""Simulated annealing solver for QUBO problems."""

from __future__ import annotations

import numpy as np


def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    """Compute x^T Q x for binary vector x."""
    return float(x @ Q @ x)


def simulated_annealing(
    Q: np.ndarray,
    max_iter: int = 5000,
    T_start: float = 10.0,
    T_end: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """
    Minimize QUBO energy via simulated annealing.

    Returns best binary solution and its energy.
    """
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    current = rng.integers(0, 2, size=n).astype(float)
    current_energy = qubo_energy(Q, current)
    best = current.copy()
    best_energy = current_energy

    cooling_rate = (T_end / T_start) ** (1.0 / max_iter)

    T = T_start
    for _ in range(max_iter):
        flip_idx = rng.integers(0, n)
        candidate = current.copy()
        candidate[flip_idx] = 1.0 - candidate[flip_idx]
        candidate_energy = qubo_energy(Q, candidate)
        delta = candidate_energy - current_energy

        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = candidate
            current_energy = candidate_energy
            if candidate_energy < best_energy:
                best = candidate
                best_energy = candidate_energy

        T *= cooling_rate

    return best.astype(int), best_energy
