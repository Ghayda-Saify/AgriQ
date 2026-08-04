"""Tests for optimizer solvers."""

import numpy as np

from src.optimizer.simulated_annealing import qubo_energy, simulated_annealing
from src.optimizer.simulated_bifurcation import simulated_bifurcation


def test_qubo_energy_diagonal():
    Q = np.diag([-1.0, -2.0, -3.0])
    x = np.array([1, 0, 1])
    assert qubo_energy(Q, x) == -4.0


def test_simulated_annealing_returns_binary():
    Q = np.array([[-1.0, 0.5], [0.5, -2.0]])
    solution, energy = simulated_annealing(Q, max_iter=500, seed=0)
    assert set(solution.tolist()) <= {0, 1}
    assert isinstance(energy, float)


def test_simulated_bifurcation_returns_binary():
    Q = np.array([[-1.0, 0.5], [0.5, -2.0]])
    solution, energy = simulated_bifurcation(Q, max_iter=200, seed=0)
    assert set(solution.tolist()) <= {0, 1}
    assert isinstance(energy, float)
