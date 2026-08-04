"""Tests for QUBO formulation."""

import numpy as np
import pandas as pd

from src.optimizer.qubo import build_qubo_matrix, decode_solution


def test_qubo_matrix_is_square():
    parcels = pd.DataFrame(
        {
            "parcel_id": ["P001", "P002"],
            "area_ha": [10.0, 8.0],
        }
    )
    suitability = pd.DataFrame(
        {
            "parcel_id": ["P001", "P001", "P002", "P002"],
            "crop": ["wheat", "barley", "wheat", "barley"],
            "suitability_score": [0.9, 0.7, 0.8, 0.6],
        }
    )
    prices = pd.DataFrame({"crop": ["wheat", "barley"], "forecast_price_usd": [300, 200]})
    demand = pd.DataFrame({"crop": ["wheat", "barley"], "demand_tonnes": [50000, 30000]})

    Q, parcel_ids, crops = build_qubo_matrix(parcels, suitability, prices, demand)
    assert Q.shape[0] == Q.shape[1]
    assert len(parcel_ids) == 2
    assert len(crops) == 2


def test_decode_solution_shape():
    binary = np.array([1, 0, 0, 1])
    result = decode_solution(binary, ["P001", "P002"], ["wheat", "barley"])
    assert len(result) == 2
    assert set(result.columns) >= {"parcel_id", "crop", "assigned"}
