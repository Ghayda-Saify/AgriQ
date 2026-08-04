"""QUBO formulation for parcel-crop allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_qubo_matrix(
    parcels: pd.DataFrame,
    suitability: pd.DataFrame,
    prices: pd.DataFrame,
    demand: pd.DataFrame,
    penalty: float = 10.0,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build QUBO matrix Q where x^T Q x is minimized.

    Decision variables x_{i,j} = 1 if parcel i is assigned crop j.
    Objective: maximize suitability * price, subject to one-crop-per-parcel.
    """
    parcel_ids = parcels["parcel_id"].tolist()
    crops = sorted(suitability["crop"].unique().tolist())
    n_parcels = len(parcel_ids)
    n_crops = len(crops)
    n_vars = n_parcels * n_crops

    price_map = prices.set_index("crop")["forecast_price_usd"].to_dict()
    demand_map = demand.set_index("crop")["demand_tonnes"].to_dict()
    area_map = parcels.set_index("parcel_id")["area_ha"].to_dict()
    suit_map = suitability.set_index(["parcel_id", "crop"])["suitability_score"].to_dict()

    Q = np.zeros((n_vars, n_vars))
    var_index: dict[tuple[str, str], int] = {}

    for i, parcel_id in enumerate(parcel_ids):
        for j, crop in enumerate(crops):
            idx = i * n_crops + j
            var_index[(parcel_id, crop)] = idx
            profit = suit_map.get((parcel_id, crop), 0.0) * price_map.get(crop, 0.0)
            Q[idx, idx] -= profit

    for i in range(n_parcels):
        for j in range(n_crops):
            for k in range(j + 1, n_crops):
                idx_j = i * n_crops + j
                idx_k = i * n_crops + k
                Q[idx_j, idx_k] += penalty
                Q[idx_k, idx_j] += penalty

    for crop in crops:
        crop_demand = demand_map.get(crop, float("inf"))
        crop_vars = [var_index[(p, crop)] for p in parcel_ids]
        total_area = sum(area_map[p] for p in parcel_ids)
        if total_area > crop_demand / 100:
            over_penalty = penalty * 0.5
            for a in crop_vars:
                for b in crop_vars:
                    if a <= b:
                        Q[a, b] += over_penalty

    labels = [f"{p}_{c}" for p in parcel_ids for c in crops]
    return Q, parcel_ids, crops


def decode_solution(
    binary: np.ndarray,
    parcel_ids: list[str],
    crops: list[str],
) -> pd.DataFrame:
    """Convert binary QUBO solution to allocation DataFrame."""
    n_crops = len(crops)
    rows: list[dict[str, object]] = []

    for i, parcel_id in enumerate(parcel_ids):
        segment = binary[i * n_crops : (i + 1) * n_crops]
        if segment.sum() == 0:
            assigned = int(np.argmax(segment))
        else:
            assigned = int(np.argmax(segment))
        rows.append({"parcel_id": parcel_id, "crop": crops[assigned], "assigned": int(segment[assigned])})

    return pd.DataFrame(rows)
