"""Repair infeasible allocation solutions to satisfy constraints."""

from __future__ import annotations

import pandas as pd


def repair_one_crop_per_parcel(
    allocation: pd.DataFrame,
    suitability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ensure each parcel is assigned exactly one crop."""
    repaired = allocation.copy()

    for parcel_id, group in repaired.groupby("parcel_id"):
        if group["assigned"].sum() == 1:
            continue

        if suitability is not None:
            scores = suitability[suitability["parcel_id"] == parcel_id].set_index("crop")[
                "suitability_score"
            ]
            best_crop = scores.idxmax()
            best_idx = group[group["crop"] == best_crop].index[0]
        else:
            best_idx = group.index[0]

        repaired.loc[group.index, "assigned"] = 0
        repaired.loc[best_idx, "assigned"] = 1
        repaired.loc[group.index, "crop"] = repaired.loc[best_idx, "crop"]

    return repaired


def repair_demand_constraint(
    allocation: pd.DataFrame,
    parcels: pd.DataFrame,
    demand: pd.DataFrame,
) -> pd.DataFrame:
    """Cap crop assignments when total allocated area exceeds demand."""
    repaired = allocation.copy()
    area_map = parcels.set_index("parcel_id")["area_ha"].to_dict()
    demand_map = demand.set_index("crop")["demand_tonnes"].to_dict()

    for crop, crop_demand in demand_map.items():
        crop_rows = repaired[repaired["crop"] == crop]
        total_area = sum(area_map.get(pid, 0) for pid in crop_rows["parcel_id"])
        max_area = crop_demand / 100.0

        if total_area > max_area and len(crop_rows) > 1:
            overflow_parcels = crop_rows["parcel_id"].tolist()[1:]
            repaired = repaired[~repaired["parcel_id"].isin(overflow_parcels)]

    return repaired.reset_index(drop=True)


def build_allocation_output(
    allocation: pd.DataFrame,
    parcels: pd.DataFrame,
    suitability: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Compute expected profit for a repaired allocation."""
    price_map = prices.set_index("crop")["forecast_price_usd"].to_dict()
    suit_map = suitability.set_index(["parcel_id", "crop"])["suitability_score"].to_dict()
    area_map = parcels.set_index("parcel_id")["area_ha"].to_dict()

    rows: list[dict[str, object]] = []
    for _, row in allocation.iterrows():
        if row.get("assigned", 1) == 0:
            continue
        parcel_id = row["parcel_id"]
        crop = row["crop"]
        area = area_map.get(parcel_id, 0.0)
        profit = area * suit_map.get((parcel_id, crop), 0.0) * price_map.get(crop, 0.0)
        rows.append(
            {
                "parcel_id": parcel_id,
                "crop": crop,
                "area_ha": area,
                "expected_profit_usd": round(profit, 2),
            }
        )

    return pd.DataFrame(rows)
