"""Build crop suitability labels from EcoCrop-style agronomic thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CROP_THRESHOLDS: dict[str, dict[str, tuple[float, float]]] = {
    "wheat": {
        "temperature": (10.0, 25.0),
        "rainfall": (250.0, 900.0),
        "ph": (6.0, 7.5),
    },
    "barley": {
        "temperature": (5.0, 20.0),
        "rainfall": (200.0, 800.0),
        "ph": (6.0, 8.0),
    },
    "lentils": {
        "temperature": (15.0, 30.0),
        "rainfall": (150.0, 600.0),
        "ph": (5.5, 7.5),
    },
    "chickpea": {
        "temperature": (15.0, 29.0),
        "rainfall": (200.0, 650.0),
        "ph": (6.0, 7.5),
    },
}


def _score_in_range(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 1.0
    distance = min(abs(value - low), abs(value - high))
    return max(0.0, 1.0 - distance / max(high - low, 1.0))


def suitability_score(
    crop: str,
    temperature: float,
    rainfall: float,
    ph: float,
) -> float:
    """Compute a 0-1 suitability score for a crop given environmental conditions."""
    thresholds = CROP_THRESHOLDS[crop]
    scores = [
        _score_in_range(temperature, *thresholds["temperature"]),
        _score_in_range(rainfall, *thresholds["rainfall"]),
        _score_in_range(ph, *thresholds["ph"]),
    ]
    return round(sum(scores) / len(scores), 3)


def build_suitability_matrix(
    parcels: pd.DataFrame,
    weather: pd.DataFrame,
    crops: list[str] | None = None,
) -> pd.DataFrame:
    """Create parcel x crop suitability scores."""
    crops = crops or list(CROP_THRESHOLDS.keys())
    annual = (
        weather.groupby("parcel_id")
        .agg(
            temperature=("temperature_c", "mean"),
            rainfall=("rainfall_mm", "sum"),
        )
        .reset_index()
    )
    merged = parcels.merge(annual, on="parcel_id")

    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        for crop in crops:
            rows.append(
                {
                    "parcel_id": row["parcel_id"],
                    "crop": crop,
                    "suitability_score": suitability_score(
                        crop,
                        row["temperature"],
                        row["rainfall"],
                        row["soil_ph"],
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build suitability matrix.")
    parser.add_argument(
        "--parcels",
        type=Path,
        default=Path("data/processed/parcels.csv"),
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=Path("data/processed/weather.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/suitability_matrix.csv"),
    )
    args = parser.parse_args()

    parcels = pd.read_csv(args.parcels)
    weather = pd.read_csv(args.weather)
    matrix = build_suitability_matrix(parcels, weather)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output, index=False)
    print(f"Wrote {len(matrix)} suitability rows to {args.output}")


if __name__ == "__main__":
    main()
