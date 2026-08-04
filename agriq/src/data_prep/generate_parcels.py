"""Generate synthetic parcel geometries and attributes for Syria governorates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GOVERNORATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "Damascus": (33.5138, 36.2765),
    "Aleppo": (36.2021, 37.1343),
    "Homs": (34.7268, 36.7234),
    "Idlib": (35.9306, 36.6339),
    "Raqqa": (35.9594, 39.0079),
}


def generate_parcels(
    n_parcels: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic parcel records spread around governorate centroids."""
    rng = np.random.default_rng(seed)
    governorates = list(GOVERNORATE_CENTROIDS.keys())
    rows: list[dict[str, object]] = []

    for i in range(n_parcels):
        gov = governorates[i % len(governorates)]
        base_lat, base_lon = GOVERNORATE_CENTROIDS[gov]
        rows.append(
            {
                "parcel_id": f"P{i + 1:03d}",
                "lat": round(base_lat + rng.normal(0, 0.15), 4),
                "lon": round(base_lon + rng.normal(0, 0.15), 4),
                "area_ha": round(rng.uniform(3.0, 25.0), 1),
                "governorate": gov,
                "soil_ph": round(rng.uniform(6.0, 8.0), 1),
                "elevation_m": int(rng.integers(200, 900)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic parcel data.")
    parser.add_argument("--n-parcels", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/parcels.csv"),
    )
    args = parser.parse_args()

    df = generate_parcels(n_parcels=args.n_parcels, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} parcels to {args.output}")


if __name__ == "__main__":
    main()
