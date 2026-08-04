"""Merge raw and processed sources into ML training dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def build_training_data(
    kaggle_path: Path,
    suitability_path: Path,
    min_score: float = 0.6,
) -> pd.DataFrame:
    """Combine Kaggle crop dataset with suitability labels."""
    kaggle = pd.read_csv(kaggle_path)
    suitability = pd.read_csv(suitability_path)

    avg_suitability = (
        suitability.groupby("crop")["suitability_score"]
        .mean()
        .reset_index()
        .rename(columns={"suitability_score": "avg_suitability"})
    )
    merged = kaggle.merge(avg_suitability, left_on="label", right_on="crop", how="left")
    merged["suitable"] = (merged["avg_suitability"].fillna(0.5) >= min_score).astype(int)
    merged = merged.drop(columns=["crop", "avg_suitability"], errors="ignore")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training dataset.")
    parser.add_argument(
        "--kaggle",
        type=Path,
        default=Path("data/raw/kaggle_crop_recommendation.csv"),
    )
    parser.add_argument(
        "--suitability",
        type=Path,
        default=Path("data/processed/suitability_matrix.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/training_data.csv"),
    )
    parser.add_argument("--min-score", type=float, default=0.6)
    args = parser.parse_args()

    training = build_training_data(args.kaggle, args.suitability, min_score=args.min_score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(args.output, index=False)
    print(f"Wrote {len(training)} training rows to {args.output}")


if __name__ == "__main__":
    main()
