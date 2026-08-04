"""Run the full AgriQ pipeline: data prep, training, optimization."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_prep.build_training_data import build_training_data
from src.data_prep.ecocrop_labels import build_suitability_matrix
from src.data_prep.fetch_nasa_power import fetch_weather_for_parcels
from src.data_prep.generate_parcels import generate_parcels
from src.models.train_lstm import train_lstm
from src.models.train_rf import train_random_forest
from src.optimizer.qubo import build_qubo_matrix, decode_solution
from src.optimizer.repair import build_allocation_output, repair_one_crop_per_parcel
from src.optimizer.simulated_annealing import simulated_annealing
from src.optimizer.simulated_bifurcation import simulated_bifurcation


def run_pipeline(
    data_dir: Path = Path("data"),
    models_dir: Path = Path("models"),
    n_parcels: int = 50,
    seed: int = 42,
) -> None:
    """Execute all pipeline stages sequentially."""
    raw = data_dir / "raw"
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("1/6 Generating parcels...")
    parcels = generate_parcels(n_parcels=n_parcels, seed=seed)
    parcels.to_csv(processed / "parcels.csv", index=False)

    print("2/6 Fetching weather...")
    weather = fetch_weather_for_parcels(parcels)
    weather.to_csv(processed / "weather.csv", index=False)

    print("3/6 Building suitability matrix...")
    suitability = build_suitability_matrix(parcels, weather)
    suitability.to_csv(processed / "suitability_matrix.csv", index=False)

    print("4/6 Building training data and training models...")
    training = build_training_data(
        raw / "kaggle_crop_recommendation.csv",
        processed / "suitability_matrix.csv",
    )
    training.to_csv(processed / "training_data.csv", index=False)
    train_random_forest(
        processed / "training_data.csv",
        models_dir / "random_forest.pkl",
        models_dir / "scaler.pkl",
    )
    train_lstm(raw / "fao_prices.csv", models_dir / "lstm_price.pth")

    prices = pd.read_csv(processed / "price_forecast.csv") if (processed / "price_forecast.csv").exists() else pd.DataFrame(
        {"crop": ["wheat", "barley"], "forecast_price_usd": [300.0, 240.0]}
    )
    demand = pd.DataFrame(
        {
            "crop": suitability["crop"].unique(),
            "demand_tonnes": [100000] * suitability["crop"].nunique(),
        }
    )
    demand.to_csv(processed / "demand_forecast.csv", index=False)

    print("5/6 Running optimizers...")
    Q, parcel_ids, crops = build_qubo_matrix(parcels, suitability, prices, demand)

    sa_solution, sa_energy = simulated_annealing(Q, seed=seed)
    sb_solution, sb_energy = simulated_bifurcation(Q, seed=seed)

    sa_alloc = decode_solution(sa_solution, parcel_ids, crops)
    sb_alloc = decode_solution(sb_solution, parcel_ids, crops)
    sa_alloc = repair_one_crop_per_parcel(sa_alloc, suitability)
    sb_alloc = repair_one_crop_per_parcel(sb_alloc, suitability)

    sa_output = build_allocation_output(sa_alloc, parcels, suitability, prices)
    sb_output = build_allocation_output(sb_alloc, parcels, suitability, prices)
    sa_output.to_csv(processed / "allocation_sa.csv", index=False)
    sb_output.to_csv(processed / "allocation_sb.csv", index=False)

    print(f"6/6 Done. SA energy={sa_energy:.2f}, SB energy={sb_energy:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgriQ end-to-end pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--n-parcels", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.models_dir, args.n_parcels, args.seed)


if __name__ == "__main__":
    main()
