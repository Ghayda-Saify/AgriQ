"""Train Random Forest classifier for crop suitability."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def train_random_forest(
    training_path: Path,
    model_path: Path,
    scaler_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train and persist RF model plus feature scaler."""
    df = pd.read_csv(training_path)
    X = df[FEATURE_COLUMNS]
    y = df["suitable"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest model.")
    parser.add_argument(
        "--training",
        type=Path,
        default=Path("data/processed/training_data.csv"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/random_forest.pkl"),
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        default=Path("models/scaler.pkl"),
    )
    args = parser.parse_args()
    train_random_forest(args.training, args.model, args.scaler)
    print(f"Saved model to {args.model} and scaler to {args.scaler}")


if __name__ == "__main__":
    main()
