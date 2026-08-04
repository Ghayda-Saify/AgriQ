"""Inference utilities for trained AgriQ models."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.train_lstm import PriceLSTM, forecast_prices


def load_rf_model(model_path: Path, scaler_path: Path):
    """Load Random Forest classifier and feature scaler."""
    with model_path.open("rb") as f:
        model = pickle.load(f)
    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_suitability(
    features: pd.DataFrame,
    model_path: Path = Path("models/random_forest.pkl"),
    scaler_path: Path = Path("models/scaler.pkl"),
) -> np.ndarray:
    """Predict crop suitability (0/1) for feature rows."""
    model, scaler = load_rf_model(model_path, scaler_path)
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = scaler.transform(features[feature_cols])
    return model.predict(X)


def predict_prices(
    history: np.ndarray,
    model_path: Path = Path("models/lstm_price.pth"),
    seq_len: int = 3,
    horizon: int = 2,
) -> list[float]:
    """Load LSTM and produce price forecasts."""
    model = PriceLSTM()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return forecast_prices(model, history, seq_len=seq_len, horizon=horizon)
