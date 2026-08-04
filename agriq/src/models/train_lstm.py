"""Train LSTM model for crop price forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PriceLSTM(nn.Module):
    """Simple univariate LSTM for price time series."""

    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


def create_sequences(values: np.ndarray, seq_len: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences from a 1D array."""
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X), np.array(y)


def train_lstm(
    prices_path: Path,
    model_path: Path,
    crop: str = "Wheat",
    seq_len: int = 3,
    epochs: int = 100,
    lr: float = 0.01,
) -> PriceLSTM:
    """Train LSTM on FAO price history for a single crop."""
    df = pd.read_csv(prices_path)
    crop_prices = df.loc[df["item"] == crop, "price_usd_per_tonne"].values.astype(np.float32)

    if len(crop_prices) <= seq_len:
        crop_prices = np.tile(crop_prices, 5).astype(np.float32)

    X, y = create_sequences(crop_prices, seq_len=seq_len)
    X_tensor = torch.tensor(X).unsqueeze(-1)
    y_tensor = torch.tensor(y).unsqueeze(-1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)

    model = PriceLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    return model


def forecast_prices(
    model: PriceLSTM,
    history: np.ndarray,
    seq_len: int,
    horizon: int,
) -> list[float]:
    """Autoregressive multi-step forecast."""
    model.eval()
    window = history[-seq_len:].tolist()
    forecasts: list[float] = []

    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(window[-seq_len:]).float().view(1, seq_len, 1)
            pred = model(x).item()
            forecasts.append(pred)
            window.append(pred)

    return forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM price forecaster.")
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("data/raw/fao_prices.csv"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/lstm_price.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/price_forecast.csv"),
    )
    parser.add_argument("--crop", default="Wheat")
    args = parser.parse_args()

    model = train_lstm(args.prices, args.model, crop=args.crop)
    df = pd.read_csv(args.prices)
    history = df.loc[df["item"] == args.crop, "price_usd_per_tonne"].values.astype(np.float32)
    forecasts = forecast_prices(model, history, seq_len=3, horizon=2)

    last_year = int(df.loc[df["item"] == args.crop, "year"].max())
    forecast_df = pd.DataFrame(
        {
            "crop": [args.crop.lower()] * len(forecasts),
            "year": [last_year + i + 1 for i in range(len(forecasts))],
            "forecast_price_usd": forecasts,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(args.output, index=False)
    print(f"Saved LSTM to {args.model} and forecast to {args.output}")


if __name__ == "__main__":
    main()
