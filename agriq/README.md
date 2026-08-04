# AgriQ

**An AI and Quantum-Optimized System for Smart Crop Planning**

AgriQ combines machine learning (Random Forest crop suitability, LSTM price forecasting) with quantum-inspired optimization (Simulated Bifurcation) to recommend optimal crop allocations across agricultural parcels in Syria.

## Project Structure

```
agriq/
├── data/           # Raw and processed datasets
├── models/         # Trained model artifacts (.pkl, .pth)
├── src/            # Source code
│   ├── data_prep/  # Data generation and preprocessing
│   ├── models/     # ML training and inference
│   ├── optimizer/  # QUBO formulation and solvers
│   ├── pipeline/   # End-to-end orchestration
│   └── dashboard/  # Streamlit UI
├── notebooks/      # Exploratory analysis
├── tests/          # Unit tests
└── paper/          # Research paper (LaTeX)
```

## Quick Start

```bash
cd agriq
pip install -r requirements.txt

# Run full pipeline
python -m src.pipeline.run_pipeline

# Launch dashboard
streamlit run src/dashboard/app.py

# Run tests
pytest tests/
```

## Pipeline Stages

1. **Generate parcels** — Synthetic farm parcels across Syrian governorates
2. **Fetch weather** — NASA POWER API (with synthetic fallback)
3. **Build suitability** — EcoCrop-style agronomic scoring
4. **Train models** — Random Forest + LSTM
5. **Optimize** — QUBO solved via Simulated Annealing and Simulated Bifurcation
6. **Visualize** — Streamlit dashboard

## Data Sources

| File | Source |
|------|--------|
| `kaggle_crop_recommendation.csv` | Kaggle Crop Recommendation dataset |
| `fao_prices.csv` | FAO commodity prices |
| `arij_yields.csv` | ARIJ yield records (Syria) |

## License

MIT
