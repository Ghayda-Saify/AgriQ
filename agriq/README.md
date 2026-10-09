# AgriQ

**Crop allocation as a QUBO: constraint encoding, solver choice, and the agricultural cost of occupation**

Market-aware crop planning for **174 real agricultural parcels (1,464 ha) in Jenin
governorate, Palestine**, formulated as a 1,914-variable QUBO and solved with
classical and quantum-inspired heuristics.

A Random Forest scores crop suitability. Allocation is written as a QUBO in which
prices fall as total production rises and irrigation water is priced rather than
capped. Two encodings are compared against two solvers at a matched wall-clock
budget.

> **Quantum-inspired, not quantum.** Simulated bifurcation is derived from a
> quantum-mechanical model but integrates *classical* Hamiltonian dynamics on an
> ordinary CPU. No quantum hardware is used. At 1,914 variables with ten 174-node
> cliques, this instance exceeds the largest fully connected problem embeddable on
> current annealing hardware (K₁₅₀ on D-Wave Pegasus).

## Main results

| Encoding | Solver | Profit (M$/yr) | Invalid parcels |
|---|---|---|---|
| one-hot | SA | **5.42 ± 0.26** | 0.0 |
| one-hot | dSB | 1.98 ± 0.42 | 129.1 |
| domain-wall | SA | 4.97 ± 0.39 | 13.0 |
| domain-wall | dSB | 5.00 ± 0.30 | 2.9 |
| — | Greedy | −1.58 | — |

20 seeds, 2.5 s per run, 174 parcels.

**The encoding decides the outcome, and its effect is solver-dependent.**
Domain-wall encoding raises discrete simulated bifurcation's profit by **+152%**
(p < 10⁻⁷) and brings it level with simulated annealing (p = 0.93), while
*costing* annealing **−8.5%** (p < 10⁻⁴). Single-solver studies cannot observe
that interaction. The best configuration overall remains classical SA with
one-hot encoding.

Other findings:

- A myopic **greedy** plan loses $1.58M by flooding the market, and still loses
  $1.55M under free disposal — so the loss is the glut, not the demand curve
  turning negative past 2D.
- **Irrigation water never binds** on Areas A/B: demand peaks at 5.50 Mm³/yr
  against a 6.0 Mm³ budget, shadow price zero. It binds only once Area C land
  becomes farmable ($1.09–1.35/m³). Water scarcity here is downstream of land
  restriction.
- Confining cultivation to Areas A/B costs **$0.23–1.22M/yr**, reported as a
  range because market size is unobserved.
- An **LSTM price forecaster loses to persistence** (RMSE 252.2 vs 130.1) and is
  reported as a negative result. The optimiser uses 2019–2023 mean prices.

## Project structure

```
agriq/
├── data/
│   ├── raw/           FAOSTAT, NASA POWER, PCBS, OSM/Oslo shapefile, Kaggle
│   └── processed/      parcels, suitability matrix, all experiment outputs
├── src/
│   ├── data_prep/      OSM/Oslo clipping, NASA POWER, SoilGrids fetchers
│   ├── models/         Random Forest suitability, NumPy LSTM, forecasts
│   ├── optimizer/      QUBO formulation, SA/dSB solvers, domain-wall transform
│   ├── pipeline/       instance builder + the five experiments + figures
│   └── dashboard/      Streamlit UI (exploratory, not part of the paper)
├── tests/              evaluation scripts and solver validation
├── notebooks/          early exploratory analysis (superseded)
└── paper/              IEEE submission, figures, Word exports
```

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.12. Only `numpy`, `pandas`, `scikit-learn`, `scipy` and `matplotlib` are
needed to reproduce the results. `requests` is required only to re-fetch weather
and soil data, `pyshp`/`pyproj` only to re-clip parcels from the shapefile, and
`python-docx` only for the Word export. The LSTM is hand-written in NumPy with
explicit BPTT — there is no deep-learning framework dependency.

## Reproducing the results

Processed data is committed, so the experiments run directly. Run from the
`agriq/` directory.

```bash
# 1. Suitability model (writes data/processed/suitability_matrix.csv)
python src/models/train_rf.py
python src/models/build_suitability.py

# 2. The 2x2 benchmark: {one-hot, domain-wall} x {SA, dSB}, 20 seeds
python src/pipeline/benchmark.py

# 3. Water shadow price (lambda sweep)
python src/pipeline/water_price.py

# 4. Cost of restricting cultivation to Areas A/B
python src/pipeline/area_c.py
python src/pipeline/area_c_market.py

# 5. Price robustness against a seed-control floor
python src/pipeline/price_robustness.py

# 6. Figures as vector PDFs at IEEE column width
python src/pipeline/make_figures.py
```

The LSTM negative result, including the effect of leaving 2024 in or out:

```bash
python tests/eval_lstm.py
```

Re-fetching raw data is only needed if you want to rebuild from source:

```bash
python src/data_prep/fetch_jenin_weather.py    # NASA POWER
python src/data_prep/fetch_soilgrids.py        # SoilGrids 250 m
python src/data_prep/clip_parcels_oslo.py      # needs the OSM extract locally
```

Building the paper:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Data sources

All inputs are public. Nothing is synthesised for the optimisation instance.

| Source | Used for |
|---|---|
| OpenStreetMap (Geofabrik Palestine extract) | parcel polygons and land-use class |
| UN OCHA oPt — Oslo Agreement layer (HDX) | Area A/B vs C classification |
| NASA POWER, 2015–2024 | climate; Jenin resolves as one grid cell |
| ISRIC SoilGrids 250 m | soil pH, organic carbon, nitrogen, clay, sand, CEC |
| FAOSTAT | producer prices (2019–2023 mean) and yields (2014–2022) |
| PCBS | regional open-field yield calibration |
| FAO-56 (Allen et al.) | net irrigation requirements |
| Kaggle Crop Recommendation + FAO EcoCrop | Random Forest training labels |

**2024 is excluded** from prices: perishables rose 1.24–5.76× over 2023 while
wheat (0.95×) and olives (1.10×) did not, a pattern consistent with a wartime
supply shock rather than demand.

Three quantities are **assumptions**, labelled as such in the paper: market size
*D<sub>j</sub>*, the production cost share, and the 2% nitrogen mineralisation
fraction. Everything else traces to a cited source.

## Project conventions

These are enforced in the code, not just documented:

- **Data loading fails loudly.** `instance.py` refuses to guess which file is
  which. An earlier silent fallback selected the *price* file when the yield file
  was missing, so wheat "yielded" 0.59 t/ha — actually $586/t ÷ 1000 — and every
  downstream number was wrong without anything crashing.
- **Real and synthetic results are never pooled.** The Random Forest reports
  0.9945 on real labelled rows and 0.8514 on synthetic EcoCrop rows separately;
  the latter measures fidelity to the envelopes, not predictive skill.
- **Every learned component is compared against a trivial baseline.** That is how
  the LSTM was rejected.
- **Solver comparisons use matched wall-clock time**, calibrated per matrix, not
  matched iteration counts. SA uses incremental O(n) delta-energy so the
  comparison measures algorithms rather than implementation quality.
- **Energy and economics are separate code paths.** The QUBO matrix is normalised
  and dimensionless; `evaluate()` recomputes profit in dollars from the decoded
  assignment, so a normalisation error cannot silently inflate a reported result.

## Notes

- `src/dashboard/` and `notebooks/` predate the current pipeline and are not used
  for any result in the paper.
- The two Random Forest `.pkl` files are gitignored (~31 MB each); regenerate with
  `src/models/train_rf.py`.
- `paper/main_no_methodology.tex` builds the paper without the Methodology
  section, for the co-author writing that section. See
  `paper/METHODOLOGY_BRIEF.md`.
