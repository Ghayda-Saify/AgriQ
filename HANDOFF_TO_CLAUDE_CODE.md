# AgriQ — Handoff Prompt for Claude Code (VS Code)

**Copy everything below the line into Claude Code as your first message.**

---

You are continuing work on **AgriQ**, a research project for an IEEE conference paper.
Repo root: `G:\Uni\AgriQ An AI and Quantum-Optimized System for Smart Crop Planning\agriq`

## 1. WHAT THIS IS

An integrated crop-planning system for Jenin, Palestine, combining:
1. **Random Forest** — crop suitability scoring
2. **LSTM** — price forecasting (evaluated, underperforms — see §5)
3. **Simulated Bifurcation (SB)** — quantum-INSPIRED (not quantum) allocation optimiser

Target: **ICFT 2026**, German Jordanian University, Amman. IEEE Xplore + Scopus.
Submit to the **NGIA-QC 2026 special session** (Next-Generation Intelligent Algorithms
Powered by Quantum Computing) — NOT Track 01, which is Edge AI/TinyML and a poor fit.

Format: **6 pages max** (7–8 incurs fees), IEEE A4 two-column, **DOUBLE-BLIND** (anonymise!),
abstract 75–150 words, 4–6 keywords, no page numbers, PDF ≤4 MB.
Submission via Microsoft CMT. Deadline: **15 August 2026**.

## 2. NON-NEGOTIABLE RULES

- **NEVER fabricate data.** Every number must trace to a cited source or be explicitly
  labelled an assumption. A previous version of this repo contained Syrian placeholder
  data (Damascus/Aleppo/Homs) masquerading as Palestinian data — do not reintroduce
  anything like that.
- **Never tune toward a desired conclusion.** The original spec demanded "SB MUST beat SA."
  It does not. That is reported honestly.
- **Always report real vs synthetic performance separately.** Never pool them.
- **Always include a trivial baseline** in any model comparison.
- If a script can't fetch data, it must FAIL LOUDLY and write nothing — never substitute
  invented values. (The original `fetch_nasa_power.py` silently wrote fake weather on
  API failure. That bug is fixed; don't recreate it.)

## 3. DATA — ALL REAL, ALL VERIFIED

| Layer | Source | File | Status |
|---|---|---|---|
| Weather, Jenin 2015–24 monthly | NASA POWER | `data/raw/nasa_power_jenin_monthly.csv` | ✅ 465 mm/yr, 20.7 °C, 90% rain Nov–Mar |
| Producer prices 1997–2024 | FAOSTAT PP (USD/t) | `data/raw/FAOSTAT_data_en_8-15-2026.csv` | ✅ all 10 crops |
| Yields 1994–2022 | FAOSTAT QCL (kg/ha) | `data/raw/FAOSTAT_data_en_8-11-2026.csv` | ✅ all 10 crops |
| Prices/yields cross-check | PCBS 2023 | `data/raw/pcbs_jenin_crop_params.csv` | ✅ agrees with FAOSTAT ~10% |
| Parcels (real field polygons) | OpenStreetMap via Geofabrik | `data/processed/jenin_parcels_zoned.csv` | ✅ 941 polygons |
| Oslo zones A/B/C | OCHA HDX | `data/raw/osloagreement/` | ✅ clipped |
| Soil (pH, N, SOC, CEC, clay, sand) | SoilGrids 250m | `data/processed/soilgrids_jenin.csv` | ✅ 258/284 parcels |
| Crop suitability training | Kaggle Crop Recommendation | `data/raw/Crop_recommendation.csv` | ✅ 2200 real rows, 22 crops |

**Key figures already established (use these, don't recompute from scratch):**
- **174 controllable parcels** (Oslo Area A/B), **1,464 ha** — the optimisation space
- **110 Area C parcels, 706 ha** — excluded by Oslo; **32.5% of Jenin farmland**
- 625 of the original 941 OSM polygons were **inside Israel** — the Oslo clip removed them
- **Area C soil is agronomically IDENTICAL to Area A/B** (pH 7.69 vs 7.71, N 2.19 vs 1.99
  g/kg) → the exclusion is political, not land-quality. Strong result.
- **Jenin agricultural water budget: 6.0 Mm³/yr from 13 wells** (PWA 2023 via PCBS)
- 10 crops + fallow: wheat, potatoes, tomatoes, cucumbers, onions, eggplants, maize,
  olives, grapes, oranges

**CRITICAL DATA WARNING — exclude FAOSTAT year 2024.**
2024 perishable prices spike 1.24×–5.76× vs 2023 (oranges 628→3611, onions 551→2736)
while storables are flat (wheat 0.95×, olives 1.10×). Consistent with wartime supply
disruption. **Use the 2019–2023 mean as the price baseline.** Using 2024 makes the
optimiser plant everything with oranges on a war-year artifact.

## 4. CODE — WHAT EXISTS AND IS VALIDATED

```
src/data_prep/fetch_jenin_weather.py   NASA POWER, fails loudly, self-checks    ✅ run
src/data_prep/fetch_soilgrids.py       SoilGrids per parcel, resumable          ✅ run
src/data_prep/clip_parcels_oslo.py     OSM ∩ Oslo zones, geodesic areas         ✅ run
src/optimizer/qubo.py                  CropQUBO3 — market-clearing formulation  ✅ validated
src/optimizer/qubo_v2_slack.py         earlier slack version (kept for paper)   superseded
src/optimizer/domain_wall.py           Chancellor domain-wall encoding          ✅ validated
src/optimizer/solvers.py               dSB (Goto 2021), SA (O(n) delta), Greedy ✅ validated
src/models/ecocrop.py                  EcoCrop synthetic label generation       ✅
src/models/train_rf.py                 Random Forest + separate evaluation      ✅ trained
src/models/lstm.py                     NumPy LSTM, explicit BPTT + Adam         ✅ validated
src/pipeline/instance.py               builds the real Jenin instance           ✅
tests/val*.py, tests/tune_dsb.py       brute-force validation harnesses         ✅
```

### Validation already done — do NOT redo, but DO cite
- QUBO exact vs brute force: **100%** at 5×3, 6×4, 7×3 (14-bit water slack, M_water=1e4)
- Domain-wall transform energy-preserving: **2430/2430** valid states
- dSB vs brute force after tuning (dt=0.25, 2000 steps, 5 restarts): **16/16 optimal**
- LSTM implementation sanity: echo-last-value corr 1.0000; sum-of-sequence corr 1.0000

## 5. RESULTS SO FAR

**Random Forest** (report the two numbers SEPARATELY — never pooled):
- Real Kaggle data, 5-fold CV: **0.9945 ± 0.0037** ← genuine predictive accuracy
- Synthetic EcoCrop rows, 5-fold CV: **0.8514 ± 0.0291** ← envelope fidelity, NOT a result
- Majority-class baseline: 0.0455
- Feature importance: rainfall .251, humidity .192, K .166, P .139, N .098, temp .082, pH .072
- ⚠️ **P and K carry 30.5% of signal and are UNOBTAINABLE for Jenin** (not in SoilGrids,
  no global source). State as a limitation.

**LSTM price forecasting — NEGATIVE RESULT, report honestly:**
| Model | RMSE USD/t | MAPE % |
|---|---|---|
| Naive (persistence) | **130.1** | **16.3** |
| AR(1) | 150.5 | 18.3 |
| Mean of last 3 | 170.1 | 19.7 |
| Linear trend | 186.0 | 20.9 |
| LSTM (pooled+augmented) | 252.2 | 26.7 |

Explanation: **3,481 parameters vs 142 real training windows.** 27 annual observations
cannot constrain that. Augmentation (×6) adds no new information. Use persistence prices
for the main results; run LSTM prices as a robustness check.

**Encoding result — THE HEADLINE FINDING:**
| Solver | One-hot penalty | Domain-wall | Gain |
|---|---|---|---|
| SA | $5.60M | $6.61M | **+18.2%** |
| dSB | $6.10M | $6.45M | +5.8% |

- Domain-wall eliminates constraint violations entirely: **0/1566 vs 136/174 parcels**
- On the fair comparison **SA beats dSB by 2.4%** — dSB does NOT win
- **Paper framing: "encoding choice dominates solver choice"** — not "our solver won"

**Penalty-conditioning analysis (a methods contribution):**
Slack-encoded hard caps have a quantisation pathology — you cannot distinguish "violated
the cap by δ" from "slack can't represent the residual to better than δ." More slack bits
widen the workable M window quadratically. Demonstrated empirically (8-bit: narrow window,
88% exact; 14-bit: 100% exact across M=1e3–1e5). Solution adopted: price both capacity
limits (demand via inverse-demand curve, water via λ $/m³) instead of penalising them.

## 6. WHAT'S NOT DONE — DO THIS, IN ORDER

1. **Apply RF to Jenin** → produce `suitability_matrix.csv` (174 × 11).
   Combine: `suitability[i,j] = RF_climate(crop j) × landuse_fit(parcel i, crop j)`
   where landuse_fit uses OSM fclass (orchard/vineyard → tree crops; farmland → annuals).
   Soil inputs: real pH/N from SoilGrids; **P and K via Monte Carlo over published ranges**
   (report an uncertainty band, never a single invented value).

2. **Rebuild the instance** with RF suitability + 2019–2023 mean prices (exclude 2024!).

3. **20-seed benchmark**: {SA, dSB} × {one-hot, domain-wall}, **matched wall-clock**,
   mean ± std, time-to-target, significance test on the dSB-vs-SA gap. Plus Greedy.

4. **Area C counterfactual**: optimise 174 A/B parcels vs all 284 incl. Area C.
   The profit gap = measured economic cost of the land-access regime.

5. **Water shadow price**: sweep λ until usage meets 6.0 Mm³. Earlier sweeps were
   non-monotonic due to solver noise — redo with domain-wall where solvers converge.

6. **Price robustness**: allocation under persistence vs LSTM prices + ±30% sweep.

7. **Three figures**: encoding comparison w/ error bars; parcel map by Oslo zone;
   water shadow-price curve.

8. **Write the paper** — 6 pages, anonymised, compile with pdflatex/latexmk.

9. **Fix intro + literature review** — see `paper/PAPER_FIXES_TODO.md`. Contains a
   fabricated citation (Matsuda/Nature Communications — the real paper is
   **Zeng et al., Communications Physics 7, 249 (2024)**), a factual error about SB
   ("emulating quantum adiabatic dynamics" — it's CLASSICAL mechanics), and an
   overclaiming title ("Quantum-Optimized" → **"Quantum-Inspired"**).

## 7. KEY CITATIONS (verified — use these exact ones)

- Goto, H., et al. (2021). High-performance combinatorial optimization based on classical
  mechanics. *Science Advances*, 7(6), eabe7953.  ← the SB/dSB primary source
- Zeng, Q-G., Cui, X-P., Liu, B., et al. (2024). Performance of quantum annealing inspired
  algorithms for combinatorial optimization problems. *Communications Physics*, 7, 249.
- Chancellor, N. (2019). Domain wall encoding of discrete variables for quantum annealing
  and QAOA. *Quantum Science and Technology*, 4(4), 045004.
- Poggio, L., et al. (2021). SoilGrids 2.0. *SOIL*, 7, 217–240.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Hochreiter & Schmidhuber (1997). Long short-term memory. *Neural Computation*, 9(8).

## 8. ENVIRONMENT NOTES

- Python 3.13 on the user's machine; numpy/pandas/requests/scikit-learn available.
- **PyTorch is NOT installed** — the LSTM is hand-written NumPy. If you install torch you
  may port it, but the negative result will not change.
- SoilGrids/NASA POWER/Overpass APIs work from the user's machine.
- Repo is a git repo — the broken original files are in history if needed.

## 9. FIRST THING TO DO

Read `paper/PAPER_FIXES_TODO.md`, then run:
```
python src/pipeline/instance.py
```
to confirm the instance still builds (should print: 174 parcels, 1,464 ha, 1,914 vars).
Then start at §6 item 1.
