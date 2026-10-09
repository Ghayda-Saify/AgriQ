# Methodology section — brief for Tala

The Methodology section has been removed from `main.tex` so you can write your
own. The previous version is preserved in `METHODOLOGY_REMOVED.tex` — use it as
a reference for the numbers, not as text you have to keep.

## Hard constraints

- **Space: ~1.75 columns.** The paper is exactly 6 pages (IEEE limit). Without
  Methodology it is 4 pages + the literature table. Anything beyond ~1.75
  columns pushes the paper over and something else must be cut.
- **Double-blind.** No author names, no institution, no "our university",
  no repository links.
- **Honesty rule (non-negotiable).** Every number traces to a cited source or
  is explicitly labelled an assumption. Nothing is invented.

## Four subsections are needed, because Results depends on all four

Results will not make sense unless Methodology has defined these. Each bullet
below is a fact Results already states.

### 1. Study Area and Data
- Parcels: OpenStreetMap land-use polygons `\cite{osm}`. 941 polygons → 625
  centroids fall outside the West Bank → 316 remain → filtered to 0.1–100 ha →
  **174 parcels (1,464 ha) in Areas A/B**, **110 parcels (706 ha) in Area C**.
- Oslo classification `\cite{ocha}` by centroid. **Note: the OCHA layer merges
  Areas A and B into one class** — this must be stated, it is why we say "A/B".
- Climate: NASA POWER 2015–2024 `\cite{nasapower}` — 465 mm/yr rainfall (90% Nov–Mar),
  20.7 °C, 68.7% RH. **Jenin is a single grid cell, so climate is identical
  across parcels** — Results relies on this to explain where parcel variation
  comes from.
- Soil: SoilGrids 250 m `\cite{poggio2021}`, available for 149 of 174 A/B
  parcels; the other 25 take the governorate mean.
- Prices: FAOSTAT `\cite{faostat}` means **2019–2023**. **2024 is excluded** —
  perishable prices rose 1.24–5.76× over 2023 while wheat (0.95×) and olives
  (1.10×) did not, i.e. a wartime supply shock, not demand. Results refers back
  to this exclusion, so the justification must be here.
- Yields: FAOSTAT means 2014–2022, rescaled — national vegetable yields include
  greenhouses, so vegetables are rescaled to the PCBS open-field level for the
  northern West Bank `\cite{pcbs}` (19.65 t/ha) and tree crops so olives match
  PCBS (1.43 t/ha).
- Water: net irrigation requirements are indicative FAO-56 values `\cite{allen1998}`.
- **Keep Table `tab:crops`** (10 crops × yield/price/water/cost share). Results
  and the reader both need it.
- **Keep Figure `fig:map`** (`figures/fig_map.pdf`) — parcels by Oslo area.
- Production cost: **label it an assumption** — a share of baseline gross
  revenue typical of the system type, held fixed when prices are perturbed.
  (The "held fixed" part matters: it is why the price-robustness test is valid.)

### 2. Suitability Model
- Random Forest, 300 trees `\cite{breiman2001}`, trained on a public dataset of
  2,200 labelled observations over 22 crops `\cite{kagglecrop}`, features
  (N, P, K, T, H, pH, R).
- 7 of our 10 crops are absent from that dataset; rows for them are generated
  from FAO EcoCrop envelopes `\cite{ecocrop}`. **Real and synthetic rows are
  evaluated separately, never pooled** — Results reports two accuracies
  (0.9945 real, 0.8514 synthetic) and this is why.
- Suitability equation: s_ij = RF_j(x_i) · φ(ℓ_i, j), where RF_j is the predicted
  probability renormalised over the ten study crops, and φ is a land-use factor
  from the OSM class (orchards favour tree crops, farmland favours annuals).
- P and K have no parcel-level source → Monte-Carlo sampled over published
  ranges, 200 draws; mean SD of s_ij is 0.039.
- Available N derived from SoilGrids total N with an **assumed** 2% mineralisation
  fraction. Jenin rainfall exceeds the training range and is clipped to the
  training maximum.
- Unsuitable = bottom 20% of that crop's suitability over the A/B parcels.

### 3. QUBO Formulation
- x_ij ∈ {0,1} = parcel i grows crop j; j = 0 is fallow. Production q_ij = a_i·y_j,
  total Q_j = Σ_i q_ij x_ij.
- Objective: maximise Σ_j (p_j Q_j − β_j Q_j²) − Σ_ij c_j a_i x_ij − λ Σ_ij w_j a_i x_ij
- β_j = p_j / (2 D_j), a linear inverse-demand curve; marginal revenue reaches
  zero at Q_j = D_j.
- **D_j is an assumption and must be labelled one**: no crop-level demand data
  exist for Jenin, so D_j = 0.6 · A_AB · y_j. **Held fixed in all counterfactuals** —
  this is what makes the Area C result a bounded range rather than a point
  estimate, and the Limitations section refers to it.
- Water priced at λ $/m³ (λ = 0.15 unless swept). Pricing rather than a hard cap
  avoids the large slack penalties a cap requires.
- Two penalty terms, each weight M = 0.5 against the objective normalised by its
  largest linear coefficient: (a) one crop per parcel, (b) unsuitable pairs.
- **Reported profit excludes the internal water charge.**

### 4. Encodings and Solvers
This subsection carries the paper's main contribution — do not compress it.
- **One-hot**: C = 11 binaries per parcel, penalty M(Σ_j x_ij − 1)²,
  **1,914 variables**. Invalid assignments remain reachable.
- **Domain-wall** `\cite{chancellor2019}`: C−1 = 10 binaries per parcel constrained
  non-increasing, x_ij = d_ij − d_i,j+1, ordering penalty 1.0, **1,740 variables**.
  **Critical sentence to keep**: it removes only the one-hot penalty; the
  suitability penalty is kept, so both encodings represent the same problem.
  (An earlier draft had a bug where both penalties were removed — a reviewer
  will look for exactly this.)
- **SA** `\cite{kirkpatrick1983}`: single-bit flips, incremental O(n) energy
  updates, geometric cooling from T₀ = 0.5·mean(Σ_k|Q_ik|) to 10⁻⁴T₀.
- **dSB** `\cite{goto2021}`: ẏ = −(a₀ − p(t))x − c₀(J·sgn(x) + h), ẋ = a₀y,
  inelastic walls at |x| = 1, a₀ = 1, linear pump p(t), time step 0.25,
  c₀ = 0.5/(√N σ_J).
- **Greedy baseline**: each parcel gets its highest-margin suitable crop,
  ignoring market coupling, then parcels are fallowed until water meets budget.
- **Fair comparison — must be stated explicitly**: all four configurations get
  the same **wall-clock budget of 2.5 s per run**, set by one-hot dSB at 1,000
  steps and **calibrated separately on each matrix** (dSB 1,000 and 1,315 steps;
  SA 62 and 54 sweeps). 20 seeds each; **two-sided Mann–Whitney U** tests
  (unpaired — the seeds are independent, so Wilcoxon signed-rank would be wrong).

## Citations already in references.bib and ready to use

`osm`, `ocha`, `nasapower`, `poggio2021`, `faostat`, `pcbs`, `allen1998`,
`breiman2001`, `kagglecrop`, `ecocrop`, `chancellor2019`, `goto2021`,
`kirkpatrick1983`

## Two things still missing from the paper (not Methodology's job, but flagging)

Both are marked `\refpending` in the Introduction and need real sources:
1. The 2021 cucumber glut and 2025 tomato price collapse (WAFA / Masrawy links).
2. The ~6.0 Mm³/yr agricultural water budget for Jenin, and the 13-wells figure.

## How to rebuild the PDF

```
cd agriq/paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Check afterwards: exactly 6 pages, no overfull boxes, no undefined references.
