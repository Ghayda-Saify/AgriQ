# AgriQ — Intro & Literature Review: changes needed

Hi! I went through the Intro and Literature Review against the experiments we
actually ran and checked every reference against its journal. The structure is
strong and most of it stays: the problem framing, the cucumber/tomato crisis
examples, the recommendation → forecasting → allocation flow, and Table 1's
layout. Below is what has to change and why. A suggested rewrite is in
`INTRO_LITREVIEW_REVISED.md`.

Everything listed here was checked against the paper itself, not guessed.

---

## 1. References — details wrong or incomplete

| As written | Problem | Correct reference |
|---|---|---|
| Matsuda, Y., et al. (2024). *Nature Communications*, 15, 11314 | Title is right; **author and journal are wrong**. This is our central SB citation. | Zeng, Q.-G., Cui, X.-P., Liu, B., Wang, Y., Mosharev, P., & Yung, M.-H. (2024). *Communications Physics*, 7, 249. https://doi.org/10.1038/s42005-024-01705-7 |
| npj Quantum Information (2025) | Real paper, but no authors/volume. | Kim, S., Ahn, S.-W., Suh, I.-S., Dowling, A. W., Lee, E., & Luo, T. (2025). *npj Quantum Information*, 11, 77. https://doi.org/10.1038/s41534-025-01020-1 |
| Atlantis Press (2025), HQOMLF | Real paper, but no authors/title. | Verma, D. S., Dafadar, M., Mishra, J. K., Kumar, A., & Priydarshi, M. K. (2025). Quantum-enhanced computational models for multi-variable optimization in precision agriculture. Atlantis Press. https://www.atlantis-press.com/article/126013752 — **please confirm the proceedings name and pages** from the article page. |
| PSO-SA-MCS study (2025), PMC | Real paper. Year is **2026**, journal is *Scientific Reports* (PMC is only the archive), title is different. | Hou, D., Li, J., Wang, Y., Guo, S., Wang, X., & Wang, X. (2026). Research on agricultural development issues based on comprehensive optimization strategy. *Scientific Reports*, 16, 12505. https://doi.org/10.1038/s41598-026-39307-8 |
| Li, [initials TBD], et al. (2025) | Real paper; title is slightly different. | Li, C., Su, Z., Nie, Y., Ye, Z., Li, J., Wang, J., Yang, Z., Li, X., Zeng, W., & Chen, Y. (2025). Scientific planning of dynamic crops in complex agricultural landscapes based on adaptive optimization hybrid SA-GA method. *Scientific Reports*, 15, 28992. https://doi.org/10.1038/s41598-025-14188-5 |
| Zhang, [initials TBD], et al. (2025), ACM | **I could not find this paper.** No search matched "34 plots / 21.3% / Branch-and-Price". Also, *Li et al.* is the paper with 34 plots, so the two may have been mixed up. | **Please send the link.** If we can't find it, we drop it. |
| Saha et al. (2025) | Correct, just incomplete. | Saha, G., Shahrin, F., Khan, F. H., Meshkat, M. M., & Azad, A. A. M. (2025). *PLOS ONE*, 20(3), e0319268. https://doi.org/10.1371/journal.pone.0319268 |
| Dey et al. (2024) | Correct, just incomplete. | Add: *Heliyon*, 10(3), e25112. https://doi.org/10.1016/j.heliyon.2024.e25112 |
| Masrawy (2025); WAFA (2021); PCBS (2023) | No titles or links. | **Please send the article links.** |

**Add these two** (we rely on them directly):
- Goto, H., Endo, K., Suzuki, M., et al. (2021). High-performance combinatorial optimization based on classical mechanics. *Science Advances*, 7(6), eabe7953. https://doi.org/10.1126/sciadv.abe7953 — the original SB paper.
- Chancellor, N. (2019). Domain wall encoding of discrete variables for quantum annealing and QAOA. *Quantum Science and Technology*, 4(4), 045004. — the encoding our main result is about.

---

## 2. Claims that don't match the cited papers

**a) Li et al. — "authors report excessive runtime at large problem sizes."**
The paper doesn't say this. It says H-SAGA has *competitive* computational
efficiency. → Remove the runtime claim.

**b) Kim et al. (npj) — "up to 10,000 variables" and "classical solvers degrading
sharply above 1,000 variables."**
The 6,561× speedup is real. I could not find either of these two details in the
paper. → Keep 6,561×, drop the two details unless you can point to where they
appear.

**c) Verma et al. (HQOMLF) — "does not integrate any AI-based prediction module"
and "the only identified prior attempt."**
HQOMLF *does* include ML: it's literally a "Hybrid Quantum Optimization and
Machine Learning Framework" (a QCNN for prediction plus QAOA for allocation).
It also isn't the only one; there are several other 2025 quantum-agriculture
papers. → Describe what it actually lacks: no QUBO formulation with market
effects, no look at encoding, and no comparison with SB.

**d) "LSTM has been applied almost exclusively to environmental or vegetation
variables rather than price or demand series."**
Only one citation backs this, and it's broadly untrue: LSTM price forecasting
for agriculture is common. A reviewer will flag it. → Soften.

---

## 3. The main gap statement contradicts the papers we cite

The intro says existing research treats recommendation, forecasting, and
allocation *"as separate problems … with no shared pipeline between them."*

Two of our own citations already do this:
- **Li et al. (2025)** combines neural-network **price forecasting** with SA-GA
  optimization, and models a **market-absorption cap** and a **water limit**.
- **Hou et al. (2026)** lowers **crop prices when production exceeds demand**.

So "nobody integrates them" and "nobody models oversupply" are both false.

**What is actually new in our work** (I checked these against the same papers):
1. Nobody formulates agricultural allocation as a **QUBO for Ising-type /
   quantum-inspired solvers** with market-clearing prices. Li and Hou use
   classical metaheuristics. Verma uses QAOA but has no market model and no
   QUBO/penalty formulation.
2. Nobody studies **how the constraint encoding affects** these solvers. This is
   our main result.
3. Nobody includes **politically restricted land access** (Oslo Area C).

→ The rewrite re-grounds the gap on these three points.

---

## 4. Claims that contradict our results

**a) "LSTM model for forecasting crop price *and demand* trends."**
We have no demand data. PCBS and FAOSTAT only give prices. And our LSTM lost to
the simple "same as last year" baseline (RMSE 252 vs 130). → Price only, and
present LSTM as something we *test*, not a component that works.

**b) §2.2 builds "LP scales poorly → metaheuristics hit a ceiling →
quantum-inspired clears it."**
Our 20-run benchmark shows otherwise. Classical simulated annealing got the
best profit with zero invalid assignments. SB under the standard (one-hot)
encoding left 129 of 174 parcels invalid. With domain-wall encoding, SB and SA
were statistically tied (p = 0.93). → The section should build toward
**"encoding is the overlooked factor,"** not "quantum-inspired wins."

**c) "Simulated Bifurcation … emulating quantum adiabatic dynamics on classical
hardware."**
Factually wrong. SB simulates **classical** mechanics; Goto's paper title says
"based on classical mechanics." → "derived from a quantum-mechanical model but
running classical dynamics."

**d) "Random Forest scores each parcel's suitability based on soil and
environmental data."**
The NASA weather data is one grid cell for all of Jenin, so climate is identical
on every parcel. What differs between parcels is soil pH/nitrogen and land-use
type (orchard vs farmland). → Reword accordingly.

**e) Title: "Quantum-Optimized."**
Overclaims, and contradicts our own "no quantum computer needed." →
**"Quantum-Inspired."**

**f) "Datasets … adapted to represent Jenin growing conditions."**
This undersells us and sounds like synthetic data. Ours is real Jenin data:
OpenStreetMap parcels, SoilGrids soil per parcel, NASA POWER weather for Jenin,
FAOSTAT/PCBS prices. → Say so.

---

## 5. Missing: our two strongest applied results

- **Oslo / Area C.** The intro mentions "fragmented land ownership" but not the
  Oslo zones. We measured **32.5% of the study parcel area is in Area C**, and excluding
  it forgoes **$0.23–1.22M/yr** within the water budget, depending on market size. Needs a short paragraph.
- **Water.** Jenin has **6.0 Mm³/yr** from 13 agricultural wells. We found water
  isn't the binding limit on accessible land, only when Area C is included
  ($1.09–1.35/m³). Needs a sentence or two.

---

## 6. Things I need from you

1. Links for **WAFA 2021** (cucumber glut, ~10× volumes) and **Masrawy 2025**
   (tomatoes at ~29% of cost). Until we have them, these numbers can't stay.
2. The **Zhang et al.** link, or we drop it.
3. The table number/link for **PCBS 2023**, and the source for the **6.0 Mm³ /
   13 wells** water figure (PWA via PCBS).
4. Confirm Verma et al.'s **proceedings name and pages**.

## 7. Length

The final paper is IEEE, 6 pages max, with numbered citations. Intro + Lit
Review currently run ~1,500 words plus a large table. For the paper they'll
need to come down to roughly 800–900 words, with Table 1 compressed. The rewrite
keeps the full length so you can see every change; we can cut once the content
is agreed.
