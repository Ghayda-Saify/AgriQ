# Intro + Literature Review — required fixes

Source reviewed: `AgriQ-1.pdf` (teammate draft). Structure, flow, gap-logic and the two
Palestinian crisis examples are good and should be KEPT. Items below must be fixed
before submission.

---

## PRIORITY 1 — Citations (fatal on their own)

| Reference as written | Problem | Correction |
|---|---|---|
| Matsuda, Y., et al. (2024). *Performance of quantum annealing inspired algorithms…* **Nature Communications, 15, 11314** | Title is REAL; author and venue are WRONG. Classic AI-hallucinated citation. This is the paper's central methodological citation. | **Zeng, Q-G., Cui, X-P., Liu, B., et al. (2024). Performance of quantum annealing inspired algorithms for combinatorial optimization problems. *Communications Physics*, 7, 249.** |
| npj Quantum Information (2025) — "6,561× faster" | No authors, no real title. 6561 = 3^8 exactly — fabricated-number pattern. | VERIFY or DELETE. Do not cite unverified. |
| Atlantis Press (2025), "HQOMLF" | No authors, no journal, unrecognised acronym. **Carries the entire novelty claim.** | VERIFY or DELETE + soften novelty claim. |
| PSO-SA-MCS study (2025) | Literally titled "study" — placeholder. | Find real citation or delete. |
| Li, [initials TBD] et al. (2025) | TBD placeholder left in. | Complete or delete. |
| Zhang, [initials TBD] et al. (2025), "ACM" | TBD placeholder; ACM is a publisher, not a journal. | Complete or delete. |
| Masrawy (2025); WAFA (2021); PCBS (2023) | Bare — no titles or URLs. | Add full details + access dates. |

**ALSO ADD (primary source for SB, currently missing entirely):**
Goto, H., Endo, K., Suzuki, M., et al. (2021). High-performance combinatorial
optimization based on classical mechanics. *Science Advances*, 7(6), eabe7953.

**ADD for domain-wall encoding:**
Chancellor, N. (2019). Domain wall encoding of discrete variables for quantum
annealing and QAOA. *Quantum Science and Technology*, 4(4), 045004.

---

## PRIORITY 2 — Factual errors

1. **"emulating quantum adiabatic dynamics on classical hardware"** — WRONG. SB simulates
   *classical* Hamiltonian dynamics; Goto's title literally says "based on classical
   mechanics." Rewrite as: *derived from a quantum-mechanical model but executing
   classical dynamics.* The NGIA-QC audience will catch this.

2. **Title "An AI and Quantum-Optimized System"** — overclaims, and contradicts the intro's
   own "without requiring physical quantum computers."
   → **"An AI and Quantum-Inspired System for Smart Crop Planning"**

---

## PRIORITY 3 — Claims the data cannot support

1. **"LSTM … forecasting crop price AND DEMAND trends"** — no demand data exists in PCBS,
   FAOSTAT or any source checked. **Delete "demand".** Price only.

2. **"Random Forest … scores each parcel's suitability based on soil and environmental
   data"** — no per-parcel soil data; NASA POWER gives ONE grid cell for all of Jenin, so
   climate is identical across parcels. As written RF cannot discriminate between parcels.
   → Rewrite to: RF scores *crop-level* agro-climatic suitability; *parcel-level*
   variation comes from OSM land-use class (orchard/vineyard vs farmland).

---

## PRIORITY 4 — Structural conflict with our results

Section 2.2 builds: LP scales poorly → metaheuristics hit a ceiling → quantum-inspired
clears it. **Our benchmark contradicts this**: on the fair comparison SA beat dSB by 2.4%.

→ Rewrite the 2.2 conclusion to set up the ACTUAL finding:
**encoding choice dominates solver choice.** Domain-wall encoding gave +18.2% (SA) and
+5.8% (dSB) and eliminated constraint violations entirely (0/1566 vs 136/174 parcels).
This is a better fit for NGIA-QC than "our solver won."

---

## PRIORITY 5 — Missing motivation for our two strongest results

1. **Oslo / Area C.** Intro says "fragmented land ownership" but never mentions Areas
   A/B/C. We measured: **32.5% of Jenin's agricultural land is outside farmer control.**
   No prior work includes this. Needs a paragraph in the intro and a gap-statement in 2.2.

2. **Water.** Palestine's binding constraint. Jenin has **6.0 Mm3/yr** from 13 agricultural
   wells (PWA 2023). We produce a water shadow price. Currently zero setup in the text.

---

## KEEP (good material)
- 2021 cucumber glut (~10x normal volumes) and 2025 tomato collapse (~29% of production
  cost) — excellent motivation, and the cucumber glut maps directly onto our
  price-depression term. Just needs verifiable sources.
- The three-strand gap structure (recommendation / forecasting / allocation).
- Table 1 layout.
