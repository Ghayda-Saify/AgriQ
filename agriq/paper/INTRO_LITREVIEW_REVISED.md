# AgriQ: An AI and Quantum-Inspired System for Smart Crop Planning

*Suggested revision of the Introduction and Literature Review. Based on the
original draft; changed passages address the points in `NOTE_TO_TEAMMATE.md`.
Items marked **[SOURCE NEEDED]** stay only if a link is provided.*

---

## 1. Introduction

Agricultural systems worldwide face a persistent challenge of balancing
production capacity with fluctuating market demand. The increasing complexity of
agricultural markets, climate variability, and resource limitations have made
traditional experience-based crop planning insufficient for achieving
sustainable production. When planting decisions are made without considering
how the market will absorb the combined harvest, agricultural systems may
experience repeated cycles of overproduction, market saturation, price
reductions, and economic losses for farmers. Beyond financial consequences,
inefficient crop allocation contributes to unnecessary consumption of water,
fertilizer, and land.

These challenges are particularly relevant in Palestine, where agricultural
production is characterized by fragmented land access, a large proportion of
smallholder farmers, and limited access to advanced forecasting tools. Many
farmers continue to rely primarily on previous-season experience when selecting
crops, which can lead to simultaneous cultivation of similar crops and
subsequent market oversupply. Recent agricultural crises illustrate the cost of
this pattern: the 2021 cucumber glut, where volumes reached roughly ten times
normal levels, and 2025 tomato prices that collapsed to approximately 29% of
production cost (WAFA, 2021; Masrawy, 2025) **[SOURCE NEEDED: both links]**.

Two further constraints shape what Palestinian farmers can plant, and where.
The first is land access. Under the Oslo interim framework, agricultural land is
divided into Areas A and B, administered by the Palestinian Authority, and
Area C, under Israeli administrative control, where Palestinian cultivation is
heavily restricted. In Jenin governorate, we find that 32.5% of the agricultural
land identified in OpenStreetMap lies in Area C. The second is water:
agricultural irrigation in Jenin draws on approximately 6.0 million m³ per year
from 13 wells (PCBS, 2023) **[SOURCE NEEDED: table reference]**. Both are hard
limits on allocation, and neither has, to our knowledge, been represented in a
crop-planning model for the region.

Machine learning has demonstrated strong capability in identifying relationships
between soil characteristics, climatic variables, and crop suitability, with
ensemble methods such as Random Forest well suited to the nonlinear,
heterogeneous features typical of agricultural data — soil nutrients, pH,
temperature, and rainfall (Breiman, 2001). Long Short-Term Memory (LSTM)
networks, in turn, are widely used to model temporal dependencies in price
series (Hochreiter & Schmidhuber, 1997), although their advantage over simple
baselines is not guaranteed when only short annual records are available. Yet a
crop that is agronomically suitable can still cause economic loss if its
combined production exceeds what the market can absorb, which points to a
further requirement beyond prediction: allocation. Determining how available
land should be distributed among competing crops is a combinatorial
optimization problem, and classical approaches such as linear programming lose
effectiveness as the objective becomes nonlinear — as it does once prices
respond to total supply — and as decision spaces grow large.

Quantum-inspired methods such as Simulated Bifurcation (SB) have been proposed as
an alternative for large combinatorial problems (Goto et al., 2021). SB is
derived from a quantum-mechanical model but runs classical dynamics, so it
executes on conventional hardware without requiring a quantum computer. To use
such a solver, the allocation problem must first be written as a Quadratic
Unconstrained Binary Optimization (QUBO) problem, and this requires a choice of
*encoding*: how each parcel's choice among several crops is represented with
binary variables, and how the rule "exactly one crop per parcel" is enforced.
This choice is usually treated as an implementation detail. In this study we
find that it is not.

This study introduces AgriQ, an AI and quantum-inspired system for smart crop
planning in Jenin, Palestine. The system combines three components: (1) a Random
Forest model that scores the agro-climatic suitability of each candidate crop,
combined with each parcel's soil properties and land-use type to produce a
parcel-level constraint that limits which crops the optimizer may assign;
(2) an LSTM model for forecasting crop prices, evaluated against standard
forecasting baselines; and (3) a QUBO allocation model, solved with Simulated
Bifurcation and Simulated Annealing, in which crop prices fall as total
production rises and irrigation water carries a cost. The system is built and
evaluated on real data for Jenin: OpenStreetMap parcel boundaries intersected
with Oslo area boundaries, SoilGrids soil measurements for each parcel, NASA
POWER weather records, and FAOSTAT and Palestinian Central Bureau of Statistics
prices and yields.

The contributions of this study are:

1. **An encoding-versus-solver benchmark on a real agricultural instance.** We
   compare two encodings (one-hot penalty and domain-wall) against two solvers
   (Simulated Bifurcation and Simulated Annealing) on 174 parcels and 1,914
   binary variables, and show that the encoding, rather than the solver,
   determines solution quality.
2. **A QUBO formulation of crop allocation with market-clearing prices and a
   priced water constraint**, which avoids the numerical problems of enforcing
   hard capacity limits through penalty terms.
3. **A measurement of the agricultural cost of restricted land access** under
   the Oslo framework, together with soil evidence that the excluded land is
   agronomically comparable to the land farmers can use.
4. **A transparent evaluation of each AI component against a trivial baseline**,
   including the finding that an LSTM does not outperform a persistence forecast
   on the available price records.

---

## 2. Literature Review

### 2.1 AI-Based Crop Recommendation and Price Forecasting

Random Forest, an ensemble learning method built on bagged decision trees,
remains a strong baseline for tabular agricultural data because it handles
nonlinear, heterogeneous features — soil nutrients, pH, temperature, rainfall —
without extensive preprocessing (Breiman, 2001). Dey et al. (2024) achieved
classification accuracies above 99% for crop recommendation using XGBoost on
soil and climatic features, but their pipeline stops at suitability: it has no
forecasting or allocation stage, so a crop flagged "suitable" may still be a
poor planting choice if the market is already saturated. Saha et al. (2025)
went further, combining satellite-based land mapping, machine-learning crop
recommendation and yield prediction (with accuracies above 95%), and an
irrigation controller; however, their LSTM component forecasts healthy
vegetation area rather than market prices, so it cannot signal oversupply before
it occurs.

Long Short-Term Memory networks (Hochreiter & Schmidhuber, 1997) model
long-range temporal dependencies and are widely applied to agricultural price
series. Their benefit is, however, sensitive to the length of the available
record. A governorate-level price history is short — a few decades of annual
observations per crop — which makes comparison against simple forecasting
baselines essential rather than optional. We therefore evaluate the LSTM against
persistence, autoregressive, moving-average, and trend baselines before using any
forecast in allocation.

### 2.2 Optimization Approaches: From Classical Methods to Quantum-Inspired Algorithms

Crop allocation has traditionally been formulated as a linear programming (LP)
problem **[Zhang et al. (2025) — reinstate with its figures only if the paper is
located]**. More recent work treats it as a nonlinear problem that integrates
prediction and market effects. Li et al. (2025) combined simulated annealing
with a genetic algorithm (H-SAGA) for crop planning across 34 plots in
mountainous northern China, feeding neural-network price forecasts into the
optimizer and constraining production by regional market absorption capacity
and total farm water supply; H-SAGA achieved profit-increment ratios 5–10
percentage points higher than standalone GA, SA, PSO, and ACO. Hou et al.
(2026) paired particle swarm optimization with simulated annealing and Monte
Carlo simulation (PSO–SA–MCS), lowering crop prices when production exceeds
demand, and increased expected profit by 12.6% over a baseline PSO; the Monte
Carlo component raises runtime complexity to O(N²·pop). These studies show that
integrating forecasting, market saturation, and resource limits into a single
allocation model is feasible with classical metaheuristics.

Quantum and quantum-inspired approaches offer a different route. Zeng et al.
(2024) benchmarked quantum-annealing-inspired algorithms, including variants of
Simulated Bifurcation, against a D-Wave quantum annealer on Max-Cut problems
with up to 20,000 nodes: ballistic SB reduced time-to-solution nearly fifty-fold
relative to quantum annealing on small and Chimera-structured graphs, while
discrete SB reached target solutions fastest on large graphs. On dense QUBO
instances, Kim et al. (2025) reported that a hybrid quantum-annealing solver was
roughly 6,561× faster than the best classical solver while matching its accuracy.
In agriculture, Verma et al. (2025) proposed a Hybrid Quantum Optimization and
Machine Learning Framework (HQOMLF) that combines a quantum convolutional neural
network for prediction with the Quantum Approximate Optimization Algorithm for
resource allocation, reporting processing speed-ups of up to 3.5×.

Two points, however, remain unaddressed. First, none of these agricultural
studies formulates allocation as a QUBO with market-clearing prices for
Ising-type or quantum-inspired solvers: Li et al. and Hou et al. use classical
metaheuristics, and Verma et al. apply QAOA without a market model. Second, the
benchmarking literature compares *solvers*, but on a real problem, results also
depend on how constraints are *encoded*. The standard approach, one-hot encoding
with a penalty term, leaves invalid assignments reachable and couples every crop
variable within a parcel to every other. Chancellor (2019) proposed domain-wall
encoding, in which a chain of binary variables makes invalid assignments
impossible by construction. Whether the choice of encoding matters as much as —
or more than — the choice of solver on a real allocation problem has not, to our
knowledge, been measured.

Table 1 summarizes the studies reviewed above.

| Study | Focus Area | Method | Limitation Relative to This Study |
|---|---|---|---|
| Breiman (2001) | Ensemble learning, general | Random Forest (foundational) | Not agricultural; establishes the method this study applies to crop suitability |
| Hochreiter & Schmidhuber (1997) | Sequence modeling, general | LSTM (foundational) | Not agricultural; establishes the method this study applies to price forecasting |
| Dey et al. (2024) | Crop recommendation | XGBoost on soil/climate features | Suitability only; no forecasting or allocation stage |
| Saha et al. (2025) | Land suitability, irrigation | ML + LSTM (vegetation), satellite data | LSTM forecasts vegetation area, not market prices; no allocation across parcels |
| Li et al. (2025) | Dynamic multi-plot crop planning | Hybrid SA–GA (H-SAGA) + neural-network price forecasts | Classical metaheuristic; no QUBO formulation or constraint-encoding analysis; no land-access restrictions |
| Hou et al. (2026) | Crop planning under uncertainty | Hybrid PSO–SA + Monte Carlo | No learned forecasts; classical metaheuristic; O(N²·pop) runtime |
| Zeng et al. (2024) | Quantum-inspired combinatorial optimization | SB and simulated CIM vs. D-Wave quantum annealer | Abstract Max-Cut benchmarks; solver comparison only, not encoding |
| Kim et al. (2025) | Large-scale dense QUBO benchmarking | Hybrid quantum annealing vs. classical solvers | Benchmark instances, not agricultural; solver comparison only |
| Verma et al. (2025) | Quantum computing in precision agriculture | QCNN + QAOA (HQOMLF) | No market-clearing prices; no QUBO penalty/encoding analysis; no comparison with SB |
| Chancellor (2019) | Encoding discrete variables for quantum optimization | Domain-wall encoding | Theoretical; not evaluated on a real-world allocation problem |

*Table 1. Summary of reviewed studies across crop recommendation, forecasting,
and optimization.*

Taken together, this literature shows that integrated crop planning — combining
prediction, market saturation, and resource limits — is established with
classical metaheuristics (Li et al., 2025; Hou et al., 2026), and that
quantum-inspired solvers perform strongly on abstract benchmarks (Zeng et al.,
2024; Kim et al., 2025). What has not been examined is the meeting point of the
two: agricultural allocation written as a QUBO with market-clearing prices, and
the effect of constraint encoding on how well quantum-inspired solvers handle it.
This study addresses that gap directly. The Random Forest stage passes parcel
suitability forward as a constraint on the optimizer; the LSTM stage is tested
against simple baselines before its forecasts are used; and the allocation stage
compares two encodings across two solvers on 174 real parcels in Jenin, while
accounting for the land-access and water constraints specific to the Palestinian
context.

---

## References

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chancellor, N. (2019). Domain wall encoding of discrete variables for quantum annealing and QAOA. *Quantum Science and Technology, 4*(4), 045004. https://doi.org/10.1088/2058-9565/ab33c2

Dey, B., Ferdous, J., & Ahmed, R. (2024). Machine learning based recommendation of agricultural and horticultural crop farming in India under the regime of NPK, soil pH and three climatic variables. *Heliyon, 10*(3), e25112. https://doi.org/10.1016/j.heliyon.2024.e25112

Goto, H., Endo, K., Suzuki, M., Sakai, Y., Kanao, T., Hamakawa, Y., Hidaka, R., Yamasaki, M., & Tatsumura, K. (2021). High-performance combinatorial optimization based on classical mechanics. *Science Advances, 7*(6), eabe7953. https://doi.org/10.1126/sciadv.abe7953

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Hou, D., Li, J., Wang, Y., Guo, S., Wang, X., & Wang, X. (2026). Research on agricultural development issues based on comprehensive optimization strategy. *Scientific Reports, 16*, 12505. https://doi.org/10.1038/s41598-026-39307-8

Kim, S., Ahn, S.-W., Suh, I.-S., Dowling, A. W., Lee, E., & Luo, T. (2025). Quantum annealing for combinatorial optimization: A benchmarking study. *npj Quantum Information, 11*, 77. https://doi.org/10.1038/s41534-025-01020-1

Li, C., Su, Z., Nie, Y., Ye, Z., Li, J., Wang, J., Yang, Z., Li, X., Zeng, W., & Chen, Y. (2025). Scientific planning of dynamic crops in complex agricultural landscapes based on adaptive optimization hybrid SA-GA method. *Scientific Reports, 15*, 28992. https://doi.org/10.1038/s41598-025-14188-5

Masrawy. (2025). **[SOURCE NEEDED: article title and URL]**

Palestinian Central Bureau of Statistics. (2023). **[SOURCE NEEDED: table title and URL]**

Saha, G., Shahrin, F., Khan, F. H., Meshkat, M. M., & Azad, A. A. M. (2025). Smart IoT-driven precision agriculture: Land mapping, crop prediction, and irrigation system. *PLOS ONE, 20*(3), e0319268. https://doi.org/10.1371/journal.pone.0319268

Verma, D. S., Dafadar, M., Mishra, J. K., Kumar, A., & Priydarshi, M. K. (2025). Quantum-enhanced computational models for multi-variable optimization in precision agriculture. Atlantis Press. https://www.atlantis-press.com/article/126013752 **[CONFIRM: proceedings name and pages]**

WAFA. (2021). **[SOURCE NEEDED: article title and URL]**

Zeng, Q.-G., Cui, X.-P., Liu, B., Wang, Y., Mosharev, P., & Yung, M.-H. (2024). Performance of quantum annealing inspired algorithms for combinatorial optimization problems. *Communications Physics, 7*, 249. https://doi.org/10.1038/s42005-024-01705-7
