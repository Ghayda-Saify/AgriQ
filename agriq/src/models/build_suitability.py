"""Build suitability_matrix.csv for the 174 real Jenin Area A/B parcels.

suitability[i,j] = RF_climate(crop j) x landuse_fit(parcel i, crop j)

RF_climate uses the trained Kaggle+EcoCrop Random Forest (rf_model.pkl).
Per-parcel real inputs: pH, N from SoilGrids. Constant-across-Jenin real
inputs: temperature, humidity, rainfall from NASA POWER (single grid cell --
RF therefore gives CROP-level climate suitability, not parcel-level; see
PAPER_FIXES_TODO.md Priority 3). P and K are UNOBTAINABLE for Jenin (no
per-parcel source exists) and are Monte-Carlo sampled over a published range
for Mediterranean/West Bank agricultural soils; we report the resulting
uncertainty (std across draws), never a single invented P/K value.

landuse_fit reuses instance.py's OSM-fclass logic verbatim: orchard/vineyard
parcels favor tree crops, farmland favors annuals.

ASSUMPTIONS flagged for the paper:
  - SoilGrids N (total N, g/kg) is converted to plant-available N
    (kg/ha, the Kaggle RF's unit) in two documented steps: (1) total N
    stock = N_gkg x bulk_density(g/cm3) x depth(cm) x 100 [standard
    soil-nutrient-stock formula, e.g. Ross (1995), a mass-balance identity,
    not a fitted parameter], using BD=1.3 g/cm3 and depth=20cm (typical
    Mediterranean topsoil); (2) available N = 2% of that stock, a commonly
    cited annual mineralization fraction for semi-arid soils (Brady & Weil,
    "The Nature and Properties of Soils"). This is an approximation, not a
    measured quantity -- flagged as a limitation.
  - The water feature is EFFECTIVE annual water, rainfall plus the crop's own
    irrigation allotment, not rainfall alone. Scoring an irrigated system on
    rainfall is the wrong question: the optimiser buys irrigation for these
    crops and then the suitability model was penalising them for not having
    it. Orange, which carries the largest requirement in the instance
    (7000 m3/ha = 700 mm), was driven to exactly zero probability on every
    parcel by that inconsistency. Feeding rainfall + irrigation also removes
    the earlier clip of Jenin's 465 mm/yr down to the Kaggle maximum of
    298.6 mm, since the EcoCrop envelopes span 300-2000 mm and Jenin's
    effective water now falls naturally inside them.
  - 16/174 Area A/B parcels have no direct SoilGrids match; they use the
    Jenin-wide SoilGrids mean pH/N (flagged, printed below).
"""
import csv, os, pickle, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CROPS = ["wheat", "potatoes", "tomatoes", "cucumbers", "onions",
         "eggplants", "maize", "olives", "grapes", "oranges"]
# Every study crop is now labelled from EcoCrop under its own project name, so
# the RF class label and the project crop name coincide.
RF_LABEL = {c: c for c in CROPS}

BULK_DENSITY_G_CM3 = 1.3             # ASSUMPTION -- typical Mediterranean topsoil
DEPTH_CM = 20.0                      # ASSUMPTION -- standard topsoil sampling depth
MINERALIZATION_FRACTION = 0.02       # ASSUMPTION -- annual available fraction of total N

# Net irrigation requirement, mm/yr. Mirrors WATER (m3/ha) in
# pipeline/instance.py divided by 10; the consistency check below fails loudly
# if the two ever drift apart.
IRRIGATION_MM = {
    "wheat": 50.0, "potatoes": 400.0, "tomatoes": 600.0, "cucumbers": 450.0,
    "onions": 350.0, "eggplants": 600.0, "maize": 500.0, "olives": 150.0,
    "grapes": 300.0, "oranges": 700.0,
}

# Published Olsen-P / exchangeable-K ranges for Mediterranean/West Bank
# agricultural soils (typical medium-fertility field ranges, mg/kg / ~kg/ha
# equivalent -- matched to the Kaggle feature's own kg/ha-equivalent scale).
# ASSUMPTION: no per-parcel source exists; Monte Carlo over this range,
# uncertainty reported explicitly. Never treated as a single true value.
P_RANGE = (10.0, 60.0)
K_RANGE = (40.0, 180.0)

N_MC = 200
SEED = 0


def load_rf():
    with open(os.path.join(ROOT, "rf_model.pkl"), "rb") as f:
        d = pickle.load(f)
    return d["model"], d["feats"]


def nasa_power_annual():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "raw", "nasa_power_jenin_monthly.csv"))))
    temp = np.mean([float(r["temperature_c"]) for r in rows])
    hum = np.mean([float(r["humidity_pct"]) for r in rows])
    by_year = {}
    for r in rows:
        by_year.setdefault(r["year"], 0.0)
        by_year[r["year"]] += float(r["rainfall_mm_month"])
    rain = float(np.mean(list(by_year.values())))
    return temp, hum, rain


def soilgrids_by_id():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "processed", "soilgrids_jenin.csv"))))
    have = [r for r in rows if r["ph"]]
    mean_ph = float(np.mean([float(r["ph"]) for r in have]))
    mean_n = float(np.mean([float(r["n_g_kg"]) for r in have]))
    return {r["osm_id"]: r for r in rows}, mean_ph, mean_n


def landuse_fit(fclass, crop):
    TREE = {"olives", "grapes", "oranges"}
    is_tree = crop in TREE
    if fclass in ("orchard", "vineyard"):
        return 0.9 if is_tree else 0.3
    return 0.9 if not is_tree else 0.35


def main():
    model, feats = load_rf()
    assert feats == ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    classes = list(model.classes_)
    class_idx = {c: i for i, c in enumerate(classes)}

    temp, hum, rain = nasa_power_annual()
    # Cross-check the irrigation table against the optimiser's own WATER dict
    # so the two definitions of a crop's water cannot silently diverge.
    sys.path.insert(0, os.path.join(ROOT, "src", "pipeline"))
    from instance import WATER as _W
    bad = {c: (IRRIGATION_MM[c], _W[c] / 10.0) for c in CROPS
           if abs(IRRIGATION_MM[c] - _W[c] / 10.0) > 1e-6}
    if bad:
        raise RuntimeError(f"IRRIGATION_MM disagrees with instance.WATER: {bad}")

    effective = {c: rain + IRRIGATION_MM[c] for c in CROPS}
    print(f"NASA POWER Jenin: temp={temp:.2f}C humidity={hum:.1f}% "
          f"rainfall={rain:.0f}mm/yr")
    print("effective annual water (rainfall + irrigation), mm: " +
          ", ".join(f"{c} {effective[c]:.0f}" for c in CROPS))

    soil, mean_ph, mean_n_gkg = soilgrids_by_id()

    # Match instance.py's load_parcels(lo=0.1, hi=100.0) exactly so this matrix
    # lines up 1:1 with the instance. Both administrative classes are built so
    # the matrix stays reusable; only class A is optimised.
    parcels = [r for r in csv.DictReader(open(os.path.join(ROOT, "data", "processed", "jenin_parcels_zoned.csv")))
               if r["zone"] in ("A", "C") and 0.1 <= float(r["area_ha"]) <= 100.0]
    missing = 0

    rng = np.random.default_rng(SEED)
    P_draws = rng.uniform(*P_RANGE, size=N_MC)
    K_draws = rng.uniform(*K_RANGE, size=N_MC)

    out_rows = []
    suit_summary = {c: [] for c in CROPS}

    for r in parcels:
        osm_id = r["osm_id"]
        fclass = r["fclass"]
        s = soil.get(osm_id)
        if s and s["ph"]:
            ph = float(s["ph"])
            n_gkg = float(s["n_g_kg"])
        else:
            missing += 1
            ph, n_gkg = mean_ph, mean_n_gkg
        total_n_stock_kgha = n_gkg * BULK_DENSITY_G_CM3 * DEPTH_CM * 100.0
        n_kgha = total_n_stock_kgha * MINERALIZATION_FRACTION
        n_kgha = float(np.clip(n_kgha, 0, 140))

        # Each crop is scored under ITS OWN water regime, so the ten crops no
        # longer share one feature vector and cannot be renormalised into a
        # simplex. suit_ij is instead an absolute match score: the ensemble's
        # confidence that conditions at parcel i, given crop j's irrigation,
        # would be labelled crop j. instance.py converts each crop's column to
        # a within-crop percentile rank across parcels, so only the ordering
        # within a column is consumed downstream and the absolute scale is
        # immaterial. All ten crops are stacked into one predict_proba call.
        blocks = [np.column_stack([
            np.full(N_MC, n_kgha), P_draws, K_draws,
            np.full(N_MC, temp), np.full(N_MC, hum),
            np.full(N_MC, ph), np.full(N_MC, effective[c]),
        ]) for c in CROPS]
        proba = model.predict_proba(np.vstack(blocks))      # (10*N_MC, classes)
        sub = np.empty((N_MC, len(CROPS)))
        for j, crop in enumerate(CROPS):
            seg = proba[j * N_MC:(j + 1) * N_MC]
            sub[:, j] = seg[:, class_idx[RF_LABEL[crop]]]

        row = {"osm_id": osm_id, "zone": r["zone"], "fclass": fclass,
               "ph": ph, "n_kg_ha": n_kgha}
        for j, crop in enumerate(CROPS):
            rf_p = sub[:, j]
            fit = landuse_fit(fclass, crop)
            suit_draws = rf_p * fit
            row[f"suit_{crop}"] = float(suit_draws.mean())
            row[f"suit_{crop}_std"] = float(suit_draws.std())
            suit_summary[crop].append(float(suit_draws.mean()))
        out_rows.append(row)

    out_path = os.path.join(ROOT, "data", "processed", "suitability_matrix.csv")
    fields = ["osm_id", "zone", "fclass", "ph", "n_kg_ha"] + \
             [f"suit_{c}" for c in CROPS] + [f"suit_{c}_std" for c in CROPS]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"\n{len(out_rows)} parcels written to {out_path} "
          f"({missing} lacked direct SoilGrids match, used Jenin-A mean pH/N)")
    print(f"\n{'crop':<11} {'min':>7} {'mean':>7} {'max':>7}")
    for c in CROPS:
        v = np.array(suit_summary[c])
        print(f"{c:<11} {v.min():7.3f} {v.mean():7.3f} {v.max():7.3f}")
    mean_std = np.mean([out_rows[i][f"suit_{c}_std"] for i in range(len(out_rows)) for c in CROPS])
    print(f"\nmean P/K-driven suitability std across all parcel x crop: {mean_std:.4f}")


if __name__ == "__main__":
    main()
