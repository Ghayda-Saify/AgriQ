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
  - Jenin's real annual rainfall (465 mm/yr) exceeds the Kaggle training
    data's rainfall range (max 298.6 mm -- that feature evidently encodes
    a shorter accumulation window, not true annual precipitation). We clip
    to the training-data maximum to avoid extrapolating the trees past any
    leaf they were fit on. Flagged as a known dataset-unit limitation.
  - 16/174 Area A/B parcels have no direct SoilGrids match; they use the
    Jenin-wide SoilGrids mean pH/N (flagged, printed below).
"""
import csv, os, pickle, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CROPS = ["wheat", "potatoes", "tomatoes", "cucumbers", "onions",
         "eggplants", "maize", "olives", "grapes", "oranges"]
# RF class labels differ from project crop names for two crops
RF_LABEL = {c: c for c in CROPS}
RF_LABEL["oranges"] = "orange"

BULK_DENSITY_G_CM3 = 1.3             # ASSUMPTION -- typical Mediterranean topsoil
DEPTH_CM = 20.0                      # ASSUMPTION -- standard topsoil sampling depth
MINERALIZATION_FRACTION = 0.02       # ASSUMPTION -- annual available fraction of total N
RAINFALL_CLIP_MM = 298.56            # Kaggle training-data max -- flagged in paper

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
    rain_clipped = min(rain, RAINFALL_CLIP_MM)
    return temp, hum, rain, rain_clipped


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

    temp, hum, rain_raw, rain = nasa_power_annual()
    print(f"NASA POWER Jenin: temp={temp:.2f}C humidity={hum:.1f}% "
          f"rainfall={rain_raw:.0f}mm/yr (clipped to {rain:.1f}mm for RF input, "
          f"training max={RAINFALL_CLIP_MM})")

    soil, mean_ph, mean_n_gkg = soilgrids_by_id()

    # Match instance.py's load_parcels(lo=0.1, hi=100.0) exactly so this matrix
    # lines up 1:1 with the instance. Both Oslo zones: A/B is the optimisation
    # space, C is needed for the Area C counterfactual.
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

        X = np.column_stack([
            np.full(N_MC, n_kgha), P_draws, K_draws,
            np.full(N_MC, temp), np.full(N_MC, hum),
            np.full(N_MC, ph), np.full(N_MC, rain),
        ])
        proba = model.predict_proba(X)   # (N_MC, n_classes)
        # Renormalize over Jenin's 10 candidate crops only: the RF was
        # trained on 29 Kaggle crops (rice, coffee, jute, ... irrelevant to
        # Jenin), which dilutes raw class probability. Conditioning on "which
        # of the 10 crops actually grown here" is the well-defined suitability
        # question -- still fully RF-driven, just renormalized over the
        # relevant support instead of the full 29-class simplex.
        crop_idx = [class_idx[RF_LABEL[c]] for c in CROPS]
        sub = proba[:, crop_idx]                      # (N_MC, 10)
        sub = sub / sub.sum(axis=1, keepdims=True)

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
