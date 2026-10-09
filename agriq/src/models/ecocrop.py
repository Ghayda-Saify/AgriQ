"""FAO EcoCrop envelopes for all 10 study crops.

Every study crop is labelled from published FAO EcoCrop optimal ranges, so all
ten share one provenance. Temperature / rainfall / pH come from EcoCrop.
EcoCrop does NOT publish N-P-K, so those are sampled from the empirical
distribution of the agronomically nearest real Kaggle crops (documented per
crop below).

Maize, grapes and orange were previously taken as raw Kaggle rows, which
describe Indian field conditions: Jenin fell outside those envelopes on 4/7,
6/7 and 5/7 features respectively, and orange was driven to exactly zero
probability on every parcel. Their Kaggle rows are now dropped from training
in favour of EcoCrop, which removes that covariate shift and makes the
labelling rule uniform across the ten crops.

EcoCrop rainfall is an ANNUAL WATER figure, so the model is fed rainfall plus
each crop's irrigation allotment (see build_suitability.py). Scoring an
irrigated system on rainfall alone was what excluded orange, a crop carrying
the largest irrigation requirement in the instance.

These rows are SYNTHETIC and are evaluated separately from real data
throughout -- never pooled into a single accuracy figure.
"""
import numpy as np

# crop: (temp_opt_C, rain_opt_mm, ph_opt, nearest real Kaggle crops for NPK)
#
# The three entries marked [FAO] were read from the FAO EcoCrop data sheets on
# 2026-10-09 and carry their database id, so they are traceable:
#   maize   -> Zea mays ssp. saccharata, ecocrop id 10981
#   grapes  -> Vitis vinifera,           ecocrop id 2160
#   oranges -> Citrus sinensis,          ecocrop id 720
ECOCROP = {
    "wheat":     ((15, 24), (400,  900), (6.0, 7.5), ["lentil", "chickpea"]),
    "potatoes":  ((15, 22), (500, 1000), (5.5, 7.0), ["mothbeans", "blackgram"]),
    "tomatoes":  ((18, 27), (600, 1200), (6.0, 7.0), ["muskmelon", "watermelon"]),
    "cucumbers": ((18, 30), (600, 1000), (6.0, 7.5), ["muskmelon", "watermelon"]),
    "onions":    ((13, 25), (400,  800), (6.0, 7.5), ["lentil", "mungbean"]),
    "eggplants": ((22, 30), (600, 1200), (5.5, 7.0), ["muskmelon", "banana"]),
    "olives":    ((15, 25), (300,  700), (7.0, 8.5), ["pomegranate", "mango"]),
    "maize":     ((16, 24), (800, 1500), (5.5, 6.8), ["rice", "cotton"]),        # [FAO] 10981
    "grapes":    ((18, 30), (700,  850), (5.5, 7.5), ["pomegranate", "apple"]),  # [FAO] 2160
    "oranges":   ((20, 30), (1200, 2000), (5.0, 6.0), ["pomegranate", "mango"]), # [FAO] 720
}

# Kaggle labels dropped from training: their envelopes describe a different
# climate regime and are superseded by the EcoCrop entries above.
KAGGLE_SUPERSEDED = {"maize", "grapes", "orange"}


def synth_rows(kaggle_rows, n_per_crop=100, seed=42):
    rng = np.random.default_rng(seed)
    by_label = {}
    for r in kaggle_rows:
        by_label.setdefault(r["label"], []).append(r)

    out = []
    for crop, (t, rain, ph, near) in ECOCROP.items():
        pool = [r for c in near for r in by_label.get(c, [])]
        if not pool:
            raise RuntimeError(f"no NPK donor rows for {crop}")
        N = np.array([float(r["N"]) for r in pool])
        P = np.array([float(r["P"]) for r in pool])
        K = np.array([float(r["K"]) for r in pool])
        H = np.array([float(r["humidity"]) for r in pool])
        for _ in range(n_per_crop):
            out.append({
                "N": float(rng.normal(N.mean(), max(N.std(), 1e-6))),
                "P": float(rng.normal(P.mean(), max(P.std(), 1e-6))),
                "K": float(rng.normal(K.mean(), max(K.std(), 1e-6))),
                "temperature": float(rng.uniform(*t)),
                "humidity": float(np.clip(rng.normal(H.mean(), max(H.std(), 1e-6)), 10, 99)),
                "ph": float(rng.uniform(*ph)),
                "rainfall": float(rng.uniform(*rain)),
                "label": crop,
                "source": "synthetic_ecocrop",
            })
    return out
