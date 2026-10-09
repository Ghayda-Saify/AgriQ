"""FAO EcoCrop envelopes for the 7 crops absent from the Kaggle dataset.

Kaggle covers only maize, grapes and orange of the 10 study crops. For the
remaining 7 we generate labelled rows from published FAO EcoCrop optimal
ranges. Temperature / rainfall / pH come from EcoCrop directly. EcoCrop does
NOT publish N-P-K, so those are sampled from the empirical distribution of the
agronomically nearest real Kaggle crops (documented per crop below).

These rows are SYNTHETIC and are evaluated separately from real data
throughout -- never pooled into a single accuracy figure.
"""
import numpy as np

# crop: (temp_opt_C, rain_opt_mm, ph_opt, nearest real Kaggle crops for NPK)
ECOCROP = {
    "wheat":     ((15, 24), (400,  900), (6.0, 7.5), ["lentil", "chickpea"]),
    "potatoes":  ((15, 22), (500, 1000), (5.5, 7.0), ["mothbeans", "blackgram"]),
    "tomatoes":  ((18, 27), (600, 1200), (6.0, 7.0), ["muskmelon", "watermelon"]),
    "cucumbers": ((18, 30), (600, 1000), (6.0, 7.5), ["muskmelon", "watermelon"]),
    "onions":    ((13, 25), (400,  800), (6.0, 7.5), ["lentil", "mungbean"]),
    "eggplants": ((22, 30), (600, 1200), (5.5, 7.0), ["muskmelon", "banana"]),
    "olives":    ((15, 25), (300,  700), (7.0, 8.5), ["pomegranate", "mango"]),
}


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
