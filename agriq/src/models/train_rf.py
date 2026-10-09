"""Random Forest crop-suitability model, with real and synthetic data
evaluated SEPARATELY (never pooled into one headline accuracy)."""
import csv, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecocrop import synth_rows, KAGGLE_SUPERSEDED

D = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

kaggle = list(csv.DictReader(open(f"{D}/data/raw/Crop_recommendation.csv")))
for r in kaggle: r["source"] = "kaggle_real"
# NPK donors are drawn from the full Kaggle set, so synthesise before filtering.
synth = synth_rows(kaggle)
# Drop the Kaggle rows for study crops now labelled from EcoCrop, so no crop
# carries two conflicting envelopes. The remaining Kaggle crops stay in as
# real-data context the classifier must discriminate against.
dropped = [r for r in kaggle if r["label"] in KAGGLE_SUPERSEDED]
real = [r for r in kaggle if r["label"] not in KAGGLE_SUPERSEDED]
print(f"dropped {len(dropped)} Kaggle rows superseded by EcoCrop: "
      f"{sorted(KAGGLE_SUPERSEDED)}")

def XY(rows):
    X = np.array([[float(r[f]) for f in FEATS] for r in rows])
    y = np.array([r["label"] for r in rows])
    return X, y

Xr, yr = XY(real); Xs, ys = XY(synth)
print(f"real rows {len(real)} ({len(set(yr))} crops) | synthetic rows {len(synth)} ({len(set(ys))} crops)")

cv = StratifiedKFold(5, shuffle=True, random_state=0)
rf = lambda: RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=2)

print("\n--- EVALUATION (reported separately, never pooled) ---")
s = cross_val_score(rf(), Xr, yr, cv=cv, n_jobs=1)
print(f"  1. REAL data, 5-fold CV      : {s.mean():.4f} +/- {s.std():.4f}   <- genuine predictive accuracy")
d = cross_val_score(DummyClassifier(strategy='most_frequent'), Xr, yr, cv=cv)
print(f"     majority-class baseline   : {d.mean():.4f}                    <- floor")
s2 = cross_val_score(rf(), Xs, ys, cv=cv, n_jobs=1)
print(f"  2. SYNTHETIC data, 5-fold CV : {s2.mean():.4f} +/- {s2.std():.4f}   <- EcoCrop envelope fidelity, NOT a result")

Xa = np.vstack([Xr, Xs]); ya = np.concatenate([yr, ys])
model = rf().fit(Xa, ya)
print(f"\ncombined model trained on {len(ya)} rows, {len(set(ya))} classes")
imp = sorted(zip(FEATS, model.feature_importances_), key=lambda t: -t[1])
print("feature importance:", ", ".join(f"{f} {v:.3f}" for f, v in imp))
import pickle
pickle.dump({"model": model, "feats": FEATS}, open("rf_model.pkl", "wb"))
print("saved rf_model.pkl")
