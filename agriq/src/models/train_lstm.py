"""Pooled LSTM price forecasting for 10 Palestinian crops, benchmarked
against four standard baselines with strict walk-forward validation."""
import csv, glob, numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstm import LSTM

D = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PP = f"{D}/data/raw/FAOSTAT_data_en_8-15-2026.csv"
ITEM = {"wheat":"Wheat","potatoes":"Potatoes","tomatoes":"Tomatoes",
 "cucumbers":"Cucumbers and gherkins",
 "onions":"Onions and shallots, dry (excluding dehydrated)",
 "eggplants":"Eggplants (aubergines)","maize":"Green corn (maize)",
 "olives":"Olives","grapes":"Grapes","oranges":"Oranges"}
CROPS = list(ITEM)
L = 5          # window length
N_TEST = 5     # last 5 years held out per crop

LOG=True
MAXYR=2023
rows = list(csv.DictReader(open(PP, encoding="utf-8-sig")))
series = {}
for c, it in ITEM.items():
    v = sorted((int(r["Year"]), (np.log(float(r["Value"])) if LOG else float(r["Value"]))) for r in rows
               if r["Item"] == it and r["Value"] and int(r["Year"])<=MAXYR)
    series[c] = v
print("series lengths:", {c: len(v) for c, v in series.items()})

# ---- windows, split BEFORE normalising (no leakage) ----------------------
def windows(vals, L):
    return [(vals[i:i+L], vals[i+L]) for i in range(len(vals)-L)]

train, test, stats = [], [], {}
for ci, c in enumerate(CROPS):
    p = [x for _, x in series[c]]
    cut = len(p) - N_TEST
    tr_p = p[:cut]
    mu, sd = float(np.mean(tr_p)), float(np.std(tr_p)) or 1.0
    stats[c] = (mu, sd)
    for seq, tgt in windows(p, L):
        rec = (ci, [(x-mu)/sd for x in seq], (tgt-mu)/sd, seq, tgt)
        (test if p.index(tgt) >= cut else train).append(rec)
print(f"train windows {len(train)}  test windows {len(test)}")

def to_XY(recs):
    X = np.zeros((len(recs), L, 1+len(CROPS))); y = np.zeros(len(recs))
    for k,(ci,seq,tgt,_,_) in enumerate(recs):
        for t in range(L):
            X[k,t,0] = seq[t]; X[k,t,1+ci] = 1.0
        y[k] = tgt
    return X, y

# ---- augmentation: jitter + magnitude warp (train only) ------------------
rng = np.random.default_rng(0)
aug = list(train)
for _ in range(5):
    for ci, seq, tgt, rs, rt in train:
        s = np.array(seq)
        j = s + rng.normal(0, 0.05*max(np.std(s),1e-3), L)   # jitter
        w = j * rng.uniform(0.95, 1.05)                      # magnitude warp
        aug.append((ci, list(w), tgt, rs, rt))
print(f"augmented train windows: {len(aug)}  (x{len(aug)/len(train):.1f})")

Xtr, ytr = to_XY(aug); Xte, yte = to_XY(test)
model = LSTM(1+len(CROPS), n_hidden=24, seed=0).fit(Xtr, ytr, epochs=350, lr=0.01, batch=64)
pred_n = model.predict(Xte)

# ---- de-normalise and compare against baselines --------------------------
def rmse(a,b): return float(np.sqrt(np.mean((np.array(a)-np.array(b))**2)))
def mape(a,b): return float(np.mean(np.abs((np.array(a)-np.array(b))/np.array(b)))*100)

actual, lstm_p, naive, mean3, trend, ar1 = [], [], [], [], [], []
for k,(ci,seq,tgt,rs,rt) in enumerate(test):
    mu, sd = stats[CROPS[ci]]
    actual.append(np.exp(rt) if LOG else rt)
    lstm_p.append(np.exp(pred_n[k]*sd+mu) if LOG else pred_n[k]*sd+mu)
    naive.append(np.exp(rs[-1]) if LOG else rs[-1])
    mean3.append(float(np.exp(np.mean(rs[-3:])) if LOG else np.mean(rs[-3:])))
    x = np.arange(L); m,b0 = np.polyfit(x, rs, 1); trend.append(float(np.exp(m*L+b0) if LOG else m*L+b0))
    a = np.array(rs[:-1]); bb = np.array(rs[1:])
    coef = float((a@bb)/(a@a)) if a@a else 1.0
    ar1.append(float(np.exp(coef*rs[-1]) if LOG else coef*rs[-1]))

print("\n" + "="*62)
print("PRICE FORECAST BENCHMARK - LOG SCALE (walk-forward, last 5 yrs x 10 crops)")
print("="*62)
print(f"{'model':<22}{'RMSE USD/t':>12}{'MAPE %':>10}")
print("-"*62)
res = [("LSTM (pooled+aug)", lstm_p), ("Naive (persistence)", naive),
       ("Mean of last 3", mean3), ("Linear trend", trend), ("AR(1)", ar1)]
for nm, p in res:
    print(f"{nm:<22}{rmse(p,actual):>12.1f}{mape(p,actual):>10.1f}")
best = min(res, key=lambda t: rmse(t[1], actual))
print("-"*62)
print(f"BEST: {best[0]}")
print("=> LSTM beats persistence" if rmse(lstm_p,actual)<rmse(naive,actual)
      else "=> LSTM does NOT beat persistence")
