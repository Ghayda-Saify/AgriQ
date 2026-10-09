"""2025 price forecasts under LSTM and persistence, with 2024 EXCLUDED.

The earlier forecast_prices.py trained through 2024 and used the 2024 value as
its persistence baseline. FAOSTAT 2024 perishable prices run 1.24x-5.76x above
2023 (wartime supply disruption) while storables stay flat, so both the LSTM
input window and the persistence anchor were contaminated: it reported an
orange persistence price of $3,611/t against a 2019-2023 mean of $601/t.

Here the series is truncated at 2023, matching the price baseline used by
src/pipeline/instance.py. Forecast target is therefore 2024-ahead from a
2019-2023 window -- one step beyond the training data, as a forecast should be.
"""
import csv, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "models"))
from lstm import LSTM

PP = os.path.join(ROOT, "data", "raw", "FAOSTAT_data_en_8-15-2026.csv")
ITEM = {"wheat": "Wheat", "potatoes": "Potatoes", "tomatoes": "Tomatoes",
        "cucumbers": "Cucumbers and gherkins",
        "onions": "Onions and shallots, dry (excluding dehydrated)",
        "eggplants": "Eggplants (aubergines)", "maize": "Green corn (maize)",
        "olives": "Olives", "grapes": "Grapes", "oranges": "Oranges"}
CROPS = list(ITEM)
L = 5
MAX_YEAR = 2023

rows = list(csv.DictReader(open(PP, encoding="utf-8-sig")))
series = {c: sorted((int(r["Year"]), float(r["Value"])) for r in rows
                    if r["Item"] == it and r["Value"] and int(r["Year"]) <= MAX_YEAR)
          for c, it in ITEM.items()}

train, stats = [], {}
for ci, c in enumerate(CROPS):
    p = [np.log(v) for _, v in series[c]]
    mu, sd = float(np.mean(p)), float(np.std(p)) or 1.0
    stats[c] = (mu, sd)
    for i in range(len(p) - L):
        train.append((ci, [(x - mu) / sd for x in p[i:i + L]], (p[i + L] - mu) / sd))

rng = np.random.default_rng(0)
aug = list(train)
for _ in range(5):
    for ci, seq, tgt in train:
        s = np.array(seq)
        j = s + rng.normal(0, 0.05 * max(np.std(s), 1e-3), L)
        aug.append((ci, list(j * rng.uniform(0.95, 1.05)), tgt))


def toX(recs):
    X = np.zeros((len(recs), L, 1 + len(CROPS))); y = np.zeros(len(recs))
    for k, (ci, seq, tgt) in enumerate(recs):
        for t in range(L):
            X[k, t, 0] = seq[t]; X[k, t, 1 + ci] = 1.0
        y[k] = tgt
    return X, y


Xtr, ytr = toX(aug)
model = LSTM(1 + len(CROPS), n_hidden=24, seed=0).fit(Xtr, ytr, epochs=350, lr=0.01, batch=64)

out = []
print(f"{'crop':<11}{'2019-23 mean':>14}{'2023 (persist)':>16}{'LSTM ahead':>12}{'vs mean':>9}")
print("-" * 62)
for ci, c in enumerate(CROPS):
    p = [np.log(v) for _, v in series[c]]
    mu, sd = stats[c]
    X = np.zeros((1, L, 1 + len(CROPS)))
    for t in range(L):
        X[0, t, 0] = (p[-L + t] - mu) / sd; X[0, t, 1 + ci] = 1.0
    lstm_v = float(np.exp(model.predict(X)[0] * sd + mu))
    persist = float(np.exp(p[-1]))
    mean5 = float(np.mean([v for y, v in series[c] if y >= 2019]))
    out.append({"crop": c, "price_mean_2019_2023_usd_t": round(mean5, 1),
                "price_persistence_2023_usd_t": round(persist, 1),
                "price_lstm_usd_t": round(lstm_v, 1)})
    print(f"{c:<11}{mean5:>14.0f}{persist:>16.0f}{lstm_v:>12.0f}{(lstm_v/mean5-1)*100:>8.1f}%")

dest = os.path.join(ROOT, "data", "processed", "price_forecast_clean.csv")
with open(dest, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(f"\nwrote {dest}")
