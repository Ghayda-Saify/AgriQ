"""LSTM price-forecast evaluation with strict walk-forward validation.

Runs the SAME protocol twice:
  (a) INCLUDING 2024 -- reproduces the earlier benchmark, whose test window
      (last 5 years per crop) contains the 2024 wartime price spike.
  (b) EXCLUDING 2024 -- the configuration the paper actually uses, matching
      the 2019-2023 price baseline in src/pipeline/instance.py. FAOSTAT 2024
      perishable prices run 1.24x-5.76x above 2023 while storables stay flat,
      so a test set containing 2024 measures reaction to a supply shock, not
      forecasting skill.

Protocol (both runs):
  * windows split BEFORE normalising -- train statistics never see test years
  * augmentation (jitter + magnitude warp) applied to TRAIN ONLY
  * four standard baselines, incl. the trivial persistence baseline
  * log scale throughout; errors reported back on the USD/t scale

Fixes vs tests/train_lstm_with_2024_contaminated.py: absolute path made
relative; train/test assignment used p.index(target), a VALUE lookup that
silently misassigns a window whenever a crop repeats a price in two years.
"""
import csv, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "models"))
from lstm import LSTM

PP = os.path.join(ROOT, "data", "raw", "FAOSTAT_data_en_8-15-2026.csv")
ITEM = {"wheat": "Wheat", "potatoes": "Potatoes", "tomatoes": "Tomatoes",
        "cucumbers": "Cucumbers and gherkins",
        "onions": "Onions and shallots, dry (excluding dehydrated)",
        "eggplants": "Eggplants (aubergines)", "maize": "Green corn (maize)",
        "olives": "Olives", "grapes": "Grapes", "oranges": "Oranges"}
CROPS = list(ITEM)
L = 5          # window length
N_TEST = 5     # last N years held out per crop


def load_series(max_year):
    rows = list(csv.DictReader(open(PP, encoding="utf-8-sig")))
    series = {}
    for c, it in ITEM.items():
        v = sorted((int(r["Year"]), np.log(float(r["Value"])))
                   for r in rows
                   if r["Item"] == it and r["Value"] and int(r["Year"]) <= max_year)
        series[c] = v
    return series


def rmse(a, b):
    return float(np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2)))


def mape(a, b):
    return float(np.mean(np.abs((np.array(a) - np.array(b)) / np.array(b))) * 100)


def run(max_year, label):
    series = load_series(max_year)
    print(f"\n{'='*64}\n{label}\n{'='*64}")
    print("series lengths:", {c: len(v) for c, v in series.items()})

    train, test, stats = [], [], {}
    for ci, c in enumerate(CROPS):
        p = [x for _, x in series[c]]
        cut = len(p) - N_TEST
        mu, sd = float(np.mean(p[:cut])), float(np.std(p[:cut])) or 1.0
        stats[c] = (mu, sd)
        for i in range(len(p) - L):
            tgt_idx = i + L                       # positional, not value lookup
            seq, tgt = p[i:i + L], p[tgt_idx]
            rec = (ci, [(x - mu) / sd for x in seq], (tgt - mu) / sd, seq, tgt)
            (test if tgt_idx >= cut else train).append(rec)
    print(f"train windows {len(train)}   test windows {len(test)}")

    def to_XY(recs):
        X = np.zeros((len(recs), L, 1 + len(CROPS)))
        y = np.zeros(len(recs))
        for k, (ci, seq, tgt, _, _) in enumerate(recs):
            for t in range(L):
                X[k, t, 0] = seq[t]
                X[k, t, 1 + ci] = 1.0
            y[k] = tgt
        return X, y

    rng = np.random.default_rng(0)
    aug = list(train)
    for _ in range(5):
        for ci, seq, tgt, rs, rt in train:
            s = np.array(seq)
            j = s + rng.normal(0, 0.05 * max(np.std(s), 1e-3), L)
            aug.append((ci, list(j * rng.uniform(0.95, 1.05)), tgt, rs, rt))
    print(f"augmented train windows: {len(aug)}  (x{len(aug)/len(train):.1f})")

    Xtr, ytr = to_XY(aug)
    Xte, _ = to_XY(test)
    model = LSTM(1 + len(CROPS), n_hidden=24, seed=0).fit(
        Xtr, ytr, epochs=350, lr=0.01, batch=64)
    n_params = model.W.size + model.b.size + model.Wy.size + model.by.size
    print(f"LSTM parameters: {n_params}  vs real (un-augmented) train windows: {len(train)}")
    pred_n = model.predict(Xte)

    actual, lstm_p, naive, mean3, trend, ar1 = [], [], [], [], [], []
    for k, (ci, seq, tgt, rs, rt) in enumerate(test):
        mu, sd = stats[CROPS[ci]]
        actual.append(np.exp(rt))
        lstm_p.append(np.exp(pred_n[k] * sd + mu))
        naive.append(np.exp(rs[-1]))
        mean3.append(float(np.exp(np.mean(rs[-3:]))))
        x = np.arange(L)
        m, b0 = np.polyfit(x, rs, 1)
        trend.append(float(np.exp(m * L + b0)))
        a, bb = np.array(rs[:-1]), np.array(rs[1:])
        coef = float((a @ bb) / (a @ a)) if a @ a else 1.0
        ar1.append(float(np.exp(coef * rs[-1])))

    res = [("LSTM (pooled+aug)", lstm_p), ("Naive (persistence)", naive),
           ("Mean of last 3", mean3), ("Linear trend", trend), ("AR(1)", ar1)]
    print(f"\n{'model':<22}{'RMSE USD/t':>12}{'MAPE %':>10}")
    print("-" * 44)
    for nm, p in sorted(res, key=lambda t: rmse(t[1], actual)):
        print(f"{nm:<22}{rmse(p, actual):>12.1f}{mape(p, actual):>10.1f}")
    print("-" * 44)
    best = min(res, key=lambda t: rmse(t[1], actual))
    print(f"BEST: {best[0]}")
    print("=> LSTM beats persistence" if rmse(lstm_p, actual) < rmse(naive, actual)
          else "=> LSTM does NOT beat persistence")
    return {nm: (rmse(p, actual), mape(p, actual)) for nm, p in res}


if __name__ == "__main__":
    a = run(2024, "(a) INCLUDING 2024 -- test window contains the wartime spike")
    b = run(2023, "(b) EXCLUDING 2024 -- configuration used in the paper")
    print(f"\n{'='*64}\nSUMMARY: RMSE USD/t\n{'='*64}")
    print(f"{'model':<22}{'incl 2024':>12}{'excl 2024':>12}")
    print("-" * 46)
    for nm in a:
        print(f"{nm:<22}{a[nm][0]:>12.1f}{b[nm][0]:>12.1f}")
