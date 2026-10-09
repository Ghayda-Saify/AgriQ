"""Produce 2025 price forecasts under BOTH models (LSTM and persistence)
so the allocation can be run under each and compared for robustness."""
import csv, numpy as np
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
CROPS=list(ITEM); L=5

rows=list(csv.DictReader(open(PP,encoding="utf-8-sig")))
series={c:sorted((int(r["Year"]),float(r["Value"])) for r in rows
                 if r["Item"]==it and r["Value"]) for c,it in ITEM.items()}

# train on ALL windows, log scale
train=[]; stats={}
for ci,c in enumerate(CROPS):
    p=[np.log(x) for _,x in series[c]]
    mu,sd=float(np.mean(p)),float(np.std(p)) or 1.0
    stats[c]=(mu,sd)
    for i in range(len(p)-L):
        train.append((ci,[(x-mu)/sd for x in p[i:i+L]],(p[i+L]-mu)/sd))

rng=np.random.default_rng(0); aug=list(train)
for _ in range(5):
    for ci,seq,tgt in train:
        s=np.array(seq); j=s+rng.normal(0,0.05*max(np.std(s),1e-3),L)
        aug.append((ci,list(j*rng.uniform(0.95,1.05)),tgt))

def toX(recs):
    X=np.zeros((len(recs),L,1+len(CROPS))); y=np.zeros(len(recs))
    for k,(ci,seq,tgt) in enumerate(recs):
        for t in range(L): X[k,t,0]=seq[t]; X[k,t,1+ci]=1.0
        y[k]=tgt
    return X,y

Xtr,ytr=toX(aug)
model=LSTM(1+len(CROPS),n_hidden=24,seed=0).fit(Xtr,ytr,epochs=350,lr=0.01,batch=64)

out=[]
print(f"{'crop':<11}{'last yr':>8}{'persistence':>13}{'LSTM 2025':>11}{'diff %':>9}")
print("-"*54)
for ci,c in enumerate(CROPS):
    p=[np.log(x) for _,x in series[c]]
    mu,sd=stats[c]
    X=np.zeros((1,L,1+len(CROPS)))
    for t in range(L):
        X[0,t,0]=(p[-L+t]-mu)/sd; X[0,t,1+ci]=1.0
    lstm_v=float(np.exp(model.predict(X)[0]*sd+mu))
    persist=float(np.exp(p[-1]))
    yr=series[c][-1][0]
    out.append({"crop":c,"last_year":yr,"price_persistence_usd_t":round(persist,1),
                "price_lstm_usd_t":round(lstm_v,1)})
    print(f"{c:<11}{yr:>8}{persist:>13.0f}{lstm_v:>11.0f}{(lstm_v/persist-1)*100:>8.1f}%")

with open(f"{D}/data/processed/price_forecast_2025.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(f"\nwrote {D}/data/processed/price_forecast_2025.csv")
