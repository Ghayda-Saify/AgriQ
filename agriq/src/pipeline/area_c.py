"""Area C counterfactual: the agricultural cost of the Oslo land-access regime.

Compares the 174 parcels Palestinian farmers control (Areas A/B) with all 284
parcels including Area C. For the comparison to isolate land access, both
regimes must face:
  * the SAME market -- demand is defined from Areas A/B in both, because
    adding land adds supply, not buyers (an earlier version let demand grow
    with land area, which forced the loss to track land share);
  * the SAME water budget (6.0 Mm3/yr) -- an unconstrained full-land plan
    used more water than Jenin has.
Profits are read from water_sweep.csv, so run water_price.py first.
Soil is compared with SoilGrids estimates (model predictions, not field
measurements); modelled available-N is deliberately excluded because it
depends on an assumed mineralisation fraction.
"""
import csv, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "src", "optimizer"))
from instance import load_parcels
from water_price import crossing, BUDGET

rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "processed", "water_sweep.csv"))))
sweep = {z: [{k: (float(v) if k != "zone" else v) for k, v in r.items()} for r in rows if r["zone"] == z]
         for z in ("A", "AC")}

res = []
for z, label in (("A", "Areas A/B"), ("AC", "A/B + Area C")):
    parcels = load_parcels(z)
    area = sum(float(p["area_ha"]) for p in parcels)
    unc = min(sweep[z], key=lambda r: r["lam"])                 # lambda = 0
    lam_star, prof, binds = crossing(sweep[z])
    res.append({"regime": label, "parcels": len(parcels), "area_ha": area,
                "profit_unconstrained": unc["profit_usd"], "water_unconstrained": unc["water_m3"],
                "shadow_price": lam_star, "profit_budget": prof, "binds": binds})
    print(f"{label:<14} {len(parcels):>4} parcels {area:>7,.0f} ha | "
          f"lambda=0: ${unc['profit_usd']/1e6:.2f}M using {unc['water_m3']/1e6:.2f} Mm3 | "
          f"within {BUDGET/1e6:.1f} Mm3: ${prof/1e6:.2f}M "
          f"({'binding, $%.2f/m3' % lam_star if binds else 'not binding'})")

a, c = res
gap_u = c["profit_unconstrained"] - a["profit_unconstrained"]
gap_b = c["profit_budget"] - a["profit_budget"]
land = c["area_ha"] - a["area_ha"]
print(f"\nforgone profit, within the water budget : ${gap_b/1e6:.2f}M/yr "
      f"({gap_b/c['profit_budget']*100:.1f}% of full-access profit)")
print(f"forgone profit, ignoring the water budget: ${gap_u/1e6:.2f}M/yr (infeasible reference)")
print(f"land withheld: {land:,.0f} ha ({land/c['area_ha']*100:.1f}% of the study parcels)")

soil = {r["osm_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data", "processed", "soilgrids_jenin.csv")))
        if r["ph"]}
print("\nSoilGrids estimates, mean over parcels with data (A/B vs C):")
for f, unit in (("ph", ""), ("soc_g_kg", " g/kg"), ("cec_cmol", " cmol/kg"), ("clay_pct", " %"), ("sand_pct", " %")):
    va = [float(soil[p["osm_id"]][f]) for p in load_parcels("A") if p["osm_id"] in soil]
    vc = [float(soil[p["osm_id"]][f]) for p in load_parcels("C") if p["osm_id"] in soil]
    print(f"  {f:<9} A/B {np.mean(va):6.2f}{unit} (n={len(va)})   C {np.mean(vc):6.2f}{unit} (n={len(vc)})")

with open(os.path.join(ROOT, "data", "processed", "area_c.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
