"""Assemble the real Jenin optimisation instance from verified sources."""
import csv, os, numpy as np, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, os.path.join(ROOT, "src", "optimizer"))
from qubo import CropQUBO3

import glob
# FAOSTAT QCL yields (the 8-11 file); PP prices are the 8-15 file
_c=[f for f in glob.glob(os.path.join(ROOT,"data","raw","FAOSTAT*.csv"))]
def _one(tag, what):
    m=[f for f in _c if tag in f]
    if len(m)!=1:
        raise RuntimeError(f"expected exactly one FAOSTAT {what} file matching '{tag}' in "
                           f"data/raw, found {len(m)} -- refusing to guess")
    return m[0]
FAO=_one("8-11","QCL yield")       # an earlier silent fallback here read PRICES as yields
FAO_PP=_one("8-15","producer-price")

CROPS=["fallow","wheat","potatoes","tomatoes","cucumbers","onions",
       "eggplants","maize","olives","grapes","oranges"]

FAO_ITEM={"wheat":"Wheat","potatoes":"Potatoes","tomatoes":"Tomatoes",
 "cucumbers":"Cucumbers and gherkins","onions":"Onions and shallots, dry (excluding dehydrated)",
 "eggplants":"Eggplants (aubergines)","maize":"Green corn (maize)","olives":"Olives",
 "grapes":"Grapes","oranges":"Oranges"}

# FAOSTAT Producer Price (USD/tonne), Palestine, mean of 2019-2023.
# 2024 is EXCLUDED: perishable prices spike 1.24x-5.76x vs 2023 (wartime
# supply disruption; storables like wheat/olives stay flat) -- see
# HANDOFF_TO_CLAUDE_CODE.md section 3. All 10 crops have real FAOSTAT PP
# data for every year 2019-2023, so no ASSUMPTION values are needed here
# (the earlier PCBS-based version had to guess olives/maize; FAOSTAT covers
# them both with real reported prices).
def fao_prices():
    rows=list(csv.DictReader(open(FAO_PP,encoding="utf-8-sig")))
    out={}
    for c,item in FAO_ITEM.items():
        v=[float(r["Value"]) for r in rows
           if r["Item"]==item and r["Element"]=="Producer Price (USD/tonne)"
           and r["Value"] and 2019<=int(r["Year"])<=2023]
        if len(v)!=5:
            raise RuntimeError(f"expected 5 years (2019-2023) of FAOSTAT PP for {c}, got {len(v)}")
        out[c]=sum(v)/len(v)
    out["fallow"]=0.0
    return out

# Net seasonal irrigation requirement, m3/ha (FAO-56 / CROPWAT, Mediterranean,
# net of ~465 mm effective rainfall measured for Jenin from NASA POWER)
WATER={"fallow":0,"wheat":500,"potatoes":4000,"tomatoes":6000,"cucumbers":4500,
 "onions":3500,"eggplants":6000,"maize":5000,"olives":1500,"grapes":3000,"oranges":7000}
# Production cost, USD/ha. ASSUMPTION: share of gross revenue typical for the
# system type (field crops .65, vegetables .55, tree crops .40). Flagged.
COST_SHARE={"fallow":0,"wheat":.65,"potatoes":.55,"tomatoes":.55,"cucumbers":.55,
 "onions":.55,"eggplants":.55,"maize":.65,"olives":.40,"grapes":.40,"oranges":.40}

# --- Reconciliation of FAOSTAT national yields to open-field Jenin levels ---
# FAOSTAT national averages include protected cultivation (greenhouses), which
# is a small share of AREA but a large share of PRODUCTION -- e.g. tomatoes at
# 136 t/ha. Our parcels are open-field OSM polygons, so vegetable yields are
# rescaled so their mean matches the PCBS open-irrigated vegetable yield for the
# North West Bank (19.65 t/ha; PCBS 2010/11 Table 17E), and tree yields so olives
# match the PCBS regional level (1.43 t/ha; Table 14E). FAOSTAT therefore sets the
# RELATIVE differences between crops; PCBS sets the ABSOLUTE level.
VEG={"potatoes","tomatoes","cucumbers","onions","eggplants","maize"}
TREE_SET={"olives","grapes","oranges"}
PCBS_VEG_OPEN_IRRIG=19.65
PCBS_OLIVE=1.43

def fao_yields():
    rows=list(csv.DictReader(open(FAO,encoding="utf-8-sig")))
    bad={(r["Element"],r["Unit"]) for r in rows} - {("Yield","kg/ha")}
    if bad:
        raise RuntimeError(f"{FAO} is not a kg/ha yield file (found {bad})")
    out={}
    for c,item in FAO_ITEM.items():
        v=[float(r["Value"]) for r in rows
           if r["Item"]==item and r["Value"] and int(r["Year"])>=2014]
        if not v:
            raise RuntimeError(f"no FAOSTAT yield records for {c}")
        out[c]=sum(v)/len(v)/1000.0   # kg/ha -> t/ha
    out["fallow"]=0.0
    vs=[out[c] for c in VEG]
    if vs: 
        f=PCBS_VEG_OPEN_IRRIG/(sum(vs)/len(vs))
        for c in VEG: out[c]*=f
    if out.get("olives",0)>0:
        f=PCBS_OLIVE/out["olives"]
        for c in TREE_SET: out[c]*=f
    return out

def load_parcels(zone="A", lo=0.1, hi=100.0):
    """`zone` may be a single administrative class ("A") or several ("AC") for a
    counterfactual, which optimises A/B and C together."""
    want=set(zone)
    rows=[r for r in csv.DictReader(open(os.path.join(ROOT,"data","processed","jenin_parcels_zoned.csv")))
          if r["zone"] in want and lo<=float(r["area_ha"])<=hi]
    return rows


# Bottom SUIT_PCTL of each crop's OWN suitability distribution is treated as
# unsuitable. A per-crop relative threshold rather than a global absolute one:
# RF confidence differs systematically BETWEEN crops (Jenin's climate/soil
# gives citrus near-zero probability everywhere), so a global cut would ban
# whole crops rather than filter poor parcel matches, which is what this
# constraint is for. Parcel-level discrimination comes from OSM land use.
SUIT_PCTL=20.0

def load_suitability(parcels):
    """RF suitability from build_suitability.py, aligned to `parcels` order."""
    path=os.path.join(ROOT,"data","processed","suitability_matrix.csv")
    if not os.path.exists(path):
        raise RuntimeError(f"{path} missing -- run src/models/build_suitability.py first")
    by_id={r["osm_id"]:r for r in csv.DictReader(open(path))}
    missing=[p["osm_id"] for p in parcels if p["osm_id"] not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} parcels have no suitability row "
                           f"(e.g. {missing[:3]}) -- rebuild suitability_matrix.csv")
    suit=np.zeros((len(parcels),len(CROPS)))
    suit[:,0]=1.0                      # fallow always permitted
    for i,p in enumerate(parcels):
        row=by_id[p["osm_id"]]
        for j,c in enumerate(CROPS):
            if c=="fallow": continue
            suit[i,j]=float(row[f"suit_{c}"])
    return suit

DEMAND_SHARE=0.60

def build(zone="A", lam=0.15, M=0.5, prices=None, M_onehot=None, demand_zone="A", market_scale=1.0):
    """Build the QUBO instance for one land regime.

    `prices`      overrides revenue prices (crop -> USD/t) for the robustness
                  analysis. Production COST stays calibrated at baseline
                  FAOSTAT prices: real input costs do not move with output
                  price, and letting cost float would cancel the shock.
    `M`           sizes the penalty terms. `M_onehot` overrides the
                  one-crop-per-parcel penalty ALONE -- domain-wall encoding
                  needs it at 0 while keeping the suitability penalty.
    `demand_zone` parcels that define market size AND the suitability
                  reference distribution, so that a larger counterfactual
                  instance faces the same market and the same thresholds.
    `market_scale` multiplies market size, for sensitivity analysis only."""
    Y=fao_yields()
    BASE=fao_prices()
    PRICE=dict(BASE) if prices is None else {**BASE, **prices}
    P=load_parcels(zone)
    area=np.array([float(r["area_ha"]) for r in P])
    C=len(CROPS)
    yld=np.zeros((len(P),C))
    for i in range(len(P)):
        for j,c in enumerate(CROPS):
            if c=="fallow": continue
            yld[i,j]=Y[c]
    # RF suitability (already includes the OSM land-use factor). Converted to
    # a within-crop percentile rank so a single scalar suit_min expresses the
    # per-crop relative threshold -- raw RF confidence is not comparable
    # across crops, ranks are. Ordering within each crop is preserved.
    suit_raw=load_suitability(P)
    ref=P if demand_zone is None else load_parcels(demand_zone)
    ref_raw=load_suitability(ref)
    suit=np.zeros_like(suit_raw)
    suit[:,0]=100.0                    # fallow always permitted
    for j in range(1,C):
        col=suit_raw[:,j]
        refcol=np.sort(ref_raw[:,j])   # rank against the REFERENCE parcels
        suit[:,j]=np.searchsorted(refcol,col,side="right")/len(refcol)*100.0
        # A rank is only meaningful relative to a non-degenerate distribution.
        # Where the RF gives a crop exactly zero probability (no tree in the
        # ensemble votes for it -- oranges, under Jenin's alkaline pH), rank
        # would call every parcel joint-best and mark the crop suitable
        # everywhere. Zero probability is unsuitable, whatever its rank.
        suit[col<=0.0,j]=0.0
    price=np.array([PRICE[c] for c in CROPS],float)
    water=np.array([WATER[c] for c in CROPS],float)
    cost=np.array([COST_SHARE[c]*BASE[c]*Y[c] for c in CROPS],float)   # baseline-calibrated
    # Market-size proxy (ASSUMPTION, flagged in paper): no crop-level demand
    # data exists for Jenin, so D_j is set to DEMAND_SHARE of the production
    # capacity of the Area A/B instance. It is computed from A/B even when
    # optimising a larger instance: adding land adds supply, not
    # buyers, so the counterfactual must face the SAME market. (Computing D_j
    # from each instance's own capacity made demand grow in proportion to
    # land and made any land-base comparison track area share by construction.)
    ref_area=np.array([float(r["area_ha"]) for r in ref])
    tot=ref_area.sum()*np.array([Y[c] for c in CROPS])
    demand=np.where(tot>0, tot*DEMAND_SHARE*market_scale, 1e12)
    demand[0]=1e12
    q=CropQUBO3(area,yld,price,cost,suit,water,demand,lam=lam,
                suit_min=SUIT_PCTL,
                M_onehot=M if M_onehot is None else M_onehot, M_suit=M)
    q.crops=CROPS; q.parcels=P; q.suit_raw=suit_raw
    return q

if __name__=="__main__":
    Y=fao_yields(); PRICE=fao_prices()
    # superseded PCBS 2023 single-year prices, shown for comparison only
    OLD={"wheat":527.1,"potatoes":586.5,"tomatoes":556.6,"cucumbers":654.4,
         "onions":548.5,"eggplants":344.8,"maize":300.0,"olives":1100.0,
         "grapes":719.5,"oranges":610.9}
    print("FAOSTAT yields (t/ha, mean 2014+) and prices (USD/t, mean 2019-2023, 2024 excluded):")
    print(f"   {'crop':<11} {'yield':>7} {'price':>9} {'was':>9}  {'water':>6}")
    for c in CROPS[1:]:
        print(f"   {c:<11} {Y[c]:7.2f} {PRICE[c]:9.1f} {OLD[c]:9.1f}  {WATER[c]:6d}")
    print("   (olives/maize previously ASSUMPTION values; FAOSTAT now covers both)")

    for z in ("A","C"):
        q=build(z)
        print(f"\nzone {z}: {q.P} parcels, {q.area.sum():,.0f} ha, "
              f"{q.n} binary variables")
        print(f"   land-use mix: {dict(collections.Counter(r['fclass'] for r in q.parcels))}")
        print(f"   RF suitability (raw) and parcels flagged unsuitable "
              f"at bottom {SUIT_PCTL:.0f}% within each crop:")
        for j,c in enumerate(CROPS[1:],1):
            raw=q.suit_raw[:,j]
            n_uns=int((q.suit[:,j]<q.suit_min).sum())
            print(f"      {c:<11} raw min/mean/max {raw.min():.4f}/{raw.mean():.4f}/"
                  f"{raw.max():.4f}   unsuitable {n_uns:3d}/{q.P}")
