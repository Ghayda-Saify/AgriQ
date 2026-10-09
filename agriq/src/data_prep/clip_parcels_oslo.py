import shapefile, csv, collections
from pyproj import Geod
geod = Geod(ellps="WGS84")

# ---- Oslo zones ---------------------------------------------------------
osl = shapefile.Reader("/sessions/brave-peaceful-planck/mnt/osloagreement/OsloAgreement")
zones = []
for i, rec in enumerate(osl.iterRecords()):
    s = osl.shape(i)
    parts = list(s.parts) + [len(s.points)]
    rings = [s.points[a:b] for a, b in zip(parts[:-1], parts[1:])]
    zones.append((rec["CLASS"], s.bbox, rings))

def in_ring(pt, ring):
    x, y = pt; inside = False; n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]; x2, y2 = ring[(i+1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2-x1)*(y-y1)/(y2-y1)+x1
            if x < xin: inside = not inside
    return inside

def classify(pt):
    x, y = pt
    for cls, bbox, rings in zones:
        if not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]): continue
        if sum(1 for r in rings if in_ring(pt, r)) % 2 == 1:
            return cls
    return None            # outside the West Bank entirely (i.e. Israel)

# ---- OSM farmland in the Jenin window -----------------------------------
LAT0,LAT1,LON0,LON1 = 32.28, 32.58, 35.03, 35.47
KEEP = {"farmland","orchard","vineyard","greenhouse_horticulture"}
r = shapefile.Reader("osm/gis_osm_landuse_a_free_1")
recs = list(r.iterRecords())
rows = []
for i, rec in enumerate(recs):
    if rec["fclass"] not in KEEP: continue
    try: s = r.shape(i)
    except Exception: continue
    if not s.points: continue
    x0,y0,x1,y1 = s.bbox
    cx, cy = (x0+x1)/2, (y0+y1)/2
    if not (LON0 <= cx <= LON1 and LAT0 <= cy <= LAT1): continue
    parts = list(s.parts)+[len(s.points)]
    area = 0.0
    for a,b in zip(parts[:-1], parts[1:]):
        ring = s.points[a:b]
        if len(ring) < 4: continue
        ar,_ = geod.polygon_area_perimeter([p[0] for p in ring],[p[1] for p in ring])
        area += ar
    ha = abs(area)/1e4
    if ha <= 0: continue
    rows.append({"osm_id":rec["osm_id"],"fclass":rec["fclass"],
                 "lat":round(cy,6),"lon":round(cx,6),"area_ha":round(ha,4),
                 "zone":classify((cx,cy)) or "OUTSIDE_WB"})

with open("jenin_parcels_zoned.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print(f"OSM agricultural polygons in Jenin window: {len(rows)}")
print(f"{'zone':<34} {'n':>5} {'area ha':>12}")
c=collections.Counter(d["zone"] for d in rows)
for z,n in c.most_common():
    a=sum(d["area_ha"] for d in rows if d["zone"]==z)
    print(f"  {z:<32} {n:>5} {a:>12,.0f}")

ctrl=[d for d in rows if d["zone"]=="A" and 0.1<=d["area_ha"]<=100]
print(f"\nCONTROLLABLE (Area A/B, 0.1-100 ha): {len(ctrl)} parcels, "
      f"{sum(d['area_ha'] for d in ctrl):,.0f} ha")
areac=[d for d in rows if d["zone"]=="C" and 0.1<=d["area_ha"]<=100]
print(f"Other administrative class         : {len(areac)} parcels, "
      f"{sum(d['area_ha'] for d in areac):,.0f} ha")
