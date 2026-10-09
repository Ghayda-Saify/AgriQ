"""
Fetch real per-parcel soil properties from ISRIC SoilGrids 250m.

Queries the SoilGrids REST API at each Jenin parcel centroid and writes
pH, total nitrogen, soil organic carbon, CEC, clay and sand.

Why this matters: without it, every parcel shares one NASA POWER climate cell,
so the Random Forest gives all 174 parcels an IDENTICAL suitability score and
cannot tell them apart. Per-parcel pH and nitrogen give the model real spatial
variation to work with.

NOT available anywhere globally: phosphorus (P) and potassium (K). Those stay
as documented assumptions -- SoilGrids does not publish them.

The script is resumable (re-run it if it stops) and NEVER fabricates a value:
failed points are recorded as blank, not filled in.

Usage:
    python fetch_soilgrids.py                 # zones A and C (284 parcels, ~6-10 min)
    python fetch_soilgrids.py --zones A       # controllable parcels only
    python fetch_soilgrids.py --limit 5       # quick test first

Citation: Poggio, L., et al. (2021). SoilGrids 2.0: producing soil information
for the globe with quantified spatial uncertainty. SOIL, 7, 217-240.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# property -> (human name, fallback divisor to convert mapped units -> usable)
PROPS = {
    "phh2o":    ("ph",         10.0),   # pH*10      -> pH
    "nitrogen": ("n_g_kg",    100.0),   # cg/kg      -> g/kg
    "soc":      ("soc_g_kg",   10.0),   # dg/kg      -> g/kg
    "cec":      ("cec_cmol",   10.0),   # mmol(c)/kg -> cmol(c)/kg
    "clay":     ("clay_pct",   10.0),   # g/kg       -> %
    "sand":     ("sand_pct",   10.0),
}
DEPTHS = ["0-5cm", "5-15cm"]

SANE = {"ph": (3.0, 10.0), "n_g_kg": (0.0, 20.0), "soc_g_kg": (0.0, 200.0),
        "cec_cmol": (0.0, 100.0), "clay_pct": (0.0, 100.0), "sand_pct": (0.0, 100.0)}


def query_point(lat: float, lon: float, session: requests.Session,
                retries: int = 4) -> dict[str, float]:
    params = [("lon", lon), ("lat", lat), ("value", "mean")]
    params += [("property", p) for p in PROPS]
    params += [("depth", d) for d in DEPTHS]

    last = None
    for attempt in range(retries):
        try:
            r = session.get(URL, params=params, timeout=90)
            if r.status_code == 429:            # rate limited
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            payload = r.json()
            out: dict[str, float] = {}
            for layer in payload["properties"]["layers"]:
                prop = layer["name"]
                if prop not in PROPS:
                    continue
                label, fallback = PROPS[prop]
                dfac = layer.get("unit_measure", {}).get("d_factor") or fallback
                vals = []
                for d in layer["depths"]:
                    v = d.get("values", {}).get("mean")
                    if v is not None:
                        vals.append(v / dfac)
                if vals:
                    out[label] = round(sum(vals) / len(vals), 4)
            return out
        except Exception as exc:                # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"SoilGrids failed at ({lat},{lon}): {last}")


def main() -> int:
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parcels", type=Path,
                    default=here / "data" / "processed" / "jenin_parcels_zoned.csv")
    ap.add_argument("--output", type=Path,
                    default=here / "data" / "processed" / "soilgrids_jenin.csv")
    ap.add_argument("--zones", default="A,C",
                    help="administrative classes to fetch (default A,C)")
    ap.add_argument("--min-ha", type=float, default=0.1)
    ap.add_argument("--max-ha", type=float, default=100.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--sleep", type=float, default=1.2,
                    help="seconds between requests (be polite to ISRIC)")
    args = ap.parse_args()

    zones = {z.strip() for z in args.zones.split(",")}
    rows = [r for r in csv.DictReader(open(args.parcels))
            if r["zone"] in zones
            and args.min_ha <= float(r["area_ha"]) <= args.max_ha]
    if args.limit:
        rows = rows[:args.limit]

    done: dict[str, dict] = {}
    if args.output.exists():
        for r in csv.DictReader(open(args.output)):
            if r.get("ph"):
                done[r["osm_id"]] = r
        print(f"resuming: {len(done)} parcels already fetched")

    todo = [r for r in rows if r["osm_id"] not in done]
    print(f"{len(rows)} parcels in scope, {len(todo)} to fetch "
          f"(~{len(todo)*args.sleep/60:.1f} min)")
    if not todo:
        print("nothing to do")
        return 0

    fields = ["osm_id", "zone", "fclass", "lat", "lon", "area_ha"] + \
             [n for n, _ in PROPS.values()]
    results = list(done.values())
    session = requests.Session()
    session.headers["User-Agent"] = "AgriQ-research/1.0"
    failed = 0

    for i, r in enumerate(todo, 1):
        try:
            soil = query_point(float(r["lat"]), float(r["lon"]), session)
        except Exception as exc:                # noqa: BLE001
            print(f"  [{i}/{len(todo)}] FAILED {r['osm_id']}: {exc}")
            soil = {}
            failed += 1
        rec = {k: r.get(k, "") for k in ("osm_id", "zone", "fclass",
                                         "lat", "lon", "area_ha")}
        rec.update({n: soil.get(n, "") for n, _ in PROPS.values()})
        results.append(rec)

        if i % 10 == 0 or i == len(todo):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader(); w.writerows(results)
            print(f"  [{i}/{len(todo)}] saved  (last pH={soil.get('ph','--')})")
        time.sleep(args.sleep)

    ok = [r for r in results if r.get("ph") not in ("", None)]
    print(f"\nWrote {len(results)} rows to {args.output}")
    print(f"  succeeded: {len(ok)}   failed: {len(results)-len(ok)}")
    if not ok:
        print("NO DATA RETRIEVED. Nothing was fabricated.", file=sys.stderr)
        return 1

    print("\nJenin soil summary (real, SoilGrids 250m):")
    bad = 0
    for name, _ in PROPS.values():
        vals = [float(r[name]) for r in ok if r.get(name) not in ("", None)]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        mean = sum(vals) / len(vals)
        flag = ""
        if name in SANE and not (SANE[name][0] <= mean <= SANE[name][1]):
            flag = "  <-- OUT OF RANGE"; bad += 1
        print(f"  {name:10s} mean {mean:8.2f}   range {lo:7.2f} - {hi:7.2f}{flag}")
    print("\nSanity: Jenin soils are calcareous, pH should be ~7.0-8.4.")
    print("  => PASS" if bad == 0 else "  => CHECK FAILED - tell Claude before using")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
