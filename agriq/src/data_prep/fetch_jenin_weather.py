"""
Fetch real monthly weather for Jenin, Palestine from NASA POWER.

Standalone: needs no parcels.csv, makes ONE API call, and NEVER fabricates
data. If the API fails, this script raises and writes nothing -- an empty
output is a far better outcome than a silently synthetic one.

Usage:
    python fetch_jenin_weather.py
    python fetch_jenin_weather.py --start 2015 --end 2024

Writes: data/raw/nasa_power_jenin_monthly.csv
"""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

import pandas as pd
import requests

# Jenin city centroid. NASA POWER grid is ~0.5 deg, so every parcel in the
# governorate falls in essentially this same cell -- one call is enough.
JENIN_LAT = 32.4597
JENIN_LON = 35.2956

URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"

PARAMS = {
    "T2M": "temperature_c",
    "T2M_MAX": "temp_max_c",
    "T2M_MIN": "temp_min_c",
    "RH2M": "humidity_pct",
    # NOTE: NASA POWER's *monthly* PRECTOTCORR is an average DAILY rate
    # (mm/day), NOT the month's total. Named explicitly so it can't be
    # summed by mistake; rainfall_mm_month is derived below.
    "PRECTOTCORR": "precip_mm_per_day",
    "ALLSKY_SFC_SW_DWN": "solar_mj_m2_day",
}

FILL = -999.0  # NASA POWER's missing-data sentinel


def fetch(lat: float, lon: float, start: int, end: int) -> pd.DataFrame:
    """One request for the whole period. Raises on any failure."""
    resp = requests.get(
        URL,
        params={
            "parameters": ",".join(PARAMS),
            "community": "AG",
            "latitude": lat,
            "longitude": lon,
            "start": str(start),
            "end": str(end),
            "format": "JSON",
        },
        timeout=120,
    )
    resp.raise_for_status()

    payload = resp.json()
    if "properties" not in payload:
        raise RuntimeError(f"Unexpected NASA POWER response: {payload}")
    block = payload["properties"]["parameter"]

    # Keys look like "202001".."202013"; the 13 suffix is the annual mean.
    records: dict[str, dict[str, float]] = {}
    for api_name, col in PARAMS.items():
        if api_name not in block:
            raise RuntimeError(f"NASA POWER did not return {api_name}")
        for key, value in block[api_name].items():
            if key.endswith("13"):  # annual aggregate, not a month
                continue
            records.setdefault(key, {})[col] = value

    rows = []
    for key in sorted(records):
        row = {"year": int(key[:4]), "month": int(key[4:]), **records[key]}
        rows.append(row)

    df = pd.DataFrame(rows)

    # Surface missing data loudly instead of letting -999 propagate into means.
    for col in PARAMS.values():
        n_missing = int((df[col] == FILL).sum())
        if n_missing:
            print(f"  WARNING: {col} has {n_missing} missing months (-999)")
            df.loc[df[col] == FILL, col] = pd.NA

    # Convert the daily precipitation rate to an actual monthly total.
    df["days_in_month"] = [
        calendar.monthrange(int(y), int(m))[1]
        for y, m in zip(df["year"], df["month"])
    ]
    df["rainfall_mm_month"] = df["precip_mm_per_day"] * df["days_in_month"]

    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--lat", type=float, default=JENIN_LAT)
    ap.add_argument("--lon", type=float, default=JENIN_LON)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "nasa_power_jenin_monthly.csv",
    )
    args = ap.parse_args()

    print(f"Fetching NASA POWER for Jenin ({args.lat}, {args.lon}) "
          f"{args.start}-{args.end} ...")
    try:
        df = fetch(args.lat, args.lon, args.start, args.end)
    except Exception as exc:  # noqa: BLE001 - we want the real reason surfaced
        print(f"\nFAILED: {exc}", file=sys.stderr)
        print("No file written. Nothing was fabricated.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nWrote {len(df)} monthly records to {args.output}")
    annual_rain = df.groupby("year")["rainfall_mm_month"].sum()
    wet = df[df["month"].isin([11, 12, 1, 2, 3])]["rainfall_mm_month"].sum()

    print("\nJenin climate normals from this pull:")
    print(f"  mean annual temperature : {df['temperature_c'].mean():.1f} C")
    print(f"  mean annual rainfall    : {annual_rain.mean():.0f} mm/yr")
    print(f"  driest / wettest year   : {annual_rain.min():.0f} / {annual_rain.max():.0f} mm")
    print(f"  Nov-Mar share of rain   : {100 * wet / df['rainfall_mm_month'].sum():.0f} %")
    print(f"  mean relative humidity  : {df['humidity_pct'].mean():.1f} %")

    ok = 15.0 <= df["temperature_c"].mean() <= 23.0 and 250 <= annual_rain.mean() <= 700
    print("\nSanity check: Jenin should be ~18-21 C, 350-550 mm/yr, "
          "with almost all rain Nov-Mar.")
    print("  => PASS" if ok else "  => FAIL - do not use; tell Claude.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
