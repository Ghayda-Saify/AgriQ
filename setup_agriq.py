#!/usr/bin/env python3
"""
setup_agriq.py
Creates the full AgriQ project structure and writes all three starter scripts.
Run this ONE file and you're done.
"""
import os
from pathlib import Path

# =============================================================================
# CONFIG: Where do you want the project folder?
# =============================================================================
PROJECT_ROOT = Path.home() / "agriq"  # Saves to ~/agriq (or C:\Users\You\agriq on Windows)
# If you want it somewhere else, change the line above. Examples:
# PROJECT_ROOT = Path("C:/Users/YourName/Desktop/agriq")   # Windows
# PROJECT_ROOT = Path("/Users/yourname/Documents/agriq")   # Mac
# PROJECT_ROOT = Path("/home/yourname/agriq")                # Linux

# =============================================================================
# 1. CREATE FOLDERS
# =============================================================================
folders = ["data/raw", "data/processed", "models", "outputs", "src"]
for f in folders:
    (PROJECT_ROOT / f).mkdir(parents=True, exist_ok=True)
print(f"✅ Created project at: {PROJECT_ROOT.absolute()}")

# =============================================================================
# 2. WRITE fetch_nasa_power.py
# =============================================================================
fetch_nasa = '''#!/usr/bin/env python3
import requests
import pandas as pd
import time
from pathlib import Path

LAT, LON = 32.4595, 35.3009
START_DATE = "20150101"
END_DATE = "20241231"
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMETERS = [
    "T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR",
    "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M"
]
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

def fetch_chunk(start, end):
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": LON,
        "latitude": LAT,
        "start": start,
        "end": end,
        "format": "JSON",
        "header": "true",
    }
    resp = requests.get(URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()

def main():
    print(f"Fetching NASA POWER for Jenin ({LAT}, {LON})...")
    years = list(range(2015, 2025))
    all_records = []
    for year in years:
        start = f"{year}0101"
        end = f"{year}1231"
        print(f"  -> {year}...")
        try:
            data = fetch_chunk(start, end)
            param_data = data["properties"]["parameter"]
            dates = param_data[PARAMETERS[0]]
            for date_str in dates:
                record = {"date": pd.to_datetime(date_str, format="%Y%m%d")}
                for p in PARAMETERS:
                    val = param_data[p].get(date_str)
                    record[p] = None if val == -999.0 else val
                all_records.append(record)
            time.sleep(0.5)
        except Exception as e:
            print(f"  !! Failed {year}: {e}")
    
    df = pd.DataFrame(all_records)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    
    out = OUTPUT_DIR / "nasa_power_jenin.csv"
    df.to_csv(out, index=False)
    print(f"\\n✅ Saved {len(df)} rows to {out}")
    print(df.head())

if __name__ == "__main__":
    main()
'''

# =============================================================================
# 3. WRITE generate_parcels.py
# =============================================================================
generate_parcels = '''#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N_PARCELS = 500
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAT_MIN, LAT_MAX = 32.20, 32.60
LON_MIN, LON_MAX = 35.10, 35.50
AREA_ZONE_WEIGHTS = {"A": 0.15, "B": 0.25, "C": 0.60}
IRRIGATION_SOURCES = ["rainfed", "spring", "well", "network"]
IRRIGATION_WEIGHTS = [0.30, 0.25, 0.30, 0.15]

def generate_soil_properties(n):
    n_total = np.clip(np.random.normal(0.12, 0.06, n), 0.02, 0.35)
    p_olsen = np.clip(np.random.normal(18, 12, n), 3, 60)
    k = np.clip(np.random.normal(220, 100, n), 50, 550)
    ph = np.clip(np.random.normal(7.9, 0.5, n), 7.0, 9.0)
    ec = np.clip(np.random.exponential(0.6, n), 0.1, 4.0)
    om = np.clip(np.random.lognormal(-0.5, 0.6, n), 0.3, 5.0)
    slope = np.clip(np.random.exponential(6, n), 0, 35)
    area = np.clip(np.random.lognormal(1.2, 0.8, n), 0.5, 80)
    return pd.DataFrame({
        "N": np.round(n_total, 3),
        "P": np.round(p_olsen, 1),
        "K": np.round(k, 1),
        "pH": np.round(ph, 2),
        "EC": np.round(ec, 2),
        "OM": np.round(om, 2),
        "slope": np.round(slope, 1),
        "area_dunam": np.round(area, 2),
    })

def main():
    print(f"Generating {N_PARCELS} parcels for Jenin...")
    df = generate_soil_properties(N_PARCELS)
    df["lat"] = np.round(np.random.uniform(LAT_MIN, LAT_MAX, N_PARCELS), 6)
    df["lon"] = np.round(np.random.uniform(LON_MIN, LON_MAX, N_PARCELS), 6)
    zones = np.random.choice(list(AREA_ZONE_WEIGHTS.keys()), size=N_PARCELS, p=list(AREA_ZONE_WEIGHTS.values()))
    df["area_zone"] = zones
    irrigation = np.random.choice(IRRIGATION_SOURCES, size=N_PARCELS, p=IRRIGATION_WEIGHTS)
    df["irrigation_source"] = irrigation
    df["has_drip"] = df["irrigation_source"].apply(lambda x: np.random.random() < 0.7 if x != "rainfed" else False)
    df.insert(0, "parcel_id", [f"JEN_{i:04d}" for i in range(N_PARCELS)])
    
    out = OUTPUT_DIR / "parcels_500.csv"
    df.to_csv(out, index=False)
    print(f"\\n✅ Saved to {out}")
    print(f"Area zones:\\n{df['area_zone'].value_counts()}")
    print(f"Irrigation:\\n{df['irrigation_source'].value_counts()}")

if __name__ == "__main__":
    main()
'''

# =============================================================================
# 4. WRITE fix_arij_yields.py
# =============================================================================
fix_arij = '''#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CROPS = [
    "wheat", "potatoes", "grapes", "oranges", "maize",
    "tomatoes", "cucumbers", "onions", "eggplants", "olives"
]

YIELD_TON_HA = {
    "wheat": 2.2, "potatoes": 22.0, "grapes": 14.0, "oranges": 18.0, "maize": 4.5,
    "tomatoes": 28.0, "cucumbers": 24.0, "onions": 18.0, "eggplants": 22.0, "olives": 2.8,
}
COST_USD_HA = {
    "wheat": 850, "potatoes": 3200, "grapes": 2600, "oranges": 2200, "maize": 650,
    "tomatoes": 3800, "cucumbers": 3000, "onions": 1600, "eggplants": 2400, "olives": 1100,
}
PRICE_USD_TON = {
    "wheat": 380, "potatoes": 420, "grapes": 850, "oranges": 520, "maize": 320,
    "tomatoes": 650, "cucumbers": 550, "onions": 450, "eggplants": 600, "olives": 2200,
}
WATER_MM = {
    "wheat": 350, "potatoes": 550, "grapes": 600, "oranges": 900, "maize": 500,
    "tomatoes": 600, "cucumbers": 550, "onions": 450, "eggplants": 550, "olives": 400,
}
DRIP_INTENSIVE = {
    "wheat": False, "potatoes": True, "grapes": True, "oranges": True, "maize": False,
    "tomatoes": True, "cucumbers": True, "onions": True, "eggplants": True, "olives": False,
}
ECOCROP = {
    "wheat":    {"t_min": 10, "t_max": 25, "r_min": 300, "ph_min": 5.5, "ph_max": 8.5},
    "potatoes": {"t_min": 15, "t_max": 25, "r_min": 400, "ph_min": 4.5, "ph_max": 7.5},
    "grapes":   {"t_min": 15, "t_max": 30, "r_min": 350, "ph_min": 5.5, "ph_max": 8.5},
    "oranges":  {"t_min": 18, "t_max": 32, "r_min": 400, "ph_min": 5.5, "ph_max": 8.5},
    "maize":    {"t_min": 15, "t_max": 30, "r_min": 400, "ph_min": 5.5, "ph_max": 8.5},
    "tomatoes": {"t_min": 15, "t_max": 28, "r_min": 400, "ph_min": 5.5, "ph_max": 7.5},
    "cucumbers":{"t_min": 18, "t_max": 28, "r_min": 400, "ph_min": 5.5, "ph_max": 7.5},
    "onions":   {"t_min": 12, "t_max": 25, "r_min": 350, "ph_min": 5.5, "ph_max": 8.0},
    "eggplants":{"t_min": 18, "t_max": 30, "r_min": 400, "ph_min": 5.5, "ph_max": 7.0},
    "olives":   {"t_min": 15, "t_max": 30, "r_min": 300, "ph_min": 5.5, "ph_max": 8.5},
}

def main():
    records = []
    for crop in CROPS:
        ec = ECOCROP[crop]
        records.append({
            "crop": crop,
            "yield_ton_ha": YIELD_TON_HA[crop],
            "cost_usd_ha": COST_USD_HA[crop],
            "price_usd_ton": PRICE_USD_TON[crop],
            "water_mm_season": WATER_MM[crop],
            "drip_intensive": DRIP_INTENSIVE[crop],
            "ecocrop_t_min_c": ec["t_min"],
            "ecocrop_t_max_c": ec["t_max"],
            "ecocrop_rain_min_mm": ec["r_min"],
            "ecocrop_ph_min": ec["ph_min"],
            "ecocrop_ph_max": ec["ph_max"],
            "profit_usd_ha": round(YIELD_TON_HA[crop] * PRICE_USD_TON[crop] - COST_USD_HA[crop], 2),
        })
    df = pd.DataFrame(records)
    out = OUTPUT_DIR / "arij_palestine_10crops.csv"
    df.to_csv(out, index=False)
    print(f"✅ Saved corrected ARIJ dataset to {out}")
    print(df[['crop', 'yield_ton_ha', 'cost_usd_ha', 'price_usd_ton', 'profit_usd_ha']].to_string(index=False))

if __name__ == "__main__":
    main()
'''

# =============================================================================
# 5. WRITE ALL FILES
# =============================================================================
scripts = {
    "src/fetch_nasa_power.py": fetch_nasa,
    "src/generate_parcels.py": generate_parcels,
    "src/fix_arij_yields.py": fix_arij,
}

for filepath, content in scripts.items():
    full_path = PROJECT_ROOT / filepath
    full_path.write_text(content, encoding="utf-8")
    print(f"✅ Wrote {full_path}")

print(f"\n{'='*60}")
print("DONE. Next steps:")
print(f"{'='*60}")
print(f"1. cd {PROJECT_ROOT}")
print(f"2. python src/generate_parcels.py")
print(f"3. python src/fix_arij_yields.py")
print(f"4. python src/fetch_nasa_power.py   (needs internet, ~5 min)")
print(f"\nAll outputs go to: {PROJECT_ROOT / 'data/processed'}")