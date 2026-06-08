"""
data_collection.py
==================
Google Earth Engine scripts for collecting CHIRPS rainfall,
TerraClimate temperature, and SRTM topographic data for the
Indus sub-basins in Pakistan.

Usage:
    python src/data_collection.py

Requirements:
    - Authenticated GEE account: run `earthengine authenticate` first
    - Replace EE_PROJECT with your GEE project ID
    - Place Indus_SubBasins_Pakistan.shp in the data/ directory
"""

import ee
import geemap
import geopandas as gpd
import pandas as pd
import os

# ── Configuration ──────────────────────────────────────────────────────────────
EE_PROJECT = "your-gee-project-id"          # ← replace with your GEE project
SHAPEFILE   = "data/Indus_SubBasins_Pakistan.shp"
CHIRPS_OUT  = "data/CHIRPS_monthly_data_2010_2024.csv"
TEMP_OUT    = "data/TerraClimate_Tmax_monthly_2010_2025.csv"
STATIC_OUT  = "data/static_features.csv"
START_YEAR, END_YEAR = 2010, 2024
# ──────────────────────────────────────────────────────────────────────────────


def init_ee(project: str) -> None:
    """Authenticate and initialize Google Earth Engine."""
    try:
        ee.Initialize(project=project)
        print(f"[GEE] Initialized with project: {project}")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
        print(f"[GEE] Authenticated and initialized.")


def load_subbasins(shp_path: str):
    """Load sub-basin shapefile and convert to EE FeatureCollection."""
    gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
    fc  = geemap.geopandas_to_ee(gdf)
    print(f"[Data] Loaded {len(gdf)} sub-basins from {shp_path}")
    return gdf, fc


# ── CHIRPS Rainfall ────────────────────────────────────────────────────────────

def extract_chirps_monthly(basins_fc, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Extract monthly CHIRPS rainfall (mm) per sub-basin via zonal mean.
    Exports to Google Drive as 'CHIRPS_monthly_data_{start}_{end}.csv'.
    Returns a placeholder DataFrame; download from Drive manually.
    """
    chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterBounds(basins_fc))

    years  = ee.List.sequence(start_year, end_year)
    months = ee.List.sequence(1, 12)

    def monthly_reducer(year):
        def reduce_month(month):
            start = ee.Date.fromYMD(year, month, 1)
            end   = start.advance(1, "month")
            img   = (chirps.filterDate(start, end)
                           .sum()
                           .set("year", year, "month", month,
                                "system:time_start", start.millis()))
            return img
        return months.map(reduce_month)

    monthly_imgs = ee.ImageCollection(years.map(monthly_reducer).flatten())

    def extract_stats(image):
        date = image.date().format("YYYY-MM-dd")
        reduced = image.reduceRegions(
            collection=basins_fc,
            reducer=ee.Reducer.mean(),
            scale=5000,
            tileScale=16
        )
        def tag(f):
            return ee.Feature(f.geometry(), {
                "HYBAS_ID":    f.get("HYBAS_ID"),
                "rainfall_mm": f.get("mean"),
                "date":        date
            })
        return reduced.map(tag)

    final_fc = monthly_imgs.map(extract_stats).flatten()

    task = ee.batch.Export.table.toDrive(
        collection=final_fc,
        description="CHIRPS_monthly_extraction",
        fileNamePrefix=f"CHIRPS_monthly_data_{start_year}_{end_year}",
        fileFormat="CSV"
    )
    task.start()
    print("[CHIRPS] Export task started. Monitor in the GEE Tasks tab.")
    return None


# ── TerraClimate Temperature ───────────────────────────────────────────────────

def fix_terraclimate_time(image):
    """Fix TerraClimate system:time_start from YYYYMM index string."""
    id_str = ee.String(image.get("system:index"))
    year   = ee.Number.parse(id_str.slice(0, 4))
    month  = ee.Number.parse(id_str.slice(4, 6))
    date   = ee.Date.fromYMD(year, month, 1)
    return image.set("system:time_start", date.millis())


def extract_terraclimate_tmax(basins_fc, start_year: int, end_year: int) -> None:
    """
    Extract monthly max temperature (°C) per sub-basin via TerraClimate.
    TerraClimate tmmx is in 0.1 °C units → divide by 10.
    Exports to Google Drive.
    """
    tc = (ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
            .filterDate(f"{start_year}-01-01", f"{end_year + 1}-01-01")
            .select("tmmx")
            .map(fix_terraclimate_time))

    def calc_zonal(image):
        temp_c = image.divide(10).rename("mean_temp_C")
        t_start = image.get("system:time_start")
        reduced = temp_c.reduceRegions(
            collection=basins_fc,
            reducer=ee.Reducer.mean(),
            scale=4638,
            crs="EPSG:4326"
        )
        def tag(f):
            return f.set({
                "date_millis": t_start,
                "year_month":  ee.Date(t_start).format("YYYY-MM")
            })
        return reduced.map(tag)

    final_fc = tc.map(calc_zonal).flatten()

    task = ee.batch.Export.table.toDrive(
        collection=final_fc,
        description="TerraClimate_Tmax_extraction",
        folder="EarthEngine_Exports",
        fileNamePrefix=f"TerraClimate_Tmax_monthly_{start_year}_{end_year + 1}",
        fileFormat="CSV"
    )
    task.start()
    print("[TerraClimate] Export task started. Monitor in the GEE Tasks tab.")


# ── SRTM Static Features ───────────────────────────────────────────────────────

def extract_static_features(basins_fc, output_path: str) -> pd.DataFrame:
    """
    Extract mean elevation and mean slope per sub-basin from SRTM 30m.
    Small enough to fetch directly (no Drive export needed).
    """
    srtm  = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(srtm)

    def add_stats(feature):
        elev  = srtm.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feature.geometry(),
            scale=1000, maxPixels=1e9
        ).get("elevation")
        slp   = slope.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feature.geometry(),
            scale=1000, maxPixels=1e9
        ).get("slope")
        return feature.set({"elev_mean": elev, "slope_mean": slp})

    stats_fc = basins_fc.map(add_stats)
    df       = geemap.ee_to_df(stats_fc)
    df       = df[["HYBAS_ID", "elev_mean", "slope_mean"]]
    df.to_csv(output_path, index=False)
    print(f"[SRTM] Static features saved to {output_path}")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)

    init_ee(EE_PROJECT)
    _, basins_fc = load_subbasins(SHAPEFILE)

    print("\n--- Extracting CHIRPS rainfall ---")
    extract_chirps_monthly(basins_fc, START_YEAR, END_YEAR)

    print("\n--- Extracting TerraClimate temperature ---")
    extract_terraclimate_tmax(basins_fc, START_YEAR, END_YEAR)

    print("\n--- Extracting SRTM static features ---")
    extract_static_features(basins_fc, STATIC_OUT)

    print("\n[Done] Check GEE Tasks tab for CHIRPS and TerraClimate exports.")
    print("       Download CSVs from Google Drive and place them in data/")


if __name__ == "__main__":
    main()
