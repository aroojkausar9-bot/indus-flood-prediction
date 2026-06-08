"""
spatial_features.py
====================
Extracts real basin coordinates from the HydroBASINS shapefile and
builds an upstream-downstream topology map.
"""

import json
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def extract_basin_coordinates(shp_path: str,
                               basin_id_field: str = "HYBAS_ID",
                               out_path: str = "data/your_basin_coordinates.csv"
                               ) -> pd.DataFrame:
    """
    Load a HydroBASINS shapefile, compute polygon centroids, and save a CSV
    with [HYBAS_ID, latitude, longitude] plus any available topology fields.

    Parameters
    ----------
    shp_path       : path to the .shp file
    basin_id_field : field name containing basin IDs (default HYBAS_ID)
    out_path       : where to save the coordinate CSV

    Returns
    -------
    DataFrame with at least [HYBAS_ID, latitude, longitude]
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    gdf["centroid"]  = gdf.geometry.centroid
    gdf["longitude"] = gdf["centroid"].x
    gdf["latitude"]  = gdf["centroid"].y

    topology_fields = [f for f in ["NEXT_DOWN", "NEXT_SINK", "MAIN_BAS",
                                    "SUB_AREA",  "UP_AREA",  "DIST_SINK"]
                       if f in gdf.columns]
    keep = [basin_id_field, "latitude", "longitude"] + topology_fields
    coords_df = gdf[keep].copy()

    if basin_id_field != "HYBAS_ID":
        coords_df = coords_df.rename(columns={basin_id_field: "HYBAS_ID"})

    coords_df.to_csv(out_path, index=False)
    print(f"[coords] Saved {len(coords_df)} basins to {out_path}")
    return coords_df


def build_topology(coords_df: pd.DataFrame,
                    out_path: str = "data/basin_topology.json") -> dict:
    """
    Build an upstream-downstream topology dictionary from NEXT_DOWN field.
    Saves JSON and returns the dict.

    Returns
    -------
    {str(basin_id): {"downstream": int, "upstream": [int, ...],
                      "latitude": float, "longitude": float}}
    """
    if "NEXT_DOWN" not in coords_df.columns:
        print("[topology] NEXT_DOWN field not found; topology not built.")
        return {}

    topology = {}
    for _, row in coords_df.iterrows():
        bid = str(int(row["HYBAS_ID"]))
        upstream = coords_df[coords_df["NEXT_DOWN"] == row["HYBAS_ID"]]["HYBAS_ID"].tolist()
        topology[bid] = {
            "downstream": int(row["NEXT_DOWN"]) if row["NEXT_DOWN"] else 0,
            "upstream":   [int(x) for x in upstream],
            "latitude":   float(row["latitude"]),
            "longitude":  float(row["longitude"]),
        }
        if "UP_AREA" in row:
            topology[bid]["upstream_area_km2"] = float(row["UP_AREA"])

    with open(out_path, "w") as fh:
        json.dump(topology, fh, indent=2)
    print(f"[topology] Saved topology for {len(topology)} basins to {out_path}")
    return topology


def plot_basin_map(shp_path: str, coords_df: pd.DataFrame,
                   save_path: str = "results/basin_map.png") -> None:
    """Quick sanity-check plot of the study basins."""
    gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    gdf.plot(ax=ax1, facecolor="lightblue", edgecolor="black", linewidth=0.5, alpha=0.7)
    ax1.scatter(coords_df["longitude"], coords_df["latitude"], c="red", s=20, zorder=5)
    ax1.set_title(f"All Indus Sub-basins (n={len(gdf)})", fontweight="bold")
    ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    study = gdf[gdf["HYBAS_ID"].isin(coords_df["HYBAS_ID"])]
    study.plot(ax=ax2, facecolor="lightgreen", edgecolor="black", linewidth=0.5, alpha=0.7)
    ax2.scatter(coords_df["longitude"], coords_df["latitude"],
                c="darkgreen", s=30, zorder=5, edgecolor="white", linewidth=0.8)
    ax2.set_title(f"Study Basins (n={len(coords_df)})", fontweight="bold")
    ax2.set_xlabel("Longitude"); ax2.set_ylabel("Latitude")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[map] Saved to {save_path}")
