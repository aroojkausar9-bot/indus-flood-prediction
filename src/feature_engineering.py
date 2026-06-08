"""
feature_engineering.py
=======================
Temporal and spatial feature engineering for the Indus Basin
flood prediction pipeline.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Flood-derived lag features — must be excluded to prevent leakage
LEAKAGE_FEATURES = [
    "flood_lag_1", "flood_lag_2", "flood_lag_3",
    "flood_lag_6", "flood_lag_12",
    "flood_roll_count_3", "flood_roll_count_6", "flood_roll_count_12",
    "flood_previous_year",
]


# ── Temporal Features ──────────────────────────────────────────────────────────

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full set of temporal features per sub-basin.

    Features created
    ----------------
    Lag            : rainfall at t-1, t-2, t-3, t-6, t-12
    Rolling stats  : 3/6/12-month mean, sum, std of rainfall
    YoY            : same month prior year, diff, ratio
    Cyclical       : sin/cos encoding of month
    Intensity      : extreme_rainfall flag, rainfall_intensity category
    Trend          : 3-month vs 12-month rolling mean ratio, 3-month pct-change
    """
    df = df.sort_values(["HYBAS_ID", "year", "month"]).reset_index(drop=True)

    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        df[f"rainfall_lag_{lag}"] = (df.groupby("HYBAS_ID")["rainfall_mm"]
                                       .shift(lag))

    # Rolling statistics
    for w in [3, 6, 12]:
        grp = df.groupby("HYBAS_ID")["rainfall_mm"]
        df[f"rainfall_roll_mean_{w}"] = grp.transform(
            lambda x: x.rolling(w, min_periods=1).mean())
        df[f"rainfall_roll_sum_{w}"]  = grp.transform(
            lambda x: x.rolling(w, min_periods=1).sum())
        df[f"rainfall_roll_std_{w}"]  = grp.transform(
            lambda x: x.rolling(w, min_periods=1).std())

    # Year-over-year
    df["rainfall_previous_year"] = (df.groupby(["HYBAS_ID", "month"])["rainfall_mm"]
                                      .shift(12))
    df["rainfall_yoy_diff"]  = df["rainfall_mm"] - df["rainfall_previous_year"]
    df["rainfall_yoy_ratio"] = df["rainfall_mm"] / (df["rainfall_previous_year"] + 1e-6)

    # Cyclical month encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Season
    df["season"] = df["month"] % 12 // 3 + 1
    df["season_name"] = df["season"].map({1: "Winter", 2: "Spring",
                                           3: "Summer", 4: "Fall"})
    df["is_monsoon"] = ((df["month"] >= 6) & (df["month"] <= 9)).astype(int)

    # Rainfall intensity category
    conds   = [df["rainfall_mm"] == 0,
               df["rainfall_mm"] < 10,
               df["rainfall_mm"] < 50,
               df["rainfall_mm"] < 100,
               df["rainfall_mm"] >= 100]
    choices = ["No Rain", "Light", "Moderate", "Heavy", "Extreme"]
    df["rainfall_intensity"] = np.select(conds, choices, default="Moderate")

    # Elevation category
    df["elevation_category"] = pd.cut(
        df["elev_mean"],
        bins=[-np.inf, 100, 500, 1000, 2000, np.inf],
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )

    # Extreme rainfall flag (top 5th percentile per basin-month)
    df["extreme_rainfall"] = (
        df["rainfall_mm"] > df.groupby(["HYBAS_ID", "month"])["rainfall_mm"]
                              .transform(lambda x: x.quantile(0.95))
    ).astype(int)

    # Trend
    df["rainfall_trend_3m"] = (df["rainfall_roll_mean_3"]
                                / (df["rainfall_roll_mean_12"] + 1e-6))
    df["rainfall_roc_3m"]   = (df.groupby("HYBAS_ID")["rainfall_mm"]
                                  .transform(lambda x: x.pct_change(3)))

    # Rainfall anomaly (departure from basin-month mean)
    df["rainfall_anomaly"] = (
        df["rainfall_mm"]
        - df.groupby(["HYBAS_ID", "month"])["rainfall_mm"].transform("mean")
    )

    return df


def impute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle NaN and Inf values introduced by lag/rolling/pct-change operations.
    Lag/rolling NaNs → 0; other NaNs → column median.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    skip = {"HYBAS_ID", "year", "month", "Flood_Flag"}
    cols = [c for c in num_cols if c not in skip]

    # Forward-fill within basin first
    for c in cols:
        df[c] = df.groupby("HYBAS_ID")[c].ffill()

    # Replace Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill remaining NaN
    for c in cols:
        if df[c].isnull().any():
            name = c.lower()
            fill_val = 0.0 if ("lag" in name or "roll" in name) else df[c].median()
            df[c] = df[c].fillna(fill_val)

    return df


def remove_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any flood-derived lag/roll features that would cause leakage."""
    to_drop = [f for f in LEAKAGE_FEATURES if f in df.columns]
    df = df.drop(columns=to_drop)
    print(f"[leakage] Removed {len(to_drop)} leakage features: {to_drop}")
    return df


# ── Spatial Features ───────────────────────────────────────────────────────────

def haversine_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute an (n × n) pairwise great-circle distance matrix in km.

    Parameters
    ----------
    coords : array of shape (n, 2) with [latitude, longitude] in degrees
    """
    R   = 6371.0
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    n   = len(coords)
    D   = np.zeros((n, n))
    for i in range(n):
        dlat = lat - lat[i]
        dlon = lon - lon[i]
        a    = np.sin(dlat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon / 2) ** 2
        D[i] = R * 2 * np.arcsin(np.sqrt(a))
    return D


def build_neighbor_map(basin_ids: np.ndarray,
                       coords: np.ndarray,
                       k: int = 5) -> dict:
    """
    Return a dict mapping each HYBAS_ID to its k nearest neighbor IDs and
    corresponding distances (km) using the Haversine metric.
    """
    from sklearn.neighbors import NearestNeighbors

    coords_rad = np.radians(coords)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="haversine")
    nbrs.fit(coords_rad)
    dists, idxs = nbrs.kneighbors(coords_rad)
    dists_km = dists * 6371.0

    neighbor_map = {}
    for i, bid in enumerate(basin_ids):
        neighbor_map[bid] = {
            "ids":       basin_ids[idxs[i][1:]].tolist(),
            "distances": dists_km[i][1:].tolist(),
        }
    return neighbor_map


def create_spatial_features(df: pd.DataFrame,
                             coords_df: pd.DataFrame,
                             k_neighbors: int = 5) -> pd.DataFrame:
    """
    Add spatial context features to the dataset.

    Parameters
    ----------
    df        : main dataset with columns [HYBAS_ID, year, month, rainfall_mm,
                Flood_Flag, ...]
    coords_df : DataFrame with columns [HYBAS_ID, latitude, longitude]
    k_neighbors : number of nearest neighbors to consider

    Returns
    -------
    DataFrame with additional spatial columns
    """
    # Merge coordinates
    df = df.merge(coords_df[["HYBAS_ID", "latitude", "longitude"]],
                  on="HYBAS_ID", how="left")
    df["latitude"]  = df["latitude"].fillna(df["latitude"].median())
    df["longitude"] = df["longitude"].fillna(df["longitude"].median())

    unique = df[["HYBAS_ID", "latitude", "longitude"]].drop_duplicates()
    basin_ids = unique["HYBAS_ID"].values
    coords    = unique[["latitude", "longitude"]].values
    nbr_map   = build_neighbor_map(basin_ids, coords, k=k_neighbors)

    df["year_month"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    records = []

    for ym, period_df in df.groupby("year_month"):
        reg_rain_avg = period_df["rainfall_mm"].mean()
        reg_rain_max = period_df["rainfall_mm"].max()
        reg_flood_rate = period_df["Flood_Flag"].mean()

        for bid in period_df["HYBAS_ID"].unique():
            row = period_df[period_df["HYBAS_ID"] == bid].iloc[0]
            nbrs = nbr_map.get(bid, {"ids": [], "distances": []})
            nbr_data = period_df[period_df["HYBAS_ID"].isin(nbrs["ids"])]

            if len(nbr_data) > 0:
                w = 1.0 / (np.array(nbrs["distances"][: len(nbr_data)]) + 1)
                w /= w.sum()
                nbr_rain_avg  = nbr_data["rainfall_mm"].mean()
                nbr_rain_max  = nbr_data["rainfall_mm"].max()
                nbr_rain_wt   = float(np.average(
                    nbr_data["rainfall_mm"].values[:len(w)], weights=w))
                nbr_flood_cnt = float(nbr_data["Flood_Flag"].sum())
                nbr_rain_std  = nbr_data["rainfall_mm"].std()
                grad          = row["rainfall_mm"] - nbr_rain_avg
            else:
                nbr_rain_avg = nbr_rain_max = nbr_rain_wt = row["rainfall_mm"]
                nbr_flood_cnt = nbr_rain_std = grad = 0.0

            records.append({
                "HYBAS_ID":                   bid,
                "year_month":                 ym,
                "neighbor_rainfall_avg":      nbr_rain_avg,
                "neighbor_rainfall_max":      nbr_rain_max,
                "neighbor_rainfall_weighted": nbr_rain_wt,
                "neighbor_flood_count":       nbr_flood_cnt,
                "neighbor_rainfall_std":      nbr_rain_std,
                "spatial_rainfall_gradient":  grad,
                "regional_rainfall_avg":      reg_rain_avg,
                "regional_rainfall_max":      reg_rain_max,
                "regional_flood_rate":        reg_flood_rate,
                "rainfall_regional_anomaly":  row["rainfall_mm"] - reg_rain_avg,
            })

    spatial_df = pd.DataFrame(records)
    out = df.merge(spatial_df, on=["HYBAS_ID", "year_month"], how="left")

    # Compound features
    out["spatial_saturation_index"] = (
        out["rainfall_mm"]
        + out["neighbor_rainfall_avg"]
    ) / 2

    out["cumulative_spatial_rainfall"] = (
        out["rainfall_mm"]
        + 0.5 * out["neighbor_rainfall_weighted"]
    )

    # Clean
    num_cols = out.select_dtypes(include=np.number).columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    for c in num_cols:
        if out[c].isnull().any():
            out[c] = out[c].fillna(out[c].median())

    print(f"[spatial] Added spatial features. New shape: {out.shape}")
    return out
