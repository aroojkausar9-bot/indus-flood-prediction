"""
preprocessing.py
================
Data cleaning, merging, flood labeling, and outlier handling for the
Indus Basin flood prediction pipeline.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ── Flood Thresholds ───────────────────────────────────────────────────────────

def calculate_flood_thresholds(historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-basin, per-month flood thresholds using the mu + 2*sigma rule.

    Parameters
    ----------
    historical_df : DataFrame with columns [HYBAS_ID, month, rainfall_mm]

    Returns
    -------
    DataFrame with columns [HYBAS_ID, month, threshold]
    """
    stats = (historical_df
             .groupby(["HYBAS_ID", "month"])["rainfall_mm"]
             .agg(["mean", "std"])
             .reset_index())
    stats["std"]       = stats["std"].fillna(0)
    stats["threshold"] = stats["mean"] + 2 * stats["std"]
    return stats[["HYBAS_ID", "month", "threshold"]]


def flag_floods_by_threshold(df: pd.DataFrame,
                              thresholds: pd.DataFrame) -> pd.DataFrame:
    """Apply thresholds to flag flood months (1) vs normal months (0)."""
    merged = df.merge(thresholds, on=["HYBAS_ID", "month"], how="left")
    merged["Flood_Flag"] = (merged["rainfall_mm"] > merged["threshold"]).astype(float)
    return merged.drop(columns=["threshold"])


# ── Data Merging ───────────────────────────────────────────────────────────────

def merge_datasets(rain_path: str,
                   tmax_path: str,
                   static_path: str,
                   flood_2010_2018_path: str) -> pd.DataFrame:
    """
    Merge rainfall, temperature, topography, and flood labels into one table.

    Parameters
    ----------
    rain_path          : path to CHIRPS_monthly_data_2010_2024.csv
    tmax_path          : path to TerraClimate_Tmax_monthly_2010_2025.csv
    static_path        : path to static_features.csv
    flood_2010_2018_path : path to Pakistan_Flood_Flags_2010_2018.csv

    Returns
    -------
    Merged DataFrame indexed by (HYBAS_ID, year, month)
    """
    rain   = pd.read_csv(rain_path)
    tmax   = pd.read_csv(tmax_path)
    static = pd.read_csv(static_path)
    floods = pd.read_csv(flood_2010_2018_path)

    # ── Standardize rainfall ──
    rain = rain[["HYBAS_ID", "date", "rainfall_mm"]].copy()
    rain[["year", "month", "day"]] = rain["date"].astype(str).str.split("-", expand=True)
    rain = rain.drop(columns=["date", "day"]).astype({"year": int, "month": int})

    # ── Standardize temperature ──
    tmax = tmax.rename(columns={"mean": "tmax_value"})
    tmax = tmax[["HYBAS_ID", "year_month", "tmax_value"]].copy()
    tmax[["year", "month"]] = tmax["year_month"].astype(str).str.split("-", expand=True)
    tmax = tmax.drop(columns=["year_month"]).astype({"year": int, "month": int})

    # ── Standardize static ──
    static = static[["HYBAS_ID", "elev_mean", "slope_mean"]].drop_duplicates("HYBAS_ID")

    # ── Standardize historical flood labels ──
    floods = floods.rename(columns={"Year": "year", "Month": "month"})
    floods = floods[["HYBAS_ID", "year", "month", "Flood_Flag"]]
    floods = floods[(floods["year"] >= 2010) & (floods["year"] <= 2018)]

    # ── Merge ──
    master = rain.merge(static, on="HYBAS_ID", how="left", validate="many_to_one")
    master = master.merge(tmax[["HYBAS_ID", "year", "month", "tmax_value"]],
                          on=["HYBAS_ID", "year", "month"], how="left")
    master = master.merge(floods, on=["HYBAS_ID", "year", "month"], how="left")

    # Fill historical flood flags (NaN means no flood record → 0)
    mask = (master["year"] >= 2010) & (master["year"] <= 2018)
    master.loc[mask, "Flood_Flag"] = master.loc[mask, "Flood_Flag"].fillna(0).astype(int)

    print(f"[merge] Final shape: {master.shape}")
    return master


def apply_future_flood_flags(master: pd.DataFrame,
                              thresholds: pd.DataFrame) -> pd.DataFrame:
    """
    For years 2019–2024, derive flood flags from the rainfall threshold rule.
    Historical (2010–2018) flags are left unchanged.
    """
    historical = master[master["year"] <= 2018].copy()
    future     = master[master["year"] >= 2019].copy()

    if len(future) == 0:
        return master

    future = flag_floods_by_threshold(future.drop(columns=["Flood_Flag"],
                                                   errors="ignore"),
                                      thresholds)
    combined = pd.concat([historical, future], ignore_index=True)
    combined["Flood_Flag"] = combined["Flood_Flag"].fillna(0).astype(int)
    return combined


# ── Missing Value Imputation ───────────────────────────────────────────────────

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
      - rainfall_mm & tmax_value: basin-month median, then global median
      - elev_mean & slope_mean  : basin median, then global median
    """
    df = df.copy()

    for col in ["rainfall_mm", "tmax_value"]:
        if col not in df.columns:
            continue
        basin_month_med = df.groupby(["HYBAS_ID", "month"])[col].transform("median")
        df[col] = df[col].fillna(basin_month_med)
        df[col] = df[col].fillna(df[col].median())

    for col in ["elev_mean", "slope_mean"]:
        if col not in df.columns:
            continue
        basin_med = df.groupby("HYBAS_ID")[col].transform("median")
        df[col] = df[col].fillna(basin_med)
        df[col] = df[col].fillna(df[col].median())

    return df


# ── Outlier Clipping ───────────────────────────────────────────────────────────

def clip_outliers(df: pd.DataFrame,
                  cols: list = None,
                  iqr_factor: float = 1.5) -> pd.DataFrame:
    """
    Winsorize continuous columns to [Q1 - k*IQR, Q3 + k*IQR].
    """
    df   = df.copy()
    cols = cols or ["rainfall_mm", "elev_mean", "slope_mean", "tmax_value"]

    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr    = q3 - q1
        lb, ub = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        n_out  = ((df[col] < lb) | (df[col] > ub)).sum()
        df[col] = df[col].clip(lb, ub)
        print(f"  [clip] {col}: clipped {n_out} outliers ({n_out/len(df)*100:.2f}%)")

    return df


# ── Full Pipeline ──────────────────────────────────────────────────────────────

def preprocess_pipeline(master_with_floods_path: str) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline on a pre-merged CSV that already
    contains the Flood_Flag column for all years.

    Parameters
    ----------
    master_with_floods_path : path to master_with_floodflags.csv

    Returns
    -------
    Clean, outlier-clipped DataFrame ready for feature engineering
    """
    df = pd.read_csv(master_with_floods_path)
    print(f"[preprocess] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")

    df = impute_missing(df)
    df = clip_outliers(df)

    # Derive flood thresholds from 2010–2018 and apply to 2019+
    historical = df[df["year"] <= 2018]
    thresholds = calculate_flood_thresholds(historical)
    df = apply_future_flood_flags(df, thresholds)

    df["Flood_Flag"] = df["Flood_Flag"].fillna(0).astype(int)
    print(f"[preprocess] Flood rate: {df['Flood_Flag'].mean()*100:.2f}%")
    print(f"[preprocess] Done. Shape: {df.shape}")
    return df
