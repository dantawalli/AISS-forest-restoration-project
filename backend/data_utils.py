import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Get the project root directory
BASE_DIR = Path(__file__).parent.parent

def clean_multiple_values(value):
    """Handle cells with multiple space-separated values by taking the first one."""
    if pd.isna(value):
        return value
    value_str = str(value).strip()
    if not value_str:
        return pd.NA
    parts = value_str.split()
    if len(parts) > 1:
        return parts[0]
    return value_str

def load_data_model():
    df = pd.read_csv(BASE_DIR / "data" / "processed" / "merged_clean_data.csv")
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Clean cells with multiple space-separated values
    for col in df.columns:
        if col not in ['country']:
            df[col] = df[col].apply(clean_multiple_values)
    
    # Convert numeric columns
    if "threshold" in df.columns:
        df["threshold"] = pd.to_numeric(df["threshold"], errors='coerce')
    if "tree_cover_loss_ha" in df.columns:
        df["tree_cover_loss_ha"] = pd.to_numeric(df["tree_cover_loss_ha"], errors='coerce')
    if "carbon_gross_emissions_MgCO2e" in df.columns:
        df["carbon_gross_emissions_MgCO2e"] = pd.to_numeric(df["carbon_gross_emissions_MgCO2e"], errors='coerce')
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors='coerce')
    
    bundle = joblib.load(BASE_DIR / "models" / "tree_cover_loss_model.pkl")
    model = bundle["model"]
    FEATURE_COLS = bundle["features"]

    latest_year = int(df["year"].max())
    
    # Filter to threshold 30 for summary statistics
    df_summary = df[df["threshold"] == 30].copy() if "threshold" in df.columns else df.copy()
    
    total_loss = float(df_summary["tree_cover_loss_ha"].sum())
    total_emissions = float(df_summary["carbon_gross_emissions_MgCO2e"].sum())
    unique_countries = int(df_summary["country"].nunique())
    
    return df, model, latest_year, total_loss, total_emissions, unique_countries

def linear_extrapolate(years, values, target_year):
    """Return linear extrapolation of values to target_year."""
    s = pd.DataFrame({"year": years, "val": values})
    
    def clean_val(v):
        if pd.isna(v):
            return v
        v_str = str(v).strip()
        if not v_str:
            return pd.NA
        parts = v_str.split()
        if len(parts) > 1:
            return parts[0]
        return v_str
    
    s["val"] = s["val"].apply(clean_val)
    s["year"] = pd.to_numeric(s["year"], errors='coerce')
    s["val"] = pd.to_numeric(s["val"], errors='coerce')
    s = s.dropna().drop_duplicates().sort_values("year")
    
    if len(s) >= 3:
        x = s["year"].values.astype(float)
        y = s["val"].values.astype(float)
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(a * target_year + b)
    
    if len(s) > 0:
        return float(s["val"].iloc[-1])
    return np.nan

def predict_tree_loss_future(country, year, df_all, model):
    """Predict tree cover loss for a country and future year."""
    # Get feature columns from model bundle
    bundle = joblib.load(BASE_DIR / "models" / "tree_cover_loss_model.pkl")
    FEATURE_COLS = bundle["features"]
    
    IMPORTANT_FEATURES = [
        "year",
        "years_since_2000",
        "threshold",
        "hard_commodities",
        "logging",
        "wildfire",
        "permanent_agriculture",
        "shifting_cultivation",
        "gfw_forest_carbon_net_flux__mg_co2e_yr-1",
        "carbon_gross_emissions_MgCO2e",
        "primary_forest_loss_ha",
        "umd_tree_cover_extent_2000__ha",
        "gfw_aboveground_carbon_stocks_2000__mg_c"
    ]
    
    dff = df_all[df_all["country"] == country].copy()
    
    if "threshold" in dff.columns:
        dff["threshold"] = pd.to_numeric(dff["threshold"], errors='coerce')
        filt = dff[dff["threshold"] == 30]
        if not filt.empty:
            dff = filt
    
    if dff.empty:
        return np.nan
    
    dff["year"] = pd.to_numeric(dff["year"], errors='coerce')
    dff = dff.sort_values("year")
    
    latest = dff.tail(1).copy()
    years_since_2000 = year - 2000
    
    base_features = [f for f in IMPORTANT_FEATURES if f not in ["year", "years_since_2000"] and not f.startswith("country_")]
    
    for col in base_features:
        if col in dff.columns:
            dff[col] = pd.to_numeric(dff[col], errors='coerce')
            extrapolated = linear_extrapolate(dff["year"], dff[col], year)
            if not np.isnan(extrapolated):
                latest[col] = extrapolated
            elif len(latest) > 0 and col in latest.columns:
                latest[col] = pd.to_numeric(latest[col].iloc[0] if len(latest) > 0 else 0, errors='coerce')
                if pd.isna(latest[col].iloc[0]):
                    latest[col] = 0
            else:
                latest[col] = 0
    
    latest["year"] = year
    latest["years_since_2000"] = years_since_2000
    
    if "tree_cover_loss_ha" in dff.columns:
        dff["tree_cover_loss_ha"] = pd.to_numeric(dff["tree_cover_loss_ha"], errors='coerce')
        last_loss = dff["tree_cover_loss_ha"].iloc[-1] if len(dff) > 0 and not pd.isna(dff["tree_cover_loss_ha"].iloc[-1]) else 0
        if "loss_lag1" in FEATURE_COLS:
            latest["loss_lag1"] = last_loss
        
        if "loss_rolling_mean_3y" in FEATURE_COLS:
            last_3_years = dff.tail(3)["tree_cover_loss_ha"].dropna()
            if len(last_3_years) > 0:
                latest["loss_rolling_mean_3y"] = last_3_years.mean()
            else:
                latest["loss_rolling_mean_3y"] = last_loss
    
    enc = pd.get_dummies(latest, columns=["country"], drop_first=True, prefix="country")
    
    # Create all feature columns at once to avoid fragmentation
    feature_data = {}
    for feat in FEATURE_COLS:
        if feat in enc.columns:
            feature_data[feat] = pd.to_numeric(enc[feat].iloc[0] if len(enc) > 0 else 0, errors='coerce')
        else:
            feature_data[feat] = 0
    
    X_future = pd.DataFrame([feature_data])
    X_future = X_future[FEATURE_COLS]
    X_future = X_future.fillna(0).astype(float)
    
    try:
        pred = model.predict(X_future)[0]
        return float(max(0, pred))
    except Exception as e:
        return np.nan
