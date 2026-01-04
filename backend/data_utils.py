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
    
    # Load predictions CSV instead of ML model
    predictions_path = BASE_DIR / "data" / "predicted_tree_cover_loss_2025_2035.csv"
    if predictions_path.exists():
        predictions_df = pd.read_csv(predictions_path)
    else:
        predictions_df = pd.DataFrame()  # Empty DataFrame if file doesn't exist

    latest_year = int(df["year"].max())
    
    # Filter to threshold 30 for summary statistics
    df_summary = df[df["threshold"] == 30].copy() if "threshold" in df.columns else df.copy()
    
    total_loss = float(df_summary["tree_cover_loss_ha"].sum())
    total_emissions = float(df_summary["carbon_gross_emissions_MgCO2e"].sum())
    unique_countries = int(df_summary["country"].nunique())
    
    return df, predictions_df, latest_year, total_loss, total_emissions, unique_countries

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

def get_prediction_from_csv(country, year, predictions_df):
    """Get tree cover loss prediction from CSV file for a country and year."""
    if predictions_df.empty:
        return np.nan
    
    try:
        # Filter for the specific country and target year
        country_predictions = predictions_df[predictions_df.iloc[:, 0] == country]
        
        if country_predictions.empty:
            return np.nan
        
        # Find prediction for target year
        target_prediction_row = country_predictions[country_predictions.iloc[:, 1] == year]
        
        if target_prediction_row.empty:
            return np.nan
        
        target_pred = float(target_prediction_row.iloc[0, 2])
        return max(0, target_pred)  # Ensure non-negative
        
    except Exception as e:
        return np.nan

def predict_tree_loss_future(country, year, df_all, predictions_df):
    """Predict tree cover loss for a country and future year using CSV predictions."""
    return get_prediction_from_csv(country, year, predictions_df)
