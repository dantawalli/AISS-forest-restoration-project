import pandas as pd
import joblib
import os

# Get the project root directory
# __file__ is at: project/app/data_utils/loader.py
# We need: project/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clean_multiple_values(value):
    """Handle cells with multiple space-separated values by taking the first one."""
    if pd.isna(value):
        return value
    value_str = str(value).strip()
    if not value_str:
        return pd.NA
    # If contains multiple space-separated numbers, take the first one
    parts = value_str.split()
    if len(parts) > 1:
        return parts[0]  # Take first value
    return value_str

def load_data_model():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "merged_clean_data.csv"))
    
    # Strip whitespace from column names (CSV has trailing spaces)
    df.columns = df.columns.str.strip()
    
    # Clean cells with multiple space-separated values (take first value)
    # Apply to all columns that might have numeric data
    for col in df.columns:
        if col not in ['country']:  # Skip non-numeric columns
            df[col] = df[col].apply(clean_multiple_values)
    
    # Convert numeric columns to numeric, coercing errors to NaN
    if "threshold" in df.columns:
        df["threshold"] = pd.to_numeric(df["threshold"], errors='coerce')
    if "tree_cover_loss_ha" in df.columns:
        df["tree_cover_loss_ha"] = pd.to_numeric(df["tree_cover_loss_ha"], errors='coerce')
    if "carbon_gross_emissions_MgCO2e" in df.columns:
        df["carbon_gross_emissions_MgCO2e"] = pd.to_numeric(df["carbon_gross_emissions_MgCO2e"], errors='coerce')
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors='coerce')
    
    bundle = joblib.load(os.path.join(BASE_DIR, "models", "tree_cover_loss_model.pkl"))
    model = bundle["model"]

    latest_year = int(df["year"].max())
    
    # Filter to threshold 30 (standard) for summary statistics to match what charts display
    df_summary = df[df["threshold"] == 30].copy() if "threshold" in df.columns else df.copy()
    
    # Calculate totals: sum across all countries and all years (cumulative total)
    total_loss = float(df_summary["tree_cover_loss_ha"].sum())
    total_emissions = float(df_summary["carbon_gross_emissions_MgCO2e"].sum())
    unique_countries = int(df_summary["country"].nunique())
    
    return df, model, latest_year, total_loss, total_emissions, unique_countries
