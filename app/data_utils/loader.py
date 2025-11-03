import pandas as pd
import joblib

def load_data_model():
    df = pd.read_csv("../data/processed/merged_clean_data.csv")
    model = joblib.load("../models/forest_loss_model.pkl")

    latest_year = int(df["year"].max())
    total_loss = df["tree_cover_loss_ha"].sum()
    total_emissions = df["carbon_gross_emissions_MgCO2e"].sum()
    unique_countries = df["country"].nunique()
    return df, model, latest_year, total_loss, total_emissions, unique_countries
