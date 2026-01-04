"""
Combined Flask app that serves both API and React frontend.
Use this for single-service deployment.
"""
import sys
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from data_utils import load_data_model, predict_tree_loss_future
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='')
CORS(app)

# Load data and model on startup
df, model, latest_year, total_loss, total_emissions, unique_countries = load_data_model()

# API routes (same as backend/app.py)
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/api/summary', methods=['GET'])
def get_summary():
    return jsonify({
        "countries": int(unique_countries),
        "total_loss": float(total_loss),
        "total_emissions": float(total_emissions),
        "latest_year": int(latest_year)
    })

@app.route('/api/countries', methods=['GET'])
def get_countries():
    countries = sorted(df["country"].dropna().unique().tolist())
    return jsonify(countries)

@app.route('/api/data', methods=['GET'])
def get_data():
    country = request.args.get('country', None)
    year_start = request.args.get('year_start', None, type=int)
    year_end = request.args.get('year_end', None, type=int)
    threshold = request.args.get('threshold', 30, type=int)
    
    dff = df.copy()
    
    if country:
        dff = dff[dff["country"] == country]
    if year_start:
        dff = dff[dff["year"] >= year_start]
    if year_end:
        dff = dff[dff["year"] <= year_end]
    if threshold:
        if "threshold" in dff.columns:
            dff["threshold"] = pd.to_numeric(dff["threshold"], errors='coerce')
            dff = dff[dff["threshold"] == threshold]
    
    result = dff.to_dict(orient='records')
    return jsonify(result)

@app.route('/api/loss-trend', methods=['GET'])
def get_loss_trend():
    country = request.args.get('country', None)
    year_start = request.args.get('year_start', None, type=int)
    year_end = request.args.get('year_end', None, type=int)
    
    if not country:
        return jsonify({"error": "Country parameter is required"}), 400
    
    dff = df[df["country"] == country].copy()
    
    if "threshold" in dff.columns:
        dff["threshold"] = pd.to_numeric(dff["threshold"], errors='coerce')
        dff = dff[dff["threshold"] == 30]
    
    if year_start:
        dff = dff[dff["year"] >= year_start]
    if year_end:
        dff = dff[dff["year"] <= year_end]
    
    dff["tree_cover_loss_ha"] = pd.to_numeric(dff["tree_cover_loss_ha"], errors='coerce')
    dff["year"] = pd.to_numeric(dff["year"], errors='coerce')
    dff = dff.dropna(subset=["tree_cover_loss_ha", "year"])
    
    if dff.empty:
        return jsonify({"error": "No data available"}), 404
    
    yearly_loss = (
        dff.groupby("year", as_index=False)["tree_cover_loss_ha"]
           .sum()
           .sort_values("year")
    )
    
    return jsonify(yearly_loss.to_dict(orient='records'))

@app.route('/api/drivers', methods=['GET'])
def get_drivers():
    country = request.args.get('country', None)
    year_start = request.args.get('year_start', None, type=int)
    year_end = request.args.get('year_end', None, type=int)
    
    if not country:
        return jsonify({"error": "Country parameter is required"}), 400
    
    dff = df[df["country"] == country].copy()
    
    if "threshold" in dff.columns:
        dff["threshold"] = pd.to_numeric(dff["threshold"], errors='coerce')
        dff = dff[dff["threshold"] == 30]
    
    if year_start:
        dff = dff[dff["year"] >= year_start]
    if year_end:
        dff = dff[dff["year"] <= year_end]
    
    driver_cols = [c for c in dff.columns if c in
                   ["hard_commodities", "logging", "wildfire",
                    "permanent_agriculture", "shifting_cultivation",
                    "settlements_infrastructure", "other_natural_disturbances"]]
    
    if not driver_cols:
        return jsonify({"error": "No driver data available"}), 404
    
    for col in driver_cols:
        dff[col] = pd.to_numeric(dff[col], errors='coerce').fillna(0)
    
    drivers_sum = dff[driver_cols].sum().to_frame(name="hectares").reset_index()
    drivers_sum.columns = ["driver", "hectares"]
    drivers_sum = drivers_sum[drivers_sum["hectares"] > 0]
    drivers_sum = drivers_sum.sort_values("hectares", ascending=False)
    
    return jsonify(drivers_sum.to_dict(orient='records'))

@app.route('/api/emissions', methods=['GET'])
def get_emissions():
    country = request.args.get('country', None)
    year_start = request.args.get('year_start', None, type=int)
    year_end = request.args.get('year_end', None, type=int)
    
    if not country:
        return jsonify({"error": "Country parameter is required"}), 400
    
    dff = df[df["country"] == country].copy()
    
    if "threshold" in dff.columns:
        dff["threshold"] = pd.to_numeric(dff["threshold"], errors='coerce')
        dff = dff[dff["threshold"] == 30]
    
    if year_start:
        dff = dff[dff["year"] >= year_start]
    if year_end:
        dff = dff[dff["year"] <= year_end]
    
    if "tree_cover_loss_ha" not in dff.columns or "carbon_gross_emissions_MgCO2e" not in dff.columns:
        return jsonify({"error": "Required columns not available"}), 404
    
    dff = dff.dropna(subset=["tree_cover_loss_ha", "carbon_gross_emissions_MgCO2e"])
    
    if dff.empty:
        return jsonify({"error": "No emission data available"}), 404
    
    return jsonify(dff[["year", "tree_cover_loss_ha", "carbon_gross_emissions_MgCO2e"]].to_dict(orient='records'))

@app.route('/api/map-data', methods=['GET'])
def get_map_data():
    animated = request.args.get('animated', 'false').lower() == 'true'
    year = request.args.get('year', None, type=int)
    
    dff = df.copy()
    dff["tree_cover_loss_ha"] = pd.to_numeric(dff["tree_cover_loss_ha"], errors="coerce")
    
    if animated and year:
        country_year_loss = (
            dff[dff["year"] == year]
            .groupby(["country", "year"], as_index=False)["tree_cover_loss_ha"]
            .sum()
            .dropna()
        )
        return jsonify(country_year_loss.to_dict(orient='records'))
    else:
        country_loss = (
            dff.groupby("country", as_index=False)["tree_cover_loss_ha"]
              .sum()
              .dropna()
        )
        return jsonify(country_loss.to_dict(orient='records'))

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    country = data.get('country', None)
    target_year = data.get('year', None)
    
    if not country or not target_year:
        return jsonify({"error": "Country and year are required"}), 400
    
    try:
        dff = df[df["country"] == country].copy()
        if "threshold" in dff.columns:
            dff["threshold"] = pd.to_numeric(dff["threshold"], errors='coerce')
            filt = dff[dff["threshold"] == 30]
            if not filt.empty:
                dff = filt
        
        if dff.empty:
            return jsonify({"error": "No data available for this country"}), 404
        
        dff["year"] = pd.to_numeric(dff["year"], errors='coerce')
        dff["tree_cover_loss_ha"] = pd.to_numeric(dff["tree_cover_loss_ha"], errors='coerce')
        hist = dff[["year", "tree_cover_loss_ha"]].dropna().copy()
        
        if hist.empty:
            return jsonify({"error": "No historical loss data"}), 404
        
        target_year = int(target_year)
        last_year = int(hist["year"].max())
        future_years = list(range(last_year + 1, target_year + 1))
        
        preds = []
        for y in future_years:
            pred = predict_tree_loss_future(country, y, df, model)
            if not np.isnan(pred):
                preds.append({"year": y, "tree_cover_loss_ha": float(pred)})
            else:
                preds.append({"year": y, "tree_cover_loss_ha": 0.0})
        
        historical = hist.to_dict(orient='records')
        avg_historical = float(hist["tree_cover_loss_ha"].mean())
        target_pred = preds[-1]["tree_cover_loss_ha"] if preds else 0.0
        change_pct = ((target_pred - avg_historical) / avg_historical * 100) if avg_historical > 0 else 0
        
        return jsonify({
            "historical": historical,
            "predictions": preds,
            "last_year": last_year,
            "target_year": target_year,
            "target_prediction": target_pred,
            "avg_historical": avg_historical,
            "change_pct": change_pct
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve React app for all non-API routes
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
