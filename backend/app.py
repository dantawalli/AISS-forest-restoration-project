import sys
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone
import json
from gee_service import initialize_gee, get_curimana_tile

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from data_utils import load_data_model, predict_tree_loss_future
from llm_engine import ForestRecommendationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle numpy/pandas types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            return list(obj)
        return super().default(obj)

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    supports_credentials=True
)

app.json_encoder = NumpyJSONEncoder

# Initialize Google Earth Engine
try:
    initialize_gee()
    logger.info("GEE initialized successfully")
except Exception as e:
    logger.error(f"GEE initialization failed: {str(e)}")

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per minute"]
)

df, landmark_df, predictions_df, latest_year, total_loss, total_emissions, unique_countries = load_data_model()

country_intelligence_df = pd.read_csv("Fynos-project/data/country_intelligence_dataset.csv")

# Initialize LLM engine if API key is available
openai_api_key = os.getenv('OPENAI_API_KEY')
print("API KEY:", openai_api_key)
llm_engine = None
if openai_api_key:
    try:
        llm_engine = ForestRecommendationEngine(openai_api_key, df)
        logger.info("LLM engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM engine: {str(e)}")
else:
    logger.warning("OPENAI_API_KEY not found in environment variables")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get summary statistics"""
    return jsonify({
        "countries": int(unique_countries),
        "total_loss": float(total_loss),
        "total_emissions": float(total_emissions),
        "latest_year": int(latest_year)
    })

@app.route('/api/countries', methods=['GET'])
def get_countries():
    """Get list of available countries"""
    countries = sorted(df["country"].dropna().unique().tolist())
    return jsonify(countries)

@app.route('/api/countries/search', methods=['GET'])
def search_countries():
    """Search countries with fuzzy matching"""
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify([])
    
    countries = sorted(df["country"].dropna().unique().tolist())
    
    # Simple fuzzy matching - case insensitive contains
    matches = [country for country in countries if query.lower() in country.lower()]
    
    # If no matches, try partial matching
    if not matches and len(query) >= 2:
        matches = [country for country in countries 
                  if any(query.lower() in part.lower() for part in country.split())]
    
    # Limit results to 20 for performance
    return jsonify(matches[:20])

@app.route('/api/country-info', methods=['GET'])
def get_country_info():
    """Get country information including common codes and alternative names"""
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    countries = sorted(df["country"].dropna().unique().tolist())
    
    # Common country code mappings
    country_codes = {
        'AUS': 'Australia',
        'USA': 'United States',
        'UK': 'United Kingdom',
        'DRC': 'Democratic Republic Of The Congo',
        'COD': 'Democratic Republic Of The Congo',
        'BRA': 'Brazil',
        'IND': 'India',
        'CHN': 'China',
        'IDN': 'Indonesia',
        'MYS': 'Malaysia',
        'THA': 'Thailand',
        'VNM': 'Vietnam',
        'PHL': 'Philippines',
        'MEX': 'Mexico',
        'CAN': 'Canada',
        'DEU': 'Germany',
        'FRA': 'France',
        'ITA': 'Italy',
        'ESP': 'Spain',
        'RUS': 'Russian Federation',
        'JPN': 'Japan',
        'KOR': 'South Korea',
        'NZL': 'New Zealand',
        'SGP': 'Singapore',
        'ARG': 'Argentina',
        'CHL': 'Chile',
        'PER': 'Peru',
        'COL': 'Colombia',
        'VEN': 'Venezuela',
        'ECU': 'Ecuador',
        'BOL': 'Bolivia',
        'PRY': 'Paraguay',
        'URY': 'Uruguay',
        'GUY': 'Guyana',
        'SUR': 'Suriname',
        'NGA': 'Nigeria',
        'ZAF': 'South Africa',
        'EGY': 'Egypt',
        'KEN': 'Kenya',
        'TZA': 'Tanzania',
        'GHA': 'Ghana',
        'CIV': 'Côte d\'Ivoire',
        'CMR': 'Cameroon',
        'GAB': 'Gabon',
        'COG': 'Congo',
        'AGO': 'Angola',
        'MDG': 'Madagascar',
        'MWI': 'Malawi',
        'ZMB': 'Zambia',
        'ZWE': 'Zimbabwe',
        'MOZ': 'Mozambique',
        'ETH': 'Ethiopia',
        'UGA': 'Uganda',
        'RWA': 'Rwanda',
        'BDI': 'Burundi',
        'TCD': 'Chad',
        'NER': 'Niger',
        'MLI': 'Mali',
        'BFA': 'Burkina Faso',
        'SEN': 'Senegal',
        'GIN': 'Guinea',
        'SLE': 'Sierra Leone',
        'LBR': 'Liberia',
        'CIV': 'Côte d\'Ivoire',
        'TGO': 'Togo',
        'BEN': 'Benin',
        'GMB': 'Gambia',
        'GNB': 'Guinea-Bissau',
        'GNQ': 'Equatorial Guinea',
        'STP': 'São Tomé and Príncipe',
        'CPV': 'Cabo Verde',
        'SYC': 'Seychelles',
        'MUS': 'Mauritius',
        'COM': 'Comoros',
        'MDG': 'Madagascar'
    }
    
    # Check if query is a country code
    if query.upper() in country_codes:
        full_name = country_codes[query.upper()]
        if full_name in countries:
            return jsonify({
                "found": True,
                "country": full_name,
                "code": query.upper(),
                "suggestions": [full_name]
            })
    
    # Check if query matches a country name exactly
    if query in countries:
        return jsonify({
            "found": True,
            "country": query,
            "code": None,
            "suggestions": [query]
        })
    
    # Find similar countries
    similar = [country for country in countries if query.lower() in country.lower()]
    
    return jsonify({
        "found": False,
        "country": None,
        "code": query.upper() if len(query) <= 3 else None,
        "suggestions": similar[:5],
        "available_codes": list(country_codes.keys())[:10]
    })

@app.route('/api/data', methods=['GET'])
def get_data():
    """Get filtered data based on query parameters"""
    country = request.args.get('country', None)
    year_start = request.args.get('year_start', None, type=int)
    year_end = request.args.get('year_end', None, type=int)
    threshold = request.args.get('threshold', 30, type=int)
    
    dff = df.copy()
    
    # Apply filters
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
    
    # Convert to JSON-serializable format
    result = dff.to_dict(orient='records')
    return jsonify(result)

@app.route('/api/primary-loss-trend', methods=['GET'])
def get_primary_loss_trend():
    """Get primary forest loss trend data for a country"""
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
    
    dff["primary_forest_loss_ha"] = pd.to_numeric(dff["primary_forest_loss_ha"], errors='coerce')
    dff["year"] = pd.to_numeric(dff["year"], errors='coerce')
    dff = dff.dropna(subset=["primary_forest_loss_ha", "year"])
    
    if dff.empty:
        return jsonify({"error": "No primary forest loss data available"}), 404
    
    # Aggregate yearly primary forest loss
    yearly_primary_loss = (
        dff.groupby("year", as_index=False)["primary_forest_loss_ha"]
           .sum()
           .sort_values("year")
    )
    
    return jsonify(yearly_primary_loss.to_dict(orient='records'))

@app.route('/api/loss-trend', methods=['GET'])
def get_loss_trend():
    """Get loss trend data for a country"""
    country = request.args.get('country', None)
    year_start = request.args.get('year_start', None, type=int)
    year_end = request.args.get('year_end', None, type=int)
    
    if not country:
        return jsonify({"error": "Country parameter is required"}), 400
    
    # Check if country exists in dataset
    available_countries = sorted(df["country"].dropna().unique().tolist())
    if country not in available_countries:
        # Try to find similar countries or check if it's a country code
        similar = [c for c in available_countries if country.lower() in c.lower()][:5]
        
        # Check common country codes
        country_codes = {
            'AUS': 'Australia', 'USA': 'United States', 'UK': 'United Kingdom',
            'DRC': 'Democratic Republic Of The Congo', 'BRA': 'Brazil',
            'IND': 'India', 'CHN': 'China', 'IDN': 'Indonesia'
        }
        
        suggested_country = country_codes.get(country.upper())
        
        return jsonify({
            "error": "Country not found in database",
            "details": {
                "provided": country,
                "suggested": suggested_country,
                "similar_countries": similar,
                "total_available": len(available_countries),
                "note": "Use /api/country-info?q=AUS to resolve country codes"
            }
        }), 404
    
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
        return jsonify({"error": "No data available for this country"}), 404
    
    # Aggregate yearly loss
    yearly_loss = (
        dff.groupby("year", as_index=False)["tree_cover_loss_ha"]
           .sum()
           .sort_values("year")
    )
    
    return jsonify(yearly_loss.to_dict(orient='records'))

@app.route('/api/drivers', methods=['GET'])
def get_drivers():
    """Get deforestation drivers data for a country"""
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
    
    # Map column names to driver names
    driver_mapping = {
        "hard_commodities": "Agriculture",
        "logging": "Logging", 
        "wildfire": "Wildfire",
        "permanent_agriculture": "Agriculture",
        "shifting_cultivation": "Agriculture",
        "settlements_infrastructure": "Infrastructure",
        "other_natural_disturbances": "Other"
    }
    
    driver_cols = [c for c in dff.columns if c in driver_mapping.keys()]
    
    if not driver_cols:
        return jsonify({"error": "No driver data available"}), 404
    
    for col in driver_cols:
        dff[col] = pd.to_numeric(dff[col], errors='coerce').fillna(0)
    
    # Aggregate by driver type
    driver_data = {}
    for col in driver_cols:
        driver_name = driver_mapping[col]
        if driver_name not in driver_data:
            driver_data[driver_name] = 0
        driver_data[driver_name] += dff[col].sum()
    
    # Convert to list of dicts and calculate percentages
    total_hectares = sum(driver_data.values())
    if total_hectares == 0:
        return jsonify({"error": "No driver data available"}), 404
    
    result = []
    for driver, hectares in driver_data.items():
        if hectares > 0:
            percentage = (hectares / total_hectares) * 100
            result.append({
                "driver": driver,
                "hectares": float(hectares),
                "percentage": round(percentage, 1)
            })
    
    # Sort by hectares descending
    result.sort(key=lambda x: x["hectares"], reverse=True)
    
    return jsonify(result)

@app.route('/api/emissions', methods=['GET'])
def get_emissions():
    """Get emissions data for a country"""
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
    """Get map data for world map visualization"""
    animated = request.args.get('animated', 'false').lower() == 'true'
    year = request.args.get('year', None, type=int)
    
    if year is None:
        year = latest_year
    
    dff = df.copy()
    dff["tree_cover_loss_ha"] = pd.to_numeric(dff["tree_cover_loss_ha"], errors="coerce")
    
    if animated:
        # Return data for all countries for the specified year
        country_year_loss = (
            dff[dff["year"] == year]
            .groupby(["country", "year"], as_index=False)["tree_cover_loss_ha"]
            .sum()
            .dropna()
        )
        return jsonify(country_year_loss.to_dict(orient='records'))
    else:
        # Return data for all countries for the specified year (same as animated for consistency)
        country_year_loss = (
            dff[dff["year"] == year]
            .groupby(["country", "year"], as_index=False)["tree_cover_loss_ha"]
            .sum()
            .dropna()
        )
        return jsonify(country_year_loss.to_dict(orient='records'))

@app.route('/api/cumulative-tree-cover-loss-trend', methods=['GET'])
def get_cumulative_tree_cover_loss_trend():
    """Get cumulative tree cover loss trend data for a country"""
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
    
    # Aggregate yearly loss and calculate cumulative sum
    yearly_loss = (
        dff.groupby("year", as_index=False)["tree_cover_loss_ha"]
           .sum()
           .sort_values("year")
    )
    
    yearly_loss["cumulative_tree_cover_loss_ha"] = yearly_loss["tree_cover_loss_ha"].cumsum()
    
    return jsonify(yearly_loss[["year", "cumulative_tree_cover_loss_ha"]].to_dict(orient='records'))

@app.route('/api/cumulative-drivers', methods=['GET'])
def get_cumulative_drivers():
    """Get cumulative deforestation drivers data for a country"""
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
    
    # Map column names to driver names
    driver_mapping = {
        "hard_commodities": "Agriculture",
        "logging": "Logging", 
        "wildfire": "Wildfire",
        "permanent_agriculture": "Agriculture",
        "shifting_cultivation": "Agriculture",
        "settlements_infrastructure": "Infrastructure",
        "other_natural_disturbances": "Other"
    }
    
    driver_cols = [c for c in dff.columns if c in driver_mapping.keys()]
    
    if not driver_cols:
        return jsonify({"error": "No driver data available"}), 404
    
    for col in driver_cols:
        dff[col] = pd.to_numeric(dff[col], errors='coerce').fillna(0)
    
    # Aggregate by driver type over all years (cumulative)
    driver_data = {}
    for col in driver_cols:
        driver_name = driver_mapping[col]
        if driver_name not in driver_data:
            driver_data[driver_name] = 0
        driver_data[driver_name] += dff[col].sum()
    
    # Convert to list of dicts and calculate percentages
    total_hectares = sum(driver_data.values())
    if total_hectares == 0:
        return jsonify({"error": "No driver data available"}), 404
    
    result = []
    for driver, hectares in driver_data.items():
        if hectares > 0:
            percentage = (hectares / total_hectares) * 100
            result.append({
                "driver": driver,
                "cumulative_hectares": float(hectares),
                "percentage": round(percentage, 1)
            })
    
    # Sort by cumulative hectares descending
    result.sort(key=lambda x: x["cumulative_hectares"], reverse=True)
    
    return jsonify(result)

@app.route('/api/cumulative-primary-loss', methods=['GET'])
def get_cumulative_primary_loss():
    """Get cumulative primary forest loss data for a country"""
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
    
    dff["primary_forest_loss_ha"] = pd.to_numeric(dff["primary_forest_loss_ha"], errors='coerce')
    dff["year"] = pd.to_numeric(dff["year"], errors='coerce')
    dff = dff.dropna(subset=["primary_forest_loss_ha", "year"])
    
    if dff.empty:
        return jsonify({"error": "No primary forest loss data available"}), 404
    
    # Aggregate yearly primary forest loss and calculate cumulative sum
    yearly_primary_loss = (
        dff.groupby("year", as_index=False)["primary_forest_loss_ha"]
           .sum()
           .sort_values("year")
    )
    
    yearly_primary_loss["cumulative_primary_forest_loss_ha"] = yearly_primary_loss["primary_forest_loss_ha"].cumsum()
    
    return jsonify(yearly_primary_loss[["year", "cumulative_primary_forest_loss_ha"]].to_dict(orient='records'))

@app.route('/api/primary-loss-all-countries', methods=['GET'])
def get_primary_loss_all_countries():
    """Get primary forest loss data for all countries"""
    year_start = request.args.get('year_start', None, type=int)
    year_end = request.args.get('year_end', None, type=int)
    
    dff = df.copy()
    
    if "threshold" in dff.columns:
        dff["threshold"] = pd.to_numeric(dff["threshold"], errors='coerce')
        dff = dff[dff["threshold"] == 30]
    
    if year_start:
        dff = dff[dff["year"] >= year_start]
    if year_end:
        dff = dff[dff["year"] <= year_end]
    
    dff["primary_forest_loss_ha"] = pd.to_numeric(dff["primary_forest_loss_ha"], errors='coerce')
    dff = dff.dropna(subset=["primary_forest_loss_ha", "country"])
    
    if dff.empty:
        return jsonify({"error": "No primary forest loss data available"}), 404
    
    # Aggregate by country and calculate total primary forest loss
    country_primary_loss = (
        dff.groupby("country", as_index=False)["primary_forest_loss_ha"]
           .sum()
           .sort_values("primary_forest_loss_ha", ascending=False)
    )
    
    return jsonify(country_primary_loss.to_dict(orient='records'))

@app.route('/api/cumulative-tree-cover-loss', methods=['GET'])
def get_cumulative_tree_cover_loss():
    """Get cumulative tree cover loss data for all countries"""
    year_start = request.args.get('year_start', None, type=int)
    year_end = request.args.get('year_end', None, type=int)
    
    dff = df.copy()
    
    if "threshold" in dff.columns:
        dff["threshold"] = pd.to_numeric(dff["threshold"], errors='coerce')
        dff = dff[dff["threshold"] == 30]
    
    if year_start:
        dff = dff[dff["year"] >= year_start]
    if year_end:
        dff = dff[dff["year"] <= year_end]
    
    dff["tree_cover_loss_ha"] = pd.to_numeric(dff["tree_cover_loss_ha"], errors='coerce')
    dff = dff.dropna(subset=["tree_cover_loss_ha", "country"])
    
    if dff.empty:
        return jsonify({"error": "No data available"}), 404
    
    # Aggregate by country and calculate cumulative loss
    country_loss = (
        dff.groupby("country", as_index=False)["tree_cover_loss_ha"]
           .sum()
           .sort_values("tree_cover_loss_ha", ascending=False)
    )
    
    country_loss["cumulative_tree_cover_loss_ha"] = country_loss["tree_cover_loss_ha"]
    
    return jsonify(country_loss[["country", "cumulative_tree_cover_loss_ha"]].to_dict(orient='records'))

@app.route('/api/predict', methods=['POST'])
def predict():
    """Get prediction for one or multiple countries and target year"""
    data = request.get_json()
    
    # Support both single country (backward compatibility) and multiple countries
    countries = data.get('countries', None)
    single_country = data.get('country', None)
    target_year = data.get('year', None)
    
    # Handle backward compatibility
    if single_country and not countries:
        countries = [single_country]
    elif not countries:
        return jsonify({"error": "Countries (array) or country (string) is required"}), 400
    
    if not target_year:
        return jsonify({"error": "Year is required"}), 400
    
    target_year = int(target_year)
    results = {}
    errors = {}
    
    for country in countries:
        try:
            # Get historical data
            dff = df[df["country"] == country].copy()
            if "threshold" in dff.columns:
                dff["threshold"] = pd.to_numeric(dff["threshold"], errors='coerce')
                filt = dff[dff["threshold"] == 30]
                if not filt.empty:
                    dff = filt
            
            if dff.empty:
                errors[country] = "No data available for this country"
                continue
            
            dff["year"] = pd.to_numeric(dff["year"], errors='coerce')
            dff["tree_cover_loss_ha"] = pd.to_numeric(dff["tree_cover_loss_ha"], errors='coerce')
            hist = dff[["year", "tree_cover_loss_ha"]].dropna().copy()
            
            if hist.empty:
                errors[country] = "No historical loss data"
                continue
            
            # Get predictions from CSV file instead of ML model
            try:
                # Path to predictions CSV
                predictions_path = Path(__file__).parent.parent / "data" / "predicted_tree_cover_loss_2025_2035.csv"
                
                if not predictions_path.exists():
                    errors[country] = "Predictions file not found"
                    continue
                
                # Read predictions CSV
                pred_df = pd.read_csv(predictions_path)
                
                # Filter for the specific country and target year
                country_predictions = pred_df[pred_df.iloc[:, 0] == country]
                
                if country_predictions.empty:
                    errors[country] = "No predictions found for this country"
                    continue
                
                # Find prediction for target year
                target_prediction_row = country_predictions[country_predictions.iloc[:, 1] == target_year]
                
                if target_prediction_row.empty:
                    errors[country] = f"No prediction available for year {target_year}"
                    continue
                
                target_pred = float(target_prediction_row.iloc[0, 2])
                target_pred = max(0, target_pred)  # Ensure non-negative
                
                # Get all predictions up to target year
                preds = []
                for _, row in country_predictions.iterrows():
                    year = int(row.iloc[1])
                    loss = float(row.iloc[2])
                    if year <= target_year:
                        preds.append({
                            "year": year, 
                            "tree_cover_loss_ha": max(0, loss)
                        })
                
                # Sort predictions by year
                preds.sort(key=lambda x: x["year"])
                
            except Exception as e:
                errors[country] = f"Error reading predictions: {str(e)}"
                continue
            
            # Get historical data for the response
            historical = hist.to_dict(orient='records')
            
            # Calculate statistics
            avg_historical = float(hist["tree_cover_loss_ha"].mean())
            last_historical_year = int(hist["year"].max())
            target_pred = preds[-1]["tree_cover_loss_ha"] if preds else 0.0
            change_pct = ((target_pred - avg_historical) / avg_historical * 100) if avg_historical > 0 else 0
            
            results[country] = {
                "historical": historical,
                "predictions": preds,
                "last_historical_year": last_historical_year,
                "target_year": target_year,
                "target_prediction": target_pred,
                "avg_historical": avg_historical,
                "change_pct": change_pct
            }
            
        except Exception as e:
            errors[country] = str(e)
    
    # Return response
    response = {
        "target_year": target_year,
        "countries_processed": len(results),
        "total_requested": len(countries),
        "results": results
    }
    
    if errors:
        response["errors"] = errors
        response["countries_with_errors"] = len(errors)
    
    # If no successful predictions, return error
    if not results:
        return jsonify({
            "error": "No successful predictions",
            "details": errors
        }), 400
    
    return jsonify(response)

@app.route('/api/recommendations', methods=['POST'])
@limiter.limit("20 per minute")
def generate_recommendations():
    """Generate AI-powered recommendations for specific countries and stakeholder groups"""
    if not llm_engine:
        return jsonify({
            "success": False,
            "error": {
                "code": "LLM_UNAVAILABLE",
                "message": "LLM service is not available. Please check API configuration."
            }
        }), 503
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['country', 'stakeholder', 'dataRange']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": {
                        "code": "MISSING_FIELD",
                        "message": f"Missing required field: {field}"
                    }
                }), 400
        
        country = data['country']
        stakeholder = data['stakeholder']
        data_range = data['dataRange']
        include_predictions = data.get('includePredictions', True)
        language = data.get('language', 'en')
        
        # Validate country
        available_countries = sorted(df["country"].dropna().unique().tolist())
        if country not in available_countries:
            return jsonify({
                "success": False,
                "error": {
                    "code": "INVALID_COUNTRY",
                    "message": "Country not found in database",
                    "details": {
                        "provided": country,
                        "available": available_countries[:10]  # Show first 10 for brevity
                    }
                }
            }), 400
        
        # Validate stakeholder
        valid_stakeholders = ['policy_governance', 'academic_research', 'environmental_ngo', 'corporate_sustainability']
        if stakeholder not in valid_stakeholders:
            return jsonify({
                "success": False,
                "error": {
                    "code": "INVALID_STAKEHOLDER",
                    "message": "Invalid stakeholder type",
                    "details": {
                        "provided": stakeholder,
                        "available": valid_stakeholders
                    }
                }
            }), 400
        
        # Validate data range
        if not isinstance(data_range, dict) or 'startYear' not in data_range or 'endYear' not in data_range:
            return jsonify({
                "success": False,
                "error": {
                    "code": "INVALID_DATA_RANGE",
                    "message": "dataRange must contain startYear and endYear"
                }
            }), 400
        
        start_year = int(data_range['startYear'])
        end_year = int(data_range['endYear'])
        
        if start_year < 2001 or end_year > 2024 or start_year > end_year:
            return jsonify({
                "success": False,
                "error": {
                    "code": "INVALID_YEAR_RANGE",
                    "message": "Year range must be between 2001-2024 with startYear <= endYear"
                }
            }), 400
        
        # Build context and generate recommendations
        context = llm_engine.build_recommendation_context(country, stakeholder, data_range)
        
        if not context:
            return jsonify({
                "success": False,
                "error": {
                    "code": "CONTEXT_BUILD_FAILED",
                    "message": "Failed to build recommendation context"
                }
            }), 500
        
        # Generate recommendations
        result = llm_engine.generate_recommendations(context)
        
        # Make JSON-safe before returning
        json_safe_result = llm_engine._json_safe(result)
        return jsonify(json_safe_result)
        
    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred"
            }
        }), 500


@app.route('/api/landmark', methods=['GET'])
def get_landmark_data():
    import pandas as pd

    df = pd.read_csv("../data/processed/landmark_country_summary.csv")

    from flask import jsonify

    return jsonify(df.to_dict(orient="records"))

@app.route('/api/country-intelligence', methods=['GET'])
def get_country_intelligence():
    """Return unified socio-environmental intelligence dataset"""

    clean_df = country_intelligence_df.copy()

    # Replace problematic values
    clean_df = clean_df.replace([np.inf, -np.inf], None)
    clean_df = clean_df.replace({np.nan: None})
    clean_df = clean_df.where(pd.notnull(clean_df), None)

    return jsonify(
        clean_df.to_dict(orient='records')
    )
@app.route('/api/insights', methods=['POST'])
@limiter.limit("30 per minute")
def generate_insights():
    """Generate deep analytical insights for research and strategic planning"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['countries', 'metrics', 'timeframe', 'analysisType']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": {
                        "code": "MISSING_FIELD",
                        "message": f"Missing required field: {field}"
                    }
                }), 400
        
        countries = data['countries']
        metrics = data['metrics']
        timeframe = data['timeframe']
        analysis_type = data['analysisType']
        
        # Validate countries
        available_countries = sorted(df["country"].dropna().unique().tolist())
        invalid_countries = [c for c in countries if c not in available_countries]
        if invalid_countries:
            return jsonify({
                "success": False,
                "error": {
                    "code": "INVALID_COUNTRIES",
                    "message": "Some countries not found in database",
                    "details": {
                        "invalid": invalid_countries,
                        "available": available_countries[:10]
                    }
                }
            }), 400
        
        # Validate analysis type
        valid_types = ['comparative', 'trend', 'correlation']
        if analysis_type not in valid_types:
            return jsonify({
                "success": False,
                "error": {
                    "code": "INVALID_ANALYSIS_TYPE",
                    "message": "Invalid analysis type",
                    "details": {
                        "provided": analysis_type,
                        "available": valid_types
                    }
                }
            }), 400
        
        # Generate insights based on analysis type
        insights = generate_analytical_insights(countries, metrics, timeframe, analysis_type)
        
        # Make JSON-safe before returning
        json_safe_insights = llm_engine._json_safe(insights)
        return jsonify({
            "success": True,
            "data": json_safe_insights
        })
        
    except Exception as e:
        logger.error(f"Error in insights endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred"
            }
        }), 500

@app.route('/api/recommendations/templates', methods=['GET'])
def get_recommendation_templates():
    """Get available recommendation templates and stakeholder profiles"""
    try:
        stakeholders = [
            {
                "id": "policy_governance",
                "name": "Policy & Governance",
                "description": "Government agencies and policy makers",
                "focusAreas": ["legislation", "enforcement", "international_cooperation"],
                "outputFormat": "policy_brief"
            },
            {
                "id": "academic_research",
                "name": "Academic Research",
                "description": "Research institutions and academic organizations",
                "focusAreas": ["research_methodology", "data_analysis", "knowledge_gaps"],
                "outputFormat": "research_report"
            },
            {
                "id": "environmental_ngo",
                "name": "Environmental NGO",
                "description": "Non-governmental environmental organizations",
                "focusAreas": ["advocacy", "community_engagement", "campaigns"],
                "outputFormat": "action_plan"
            },
            {
                "id": "corporate_sustainability",
                "name": "Corporate Sustainability",
                "description": "Business organizations focused on sustainability",
                "focusAreas": ["supply_chain", "esg_reporting", "sustainability_strategy"],
                "outputFormat": "business_case"
            }
        ]
        
        metrics = [
            {
                "id": "deforestation_rate",
                "name": "Deforestation Rate",
                "description": "Annual rate of forest loss",
                "unit": "hectares_per_year"
            },
            {
                "id": "carbon_emissions",
                "name": "Carbon Emissions",
                "description": "CO2 emissions from deforestation",
                "unit": "Mg_CO2e"
            },
            {
                "id": "biodiversity_impact",
                "name": "Biodiversity Impact",
                "description": "Impact on biodiversity and ecosystems",
                "unit": "impact_score"
            },
            {
                "id": "economic_impact",
                "name": "Economic Impact",
                "description": "Economic consequences of forest loss",
                "unit": "USD"
            }
        ]
        
        return jsonify({
            "success": True,
            "data": {
                "stakeholders": stakeholders,
                "metrics": metrics,
                "analysisTypes": [
                    {"id": "comparative", "name": "Comparative Analysis"},
                    {"id": "trend", "name": "Trend Analysis"},
                    {"id": "correlation", "name": "Correlation Analysis"}
                ]
            }
        })
        
    except Exception as e:
        logger.error(f"Error in templates endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred"
            }
        }), 500

def generate_analytical_insights(countries, metrics, timeframe, analysis_type):
    """Generate analytical insights based on type"""
    insights = {
        "countries": countries,
        "metrics": metrics,
        "timeframe": timeframe,
        "analysisType": analysis_type,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "insights": []
    }
    
    try:
        if analysis_type == "comparative":
            # Comparative analysis
            country_data = {}
            for country in countries:
                country_df = df[df['country'] == country]
                if not country_df.empty:
                    country_data[country] = {
                        'total_loss': float(country_df['tree_cover_loss_ha'].sum()),
                        'avg_annual_loss': float(country_df.groupby('year')['tree_cover_loss_ha'].sum().mean()),
                        'latest_year_loss': float(country_df[country_df['year'] == country_df['year'].max()]['tree_cover_loss_ha'].sum())
                    }
            
            # Sort countries by total loss
            sorted_countries = sorted(country_data.items(), key=lambda x: x[1]['total_loss'], reverse=True)
            
            insights["insights"] = [
                {
                    "type": "ranking",
                    "title": "Forest Loss Ranking",
                    "description": f"Countries ranked by total forest loss",
                    "data": [{"country": c, "total_loss": d['total_loss'], "rank": i+1} 
                            for i, (c, d) in enumerate(sorted_countries)]
                },
                {
                    "type": "comparison",
                    "title": "Comparative Analysis",
                    "description": "Comparison of forest loss metrics across countries",
                    "data": country_data
                }
            ]
            
        elif analysis_type == "trend":
            # Trend analysis
            for country in countries:
                country_df = df[df['country'] == country]
                if not country_df.empty:
                    yearly_loss = country_df.groupby('year')['tree_cover_loss_ha'].sum().reset_index()
                    
                    # Calculate trend
                    if len(yearly_loss) >= 3:
                        years = yearly_loss['year'].values.reshape(-1, 1)
                        losses = yearly_loss['tree_cover_loss_ha'].values
                        from sklearn.linear_model import LinearRegression
                        lr = LinearRegression().fit(years, losses)
                        trend_slope = lr.coef_[0]
                        
                        insights["insights"].append({
                            "type": "trend",
                            "country": country,
                            "title": f"{country} Forest Loss Trend",
                            "description": f"Trend analysis for {country}",
                            "data": {
                                "slope": float(trend_slope),
                                "direction": "increasing" if trend_slope > 0 else "decreasing",
                                "yearly_data": yearly_loss.to_dict('records')
                            }
                        })
                        
        elif analysis_type == "correlation":
            # Correlation analysis
            insights["insights"] = [
                {
                    "type": "correlation",
                    "title": "Driver Correlation Analysis",
                    "description": "Correlation between deforestation drivers and forest loss",
                    "data": {
                        "note": "Detailed correlation analysis requires additional processing",
                        "available_drivers": ["agriculture", "logging", "wildfire", "infrastructure"]
                    }
                }
            ]
    
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        insights["error"] = "Failed to generate complete insights"
    
    return insights
@app.route('/api/subnational/<country>/<region>', methods=['GET'])
def get_subnational_data(country, region):
    import pandas as pd
    import unicodedata

    # 👉 LOAD DATA (this was missing!)
    df_sub = pd.read_csv("../data/processed/subnational_timeseries.csv")

    print("Incoming:", country, region)
    print("Unique regions:", df_sub["region"].unique())

    # 👉 Normalize function
    def normalize(text):
        if pd.isna(text):
            return ""
        return unicodedata.normalize('NFD', str(text)) \
            .encode('ascii', 'ignore') \
            .decode('utf-8') \
            .lower() \
            .replace(" ", "") \
            .strip()

    # 👉 Apply normalization
    df_sub["country_norm"] = df_sub["country"].apply(normalize)
    df_sub["region_norm"] = df_sub["region"].apply(normalize)

    country_norm = normalize(country)
    region_norm = normalize(region)

    # 👉 Filter
    filtered = df_sub[
        (df_sub["country_norm"] == country_norm) &
        (df_sub["region_norm"] == region_norm)
    ]

    return jsonify(filtered.to_dict(orient="records"))

@app.route('/api/landmark-summary', methods=['GET'])
def get_landmark_summary():
    """Return LANDMARK Indigenous territories summary by country"""
    return jsonify(landmark_df.to_dict(orient='records'))

@app.route('/api/satellite/curimana', methods=['GET'])
def get_curimana_satellite():
    """Return satellite tile for Curimana"""
    try:
        result = get_curimana_tile()
        return jsonify({
            "success": True,
            "data": result
        })
    except Exception as e:
        logger.error(f"Error fetching Curimana tile: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def handle_ndvi_area_options():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        return response


@app.route('/api/ndvi-area', methods=['GET', 'POST', 'OPTIONS'])
def ndvi_area():

    if request.method == 'OPTIONS':
        return handle_ndvi_area_options()

    import ee
    initialize_gee()

    data = request.get_json()

    # ✅ fallback region
    region = data.get("region", "lowland")

    try:
        geom = ee.Geometry(data["geometry"])

        collection = ee.ImageCollection("COPERNICUS/S2_SR") \
            .filterBounds(geom) \
            .filterDate('2023-01-01', '2023-12-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .select(['B8', 'B4', 'B3'])

        image = collection.median()

        # =========================
        # 📊 INDICES
        # =========================
        ndvi = image.normalizedDifference(['B8', 'B4'])
        ndwi = image.normalizedDifference(['B3', 'B8'])

        mean_ndvi = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=10
        )

        mean_ndwi = ndwi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=10
        )

        # =========================
        # 🏔️ DEM (ELEVATION)
        # =========================
        dem = ee.Image("USGS/SRTMGL1_003")

        mean_elevation = dem.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=30
        )

        # =========================
        # 🆕 SLOPE
        # =========================
        slope = ee.Terrain.slope(dem)

        mean_slope = slope.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=30
        )

        # =========================
        # 📊 SAFE EXTRACTION
        # =========================
        elevation_dict = mean_elevation.getInfo()
        elevation_value = list(elevation_dict.values())[0] if elevation_dict else None

        slope_dict = mean_slope.getInfo()
        slope_value = list(slope_dict.values())[0] if slope_dict else 0

        # =========================
        # 🏔️ SLOPE CLASSIFICATION (NEW)
        # =========================
        if slope_value is None:
            slope_class = "unknown"
        elif slope_value < 5:
            slope_class = "flat"
        elif slope_value < 15:
            slope_class = "moderate"
        elif slope_value < 30:
            slope_class = "steep"
        else:
            slope_class = "very_steep"

        # =========================
        # 🌍 REGION DETECTION
        # =========================
        if elevation_value is not None:
            if elevation_value < 400:
                region = "lowland"
            elif elevation_value < 800:
                region = "central"
            else:
                region = "andean"

        # =========================
        # 📊 NDVI / NDWI
        # =========================
        ndvi_dict = mean_ndvi.getInfo()
        ndwi_dict = mean_ndwi.getInfo()

        ndvi_value = list(ndvi_dict.values())[0]
        ndwi_value = list(ndwi_dict.values())[0]

        # =========================
        # 💧 WATER
        # =========================
        water_presence = ndwi_value is not None and ndwi_value > 0.2

        if ndvi_value is None:
            return jsonify({"error": "NDVI calculation failed"}), 500

        # =========================
        # 🌍 ECOSYSTEM
        # =========================
        if water_presence:
            ecosystem_type = "bajial"
        elif ndvi_value < 0.35:
            ecosystem_type = "pastizales"
        elif ndvi_value < 0.6:
            ecosystem_type = "purma"
        else:
            ecosystem_type = "bosque_alto"

        # =========================
        # 🌱 RESTORATION PLAN (UNCHANGED)
        # =========================
        restoration_plan = None

        if ecosystem_type == "pastizales":
            restoration_plan = {
                "phase_1": {"name": "Soil Recovery", "species": ["Mucuna", "Kudzu", "Guaba"]},
                "phase_2": {"name": "Transition", "species": ["Papaya", "Plátano"]},
                "phase_3": {"name": "Stabilization", "species": ["Caoba", "Tornillo"]},
                "animals": ["Chickens"]
            }

        elif ecosystem_type == "purma":

            if ndvi_value > 0.5:

                if region == "lowland":

                    if ndwi_value is not None and ndwi_value < 0:
                        restoration_plan = {
                            "strategy": "Dry Lowland Agroforestry Recovery",
                            "phase_1": {"name": "Soil recovery & survival crops (0–1 years)", "species": ["Yuca", "Maíz", "Frijol", "Ají Charapita"]},
                            "phase_2": {"name": "Agroforestry transition (2–4 years)", "species": ["Guaba", "Bolaina", "Capirona"]},
                            "phase_3": {"name": "Forest-based system (5+ years)", "species": ["Castaña", "Shihuahuaco", "Pashaco"]},
                            "animals": ["Chickens", "Bees"]
                        }
                    else:
                        restoration_plan = {
                            "strategy": "Flood-adapted system",
                            "phase_1": {"name": "Flood cycle crops", "species": ["Arroz de bajial", "Chiclayo"]},
                            "phase_2": {"name": "Water-tolerant crops", "species": ["Camu Camu", "Açaí"]},
                            "phase_3": {"name": "Wetland forest", "species": ["Aguaje", "Renaco"]},
                            "animals": ["Fish"]
                        }

                elif region == "central":
                    restoration_plan = {
                        "phase_1": {"name": "Fast crops", "species": ["Yuca", "Maíz"]},
                        "phase_2": {"name": "Agroforestry", "species": ["Cacao", "Cítricos"]},
                        "phase_3": {"name": "Timber", "species": ["Tornillo"]},
                        "animals": ["Bees"]
                    }

                else:
                    restoration_plan = {
                        "phase_1": {"name": "Andean crops", "species": ["Arracacha"]},
                        "phase_2": {"name": "Specialty", "species": ["Granadilla"]},
                        "phase_3": {"name": "Coffee", "species": ["Café"]},
                        "animals": ["Bees"]
                    }

            else:
                if region == "lowland":
                    restoration_plan = {
                        "phase_1": {"name": "Soil recovery", "species": ["Mucuna", "Kudzu"]},
                        "phase_2": {"name": "Transition", "species": ["Yuca", "Maíz"]},
                        "phase_3": {"name": "Forest", "species": ["Pijuayo", "Aguaje"]},
                        "animals": ["Chickens"]
                    }
                else:
                    restoration_plan = {
                        "phase_1": {"name": "Enrichment", "species": ["Cacao"]},
                        "phase_2": {"name": "Timber", "species": ["Bolaina"]},
                        "phase_3": {"name": "Climax", "species": ["Caoba"]},
                        "animals": ["Bees"]
                    }

        elif ecosystem_type == "bajial":
            restoration_plan = {
                "phase_1": {"name": "Hydric", "species": ["Aguaje"]},
                "phase_2": {"name": "Flood crops", "species": ["Camu Camu"]},
                "phase_3": {"name": "Wetland forest", "species": ["Putumayo"]},
                "animals": ["Fish"]
            }

        else:
            restoration_plan = {
                "phase_1": {"name": "Gap closure", "species": ["Balsa"]},
                "phase_2": {"name": "Structure", "species": ["Pijuayo"]},
                "phase_3": {"name": "Primary forest", "species": ["Shihuahuaco"]},
                "animals": ["Bees"]
            }

        # =========================
        # 🌱 SLOPE ADJUSTMENT (NEW SAFE LAYER)
        # =========================
        if restoration_plan and isinstance(restoration_plan, dict):

            restoration_plan["terrain"] = {
                "slope_degrees": slope_value,
                "slope_class": slope_class
            }

            if slope_class == "moderate":
                restoration_plan["recommendation_note"] = "Use contour planting to reduce runoff."

            elif slope_class == "steep":
                restoration_plan["recommendation_note"] = "High erosion risk. Prioritize trees and avoid annual crops."

            elif slope_class == "very_steep":
                restoration_plan["recommendation_note"] = "Avoid agriculture. Focus on forest restoration."

                if "phase_3" in restoration_plan:
                    restoration_plan["phase_3"]["priority"] = "critical_ecosystem_restoration"

        # =========================
        # 📊 STATUS (UNCHANGED)
        # =========================
        if ndvi_value < 0.35:
            status = "Severely degraded"
            vegetation = "Very low"
            soil = "Highly degraded"
            moisture = "Low"
            risk = "High erosion risk"
            timeline = 36
            carbon = 2.5
            biodiversity = 20

        elif ndvi_value < 0.5:
            status = "Moderately degraded"
            vegetation = "Moderate"
            soil = "Degraded"
            moisture = "Medium-low"
            risk = "Moderate erosion"
            timeline = 24
            carbon = 5.8
            biodiversity = 45

        elif ndvi_value < 0.7:
            status = "Recovering ecosystem"
            vegetation = "Good"
            soil = "Stable"
            moisture = "Medium"
            risk = "Low risk"
            timeline = 12
            carbon = 9.5
            biodiversity = 70

        else:
            status = "Healthy ecosystem"
            vegetation = "Dense"
            soil = "Healthy"
            moisture = "Good"
            risk = "Minimal"
            timeline = 6
            carbon = 12.0
            biodiversity = 90

        return jsonify({
            "ndvi": ndvi_value,
            "ndwi": ndwi_value,
            "water_presence": water_presence,
            "ecosystem_type": ecosystem_type,
            "restoration_plan": restoration_plan,
            "elevation": elevation_value,
            "slope": slope_value,
            "slope_class": slope_class,  # ✅ NEW
            "region_detected": region,
            "status": status,
            "vegetation": vegetation,
            "soil": soil,
            "moisture": moisture,
            "risk": risk,
            "timeline_months": timeline,
            "carbon_estimate_tons_per_ha": carbon,
            "biodiversity_score": biodiversity
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/ndvi-point', methods=['POST', 'OPTIONS'])
def ndvi_point():
    try:
        initialize_gee()
        data = request.get_json()

        lat = data.get("lat")
        lon = data.get("lon")

        if lat is None or lon is None:
            return jsonify({"error": "Missing lat/lon"}), 400

        from gee_service import get_ndvi_at_point

        result = get_ndvi_at_point(lat, lon)

        return jsonify(result)

    except Exception as e:
        logger.error(f"NDVI point error: {str(e)}")
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
