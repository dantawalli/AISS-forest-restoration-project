from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib

# --- Load model and data ---
import os

# Get the project root directory (go up from app/components/ to project root)
# __file__ is at: project/app/components/prediction_box.py
# We need: project/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

bundle = joblib.load(os.path.join(BASE_DIR, "models", "tree_cover_loss_model.pkl"))
model = bundle["model"]
FEATURE_COLS = bundle["features"]
df_all = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "merged_clean_data.csv"))

# Strip whitespace from column names (CSV has trailing spaces)
df_all.columns = df_all.columns.str.strip()

# Clean cells with multiple space-separated values (take first value)
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

# Apply cleaning to all columns except country
for col in df_all.columns:
    if col not in ['country']:  # Skip non-numeric columns
        df_all[col] = df_all[col].apply(clean_multiple_values)

# Convert numeric columns to numeric types after cleaning
if "threshold" in df_all.columns:
    df_all["threshold"] = pd.to_numeric(df_all["threshold"], errors='coerce')
if "year" in df_all.columns:
    df_all["year"] = pd.to_numeric(df_all["year"], errors='coerce')
if "tree_cover_loss_ha" in df_all.columns:
    df_all["tree_cover_loss_ha"] = pd.to_numeric(df_all["tree_cover_loss_ha"], errors='coerce')


# --- Linear extrapolation helper ---
def linear_extrapolate(years, values, target_year):
    """Return linear extrapolation of values to target_year."""
    # Convert to DataFrame
    s = pd.DataFrame({"year": years, "val": values})
    
    # Clean values that might contain multiple space-separated numbers
    def clean_val(v):
        if pd.isna(v):
            return v
        v_str = str(v).strip()
        if not v_str:
            return pd.NA
        # If contains multiple space-separated numbers, take the first one
        parts = v_str.split()
        if len(parts) > 1:
            return parts[0]
        return v_str
    
    s["val"] = s["val"].apply(clean_val)
    
    # Convert both columns to numeric, coercing errors to NaN
    s["year"] = pd.to_numeric(s["year"], errors='coerce')
    s["val"] = pd.to_numeric(s["val"], errors='coerce')
    
    # Drop NaN values and duplicates
    s = s.dropna().drop_duplicates().sort_values("year")
    
    if len(s) >= 3:
        # Ensure values are float64
        x = s["year"].values.astype(float)
        y = s["val"].values.astype(float)
        
        # Perform linear regression
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(a * target_year + b)
    
    # If not enough data, return last value or NaN
    if len(s) > 0:
        return float(s["val"].iloc[-1])
    return np.nan


# --- Prediction helper ---
# Important features only (from notebook analysis)
# Focus on features that can be extrapolated, excluding lag features that cause data leakage
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

def predict_tree_loss_future(country, year):
    """Predict tree cover loss for a country and future year using only important features."""
    dff = df_all[df_all["country"] == country].copy()
    
    # Filter to threshold 30 if available
    if "threshold" in dff.columns:
        dff["threshold"] = pd.to_numeric(dff["threshold"], errors='coerce')
        filt = dff[dff["threshold"] == 30]
        if not filt.empty:
            dff = filt
    
    if dff.empty:
        return np.nan
    
    # Ensure year is numeric
    dff["year"] = pd.to_numeric(dff["year"], errors='coerce')
    dff = dff.sort_values("year")
    
    # Get latest data point
    latest = dff.tail(1).copy()
    
    # Calculate years_since_2000
    years_since_2000 = year - 2000
    
    # Prepare base features (only important, extrapolatable ones)
    base_features = [f for f in IMPORTANT_FEATURES if f not in ["year", "years_since_2000"] and not f.startswith("country_")]
    
    # Extrapolate only important features to target year
    for col in base_features:
        if col in dff.columns:
            # Convert to numeric
            dff[col] = pd.to_numeric(dff[col], errors='coerce')
            # Extrapolate
            extrapolated = linear_extrapolate(dff["year"], dff[col], year)
            if not np.isnan(extrapolated):
                latest[col] = extrapolated
            elif len(latest) > 0 and col in latest.columns:
                # Use last known value if extrapolation fails
                latest[col] = pd.to_numeric(latest[col].iloc[0] if len(latest) > 0 else 0, errors='coerce')
                if pd.isna(latest[col].iloc[0]):
                    latest[col] = 0
            else:
                latest[col] = 0
    
    # Set year and years_since_2000
    latest["year"] = year
    latest["years_since_2000"] = years_since_2000
    
    # For temporal features that can't be extrapolated, use last known values or calculate
    # Calculate loss_lag1 (last year's loss)
    if "tree_cover_loss_ha" in dff.columns:
        dff["tree_cover_loss_ha"] = pd.to_numeric(dff["tree_cover_loss_ha"], errors='coerce')
        last_loss = dff["tree_cover_loss_ha"].iloc[-1] if len(dff) > 0 and not pd.isna(dff["tree_cover_loss_ha"].iloc[-1]) else 0
        if "loss_lag1" in FEATURE_COLS:
            latest["loss_lag1"] = last_loss
        
        # Calculate loss_rolling_mean_3y (3-year rolling average)
        if "loss_rolling_mean_3y" in FEATURE_COLS:
            last_3_years = dff.tail(3)["tree_cover_loss_ha"].dropna()
            if len(last_3_years) > 0:
                latest["loss_rolling_mean_3y"] = last_3_years.mean()
            else:
                latest["loss_rolling_mean_3y"] = last_loss
    
    # Encode country and align with all required features
    enc = pd.get_dummies(latest, columns=["country"], drop_first=True, prefix="country")
    
    # Build X_future with all features required by model
    X_future = pd.DataFrame(index=[0])
    
    # Add all required features from FEATURE_COLS
    for feat in FEATURE_COLS:
        if feat in enc.columns:
            X_future[feat] = pd.to_numeric(enc[feat].iloc[0] if len(enc) > 0 else 0, errors='coerce')
        else:
            X_future[feat] = 0  # Missing features set to 0
    
    # Ensure correct order and all numeric
    X_future = X_future[FEATURE_COLS]
    X_future = X_future.fillna(0).astype(float)
    
    # Predict
    try:
        pred = model.predict(X_future)[0]
        return float(max(0, pred))  # Ensure non-negative
    except Exception as e:
        return np.nan


def render_prediction_box():
    """UI component"""
    max_year = int(df_all["year"].max())
    future_years = list(range(max_year + 1, max_year + 12))  # 10 years ahead
    
    return html.Div([
        html.H3("🌲 Tree-Cover Loss Forecast"),
        html.P("Predict future tree cover loss based on historical trends and machine learning models.", 
               style={"color": "#666", "marginBottom": "20px"}),

        html.Div([
            html.Div([
                html.Label("Select Country:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                dcc.Dropdown(
                    id="pred-country",
                    options=[{"label": c, "value": c}
                             for c in sorted(df_all["country"].dropna().unique())],
                    value="Brazil",
                    style={"width": "300px"}
                ),
            ], style={"display": "inline-block", "marginRight": "30px"}),
            
            html.Div([
                html.Label("Target Year:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                dcc.Dropdown(
                    id="pred-year",
                    options=[{"label": str(y), "value": y} for y in future_years],
                    value=future_years[-1],
                    style={"width": "150px"}
                ),
            ], style={"display": "inline-block"}),
        ], style={"marginBottom": "20px"}),

        html.Div(id="prediction-output",
                 style={"fontSize": "18px", "fontWeight": "bold",
                        "color": "#2E7D32", "marginBottom": "20px",
                        "padding": "15px", "backgroundColor": "#E8F5E9",
                        "borderRadius": "8px"}),

        dcc.Graph(id="prediction-trend-chart")
    ], style={
        "backgroundColor": "#F9F9F9",
        "padding": "25px",
        "borderRadius": "12px",
        "boxShadow": "0 0 10px rgba(0,0,0,0.1)"
    })


@callback(
    Output("prediction-output", "children"),
    Output("prediction-trend-chart", "figure"),
    Input("pred-country", "value"),
    Input("pred-year", "value")
)
def predict_and_plot(country, target_year):
    """Generate predictions and plot historical vs predicted trends."""
    try:
        # Check if inputs are selected
        if country is None or target_year is None:
            return html.Div([
                html.P("Please select country and target year", style={"color": "#D32F2F", "fontSize": "16px"})
            ]), go.Figure()
        
        # --- Get historical data
        dff = df_all[df_all["country"] == country].copy()
        if "threshold" in dff.columns:
            dff["threshold"] = pd.to_numeric(dff["threshold"], errors='coerce')
            filt = dff[dff["threshold"] == 30]
            if not filt.empty:
                dff = filt
        if dff.empty:
            return html.Div([
                html.P(f"⚠️ No data available for {country}.", style={"color": "#D32F2F"})
            ]), go.Figure()

        # Ensure year and tree_cover_loss_ha are numeric
        dff["year"] = pd.to_numeric(dff["year"], errors='coerce')
        dff["tree_cover_loss_ha"] = pd.to_numeric(dff["tree_cover_loss_ha"], errors='coerce')
        
        hist = dff[["year", "tree_cover_loss_ha"]].dropna().copy()
        if hist.empty:
            return html.Div([
                html.P(f"⚠️ No historical loss data for {country}.", style={"color": "#D32F2F"})
            ]), go.Figure()

        # --- Predict future - ensure target_year is int
        target_year = int(target_year) if target_year is not None else int(df_all["year"].max()) + 10
        last_year = int(hist["year"].max())
        future_years = list(range(last_year + 1, target_year + 1))
        
        preds = []
        for y in future_years:
            pred = predict_tree_loss_future(country, y)
            if not np.isnan(pred):
                preds.append(pred)
            else:
                preds.append(0)  # Fallback to 0 if prediction fails
        
        future = pd.DataFrame({"year": future_years, "tree_cover_loss_ha": preds})

        # Join for continuity
        combined = pd.concat([hist, future]).reset_index(drop=True)
        
        # Ensure year column is numeric for comparisons
        combined["year"] = pd.to_numeric(combined["year"], errors='coerce')
        combined["tree_cover_loss_ha"] = pd.to_numeric(combined["tree_cover_loss_ha"], errors='coerce')
        combined = combined.dropna()

        # --- Plot
        fig = go.Figure()

        # Blue: historical
        hist_mask = combined["year"] <= last_year
        if hist_mask.any():
            fig.add_trace(go.Scatter(
                x=combined.loc[hist_mask, "year"].values,
                y=combined.loc[hist_mask, "tree_cover_loss_ha"].values,
                mode="lines+markers",
                line=dict(color="#2E86AB", width=3),
                marker=dict(size=4),
                name="Historical Data"
            ))

        # Red: predicted
        pred_mask = combined["year"] >= last_year
        if pred_mask.any():
            fig.add_trace(go.Scatter(
                x=combined.loc[pred_mask, "year"].values,
                y=combined.loc[pred_mask, "tree_cover_loss_ha"].values,
                mode="lines+markers",
                line=dict(color="#E74C3C", width=3, dash="dash"),
                marker=dict(size=4, symbol="diamond"),
                name="Predicted"
            ))

        # Add vertical marker at last historical year
        fig.add_vline(
            x=last_year, 
            line_width=2, 
            line_dash="dot", 
            line_color="gray",
            annotation_text=f"Last Data: {last_year}",
            annotation_position="top"
        )

        # Highlight target year prediction
        if target_year in future_years:
            target_idx = future_years.index(target_year)
            target_pred = preds[target_idx]
            fig.add_trace(go.Scatter(
                x=[target_year],
                y=[target_pred],
                mode="markers",
                marker=dict(size=15, color="#FF6B35", symbol="star"),
                name=f"Target: {target_year}",
                showlegend=False
            ))

        fig.update_layout(
            title=f"Tree-Cover Loss Trend — {country}",
            xaxis_title="Year",
            yaxis_title="Tree-Cover Loss (hectares)",
            plot_bgcolor="white",
            paper_bgcolor="#F9F9F9",
            margin=dict(l=60, r=20, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hovermode="x unified"
        )

        # Format prediction output
        target_pred = preds[-1] if preds else 0
        avg_historical = hist["tree_cover_loss_ha"].mean()
        change_pct = ((target_pred - avg_historical) / avg_historical * 100) if avg_historical > 0 else 0
        
        output = html.Div([
            html.H4(f"📊 Prediction for {country} in {target_year}", 
                   style={"marginBottom": "10px", "color": "#2E7D32"}),
            html.P([
                html.Strong(f"Predicted Loss: {target_pred:,.0f} hectares", 
                           style={"fontSize": "20px", "color": "#1B5E20"}),
            ]),
            html.P([
                f"Historical Average (2001-{last_year}): {avg_historical:,.0f} ha",
                html.Br(),
                f"Change: {change_pct:+.1f}% from historical average"
            ], style={"color": "#666", "marginTop": "10px"})
        ])

        return output, fig
        
    except Exception as e:
        error_msg = html.Div([
            html.P(f"⚠️ Error generating prediction: {str(e)}", style={"color": "#D32F2F"})
        ])
        return error_msg, go.Figure()
