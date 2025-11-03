from dash import html
import numpy as np

def render_prediction_box(dff, model):
    feature_cols = ["carbon_gross_emissions_MgCO2e","fire","forestry",
                    "shifting_agriculture","urbanization","year"]
    latest = dff.dropna(subset=feature_cols)
    if not latest.empty:
        X = latest[feature_cols].tail(1)
        pred = model.predict(X)[0]
        pred_text = f"Predicted Tree Cover Loss (Next Year): {pred:,.0f} ha"
    else:
        pred_text = "Prediction unavailable (insufficient data)."

    return html.Div([
        html.H3("🤖 Model-Based Prediction"),
        html.P(pred_text, style={"fontSize": "20px", "fontWeight": "bold", "color": "green"})
    ])
