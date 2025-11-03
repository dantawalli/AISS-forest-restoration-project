from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib

# --- Load model and data ---
bundle = joblib.load("../models/tree_cover_loss_model.pkl")
model = bundle["model"]
FEATURE_COLS = bundle["features"]
df_all = pd.read_csv("../data/processed/merged_clean_data.csv")


# --- Linear extrapolation helper ---
def linear_extrapolate(years, values, target_year):
    s = pd.DataFrame({"year": years, "val": values}).dropna().drop_duplicates().sort_values("year")
    if len(s) >= 3:
        x, y = s["year"].astype(float).values, s["val"].astype(float).values
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(a * target_year + b)
    return float(s["val"].iloc[-1]) if len(s) else np.nan


# --- Prediction helper ---
def predict_tree_loss_future(country, year):
    dff = df_all[df_all["country"] == country].copy()
    if "threshold" in dff.columns:
        filt = dff[dff["threshold"] == 30]
        if not filt.empty:
            dff = filt
    if dff.empty:
        return np.nan

    latest = dff.sort_values("year").tail(1).copy()
    numeric_cols = [c for c in FEATURE_COLS if c not in ["year"] and not c.startswith("country_")]

    for col in numeric_cols:
        if col in dff.columns:
            latest[col] = linear_extrapolate(dff["year"], dff[col], year)

    latest["year"] = year

    enc = pd.get_dummies(latest, columns=["country"], drop_first=True)
    for c in FEATURE_COLS:
        if c not in enc.columns:
            enc[c] = 0
    X_future = enc[FEATURE_COLS].fillna(0)
    return float(model.predict(X_future)[0])


def render_prediction_box():
    """UI component"""
    return html.Div([
        html.H3("🌲 Tree-Cover Loss Forecast (2001–2035)"),

        html.Div([
            html.Label("Select Country:"),
            dcc.Dropdown(
                id="pred-country",
                options=[{"label": c, "value": c}
                         for c in sorted(df_all["country"].dropna().unique())],
                value="Brazil",
                style={"width": "300px"}
            ),
        ], style={"marginBottom": "20px"}),

        html.Div(id="prediction-output",
                 style={"fontSize": "20px", "fontWeight": "bold",
                        "color": "green", "marginBottom": "10px"}),

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
    Input("pred-country", "value")
)
def predict_and_plot(country):
    # --- historical
    dff = df_all[df_all["country"] == country].copy()
    if "threshold" in dff.columns:
        filt = dff[dff["threshold"] == 30]
        if not filt.empty:
            dff = filt
    if dff.empty:
        return f"⚠️ No data for {country}.", go.Figure()

    hist = dff[["year", "tree_cover_loss_ha"]].dropna().copy()

    # --- predict future
    last_year = int(hist["year"].max())
    future_years = list(range(last_year + 1, last_year + 11))
    preds = [predict_tree_loss_future(country, y) for y in future_years]
    future = pd.DataFrame({"year": future_years, "tree_cover_loss_ha": preds})

    # join for continuity
    combined = pd.concat([hist, future]).reset_index(drop=True)

    # --- Plot continuous line with color split
    fig = go.Figure()

    # Blue: historical
    fig.add_trace(go.Scatter(
        x=combined.loc[combined["year"] <= last_year, "year"],
        y=combined.loc[combined["year"] <= last_year, "tree_cover_loss_ha"],
        mode="lines",
        line=dict(color="#2E86AB", width=3),
        name="Historical"
    ))

    # Red: predicted (starts from last year to ensure continuity)
    fig.add_trace(go.Scatter(
        x=combined.loc[combined["year"] >= last_year, "year"],
        y=combined.loc[combined["year"] >= last_year, "tree_cover_loss_ha"],
        mode="lines",
        line=dict(color="#E74C3C", width=3),
        name="Predicted"
    ))

    # Add vertical marker
    fig.add_vline(x=last_year, line_width=1.5, line_dash="dot", line_color="gray")

    fig.update_layout(
        title=f"Tree-Cover Loss Trend — {country}",
        xaxis_title="Year",
        yaxis_title="Tree-Cover Loss (ha)",
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        margin=dict(l=20, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    pred_value = preds[-1]
    text = f"🌲 Projected Tree-Cover Loss in {country} ({last_year + 10}): {pred_value:,.0f} ha"

    return text, fig
