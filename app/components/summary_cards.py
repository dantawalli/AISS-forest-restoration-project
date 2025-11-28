from dash import html
import pandas as pd

def render_summary_cards(countries, total_loss, total_emissions):
    """Render top-level summary KPI cards for the dashboard."""
    # Convert to numeric and handle None/NaN
    try:
        countries = int(countries) if countries is not None and not pd.isna(countries) else 0
    except (ValueError, TypeError):
        countries = 0
    
    try:
        total_loss = float(total_loss) if total_loss is not None and not pd.isna(total_loss) else 0.0
    except (ValueError, TypeError):
        total_loss = 0.0
    
    try:
        total_emissions = float(total_emissions) if total_emissions is not None and not pd.isna(total_emissions) else 0.0
    except (ValueError, TypeError):
        total_emissions = 0.0

    return html.Div([
        html.Div([
            html.H3("🌍"),
            html.H2(f"{countries:,}"),
            html.P("Countries")
        ], className="kpi-card"),

        html.Div([
            html.H3("🌲"),
            html.H2(f"{total_loss:,.0f} ha"),
            html.P("Total Tree Cover Loss (≥30% canopy, 2001-2024)")
        ], className="kpi-card"),

        html.Div([
            html.H3("🌬️"),
            html.H2(f"{total_emissions:,.0f} Mg CO₂e"),
            html.P("Total Carbon Emissions (≥30% canopy, 2001-2024)")
        ], className="kpi-card"),
    ], className="kpi-container")
