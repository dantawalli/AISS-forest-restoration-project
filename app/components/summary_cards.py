from dash import html

def render_summary_cards(countries, total_loss, total_emissions):
    """Render top-level summary KPI cards for the dashboard."""
    # Fallbacks to handle None or NaN
    countries = countries or 0
    total_loss = total_loss or 0
    total_emissions = total_emissions or 0

    return html.Div([
        html.Div([
            html.H3("🌍"),
            html.H2(f"{countries:,}"),
            html.P("Countries")
        ], className="card kpi-card"),

        html.Div([
            html.H3("🌲"),
            html.H2(f"{total_loss:,.0f} ha"),
            html.P("Total Tree Cover Loss")
        ], className="card kpi-card"),

        html.Div([
            html.H3("🌬️"),
            html.H2(f"{total_emissions:,.0f} Mg CO₂e"),
            html.P("Total Carbon Emissions")
        ], className="card kpi-card"),
    ], className="kpi-container", style={
        "display": "flex",
        "justifyContent": "space-around",
        "alignItems": "center",
        "margin": "20px 0",
        "gap": "15px"
    })
