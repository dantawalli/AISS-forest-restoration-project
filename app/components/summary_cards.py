from dash import html

def render_summary_cards(countries, total_loss, total_emissions):
    return html.Div([
        html.Div([
            html.H3(f"{countries:,}"),
            html.P("Countries")
        ], className="card"),
        html.Div([
            html.H3(f"{total_loss:,.0f} ha"),
            html.P("Total Tree Cover Loss")
        ], className="card"),
        html.Div([
            html.H3(f"{total_emissions:,.0f} Mg CO₂e"),
            html.P("Total Carbon Emissions")
        ], className="card")
    ], className="kpi-container")
