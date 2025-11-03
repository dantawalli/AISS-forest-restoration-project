from dash import dcc, html
import plotly.express as px

def render_drivers_chart(dff, country):
    driver_cols = [c for c in dff.columns if c in
                   ["fire","forestry","shifting_agriculture","urbanization"]]
    if not driver_cols:
        return html.P("No driver data available for this country.")

    drivers_sum = dff[driver_cols].sum().reset_index()
    drivers_sum.columns = ["driver", "hectares"]
    fig = px.bar(drivers_sum, x="driver", y="hectares", text_auto='.2s',
                 title=f"Share of Deforestation Drivers — {country}", color="driver")
    return dcc.Graph(figure=fig)
