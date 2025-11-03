from dash import dcc
import plotly.express as px

def render_emissions_chart(dff, country):
    fig = px.scatter(dff, x="tree_cover_loss_ha", y="carbon_gross_emissions_MgCO2e",
                     color="year", trendline="ols",
                     title=f"Carbon Emissions vs Forest Loss — {country}")
    return dcc.Graph(figure=fig)
