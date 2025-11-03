from dash import dcc
import plotly.express as px

def render_loss_trend(dff, country):
    fig = px.line(dff, x="year", y="tree_cover_loss_ha",
                  title=f"Tree Cover Loss Over Time — {country}", markers=True)
    return dcc.Graph(figure=fig)
