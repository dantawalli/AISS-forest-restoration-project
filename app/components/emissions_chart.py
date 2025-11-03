from dash import dcc, html
import plotly.express as px

def render_emissions_chart(dff, country):
    """
    Displays relationship between tree-cover loss and
    forest-carbon emissions for a selected country
    (filtered to threshold = 30 % canopy).
    """

    # --- Filter to standard canopy threshold ---
    if "threshold" in dff.columns:
        dff = dff[dff["threshold"] == 30]

    # --- Ensure numeric values ---
    if "tree_cover_loss_ha" in dff.columns and "carbon_gross_emissions_MgCO2e" in dff.columns:
        dff = dff.dropna(subset=["tree_cover_loss_ha", "carbon_gross_emissions_MgCO2e"])
    else:
        return html.P("⚠️ Required columns not available for this country.")

    if dff.empty:
        return html.P("⚠️ No emission data available for this country at 30% canopy threshold.")

    # --- Plot ---
    fig = px.scatter(
        dff,
        x="tree_cover_loss_ha",
        y="carbon_gross_emissions_MgCO2e",
        color="year",
        trendline="ols",
        title=f"🌬️ Carbon Emissions vs Tree-Cover Loss — {country} (≥30 % canopy)",
        labels={
            "tree_cover_loss_ha": "Tree-Cover Loss (ha)",
            "carbon_gross_emissions_MgCO2e": "Carbon Emissions (Mg CO₂e)"
        },
        color_continuous_scale="Viridis"
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0)))
    fig.update_layout(
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        margin=dict(l=20, r=20, t=60, b=40)
    )

    return dcc.Graph(figure=fig)
