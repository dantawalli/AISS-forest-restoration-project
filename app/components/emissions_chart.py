from dash import dcc, html
import plotly.express as px
import pandas as pd

def render_emissions_chart(dff, country):
    """
    Displays relationship between tree-cover loss and
    forest-carbon emissions for a selected country
    (filtered to threshold = 30 % canopy).
    """

    # --- Convert threshold to numeric if it exists ---
    if "threshold" in dff.columns:
        dff["threshold"] = pd.to_numeric(dff["threshold"], errors='coerce')
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

    fig.update_traces(
        marker=dict(size=10, opacity=0.7, line=dict(width=1, color="#ffffff")),
        line=dict(color="#1a73e8", width=2)
    )
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Roboto, sans-serif", size=12, color="#202124"),
        xaxis=dict(
            title=dict(font=dict(size=14, color="#5f6368")),
            gridcolor="#dadce0",
            gridwidth=1,
            showline=True,
            linecolor="#dadce0"
        ),
        yaxis=dict(
            title=dict(font=dict(size=14, color="#5f6368")),
            gridcolor="#dadce0",
            gridwidth=1,
            showline=True,
            linecolor="#dadce0"
        ),
        title=dict(
            font=dict(size=20, family="Google Sans, Roboto, sans-serif", color="#202124"),
            x=0.5,
            xanchor="center"
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode="closest",
        coloraxis_colorbar=dict(
            title=dict(font=dict(size=12, color="#5f6368")),
            tickfont=dict(size=11, color="#5f6368")
        )
    )

    return dcc.Graph(figure=fig)
