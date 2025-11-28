from dash import dcc, html
import plotly.express as px
import pandas as pd

def render_loss_trend(dff, country):
    """
    Plots tree-cover loss over time for a single country,
    filtered to the standard 30% canopy threshold.
    """
    # --- Convert threshold to numeric if it exists ---
    if "threshold" in dff.columns:
        dff["threshold"] = pd.to_numeric(dff["threshold"], errors='coerce')
        dff = dff[dff["threshold"] == 30]

    # --- Check if required columns exist ---
    if "tree_cover_loss_ha" not in dff.columns or "year" not in dff.columns:
        return html.P("⚠️ Required columns (tree_cover_loss_ha, year) not available for this country.")

    # --- Convert to numeric and ensure numeric values ---
    if "tree_cover_loss_ha" in dff.columns:
        dff["tree_cover_loss_ha"] = pd.to_numeric(dff["tree_cover_loss_ha"], errors='coerce')
    if "year" in dff.columns:
        dff["year"] = pd.to_numeric(dff["year"], errors='coerce')
    
    # --- Drop NaN values ---
    dff = dff.dropna(subset=["tree_cover_loss_ha", "year"])

    if dff.empty:
        return html.P("⚠️ No tree cover loss data available for this country at 30% canopy threshold.")

    # --- Aggregate yearly loss (some datasets have multiple rows per year) ---
    yearly_loss = (
        dff.groupby("year", as_index=False)["tree_cover_loss_ha"]
           .sum()
           .sort_values("year")
    )

    # --- Check if aggregated data is empty ---
    if yearly_loss.empty or yearly_loss["tree_cover_loss_ha"].sum() == 0:
        return html.P("⚠️ No tree cover loss data available for the selected year range and country.")

    # --- Build chart ---
    fig = px.line(
        yearly_loss, x="year", y="tree_cover_loss_ha",
        title=f"🌲 Tree Cover Loss Over Time — {country} (≥30% canopy)",
        markers=True,
        color_discrete_sequence=["#2E86AB"]
    )

    # --- Clean layout - Material Design Style ---
    fig.update_traces(line=dict(width=3, color="#1a73e8"))
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
        hovermode="x unified"
    )

    return dcc.Graph(figure=fig)

