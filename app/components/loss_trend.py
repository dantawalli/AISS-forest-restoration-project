from dash import dcc
import plotly.express as px

def render_loss_trend(dff, country):
    """
    Plots tree-cover loss over time for a single country,
    filtered to the standard 30% canopy threshold.
    """
    # --- Filter to threshold = 30 (standard GFW definition) ---
    if "threshold" in dff.columns:
        dff = dff[dff["threshold"] == 30]

    # --- Aggregate yearly loss (some datasets have multiple rows per year) ---
    yearly_loss = (
        dff.groupby("year", as_index=False)["tree_cover_loss_ha"]
           .sum()
           .sort_values("year")
    )

    # --- Build chart ---
    fig = px.line(
        yearly_loss, x="year", y="tree_cover_loss_ha",
        title=f"🌲 Tree Cover Loss Over Time — {country} (≥30% canopy)",
        markers=True,
        color_discrete_sequence=["#2E86AB"]
    )

    # --- Clean layout ---
    fig.update_traces(line=dict(width=3))
    fig.update_layout(
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        xaxis_title="Year",
        yaxis_title="Tree Cover Loss (ha)",
        margin=dict(l=20, r=20, t=60, b=40)
    )

    return dcc.Graph(figure=fig)

