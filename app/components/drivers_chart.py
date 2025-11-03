from dash import dcc, html
import plotly.express as px

def render_drivers_chart(dff, country):
    """
    Displays the contribution of different deforestation drivers
    for a given country (filtered to threshold = 30% canopy).
    """

    # --- Filter to standard canopy threshold ---
    if "threshold" in dff.columns:
        dff = dff[dff["threshold"] == 30]

    # --- Identify available driver columns ---
    driver_cols = [c for c in dff.columns if c in
                   ["fire", "forestry", "shifting_agriculture",
                    "urbanization", "commodity_driven_deforestation",
                    "unknown", "hard_commodities", "logging",
                    "permanent_agriculture", "shifting_cultivation",
                    "settlements_infrastructure", "other_natural_disturbances"]]

    if not driver_cols:
        return html.P("⚠️ No driver data available for this country at 30% canopy threshold.")

    # --- Aggregate yearly driver totals ---
    drivers_sum = dff[driver_cols].sum().reset_index()
    drivers_sum.columns = ["driver", "hectares"]
    drivers_sum = drivers_sum.sort_values("hectares", ascending=False)

    # --- Plot ---
    fig = px.bar(
        drivers_sum,
        x="driver", y="hectares",
        title=f"🔥 Deforestation Drivers — {country} (≥30% canopy)",
        text_auto=".2s",
        color="driver",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_traces(marker_line_width=0, opacity=0.85)
    fig.update_layout(
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        xaxis_title="Driver Type",
        yaxis_title="Tree Cover Loss (ha)",
        margin=dict(l=20, r=20, t=60, b=40)
    )

    return dcc.Graph(figure=fig)
