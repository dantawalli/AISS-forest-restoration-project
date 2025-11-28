from dash import dcc, html
import plotly.express as px
import pandas as pd

def render_drivers_chart(dff, country):
    """
    Displays the contribution of different deforestation drivers
    for a given country (filtered to threshold = 30% canopy).
    """

    # --- Convert threshold to numeric if it exists ---
    if "threshold" in dff.columns:
        dff["threshold"] = pd.to_numeric(dff["threshold"], errors='coerce')
        dff = dff[dff["threshold"] == 30]

    # --- Identify available driver columns (using actual column names from the dataset) ---
    driver_cols = [c for c in dff.columns if c in
                   ["hard_commodities", "logging", "wildfire",
                    "permanent_agriculture", "shifting_cultivation",
                    "settlements_infrastructure", "other_natural_disturbances"]]

    if not driver_cols:
        return html.P("⚠️ No driver data available for this country at 30% canopy threshold.")

    # --- Convert driver columns to numeric and ensure they're numeric ---
    for col in driver_cols:
        dff[col] = pd.to_numeric(dff[col], errors='coerce').fillna(0)

    # --- Aggregate yearly driver totals ---
    # .sum() on DataFrame returns a Series with column names as index
    # Convert to DataFrame properly
    drivers_sum = dff[driver_cols].sum().to_frame(name="hectares").reset_index()
    drivers_sum.columns = ["driver", "hectares"]
    
    # Filter out drivers with zero values
    drivers_sum = drivers_sum[drivers_sum["hectares"] > 0]
    drivers_sum = drivers_sum.sort_values("hectares", ascending=False)

    if drivers_sum.empty:
        return html.P("⚠️ No driver data with values greater than zero for this country at 30% canopy threshold.")

    # --- Material Design Color Palette - Assign unique colors to each driver ---
    driver_color_map = {
        "hard_commodities": "#1a73e8",           # Google Blue
        "logging": "#34a853",                     # Google Green
        "wildfire": "#ea4335",                    # Google Red
        "permanent_agriculture": "#fbbc04",       # Google Yellow
        "shifting_cultivation": "#ff9800",        # Material Orange
        "settlements_infrastructure": "#9c27b0",  # Material Purple
        "other_natural_disturbances": "#4285f4",  # Light Blue
        "fire": "#ea4335",                        # Red (if fire exists as separate)
        "forestry": "#607d8b",                    # Material Blue Grey
        "shifting_agriculture": "#ff9800",        # Orange (alternative name)
        "urbanization": "#795548",                # Material Brown
        "commodity_driven_deforestation": "#1a73e8",  # Blue
        "unknown": "#9e9e9e"                      # Material Grey
    }
    
    # Create color mapping dictionary for each driver in the data
    # Use exact driver names from the dataframe
    color_map = {}
    for driver in drivers_sum["driver"].values:
        # Try exact match first, then lowercase, then with underscores
        if driver in driver_color_map:
            color_map[driver] = driver_color_map[driver]
        elif driver.lower() in driver_color_map:
            color_map[driver] = driver_color_map[driver.lower()]
        elif driver.lower().replace(" ", "_") in driver_color_map:
            color_map[driver] = driver_color_map[driver.lower().replace(" ", "_")]
        else:
            color_map[driver] = "#5f6368"  # Default gray

    # --- Plot ---
    fig = px.bar(
        drivers_sum,
        x="driver", y="hectares",
        title=f"🔥 Deforestation Drivers — {country} (≥30% canopy)",
        text_auto=".2s",
        color="driver",
        color_discrete_map=color_map
    )
    
    fig.update_traces(
        marker_line_width=0, 
        opacity=0.85
    )
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Roboto, sans-serif", size=12, color="#202124"),
        xaxis=dict(
            title=dict(font=dict(size=14, color="#5f6368"), text="Driver Type"),
            gridcolor="#dadce0",
            gridwidth=1,
            showline=True,
            linecolor="#dadce0"
        ),
        yaxis=dict(
            title=dict(font=dict(size=14, color="#5f6368"), text="Tree Cover Loss (ha)"),
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
