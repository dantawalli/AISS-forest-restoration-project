from dash import dcc
import plotly.express as px
import pandas as pd

def render_global_map(df, animated=False):
    """
    Generates a world map showing tree-cover loss per country.
    If animated=True, includes a year slider to visualize temporal changes.
    """

    # Ensure numeric
    df = df.copy()
    df["tree_cover_loss_ha"] = pd.to_numeric(df["tree_cover_loss_ha"], errors="coerce")

    if animated:
        # Aggregate by country + year for time-lapse
        country_year_loss = (
            df.groupby(["country", "year"], as_index=False)["tree_cover_loss_ha"]
              .sum()
              .dropna()
        )

        fig = px.choropleth(
            country_year_loss,
            locations="country",
            locationmode="country names",
            color="tree_cover_loss_ha",
            animation_frame="year",
            color_continuous_scale="YlOrRd",
            title="🌎 Global Tree Cover Loss (Animated by Year)",
            labels={"tree_cover_loss_ha": "Tree Cover Loss (ha)"}
        )
        fig.update_layout(
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
            margin=dict(l=0, r=0, t=50, b=0),
            height=600,
            coloraxis_colorbar_title="ha lost"
        )

    else:
        # Static map (total loss)
        country_loss = (
            df.groupby("country", as_index=False)["tree_cover_loss_ha"]
              .sum()
              .dropna()
        )

        fig = px.choropleth(
            country_loss,
            locations="country",
            locationmode="country names",
            color="tree_cover_loss_ha",
            color_continuous_scale="YlOrRd",
            title="🌎 Global Tree Cover Loss by Country (2001–2024)",
            labels={"tree_cover_loss_ha": "Tree Cover Loss (ha)"}
        )
        fig.update_layout(
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
            margin=dict(l=0, r=0, t=50, b=0),
            height=600
        )

    return dcc.Graph(figure=fig)
