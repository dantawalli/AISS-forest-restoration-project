from dash import Dash, html, dcc, Input, Output
from app.components.summary_cards import render_summary_cards
from app.components.loss_trend import render_loss_trend
from app.components.drivers_chart import render_drivers_chart
from app.components.emissions_chart import render_emissions_chart
from app.components.prediction_box import render_prediction_box
from app.components.map_view import render_global_map
from app.data_utils.loader import load_data_model
from app.components.about_tab import render_about_tab

# --- Load data & model ---
df, model, latest_year, total_loss, total_emissions, unique_countries = load_data_model()

# --- Initialize App ---
app = Dash(__name__, title="AI for Sustainable Forest Restoration", suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    html.Div([
        html.H1("🌲 AI for Sustainable Forest Restoration Dashboard", className="text-center"),

        render_summary_cards(unique_countries, total_loss, total_emissions),

        html.Div([
            html.Div([
                html.H4("Select Country"),
                dcc.Dropdown(
                    id="country-dd",
                    options=[{"label": c, "value": c} for c in sorted(df["country"].dropna().unique())],
                    value="Brazil",
                    style={"width": "100%"}
                )
            ], className="filter-group"),

            html.Div([
                html.H4("Select Year Range"),
                dcc.RangeSlider(
                    id="year-slider",
                    min=int(df["year"].min()), max=int(df["year"].max()),
                    step=1, value=[2001, latest_year],
                    marks={y: str(y) for y in range(2001, latest_year + 1, 5)},
                    tooltip={"placement": "bottom"}
                )
            ], className="filter-group", style={"flex": "2", "minWidth": "400px"})
        ], className="filter-container"),

        dcc.Tabs(id="tabs", value="tab-map-static", children=[
            dcc.Tab(label="ℹ️ About & SDG Context", value="tab-about"),
            dcc.Tab(label="🌀 Animated Map (2001–2024)", value="tab-map-animated"),
            dcc.Tab(label="📉 Loss Trends", value="tab-loss"),
            dcc.Tab(label="🔥 Deforestation Drivers", value="tab-drivers"),
            dcc.Tab(label="🌬️ Carbon & Climate Impact", value="tab-carbon"),
            dcc.Tab(label="🤖 Predictions", value="tab-predict"),
        ]),

        html.Div(id="tab-content", className="tab-content-container")
    ], className="main-container")
])


# --- Callbacks ---
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("country-dd", "value"),
    Input("year-slider", "value")
)
def update_tabs(tab, country, year_range):
    dff = df[(df["country"] == country) & (df["year"].between(year_range[0], year_range[1]))]
    if tab == "tab-about":
        return render_about_tab()
    elif tab == "tab-map-animated":
        return render_global_map(df, animated=True)
    elif tab == "tab-loss":
        return render_loss_trend(dff, country)
    elif tab == "tab-drivers":
        return render_drivers_chart(dff, country)
    elif tab == "tab-carbon":
        return render_emissions_chart(dff, country)
    elif tab == "tab-predict":
        return render_prediction_box()
    else:
        return html.P("Select a tab to view data.")


if __name__ == "__main__":
    app.run(debug=True)
