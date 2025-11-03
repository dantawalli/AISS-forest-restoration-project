from dash import html

def render_about_tab():
    """
    Returns a professional overview section describing the project,
    dataset, methodology, and SDG alignment.
    """
    return html.Div([
        html.H2("🌍 About This Project"),
        html.P("""
            This dashboard is part of the Erasmus Mundus Joint Master’s in Artificial Intelligence
            for Sustainable Societies (AISS) — Introduction to Data Science course. 
            It visualizes forest change dynamics using data from Global Forest Watch (Hansen et al., 2013),
            integrating tree-cover loss, deforestation drivers, and carbon emissions between 2001 and 2024.
        """),

        html.H3("🎯 Project Objective"),
        html.P("""
            The main goal is to explore how forest degradation, primary forest loss, and carbon fluxes 
            evolve over time, and to identify which human or natural drivers contribute most to 
            deforestation. Machine-learning models are applied to predict future forest loss trends.
        """),

        html.H3("📊 Data Sources"),
        html.Ul([
            html.Li("Global Forest Watch country-level datasets (2001–2024)."),
            html.Li("Tree Cover Loss, Primary Forest Loss, Drivers of Deforestation, Carbon Emissions."),
            html.Li("University of Maryland GLAD Laboratory & Google (Hansen et al., Science 2013)."),
            html.Li("Data processed and merged using Python (Pandas, Plotly, Scikit-Learn).")
        ]),

        html.H3("🌱 Sustainable Development Goals Alignment"),
        html.P("""
            This project supports:
        """),
        html.Ul([
            html.Li("🟢 SDG 15 — Life on Land: monitoring deforestation and forest restoration progress."),
            html.Li("🟢 SDG 13 — Climate Action: analyzing CO₂ emissions from forest loss."),
        ]),

        html.H3("🧠 Methodology Summary"),
        html.Ul([
            html.Li("Data cleaning and integration of four country-level GFW datasets."),
            html.Li("Exploratory Data Analysis (EDA) using Plotly & Seaborn."),
            html.Li("Regression modeling (Random Forest) for loss prediction up to 2030."),
            html.Li("Dash App for interactive visualization and stakeholder insights."),
        ]),

        html.H3("📚 References"),
        html.Ul([
            html.Li("Hansen et al. (2013). High-Resolution Global Maps of 21st-Century Forest Cover Change. Science 342(6160): 850–853."),
            html.Li("Harris et al. (2021). Global Forest Watch Carbon Dataset Documentation (v2.0)."),
            html.Li("World Resources Institute (2024). Global Forest Watch Open Data Portal."),
        ]),

        html.P("Developed by Buhari Nasir Ahmad and Aladino — AISS EMJM Cohort 2025 · Lusófona University. Tallinn University · Tampere University",
               style={"marginTop": "40px", "fontStyle": "italic", "color": "#555"})
    ], style={
        "padding": "40px",
        "maxWidth": "900px",
        "margin": "auto",
        "backgroundColor": "#F9F9F9",
        "borderRadius": "12px",
        "boxShadow": "0 0 10px rgba(0,0,0,0.1)"
    })
