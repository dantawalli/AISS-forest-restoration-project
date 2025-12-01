from dash import html

def render_about_tab():
    """
    Returns a professional overview section describing the project,
    dataset, methodology, and SDG alignment.
    """
    return html.Div([
        html.H2("🌍 About This Project"),
        html.P("""
            This dashboard is part of the Erasmus Mundus Joint Master's in Artificial Intelligence
            for Sustainable Societies (AISS) — Introduction to Data Science course. 
            It visualizes forest change dynamics using data from Global Forest Watch (Hansen et al., 2013),
            integrating tree-cover loss, deforestation drivers, and carbon emissions between 2001 and 2024.
            The interactive web application is deployed on Render.com and provides real-time insights for 
            policymakers, researchers, and conservation organizations.
        """),

        html.H3("🎯 Project Objective"),
        html.P("""
            The main goal is to explore how forest degradation, primary forest loss, and carbon fluxes 
            evolve over time, and to identify which human or natural drivers contribute most to 
            deforestation. Machine-learning models are applied to predict future forest loss trends up to 2035,
            enabling proactive conservation planning and evidence-based policy decision-making.
        """),

        html.H3("📊 Data Sources"),
        html.Ul([
            html.Li("Global Forest Watch country-level datasets (2001–2024) covering 167 countries and territories."),
            html.Li("Four integrated datasets: Tree Cover Loss, Primary Forest Loss, Drivers of Deforestation, and Carbon Emissions."),
            html.Li("University of Maryland GLAD Laboratory & Google (Hansen et al., Science 2013)."),
            html.Li("Data processed and merged using Python (Pandas, Plotly, Scikit-Learn, XGBoost)."),
            html.Li("All visualizations filtered to 30% canopy threshold for consistency and comparability.")
        ]),
        html.P("Developed by Buhari Nasir Ahmad and Juan Aladino Valdiviezo Alegria(JAVA) — AISS EMJM Cohort 2025",
               style={"marginTop": "40px", "fontStyle": "italic", "color": "#555"})
    ], style={
        "padding": "40px",
        "maxWidth": "900px",
        "margin": "auto",
        "backgroundColor": "#F9F9F9",
        "borderRadius": "12px",
        "boxShadow": "0 0 10px rgba(0,0,0,0.1)"
    })
