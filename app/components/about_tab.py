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

        html.H3("🌱 Sustainable Development Goals Alignment"),
        html.P("""
            This project directly supports:
        """),
        html.Ul([
            html.Li("🟢 SDG 15 — Life on Land: monitoring deforestation and forest restoration progress through comprehensive data visualization and trend analysis."),
            html.Li("🟢 SDG 13 — Climate Action: analyzing CO₂ emissions from forest loss to quantify climate impact and inform carbon mitigation strategies."),
        ]),

        html.H3("🧠 Methodology Summary"),
        html.P("""
            The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:
        """),
        html.Ul([
            html.Li("Data cleaning and integration of four country-level GFW datasets with domain-informed imputation strategies."),
            html.Li("Exploratory Data Analysis (EDA) using Plotly & Seaborn to identify patterns and relationships."),
            html.Li("Feature engineering: temporal features, cumulative metrics, and feature selection (reduced from 180 to 100 features)."),
            html.Li("Model development: evaluated five regression models (Linear, Ridge, Lasso, Random Forest, XGBoost) with hyperparameter tuning."),
            html.Li("Time-aware data splitting: training on 2001-2019, testing on 2020-2024 to prevent temporal data leakage."),
            html.Li("Deployed XGBoost model for predictions with feature extrapolation for future year forecasting."),
            html.Li("Interactive Dash application with six visualization tabs for comprehensive forest monitoring and analysis."),
        ]),

        html.H3("🚀 Deployment"),
        html.P("""
            The dashboard is deployed as a production web application:
        """),
        html.Ul([
            html.Li("Platform: Render.com (Web Service, Python 3.12.3, Free Tier)"),
            html.Li("Server: Gunicorn WSGI server with 2 workers and 2 threads per worker"),
            html.Li("Auto-deployment: Continuous integration from GitHub repository"),
            html.Li("HTTPS/SSL: Automatic SSL certificates for secure access"),
        ]),

        html.H3("📚 References"),
        html.Ul([
            html.Li("Hansen et al. (2013). High-Resolution Global Maps of 21st-Century Forest Cover Change. Science 342(6160): 850–853."),
            html.Li("Harris et al. (2021). Global Forest Watch Carbon Dataset Documentation (v2.0)."),
            html.Li("World Resources Institute (2024). Global Forest Watch Open Data Portal."),
        ]),

        html.P("Developed by Buhari Nasir Ahmad and Aladino — AISS EMJM Cohort 2025 · Lusófona University · Tallinn University · Tampere University",
               style={"marginTop": "40px", "fontStyle": "italic", "color": "#555"})
    ], style={
        "padding": "40px",
        "maxWidth": "900px",
        "margin": "auto",
        "backgroundColor": "#F9F9F9",
        "borderRadius": "12px",
        "boxShadow": "0 0 10px rgba(0,0,0,0.1)"
    })
