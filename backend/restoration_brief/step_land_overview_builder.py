class StepLandOverviewBuilder:

    def build(self, diagnosis):
        return {
            "title": "Landscape Overview",

            # Main cards
            "area": diagnosis.get("area_hectares"),
            "ecosystem": diagnosis.get("ecosystem"),
            "risk": diagnosis.get("erosion_risk"),
            "restoration_potential": diagnosis.get("restoration_potential"),

            # Supporting information
            "climate_zone": diagnosis.get("climate_zone"),
            "elevation_m": diagnosis.get("elevation"),
            "slope_percent": diagnosis.get("slope"),
            "soil_type": diagnosis.get("soil_type"),
        }