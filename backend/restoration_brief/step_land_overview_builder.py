class StepLandOverviewBuilder:

    def build(self, diagnosis):
        return {
            "title": "Landscape Overview",

            # Main cards
            "area": {
                "value": diagnosis.get("area_hectares"),
                "unit": "ha",
                "title": "Project Area",
                "description": (
                    f"The selected restoration site covers "
                    f"{diagnosis.get('area_hectares')} hectares."
                ),
            },
            "ecosystem": {
                "value": diagnosis.get("ecosystem"),
                "title": "Current Ecosystem",
                "description": (
                    "The site is currently classified as "
                    f"{diagnosis.get('ecosystem')}, indicating the present ecological condition detected from satellite analysis."
                ),
            },
            "risk": {
                "value": diagnosis.get("degradation_level"),
                "title": "Restoration Risk",
                "description": (
                    "The site presents "
                    f"{str(diagnosis.get('degradation_level', '')).lower()} "
                    "restoration challenges. Appropriate species selection and a phased "
                    "implementation strategy can significantly improve long-term success."
                ),
                "score": diagnosis.get("risk_score"),
                "status": str(diagnosis.get("degradation_level", "")).lower(),
            },
            "restoration_potential": {
                "value": diagnosis.get("restoration_potential"),
                "title": "Restoration Potential",
                "description": (
                    "The environmental conditions indicate "
                    f"{str(diagnosis.get('restoration_potential', '')).lower()} "
                    "potential for successful restoration with appropriate ecological planning."
                ),
                "score": None,
                "status": str(diagnosis.get("restoration_potential", "")).lower(),
            },

            # Supporting information
            "climate_zone": diagnosis.get("climate_zone"),
            "elevation_m": diagnosis.get("elevation"),
            "slope_percent": diagnosis.get("slope"),
            "soil_type": diagnosis.get("soil_type"),
        }