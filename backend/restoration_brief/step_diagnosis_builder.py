from .scoring_engine import qualitative_score

class StepDiagnosisBuilder:
    def build(self, diagnosis):
        return {
            "title": "Landscape Diagnosis",

            "landscape_health": {
                "value": diagnosis.get("degradation_level"),
                "title": "Landscape Health",
                "description": (
                    "Your land is naturally recovering and retains good ecological resilience. "
                    "Restoration is highly feasible, although some environmental limitations "
                    "should be addressed to maximize long-term success."
                ),
                "score": qualitative_score(
                    diagnosis.get("degradation_level")
                ),
                "status": str(diagnosis.get("degradation_level", "")).lower(),
            },

            "restoration_potential": {
                "value": diagnosis.get("restoration_potential"),
                "title": "Restoration Potential",
                "description": (
                    "Excellent ecological capacity for successful restoration."
                ),
                "score": qualitative_score(
                    diagnosis.get("restoration_potential")
                ),
                "status": str(diagnosis.get("restoration_potential", "")).lower(),
            },

            "water_availability": {
                "value": diagnosis.get("water_availability"),
                "title": "Water Availability",
                "description": (
                    "Seasonal water deficit may slow establishment of vegetation."
                ),
                "score": qualitative_score(
                    diagnosis.get("water_availability")
                ),
                "status": str(diagnosis.get("water_availability", "")).lower(),
            },

            "key_constraints": diagnosis.get("constraints", [])
        }