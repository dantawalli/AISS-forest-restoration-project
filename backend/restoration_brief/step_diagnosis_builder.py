class StepDiagnosisBuilder:
    def build(self, diagnosis):
        return {
            "title": "Landscape Diagnosis",
            "restoration_potential": diagnosis.get("restoration_potential"),
            "degradation_level": diagnosis.get("degradation_level"),
            "water_availability": diagnosis.get("water_availability"),
            "key_constraints": diagnosis.get("constraints", [])
        }