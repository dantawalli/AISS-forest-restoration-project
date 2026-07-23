class StepImpactBuilder:
    def build(self, diagnosis):
        return {
            "title": "Expected Restoration Impact",
            "expected_outcomes": diagnosis.get("expected_outcomes", []),
            "confidence": diagnosis.get("confidence", "Medium")
        }