class StepStrategyBuilder:
    def build(self, recommendations):
        return {
            "title": "Recommended Restoration Strategy",
            "strategy": recommendations
        }