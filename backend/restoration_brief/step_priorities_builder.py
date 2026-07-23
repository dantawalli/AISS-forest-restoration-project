class StepPrioritiesBuilder:
    def build(self, diagnosis):
        return {
            "title": "Restoration Priorities",
            "priorities": diagnosis.get("priorities", [])
        }