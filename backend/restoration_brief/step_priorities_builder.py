class StepPrioritiesBuilder:
    def build(self, diagnosis):
        priorities = diagnosis.get("priorities", [])

        return {
            "title": "Restoration Priorities",
            "priorities": priorities,
        }