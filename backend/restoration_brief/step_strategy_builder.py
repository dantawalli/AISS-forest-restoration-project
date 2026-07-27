class StepStrategyBuilder:

    def build(self, recommendations):

        strategy = recommendations.get("strategy", {})
        landscape = recommendations.get("landscape", {})

        return {
            "title": "AI Restoration Strategy",

            "summary": landscape.get("summary", ""),

            "sections": [
                {
                    "title": "Landscape Assessment",
                    "description": landscape.get("summary", ""),
                    "status": "completed",
                },
                {
                    "title": "Restoration Strategy",
                    "description": strategy.get("restoration_type", ""),
                    "status": "ready",
                },
                {
                    "title": "Restoration Objective",
                    "description": strategy.get("objective", ""),
                    "status": "planned",
                },
                {
                    "title": "Recommended Approach",
                    "description": strategy.get("approach", ""),
                    "status": "recommended",
                },
            ],
        }