class ExecutiveSummaryBuilder:
    def build(self, diagnosis, recommendations):
        return {
            "title": "Executive Summary",
            "summary": (
                "This restoration assessment provides an AI-generated overview "
                "of the landscape condition, restoration opportunities, and "
                "recommended ecological interventions."
            )
        }