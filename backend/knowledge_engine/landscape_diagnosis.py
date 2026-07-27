class LandscapeDiagnosisEngine:
    """
    Converts raw environmental variables into
    an interpretable landscape diagnosis.

    This engine does NOT recommend species.

    It describes the current condition of
    the landscape so that downstream engines
    (Compatibility, Ranking, AI)
    can make better decisions.
    """

    def diagnose(self, site):

        site["vegetation_condition"] = self._vegetation_condition(site)
        site["erosion_risk"] = self._erosion_risk(site)
        site["water_availability"] = self._water_availability(site)
        site["terrain"] = self._terrain(site)
        site["restoration_priority"] = self._priority(site)
        site["restoration_potential"] = self._restoration_potential(site)
        site["degradation_level"] = self._degradation_level(site)
        site["constraints"] = self._constraints(site)
        site["priorities"] = self._priorities(site)
        site["expected_outcomes"] = self._expected_outcomes(site)
        site["confidence"] = self._confidence(site)

        return site

    # -------------------------------------

    def _vegetation_condition(self, site):

        ndvi = site.get("ndvi")

        if ndvi is None:
            return "Unknown"

        if ndvi < 0.20:
            return "Highly Degraded"

        if ndvi < 0.40:
            return "Degraded"

        if ndvi < 0.60:
            return "Recovering"

        return "Healthy"

    # -------------------------------------

    def _erosion_risk(self, site):

        slope = site.get("slope")

        if slope is None:
            return "Unknown"

        if slope >= 30:
            return "Very High"

        if slope >= 20:
            return "High"

        if slope >= 10:
            return "Moderate"

        return "Low"

    # -------------------------------------

    def _water_availability(self, site):

        ndwi = site.get("ndwi")

        if ndwi is None:
            return "Unknown"

        if ndwi < -0.2:
            return "Low"

        if ndwi < 0.1:
            return "Moderate"

        return "High"

    # -------------------------------------

    def _terrain(self, site):

        slope = site.get("slope")

        if slope is None:
            return "Unknown"

        if slope < 5:
            return "Flat"

        if slope < 15:
            return "Gentle"

        if slope < 30:
            return "Hilly"

        return "Steep"

    # -------------------------------------

    def _priority(self, site):

        vegetation = self._vegetation_condition(site)
        erosion = self._erosion_risk(site)

        if vegetation == "Highly Degraded":
            return "Very High"

        if erosion == "Very High":
            return "High"

        return "Medium"

    def _restoration_potential(self, site):

        ndvi = site.get("ndvi")
        ecosystem = site.get("ecosystem")

        if ndvi is None:
            return "Unknown"

        if ecosystem == "purma":
            if ndvi >= 0.45:
                return "High"
            return "Medium"

        if ecosystem == "pastizales":
            if ndvi >= 0.30:
                return "Medium"
            return "Low"

        if ecosystem == "bosque_alto":
            return "Very High"

        if ecosystem == "bajial":
            return "High"

        return "Medium"

    def _degradation_level(self, site):

        ndvi = site.get("ndvi")

        if ndvi is None:
            return "Unknown"

        if ndvi < 0.20:
            return "Severe"

        if ndvi < 0.40:
            return "High"

        if ndvi < 0.60:
            return "Moderate"

        if ndvi < 0.75:
            return "Low"

        return "Minimal"

    def _constraints(self, site):

        constraints = []

        if site.get("water_availability") == "Low":
            constraints.append("Seasonal water deficit")

        if site.get("erosion_risk") in ["High", "Very High"]:
            constraints.append("High erosion risk")

        if site.get("vegetation_condition") == "Highly Degraded":
            constraints.append("Severe vegetation loss")

        if site.get("terrain") == "Steep":
            constraints.append("Difficult terrain")

        return constraints

    def _priorities(self, site):

        priorities = []

        if site.get("vegetation_condition") in ["Highly Degraded", "Degraded"]:
            priorities.append({
                "title": "Restore vegetation cover",
                "reason": "FYNOS AI detected that vegetation cover has been significantly reduced, limiting the landscape's ability to recover naturally.",
                "description": "Reintroduce native vegetation to improve ecosystem resilience, stabilize the landscape, and create suitable conditions for long-term forest recovery.",
            })

        if site.get("erosion_risk") in ["High", "Very High"]:
            priorities.append({
                "title": "Stabilize soil",
                "reason": "FYNOS AI detected a high risk of soil erosion, which could reduce restoration success if left unmanaged.",
                "description": "Protect exposed soil using vegetation, ground cover, or erosion control practices before establishing new plantings.",
            })

        if site.get("water_availability") == "Low":
            priorities.append({
                "title": "Improve water retention",
                "reason": "FYNOS AI detected limited water availability across your landscape, which could reduce the success of restoration if not addressed early.",
                "description": "Implement practices that improve soil moisture retention and reduce water stress before planting.",
            })

        if site.get("ecosystem") == "purma":
            priorities.append({
                "title": "Accelerate secondary forest succession",
                "reason": "FYNOS AI identified your landscape as secondary forest (purma), where supporting natural regeneration is often the fastest and most effective restoration approach.",
                "description": "Protect existing natural regeneration and enrich the area with native species where needed to accelerate forest recovery.",
            })

        if not priorities:
            priorities.append({
                "title": "Maintain ecosystem resilience",
                "reason": "FYNOS AI did not identify any immediate ecological constraints that require urgent intervention.",
                "description": "Continue protecting the existing ecosystem while monitoring changes and strengthening its long-term resilience through sustainable management.",
            })

        return priorities

    def _expected_outcomes(self, site):

        outcomes = []

        if site.get("restoration_potential") in ["High", "Very High"]:
            outcomes.append("Increase biodiversity")

        if site.get("vegetation_condition") != "Healthy":
            outcomes.append("Improve canopy cover")

        if site.get("erosion_risk") != "Low":
            outcomes.append("Reduce soil erosion")

        if site.get("water_availability") == "Low":
            outcomes.append("Increase soil moisture retention")

        outcomes.append("Increase carbon sequestration")

        return outcomes

    def _confidence(self, site):

        score = 0.5

        if site.get("ndvi") is not None:
            score += 0.1

        if site.get("ndwi") is not None:
            score += 0.1

        if site.get("elevation") is not None:
            score += 0.1

        if site.get("slope") is not None:
            score += 0.1

        if site.get("ecosystem") is not None:
            score += 0.1

        return round(min(score, 1.0), 2)