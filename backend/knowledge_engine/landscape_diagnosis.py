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

        diagnosis = {
            "vegetation_condition":
                self._vegetation_condition(site),

            "erosion_risk":
                self._erosion_risk(site),

            "water_availability":
                self._water_availability(site),

            "terrain":
                self._terrain(site),

            "restoration_priority":
                self._priority(site)
        }

        return diagnosis

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