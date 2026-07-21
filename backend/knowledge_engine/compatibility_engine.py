from knowledge_engine.species_registry import SpeciesRegistry

class CompatibilityEngine:
    """
    Evaluates species compatibility with a site's environmental conditions.

    Version 2:
    - Dynamic scoring
    - Rainfall
    - Altitude
    - Temperature
    - Soil pH
    - Soil Texture

    Future criteria will be added incrementally:
    - Drainage
    - Flood tolerance
    - Drought tolerance
    - Slope
    - Ecosystem
    - Successional stage
    - Restoration objectives
    """

    POINTS_PER_CRITERION = 20

    def __init__(self, registry):
        self.registry = registry

    def score_species(self, site_conditions):
        """
        Score every species against the provided site conditions.
        """

        results = []

        for species in self.registry.get_all_species():

            metadata = species["metadata"]

            score = 0
            maximum_score = 0

            # ======================================================
            # Rainfall
            # ======================================================

            if "rainfall" in site_conditions:
                maximum_score += self.POINTS_PER_CRITERION

                score += self._score_rainfall(
                    metadata,
                    site_conditions
                )

            # ======================================================
            # Altitude
            # ======================================================

            if "altitude" in site_conditions:
                maximum_score += self.POINTS_PER_CRITERION

                score += self._score_altitude(
                    metadata,
                    site_conditions
                )

            # ======================================================
            # Temperature
            # ======================================================

            if "temperature" in site_conditions:
                maximum_score += self.POINTS_PER_CRITERION

                score += self._score_temperature(
                    metadata,
                    site_conditions
                )

            # ======================================================
            # Soil pH
            # ======================================================

            if "soil_ph" in site_conditions:
                maximum_score += self.POINTS_PER_CRITERION

                score += self._score_soil_ph(
                    metadata,
                    site_conditions
                )

            # ======================================================
            # Soil Texture
            # ======================================================

            if "soil_texture" in site_conditions:
                maximum_score += self.POINTS_PER_CRITERION

                score += self._score_soil_texture(
                    metadata,
                    site_conditions
                )

            # ======================================================
            # Final Compatibility
            # ======================================================

            compatibility = (
                round((score / maximum_score) * 100)
                if maximum_score > 0
                else 0
            )

            results.append({
                "scientific_name": species["scientific_name"],
                "score": score,
                "maximum_score": maximum_score,
                "compatibility": compatibility,
                "species": species
            })

        results.sort(
            key=lambda x: x["compatibility"],
            reverse=True
        )

        return results

    # ==========================================================
    # Rainfall
    # ==========================================================

    def _score_rainfall(self, metadata, site_conditions):

        rainfall = site_conditions.get("rainfall")

        if rainfall is None:
            return 0

        requirements = (
            metadata["ecological_requirements"]["rainfall"]["annual_mm"]
        )

        minimum = requirements["min"]
        maximum = requirements["max"]

        if minimum is None or maximum is None:
            return 0

        if minimum <= rainfall <= maximum:
            return self.POINTS_PER_CRITERION

        return 0

    # ==========================================================
    # Altitude
    # ==========================================================

    def _score_altitude(self, metadata, site_conditions):

        altitude = site_conditions.get("altitude")

        if altitude is None:
            return 0

        requirements = (
            metadata["ecological_requirements"]["altitude"]["meters"]
        )

        minimum = requirements["min"]
        maximum = requirements["max"]

        if minimum is None or maximum is None:
            return 0

        if minimum <= altitude <= maximum:
            return self.POINTS_PER_CRITERION

        return 0

    # ==========================================================
    # Temperature
    # ==========================================================

    def _score_temperature(self, metadata, site_conditions):

        temperature = site_conditions.get("temperature")

        if temperature is None:
            return 0

        requirements = (
            metadata["ecological_requirements"]["temperature"]["annual_mean_c"]
        )

        minimum = requirements["min"]
        maximum = requirements["max"]

        if minimum is None or maximum is None:
            return 0

        if minimum <= temperature <= maximum:
            return self.POINTS_PER_CRITERION

        return 0

    # ==========================================================
    # Soil pH
    # ==========================================================

    def _score_soil_ph(self, metadata, site_conditions):

        soil_ph = site_conditions.get("soil_ph")

        if soil_ph is None:
            return 0

        requirements = (
            metadata["ecological_requirements"]["soil"]["ph"]
        )

        minimum = requirements["min"]
        maximum = requirements["max"]

        if minimum is None or maximum is None:
            return 0

        if minimum <= soil_ph <= maximum:
            return self.POINTS_PER_CRITERION

        return 0

    # ==========================================================
    # Soil Texture
    # ==========================================================

    def _score_soil_texture(self, metadata, site_conditions):

        texture = site_conditions.get("soil_texture")

        if texture is None:
            return 0

        accepted = (
            metadata["ecological_requirements"]["soil"]["texture"]
        )

        if not accepted:
            return 0

        if texture.lower() in [t.lower() for t in accepted]:
            return self.POINTS_PER_CRITERION

        return 0