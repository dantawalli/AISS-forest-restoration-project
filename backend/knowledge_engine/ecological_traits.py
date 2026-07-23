class EcologicalTraitsEngine:
    """
    Scores ecological traits independently from environmental compatibility.

    These traits will later receive dynamic weights based on the
    Landscape Diagnosis Engine.
    """

    POINTS_PER_TRAIT = 20

    def score(self, metadata, site_conditions):

        score = 0

        score += self._score_nitrogen_fixation(metadata)

        return score

    # ==========================================================
    # Nitrogen Fixation
    # ==========================================================

    def _score_nitrogen_fixation(self, metadata):

        if (
            metadata["ecological_functions"]
            ["nitrogen_fixation"]
            ["capable"]
        ):
            return self.POINTS_PER_TRAIT

        return 0