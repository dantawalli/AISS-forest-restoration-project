class SpeciesRanker:
    """
    Produces the final ranked list of species.

    This class is intentionally separated from the CompatibilityEngine.

    Responsibilities
    ----------------
    - Receive compatibility results
    - Sort species
    - Return the best candidates
    - Later apply ecological priorities
    - Later apply restoration objectives
    - Later apply AI weighting
    """

    def __init__(self):
        pass

    def rank(self, compatibility_results, top_n=10):
        """
        Returns the best ranked species.

        Parameters
        ----------
        compatibility_results : list

        top_n : int

        Returns
        -------
        list
        """

        ranked = sorted(
            compatibility_results,
            key=lambda species: (
                species["compatibility"],
                species["score"]
            ),
            reverse=True
        )

        return ranked[:top_n]