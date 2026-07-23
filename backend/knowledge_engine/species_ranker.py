class SpeciesRanker:
    """
    Produces the final ranked list of species.
    """

    def __init__(self):
        pass

    def rank(self, compatibility_results, top_n=10):

        ranked = sorted(
            compatibility_results,
            key=lambda species: (
                species["compatibility"],
                species["score"]
            ),
            reverse=True
        )

        # Remove duplicates by scientific name
        unique_species = []
        seen = set()

        for item in ranked:

            scientific_name = item["species"]["scientific_name"]

            if scientific_name in seen:
                continue

            seen.add(scientific_name)

            species = item["species"]

            # Preferred display name:
            # Spanish → English → Scientific

            display_name = (
                (species.get("common_names", {}).get("spanish") or [None])[0]
                or (species.get("common_names", {}).get("english") or [None])[0]
                or scientific_name
            )

            species["display_name"] = display_name

            unique_species.append(item)

            if len(unique_species) >= top_n:
                break
        print(unique_species[0]["species"].keys())
        return unique_species