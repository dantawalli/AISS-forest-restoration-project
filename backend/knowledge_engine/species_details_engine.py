class SpeciesDetailsEngine:

    def __init__(self, registry):
        self.registry = registry

    def get_species(self, metadata_path):
        species = next(
            (
                s for s in self.registry.species
                if s["metadata_path"] == metadata_path
            ),
            None
        )

        if species is None:
            return None

        return {
            "scientific_name": species["scientific_name"],
            "family": species["family"],
            "life_form": species["life_form"],
            "category": species["category"],
            "duration": species["duration"],
            "metadata": species["metadata"],
            "metadata_version": "2.0",
            "knowledge_source": "FYNOS Species Knowledge Base",
        }