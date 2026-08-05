from knowledge_engine.species_registry import SpeciesRegistry


class StepProductiveSpeciesBuilder:

    def __init__(self, species_root="knowledge/species"):

        self.registry = SpeciesRegistry(species_root)
        self.registry.load_species()

    def build(self, diagnosis, recommendations):

        print("StepProductiveSpeciesBuilder.build() called")

        productive_species = self.registry.get_species_by_category(
            "Productive_species"
        )

        print(f"Loaded {len(productive_species)} productive species.")

        return {
            "title": "Recommended Productive Species",
            "species": productive_species,
        }