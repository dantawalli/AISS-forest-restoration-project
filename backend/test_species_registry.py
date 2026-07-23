from knowledge_engine.species_registry import SpeciesRegistry

# Create registry
registry = SpeciesRegistry("knowledge/species")

# Load all metadata.json files
registry.load_species()

print("=" * 60)
print(f"Species loaded: {len(registry.get_all_species())}")
print("=" * 60)

# Print first five species
for species in registry.get_all_species()[:5]:

    print(
        species["scientific_name"],
        "|",
        species["category"],
        "|",
        species["duration"]
    )