from knowledge_engine.species_registry import SpeciesRegistry
from knowledge_engine.restoration_engine import RestorationEngine

def main():

    registry = SpeciesRegistry("knowledge/species")
    registry.load_species()

    engine = RestorationEngine(registry)

    site = {
        "rainfall": 2400,
        "altitude": 200,
        "temperature": 26,
        "soil_ph": 6.5,
        "soil_texture": "loam",

        # Google Earth Engine variables
        "ndvi": 0.22,
        "ndwi": -0.08,
        "slope": 18
    }

    # Evaluate compatibility
    result = engine.analyze(site)

    diagnosis = result["landscape_diagnosis"]

    ranked_species = result["recommended_species"]

    print(f"\nSpecies evaluated: {len(ranked_species)}\n")

    diagnosis = result["landscape_diagnosis"]

    print("Landscape Diagnosis")
    print("--------------------")

    for key, value in diagnosis.items():
        print(f"{key}: {value}")

    print()

    for species in ranked_species:
        print(
            f'{species["scientific_name"]} - '
            f'{species["compatibility"]}% '
            f'({species["score"]} pts)'
        )


if __name__ == "__main__":
    main()