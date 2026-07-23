from knowledge_engine.species_registry import SpeciesRegistry
from backend.restoration_engine.restoration_engine import RestorationEngine


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
        "slope": 18,
    }

    # Run restoration analysis
    result = engine.analyze(site)

    diagnosis = result["landscape_diagnosis"]
    ranked_species = result["recommended_species"]
    recommendations = result["recommendations"]
    restoration_brief = result["restoration_brief"]

    print(f"\nSpecies evaluated: {len(ranked_species)}\n")

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

    print("\nRecommendation Object")
    print("----------------------")

    print()

    for key, value in recommendations.items():
        print(f"{key}: {value}")

        print("\n")
        print("=" * 70)
        print("AI RESTORATION BRIEF")
        print("=" * 70)

        print(f"\nTitle: {restoration_brief['title']}")
        print(f"Version: {restoration_brief['version']}")

        print("\nEXECUTIVE SUMMARY")
        print("-" * 70)

        print(f"Headline: {restoration_brief['executive_summary']['headline']}")
        print(f"Summary : {restoration_brief['executive_summary']['summary']}")


if __name__ == "__main__":
    main()