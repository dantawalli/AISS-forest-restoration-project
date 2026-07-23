from knowledge_engine.species_registry import SpeciesRegistry


REQUIRED_SECTIONS = [
    "species_profile",
    "taxonomy",
    "distribution",
    "source_information",
    "fynos_classification",
    "ecological_requirements",
    "ecological_functions",
    "ecosystem_services",
    "restoration",
    "agroforestry",
    "silviculture",
    "propagation",
    "phenology",
    "species_interactions",
    "products",
    "threats",
    "quantitative_data",
    "remote_sensing_signatures",
    "ai_reasoning",
    "research",
    "sources",
    "quality_assessment",
]


def main():

    registry = SpeciesRegistry("knowledge/species")
    registry.load_species()

    print(f"\nChecking {len(registry.get_all_species())} species...\n")

    errors = 0

    for species in registry.get_all_species():

        metadata = species["metadata"]

        missing = [
            section
            for section in REQUIRED_SECTIONS
            if section not in metadata
        ]

        if missing:
            errors += 1
            print(f"❌ {species['scientific_name']}")
            print(f"   Missing: {missing}")

    if errors == 0:
        print("✅ All species follow the FYNOS Metadata Standard v2.0")
    else:
        print(f"\n{errors} species have missing sections.")


if __name__ == "__main__":
    main()