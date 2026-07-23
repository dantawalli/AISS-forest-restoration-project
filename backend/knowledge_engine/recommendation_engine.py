from schemas.restoration_project_schema import RestorationProjectSchema


class RecommendationEngine:

    def __init__(self):
        pass

    def generate(
        self,
        site_conditions,
        landscape_diagnosis,
        ranked_species,
    ):

        project = RestorationProjectSchema.empty()

        project["project"] = {
            "name": "Prototype Restoration Project",
            "version": "0.1",
            "status": "Draft",
            "engine": "FYNOS Knowledge Engine",
        }

        project["site"] = site_conditions

        project["landscape"] = {
            "summary": (
                f"The site is classified as a {site_conditions['ecosystem']} ecosystem "
                f"with {landscape_diagnosis['restoration_potential'].lower()} restoration potential "
                f"and {landscape_diagnosis['erosion_risk'].lower()} erosion risk."
            )
        }

        project["strategy"] = {
            "restoration_type": "Agroforestry Restoration",
            "objective": "Restore ecosystem functionality",
            "approach": "Native species mixed planting",
            "status": "Prototype",
        }

        project["recommended_species"] = []

        for item in ranked_species[:5]:
            species = item["species"]

            species_output = species.copy()

            species_output["compatibility"] = item["compatibility"]
            species_output["score"] = item["score"]

            project["recommended_species"].append(species_output)

        project["implementation"] = {
            "summary": (
                "Begin with site preparation, establish native species, "
                "and progressively restore ecosystem structure through phased planting."
            )
        }

        project["monitoring"] = {
            "summary": (
                f"Monitor vegetation recovery monthly using NDVI "
                f"(current value: {site_conditions['ndvi']:.2f}) "
                "together with field observations."
            )
        }

        project["risks"] = {
            "erosion": landscape_diagnosis["erosion_risk"],
            "water": landscape_diagnosis["water_availability"],
        }

        project["next_actions"] = [
            "Validate species",
            "Generate restoration brief",
            "Estimate costs",
        ]

        return project