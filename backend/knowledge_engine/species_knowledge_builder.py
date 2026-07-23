class SpeciesKnowledgeBuilder:

    def build(
            self,
            profile,
            landscape,
            strategy=None
    ):

        knowledge = {
            "scientific_name": profile["scientific_name"],
            "planting_role": None,
            "ecosystem_role": None,
            "priority": None,
            "expected_benefits": [],
            "recommended_actions": []
        }

        # Planting role
        if profile["nitrogen_fixer"]:
            knowledge["planting_role"] = "Nitrogen enrichment species"
            knowledge["expected_benefits"].append(
                "Improves soil fertility through biological nitrogen fixation"
            )
            knowledge["recommended_actions"].append(
                "Use together with slower-growing native canopy species"
            )

        # Ecosystem role
        if profile["ecosystem_engineer"]:
            knowledge["ecosystem_role"] = "Ecosystem engineer"
            knowledge["expected_benefits"].append(
                "Supports ecosystem recovery and biodiversity"
            )
            knowledge["recommended_actions"].append(
                "Prioritize planting during the first restoration phase"
            )

        # Restoration priority
        if profile["restoration_value"] == "Very High":
            knowledge["priority"] = "Highest priority for restoration"

        elif profile["restoration_value"] == "High":
            knowledge["priority"] = "High priority for restoration"

        # Carbon benefits
        if profile["carbon_value"] == "Very High":
            knowledge["expected_benefits"].append(
                "High long-term carbon sequestration"
            )

        elif profile["carbon_value"] == "High":
            knowledge["expected_benefits"].append(
                "Good carbon sequestration potential"
            )

        # Agroforestry value
        if profile["agroforestry_value"] == "Very High":
            knowledge["expected_benefits"].append(
                "Highly suitable for diversified agroforestry systems"
            )
            knowledge["recommended_actions"].append(
                "Integrate into productive restoration and mixed-species designs"
            )

        elif profile["agroforestry_value"] == "High":
            knowledge["expected_benefits"].append(
                "Suitable for agroforestry integration"
            )

        # Landscape-specific recommendations

        if landscape["erosion_risk"] == "High":
            knowledge["recommended_actions"].append(
                "Prioritize planting on erosion-prone slopes"
            )

        if landscape["vegetation_condition"] == "Degraded":
            knowledge["recommended_actions"].append(
                "Suitable for early restoration of degraded landscapes"
            )
        # Strategy-specific recommendations

        if strategy:

            restoration_type = strategy.get("restoration_type")

            if restoration_type == "Agroforestry Restoration":
                knowledge["expected_benefits"].append(
                    "Contributes to productive and economically resilient restoration systems"
                )

                knowledge["recommended_actions"].append(
                    "Combine with fruit, timber and soil-improving species to maximize agroforestry performance"
                )

        return knowledge