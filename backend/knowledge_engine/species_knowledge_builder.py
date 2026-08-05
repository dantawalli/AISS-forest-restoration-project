class SpeciesKnowledgeBuilder:

    def build(
        self,
        profile,
        landscape,
        strategy=None,
    ):

        fynos = profile.get("fynos_classification", {})

        knowledge = {
            "scientific_name": profile.get("scientific_name"),
            "planting_role": None,
            "ecosystem_role": None,
            "priority": None,
            "expected_benefits": [],
            "recommended_actions": [],
            "why_selected": "",
        }

        # ==========================================================
        # FYNOS Classification
        # ==========================================================

        restoration_value = fynos.get("restoration_value")
        agroforestry_value = fynos.get("agroforestry_value")
        carbon_value = fynos.get("carbon_value")
        nitrogen_fixer = fynos.get("nitrogen_fixer")
        ecosystem_engineer = fynos.get("ecosystem_engineer")

        # ==========================================================
        # Planting Role
        # ==========================================================

        if nitrogen_fixer:
            knowledge["planting_role"] = "Nitrogen enrichment species"

            knowledge["expected_benefits"].append(
                "Improves soil fertility through biological nitrogen fixation"
            )

            knowledge["recommended_actions"].append(
                "Use together with slower-growing native canopy species"
            )

        # ==========================================================
        # Ecosystem Role
        # ==========================================================

        if ecosystem_engineer:
            knowledge["ecosystem_role"] = "Ecosystem engineer"

            knowledge["expected_benefits"].append(
                "Supports ecosystem recovery and biodiversity"
            )

            knowledge["recommended_actions"].append(
                "Prioritize planting during the first restoration phase"
            )

        # ==========================================================
        # Restoration Priority
        # ==========================================================

        if restoration_value == "Very High":
            knowledge["priority"] = "Highest priority for restoration"

        elif restoration_value == "High":
            knowledge["priority"] = "High priority for restoration"

        # ==========================================================
        # Carbon Benefits
        # ==========================================================

        if carbon_value == "Very High":
            knowledge["expected_benefits"].append(
                "High long-term carbon sequestration"
            )

        elif carbon_value == "High":
            knowledge["expected_benefits"].append(
                "Good carbon sequestration potential"
            )

        # ==========================================================
        # Agroforestry Benefits
        # ==========================================================

        if agroforestry_value == "Very High":

            knowledge["expected_benefits"].append(
                "Highly suitable for diversified agroforestry systems"
            )

            knowledge["recommended_actions"].append(
                "Integrate into productive restoration and mixed-species designs"
            )

        elif agroforestry_value == "High":

            knowledge["expected_benefits"].append(
                "Suitable for agroforestry integration"
            )

        # ==========================================================
        # Landscape-specific Recommendations
        # ==========================================================

        if landscape.get("erosion_risk") == "High":

            knowledge["recommended_actions"].append(
                "Prioritize planting on erosion-prone slopes"
            )

        if landscape.get("vegetation_condition") == "Degraded":

            knowledge["recommended_actions"].append(
                "Suitable for early restoration of degraded landscapes"
            )

        # ==========================================================
        # Strategy-specific Recommendations
        # ==========================================================

        if strategy:

            restoration_type = strategy.get("restoration_type")

            if restoration_type == "Agroforestry Restoration":

                knowledge["expected_benefits"].append(
                    "Contributes to productive and economically resilient restoration systems"
                )

                knowledge["recommended_actions"].append(
                    "Combine with fruit, timber and soil-improving species to maximize agroforestry performance"
                )

        # ==========================================================
        # Why Selected
        # ==========================================================

        reasons = []

        if knowledge["planting_role"]:
            reasons.append(knowledge["planting_role"].lower())

        if knowledge["ecosystem_role"]:
            reasons.append(knowledge["ecosystem_role"].lower())

        if restoration_value in ["Very High", "High"]:
            reasons.append(
                f"its {restoration_value.lower()} restoration value"
            )

        if agroforestry_value in ["Very High", "High"]:
            reasons.append(
                f"its {agroforestry_value.lower()} agroforestry value"
            )

        if reasons:

            if len(reasons) == 1:

                knowledge["why_selected"] = (
                    f"This species was recommended because of {reasons[0]}."
                )

            else:

                knowledge["why_selected"] = (
                    "This species was recommended because of "
                    + ", ".join(reasons[:-1])
                    + " and "
                    + reasons[-1]
                    + "."
                )

        return knowledge