class SpeciesReasoningMapper:

    def map_identity(self, context_data):
        return {
            "scientific_name": context_data.get("identity", {}).get("scientific_name", ""),
            "common_names": context_data.get("identity", {}).get("common_names", {}),
        }

    def map_capabilities(self, context_data):
        ecology = context_data.get("ecology", {})
        services = context_data.get("ecosystem_services", {})

        return {
            "ecological_functions": {
                "pollinator_support": ecology.get("pollinator_support"),
                "wildlife_support": ecology.get("wildlife_support"),
                "microclimate_regulation": ecology.get("microclimate_regulation"),
                "soil_improvement": ecology.get("soil_improvement"),
                "erosion_control": ecology.get("erosion_control"),
                "water_regulation": ecology.get("water_regulation"),
            },
            "ecosystem_services": services,
        }

    def map_constraints(self, context_data):
        environment = context_data.get("environment", {})
        risks = context_data.get("risks", {})

        return {
            "environmental_constraints": {
                "stress_tolerance": environment.get("stress_tolerance"),
            },
            "risks": risks,
        }

    def map_applications(self, context_data):
        restoration = context_data.get("restoration", {})

        return {
            "objectives": restoration.get("objectives", []),
            "recommended_roles": restoration.get("recommended_roles", []),
            "recommended_landscapes": restoration.get("recommended_landscapes", []),
            "restoration_phase": restoration.get("restoration_phase", ""),
        }

    def map_relationships(self, context_data):
        ecology = context_data.get("ecology", {})

        return {
            "forest_layer": ecology.get("forest_layer", ""),
            "canopy_position": ecology.get("canopy_position", ""),
            "successional_stage": ecology.get("successional_stage", ""),
        }

    def map_decision_rules(self, context_data):
        restoration = context_data.get("restoration", {})
        environment = context_data.get("environment", {})

        return {
            "recommended_for": restoration.get("recommended_landscapes", []),
            "restoration_phase": restoration.get("restoration_phase", ""),
            "light_requirements": environment.get("light_requirements", []),
            "water_requirements": environment.get("water_requirements", ""),
        }

    def map_operational_guidance(self, context_data):
        propagation = context_data.get("propagation", {})
        restoration = context_data.get("restoration", {})

        return {
            "seed_propagation": propagation.get("seed_propagation", {}),
            "nursery": propagation.get("nursery", {}),
            "vegetative_propagation": propagation.get("vegetative_propagation", {}),
            "silviculture": context_data.get("silviculture", {}),
            "summary": restoration.get("summary", ""),

        }

    def map_fynos(self, context_data):
        fynos = context_data.get("fynos", {})

        return {
            "priority": fynos.get("priority", ""),
            "recommended_for": fynos.get("recommended_for", []),
        }

    def map_provenance(self, context_data):
        reference = context_data.get("reference", {})

        return {
            "title": reference.get("title", ""),
            "authors": reference.get("authors", []),
            "institution": reference.get("institution", ""),
            "year": reference.get("year", ""),
        }
