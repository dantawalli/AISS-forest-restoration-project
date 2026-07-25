class SpeciesMetadataMapper:
    """
    Maps a Species metadata.json (Source of Truth)
    into the standardized Species Context format.
    """

    def map_identity(self, metadata: dict) -> dict:
        profile = metadata.get("species_profile", {})

        return {
            "scientific_name": profile.get("scientific_name", ""),
            "common_names": profile.get("common_names", {}),
        }

    def map_ecology(self, metadata: dict) -> dict:
        ecological = metadata.get("ecological_functions", {})

        return {
            "successional_stage": ecological.get("successional_stage", ""),
            "forest_layer": ecological.get("forest_layer", ""),
            "canopy_position": ecological.get("canopy_position", ""),
            "pollinator_support": ecological.get("pollinator_support", ""),
            "wildlife_support": ecological.get("wildlife_support", ""),
            "microclimate_regulation": ecological.get("microclimate_regulation", ""),
            "soil_improvement": ecological.get("soil_improvement", ""),
            "erosion_control": ecological.get("erosion_control", ""),
            "water_regulation": ecological.get("water_regulation", ""),
        }

    def map_environment(self, metadata: dict) -> dict:
        ecological = metadata.get("ecological_requirements", {})
        soil = ecological.get("soil", {})

        temperature = ecological.get("temperature", {}).get("annual_mean_c", {})
        rainfall = ecological.get("rainfall", {}).get("annual_mm", {})
        elevation = ecological.get("altitude", {}).get("meters", {})
        ph = soil.get("ph", {})

        return {
            "temperature_c": [
                temperature.get("min"),
                temperature.get("max"),
            ],

            "rainfall_mm": [
                rainfall.get("min"),
                rainfall.get("max"),
            ],

            "elevation_m": [
                elevation.get("min"),
                elevation.get("max"),
            ],

            "climate_zones": ecological.get("climate_zones", []),

            "soil": {
                "preferred_types": soil.get("preferred_types", []),
                "texture": soil.get("texture", []),
                "drainage": soil.get("drainage", ""),
                "fertility": soil.get("fertility", ""),
                "ph": [
                    ph.get("min"),
                    ph.get("max"),
                ],
            },

            "light_requirements": ecological.get("light", {}),

            "water_requirements": ecological.get("hydrology", {}),

            "stress_tolerance": ecological.get("tolerances", {}),
        }

    def map_restoration(self, metadata: dict) -> dict:
        restoration = metadata.get("restoration", {})

        return {
            "objectives": restoration.get("restoration_objectives", []),
            "recommended_roles": restoration.get("recommended_roles", []),
            "restoration_phase": restoration.get("restoration_phase", []),
            "recommended_landscapes": restoration.get("recommended_land_types", []),
            "species_mixtures": restoration.get("recommended_species_mixtures", []),
            "summary": restoration.get("restoration_notes", ""),
        }

    def map_ecosystem_services(self, metadata: dict) -> dict:
        services = metadata.get("ecosystem_services", {})

        return {
            "provisioning": services.get("provisioning", []),
            "regulating": services.get("regulating", []),
            "supporting": services.get("supporting", []),
            "cultural": services.get("cultural", []),
        }

    def map_propagation(self, metadata: dict) -> dict:
        propagation = metadata.get("propagation", {})

        return {
            "seed_propagation": propagation.get("seed", {}),
            "nursery": propagation.get("nursery", {}),
            "vegetative_propagation": propagation.get("vegetative", {}),
        }

    def map_silviculture(self, source_data):

        silviculture = source_data.get("silviculture", {})

        return {
            "spacing": silviculture.get("spacing", []),
            "growth_rate": silviculture.get("growth_rate", ""),
            "rotation_years": silviculture.get("rotation_years"),
            "harvest_age_years": silviculture.get("harvest_age_years"),
            "planting_methods": silviculture.get("planting_methods", []),
            "pruning": silviculture.get("pruning", ""),
            "fertilization": silviculture.get("fertilization", ""),
            "maintenance": silviculture.get("maintenance", []),
        }

    def map_risks(self, metadata: dict) -> dict:
        threats = metadata.get("threats", {})

        return {
            "biotic_threats": threats.get("biotic", {}),
            "abiotic_threats": threats.get("abiotic", []),
            "human_threats": threats.get("human", []),
        }

    def map_fynos(self, metadata: dict) -> dict:
        fynos = metadata.get("fynos_classification", {})

        recommended_for = []

        if fynos.get("recommended_for_restoration"):
            recommended_for.append("restoration")

        if fynos.get("recommended_for_agroforestry"):
            recommended_for.append("agroforestry")

        if fynos.get("recommended_for_reforestation"):
            recommended_for.append("reforestation")

        if fynos.get("recommended_for_regeneration"):
            recommended_for.append("regeneration")

        return {
            "priority_level": fynos.get("priority_level", ""),
            "restoration_value": fynos.get("restoration_value", ""),
            "agroforestry_value": fynos.get("agroforestry_value", ""),
            "carbon_value": fynos.get("carbon_value", ""),
            "biodiversity_value": fynos.get("biodiversity_value", ""),
            "food_security_value": fynos.get("food_security_value", ""),
            "income_generation_value": fynos.get("income_generation_value", ""),
            "ecosystem_engineer": fynos.get("ecosystem_engineer", False),
            "foundation_species": fynos.get("foundation_species", False),
            "recommended_for": recommended_for,
        }

    def map_reference(self, metadata: dict) -> dict:
        source = metadata.get("source_information", {})

        return {
            "title": source.get("title", ""),
            "authors": source.get("authors", []),
            "institution": source.get("institution", ""),
            "year": source.get("year"),
            "language": source.get("language", ""),
        }