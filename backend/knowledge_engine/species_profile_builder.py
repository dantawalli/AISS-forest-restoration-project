class SpeciesProfileBuilder:

    def build(self, species):
        metadata = species.get("metadata", {})
        species_profile = metadata.get("species_profile", {})

        return {
            # ================= BASIC INFORMATION =================
            "scientific_name": species.get("scientific_name"),
            "family": species.get("family"),
            "life_form": species.get("life_form"),
            "life_cycle": species_profile.get("life_cycle"),
            "native": species_profile.get("native"),
            "common_names": species_profile.get("common_names", {}),

            # ================= ECOLOGY =================
            "ecological_requirements": metadata.get(
                "ecological_requirements", {}
            ),
            "ecological_functions": metadata.get(
                "ecological_functions", {}
            ),

            # ================= SILVICULTURE =================
            "silviculture": metadata.get("silviculture", {}),

            # ================= PROPAGATION =================
            "propagation": metadata.get("propagation", {}),

            # ================= PRODUCTS =================
            "products": metadata.get("products", {}),

            # ================= FYNOS INTELLIGENCE =================
            "fynos_classification": metadata.get(
                "fynos_classification", {}
            ),

            # ================= ECOSYSTEM =================
            "ecosystem_services": metadata.get(
                "ecosystem_services", {}
            ),
            "restoration_functions": metadata.get(
                "restoration_functions", {}
            ),

            # ================= REFERENCES =================
            "sources": metadata.get("sources", []),
        }