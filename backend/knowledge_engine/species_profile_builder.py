class SpeciesProfileBuilder:

    def build(self, species):

        metadata = species["metadata"]

        return {
            "scientific_name": species["scientific_name"],
            "family": species["family"],
            "life_form": species["life_form"],

            "restoration_value":
                metadata["fynos_classification"]["restoration_value"],

            "agroforestry_value":
                metadata["fynos_classification"]["agroforestry_value"],

            "carbon_value":
                metadata["fynos_classification"]["carbon_value"],

            "nitrogen_fixer":
                metadata["fynos_classification"]["nitrogen_fixer"],

            "ecosystem_engineer":
                metadata["fynos_classification"]["ecosystem_engineer"],

            "recommended_for":
                metadata["fynos_classification"],

            "ecological_requirements":
                metadata["ecological_requirements"],

            "ecosystem_services":
                metadata["ecosystem_services"],

            "restoration_functions":
                metadata.get("restoration_functions"),

            "propagation":
                metadata["propagation"]
        }