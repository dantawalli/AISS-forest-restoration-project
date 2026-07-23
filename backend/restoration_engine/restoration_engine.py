from knowledge_engine.landscape_diagnosis import LandscapeDiagnosisEngine
from knowledge_engine.compatibility_engine import CompatibilityEngine
from knowledge_engine.species_ranker import SpeciesRanker
from knowledge_engine.recommendation_engine import RecommendationEngine
from knowledge_engine.knowledge_engine import KnowledgeEngine
from restoration_brief.restoration_brief_builder import RestorationBriefBuilder
from knowledge_engine.environmental_data_provider import EnvironmentalDataProvider


class RestorationEngine:

    def __init__(self, registry):

        self.landscape = LandscapeDiagnosisEngine()
        self.environment = EnvironmentalDataProvider()
        self.compatibility = CompatibilityEngine(registry)
        self.knowledge = KnowledgeEngine(registry)
        self.ranker = SpeciesRanker()
        self.recommendation = RecommendationEngine()
        self.brief = RestorationBriefBuilder()

    def analyze(self, site_conditions):

        # Diagnose the landscape
        site_conditions = self.environment.enrich(site_conditions)


        # Enrich environmental variables
        site_conditions = self.environment.enrich(site_conditions)

        # Diagnose the landscape
        diagnosis = self.landscape.diagnose(site_conditions)

        print("\n=== DIAGNOSIS ===")
        for key, value in diagnosis.items():
            print(f"{key}: {value}")
        print("=================\n")

        site_conditions = diagnosis

        # Evaluate species compatibility
        compatibility_results = self.compatibility.score_species(
            site_conditions
        )

        # Rank candidate species
        ranked_species = self.ranker.rank(
            compatibility_results
        )

        # Build the restoration project
        recommendations = self.recommendation.generate(
            site_conditions=site_conditions,
            landscape_diagnosis=diagnosis,
            ranked_species=ranked_species
        )

        # Build standardized species profiles
        recommendations["selected_species"] = []

        for species in recommendations["recommended_species"]:

            details = self.knowledge.species.get_species(
                species["metadata_path"]
            )

            profile = self.knowledge.species_profile.build(details)

            knowledge = self.knowledge.species_knowledge.build(
                profile=profile,
                landscape=diagnosis,
                strategy=recommendations["strategy"]
            )

            recommendations["selected_species"].append({
                "profile": profile,
                "knowledge": knowledge
            })

        brief = self.brief.build(
            diagnosis=diagnosis,
            species=recommendations["selected_species"],
            recommendations=recommendations
        )
        print("\n>>>>>>>> LEAVING RESTORATION ENGINE <<<<<<<<\n")
        return {
            "landscape_diagnosis": diagnosis,
            "recommended_species": recommendations["recommended_species"],
            "recommendations": recommendations,
            "restoration_brief": brief
        }