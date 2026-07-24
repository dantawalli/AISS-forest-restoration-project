from knowledge_engine.landscape_diagnosis import LandscapeDiagnosisEngine
from knowledge_engine.compatibility_engine import CompatibilityEngine
from knowledge_engine.species_ranker import SpeciesRanker
from knowledge_engine.recommendation_engine import RecommendationEngine
from knowledge_engine.knowledge_engine import KnowledgeEngine
from knowledge_engine.environmental_data_provider import EnvironmentalDataProvider
from restoration_brief.restoration_brief_builder import RestorationBriefBuilder
from knowledge_engine.restoration_llm_engine import RestorationLLMEngine

class RestorationEngine:

    def __init__(self, registry, api_key):

        self.environment = EnvironmentalDataProvider()
        self.landscape = LandscapeDiagnosisEngine()

        self.compatibility = CompatibilityEngine(registry)
        self.ranker = SpeciesRanker()
        self.recommendation = RecommendationEngine()

        self.knowledge = KnowledgeEngine(registry)
        self.brief = RestorationBriefBuilder()
        self.llm = RestorationLLMEngine(api_key)

    def analyze(self, site_conditions):

        # Enrich environmental variables
        site_conditions = self.environment.enrich(site_conditions)

        # Diagnose the landscape
        diagnosis = self.landscape.diagnose(site_conditions)

        # Evaluate species compatibility
        compatibility = self.compatibility.score_species(
            diagnosis
        )

        # Rank candidate species
        ranked_species = self.ranker.rank(
            compatibility
        )

        # Build restoration strategy
        recommendations = self.recommendation.generate(
            site_conditions=diagnosis,
            landscape_diagnosis=diagnosis,
            ranked_species=ranked_species
        )

        selected_species = []

        for species in recommendations["recommended_species"]:

            details = self.knowledge.species.get_species(
                species["metadata_path"]
            )

            profile = self.knowledge.species_profile.build(
                details
            )

            knowledge = self.knowledge.species_knowledge.build(
                profile=profile,
                landscape=diagnosis,
                strategy=recommendations["strategy"]
            )

            selected_species.append({
                "profile": profile,
                "knowledge": knowledge
            })

        recommendations["selected_species"] = selected_species

        restoration_brief = self.brief.build(
            diagnosis=diagnosis,
            species=selected_species,
            recommendations=recommendations
        )

        llm_output = self.llm.generate(
            diagnosis=diagnosis,
            recommendations=recommendations,
            species=selected_species,
            restoration_brief=restoration_brief,
        )

        return {
            "landscape_diagnosis": diagnosis,
            "recommended_species": recommendations["recommended_species"],
            "selected_species": selected_species,
            "recommendations": recommendations,
            "restoration_brief": restoration_brief,

            "impact_summary": llm_output.get("impact_summary", ""),
            "executive_summary": llm_output.get("executive_summary", ""),
            "farmer_guidance": llm_output.get("farmer_guidance", ""),
        }