from knowledge_engine.landscape_diagnosis import LandscapeDiagnosisEngine
from knowledge_engine.compatibility_engine import CompatibilityEngine
from knowledge_engine.species_ranker import SpeciesRanker


class RestorationEngine:
    """
    Main orchestration engine for FYNOS AI.

    This class coordinates the complete restoration
    intelligence workflow.

    Pipeline:

        Site Conditions
              ↓
        Landscape Diagnosis
              ↓
        Compatibility Analysis
              ↓
        Species Ranking
              ↓
        (Future) LLM Recommendation
              ↓
        Restoration Brief
    """

    def __init__(self, registry):

        self.landscape = LandscapeDiagnosisEngine()
        self.compatibility = CompatibilityEngine(registry)
        self.ranker = SpeciesRanker()

    def __init__(self, registry):

        self.landscape = LandscapeDiagnosisEngine()
        self.compatibility = CompatibilityEngine(registry)
        self.ranker = SpeciesRanker()

    def analyze(self, site_conditions):

        # Step 1
        diagnosis = self.landscape.diagnose(site_conditions)

        # Step 2
        compatibility_results = self.compatibility.score_species(
            site_conditions
        )

        # Step 3
        ranked_species = self.ranker.rank(
            compatibility_results
        )

        return {
            "landscape_diagnosis": diagnosis,
            "recommended_species": ranked_species
        }