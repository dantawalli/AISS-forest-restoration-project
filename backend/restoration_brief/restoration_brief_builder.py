from restoration_brief.executive_summary_builder import ExecutiveSummaryBuilder
from restoration_brief.step_land_overview_builder import StepLandOverviewBuilder
from restoration_brief.step_diagnosis_builder import StepDiagnosisBuilder
from restoration_brief.step_priorities_builder import StepPrioritiesBuilder
from restoration_brief.step_strategy_builder import StepStrategyBuilder
from restoration_brief.step_species_builder import StepSpeciesBuilder
from restoration_brief.step_impact_builder import StepImpactBuilder


class RestorationBriefBuilder:

    def __init__(self):
        self.executive_summary = ExecutiveSummaryBuilder()
        self.land_overview = StepLandOverviewBuilder()
        self.diagnosis = StepDiagnosisBuilder()
        self.priorities = StepPrioritiesBuilder()
        self.strategy = StepStrategyBuilder()
        self.species = StepSpeciesBuilder()
        self.impact = StepImpactBuilder()

    def build(
            self,
            diagnosis,
            productive_species,
            restoration_species,
            recommendations,
    ):

        restoration_brief = {
            "executive_summary": self.executive_summary.build(diagnosis, recommendations),
            "landscape_overview": self.land_overview.build(diagnosis),
            "landscape_diagnosis": self.diagnosis.build(diagnosis),
            "restoration_priorities": self.priorities.build(diagnosis),
            "restoration_strategy": self.strategy.build(recommendations),
            "productive_species": {
                "title": "Recommended Productive Species",
                "species": productive_species,
            },

            "recommended_species": self.species.build(
                restoration_species
            ),
            "expected_impact": self.impact.build(diagnosis),
        }

        return restoration_brief