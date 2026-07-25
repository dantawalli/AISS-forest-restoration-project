from .base_reasoning_context_builder import BaseReasoningContextBuilder
from .species_reasoning_mapper import SpeciesReasoningMapper


class SpeciesReasoningContextBuilder(BaseReasoningContextBuilder):

    CONTEXT_VERSION = "1.0"

    def build(self, context_data: dict) -> dict:
        mapper = SpeciesReasoningMapper()

        return {
            "reasoning_context_version": self.CONTEXT_VERSION,

            "identity": mapper.map_identity(context_data),

            "capabilities": mapper.map_capabilities(context_data),

            "constraints": mapper.map_constraints(context_data),

            "applications": mapper.map_applications(context_data),

            "relationships": mapper.map_relationships(context_data),

            "decision_rules": mapper.map_decision_rules(context_data),

            "operational_guidance": mapper.map_operational_guidance(context_data),

            "fynos": mapper.map_fynos(context_data),

            "provenance": mapper.map_provenance(context_data),
        }