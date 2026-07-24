from .base_context_builder import BaseContextBuilder
from .species_metadata_mapper import SpeciesMetadataMapper


class SpeciesContextBuilder(BaseContextBuilder):
    """
    Builds the standardized AI Context Object for a Species Knowledge Object.

    Responsibilities
    ----------------
    - Orchestrate the construction of a Species Context Object.
    - Delegate all metadata interpretation to SpeciesMetadataMapper.
    - Return a standardized AI-ready context following the
      FYNOS AI Context Specification v1.0.
    """

    def build(self, source_data: dict) -> dict:
        """
        Build the standardized Species Context Object.

        Parameters
        ----------
        source_data : dict
            Species metadata.json (Source of Truth).

        Returns
        -------
        dict
            Standardized Species Context Object.
        """

        mapper = SpeciesMetadataMapper()

        return {
            "context_version": self.CONTEXT_VERSION,

            "identity": mapper.map_identity(source_data),

            "ecology": mapper.map_ecology(source_data),

            "environment": mapper.map_environment(source_data),

            "restoration": mapper.map_restoration(source_data),

            "ecosystem_services": mapper.map_ecosystem_services(source_data),

            "propagation": mapper.map_propagation(source_data),

            "risks": mapper.map_risks(source_data),

            "fynos": mapper.map_fynos(source_data),

            "reference": mapper.map_reference(source_data),

        }