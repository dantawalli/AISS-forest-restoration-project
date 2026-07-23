from knowledge_engine.species_details_engine import SpeciesDetailsEngine
from knowledge_engine.species_profile_builder import SpeciesProfileBuilder
from knowledge_engine.species_knowledge_builder import SpeciesKnowledgeBuilder

class KnowledgeEngine:

    def __init__(self, registry):

        self.registry = registry

        # Knowledge Modules
        self.species = SpeciesDetailsEngine(registry)
        self.species_profile = SpeciesProfileBuilder()
        self.species_knowledge = SpeciesKnowledgeBuilder()

        # Future modules
        self.ethnobotany = None
        self.restoration = None
        self.soils = None
        self.climate = None
        self.carbon = None
        self.publications = None