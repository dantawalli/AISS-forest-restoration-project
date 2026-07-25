import json

from knowledge_engine.compiler.species_compiler import SpeciesCompiler


compiler = SpeciesCompiler()

metadata = compiler.compile(
    "knowledge/species/Restoration_species/long_term/Swietenia_macrophylla"
)

print(json.dumps(metadata, indent=2, ensure_ascii=False))