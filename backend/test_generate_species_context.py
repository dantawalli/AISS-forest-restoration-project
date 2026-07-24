import json
from pathlib import Path

from knowledge_engine.context_builder.species_context_builder import SpeciesContextBuilder


# Update this path to one of your real species metadata.json files
metadata_path = Path("knowledge/species/Restoration_species/long_term/Swietenia_macrophylla/metadata.json")

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

builder = SpeciesContextBuilder()

context = builder.build(metadata)

output_path = metadata_path.parent / "context.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(context, f, indent=2, ensure_ascii=False)

print(f"✅ Context generated: {output_path}")