from pathlib import Path
import json

from knowledge_engine.context_builder.species_context_builder import (
    SpeciesContextBuilder,
)
from knowledge_engine.reasoning_context_builder.species_reasoning_context_builder import (
    SpeciesReasoningContextBuilder,
)


class SpeciesCompiler:

    def __init__(self):
        self.context_builder = SpeciesContextBuilder()
        self.reasoning_builder = SpeciesReasoningContextBuilder()

    def compile(self, species_folder: str):
        species_path = Path(species_folder)

        metadata_path = species_path / "metadata.json"

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        context = self.context_builder.build(metadata)

        context_path = species_path / "context.json"

        with open(context_path, "w", encoding="utf-8") as f:
            json.dump(
                context,
                f,
                indent=2,
                ensure_ascii=False,
            )

        reasoning_context = self.reasoning_builder.build(context)

        reasoning_context_path = species_path / "reasoning_context.json"

        with open(reasoning_context_path, "w", encoding="utf-8") as f:
            json.dump(
                reasoning_context,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return reasoning_context