import json
import os

from knowledge_engine.context_builder.species_context_builder import SpeciesContextBuilder
from knowledge_engine.reasoning_context_builder.species_reasoning_context_builder import (
    SpeciesReasoningContextBuilder,
)


def test_species_reasoning_context_builder():

    with open(
            "knowledge/species/Restoration_species/long_term/Swietenia_macrophylla/metadata.json",
            "r",
            encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    context_builder = SpeciesContextBuilder()
    context = context_builder.build(metadata)

    reasoning_builder = SpeciesReasoningContextBuilder()
    reasoning = reasoning_builder.build(context)

    output_dir = "knowledge/species/Restoration_species/long_term/Swietenia_macrophylla"

    os.makedirs(output_dir, exist_ok=True)

    with open(
        os.path.join(output_dir, "context.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(context, f, indent=2, ensure_ascii=False)

    with open(
        os.path.join(output_dir, "reasoning_context.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(reasoning, f, indent=2, ensure_ascii=False)

    print(json.dumps(reasoning, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_species_reasoning_context_builder()