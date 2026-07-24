from knowledge_engine.context_builder.species_context_builder import SpeciesContextBuilder


sample_metadata = {
    "species_profile": {
        "scientific_name": "Cedrelinga cateniformis",
        "common_names": {
            "english": ["Tornillo"]
        }
    }
}

builder = SpeciesContextBuilder()

context = builder.build(sample_metadata)

print(context)