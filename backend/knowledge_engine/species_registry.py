import json
from pathlib import Path


class SpeciesRegistry:
    """
    Loads every species metadata.json from the knowledge base.

    The registry becomes the single source of truth for all species
    available inside FYNOS AI.
    """

    def __init__(self, species_root: str):
        self.species_root = Path(species_root)
        self.species = []

    def load_species(self):
        """
        Scan recursively for every metadata.json file.
        """

        self.species = []

        for metadata_file in self.species_root.rglob("metadata.json"):

            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                relative = metadata_file.relative_to(self.species_root)

                parts = relative.parts

                # Example:
                # Restoration_species / long_term / Swietenia_macrophylla

                category = parts[0] if len(parts) > 0 else "unknown"
                duration = parts[1] if len(parts) > 1 else "unknown"
                folder = parts[2] if len(parts) > 2 else metadata_file.parent.name

                metadata["_category"] = category
                metadata["_duration"] = duration
                metadata["_folder"] = folder
                metadata["_metadata_path"] = str(metadata_file)

                species_data = {
                    "scientific_name": metadata["species_profile"]["scientific_name"],

                    # Common names
                    "common_names": metadata["species_profile"].get("common_names", {}),

                    "family": metadata["species_profile"]["family"],
                    "life_form": metadata["species_profile"]["life_form"],
                    "growth_form": metadata["species_profile"]["growth_form"],

                    "category": metadata["_category"],
                    "duration": metadata["_duration"],

                    "folder": metadata["_folder"],
                    "metadata_path": metadata["_metadata_path"],

                    "metadata": metadata,
                }

                self.species.append(species_data)

            except Exception as e:
                print(f"Could not load {metadata_file}: {e}")

        print(f"Loaded {len(self.species)} species.")

        return self.species

    def get_all_species(self):
        return self.species

    def get_species_by_category(self, category):

        return [
            s for s in self.species
            if s["_category"].lower() == category.lower()
        ]

    def get_species_by_duration(self, duration):

        return [
            s for s in self.species
            if s["_duration"].lower() == duration.lower()
        ]

    def find_species(self, scientific_name):

        scientific_name = scientific_name.lower()

        for species in self.species:

            if species.get("scientific_name", "").lower() == scientific_name:
                return species

        return None