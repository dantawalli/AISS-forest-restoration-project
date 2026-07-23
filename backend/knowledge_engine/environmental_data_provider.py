class EnvironmentalDataProvider:
    """
    Enriches site_conditions with environmental variables
    from external datasets.

    Current version:
    - Rule-based environmental enrichment

    Future:
    - WorldClim
    - SoilGrids
    - CHIRPS
    - Google Earth Engine
    """

    def enrich(self, site_conditions):

        site_conditions["rainfall"] = self._rainfall(site_conditions)
        site_conditions["temperature"] = self._temperature(site_conditions)
        site_conditions["soil_texture"] = self._soil_texture(site_conditions)
        site_conditions["soil_ph"] = self._soil_ph(site_conditions)

        return site_conditions

    def _rainfall(self, site):
        zone = site.get("climate_zone")

        rainfall = {
            "lowland": 2500,
            "central": 1800,
            "andean": 1200,
        }

        return rainfall.get(zone)

    def _temperature(self, site):
        zone = site.get("climate_zone")

        temperature = {
            "lowland": 26,
            "central": 22,
            "andean": 16,
        }

        return temperature.get(zone)

    def _soil_texture(self, site):
        ecosystem = site.get("ecosystem")

        textures = {
            "pastizales": "loam",
            "purma": "clay loam",
            "bosque_alto": "clay",
            "bajial": "silty clay",
        }

        return textures.get(ecosystem)

    def _soil_ph(self, site):
        ecosystem = site.get("ecosystem")

        ph = {
            "pastizales": 5.6,
            "purma": 5.3,
            "bosque_alto": 5.0,
            "bajial": 5.8,
        }

        return ph.get(ecosystem)