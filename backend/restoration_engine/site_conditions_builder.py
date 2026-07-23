from shapely.geometry import shape


class SiteConditionsBuilder:

    def build(
        self,
        geometry,
        ndvi,
        ndwi,
        elevation,
        slope,
    ):

        ecosystem = self._ecosystem(ndvi, ndwi)
        climate_zone = self._climate_zone(elevation)

        area_hectares = self._calculate_area_hectares(geometry)

        return {
            "geometry": geometry,

            # Spatial metadata
            "area_hectares": area_hectares,

            # Remote sensing
            "ndvi": ndvi,
            "ndwi": ndwi,

            # Terrain
            "elevation": elevation,
            "slope": slope,
            "altitude": elevation,
            "slope_class": self._slope_class(slope),

            # Derived by SiteConditionsBuilder
            "ecosystem": ecosystem,
            "climate_zone": climate_zone,
            "soil_type": self._soil_type(ecosystem),
            "water_presence": self._water_presence(ndwi),

            # Filled later by LandscapeDiagnosisEngine
            "vegetation_condition": None,
            "erosion_risk": None,
            "water_availability": None,
            "terrain": None,
            "restoration_priority": None,

            # Filled later by RecommendationEngine
            "restoration_potential": None,
            "degradation_level": None,
            "constraints": [],
            "priorities": [],
            "expected_outcomes": [],
            "confidence": None,
        }

    def _calculate_area_hectares(self, geometry):
        """
        Calculates polygon area in hectares.

        NOTE:
        Shapely computes area in coordinate units. Since our polygons are
        currently received in WGS84 (lat/lon), this is an approximation.

        Later we will replace this with a proper projected-area calculation
        using Earth Engine or pyproj.
        """

        if geometry is None:
            return None

        try:
            polygon = shape(geometry)

            # Approximate conversion:
            # 1 degree² ≈ 12,321 km² near the equator
            area_deg2 = polygon.area
            area_m2 = area_deg2 * 12321000000
            area_ha = area_m2 / 10000

            return round(area_ha, 2)

        except Exception:
            return None

    def _ecosystem(self, ndvi, ndwi):
        if ndwi is not None and ndwi > 0.2:
            return "bajial"

        if ndvi is None:
            return "unknown"

        if ndvi < 0.35:
            return "pastizales"

        if ndvi < 0.60:
            return "purma"

        return "bosque_alto"

    def _climate_zone(self, elevation):
        if elevation is None:
            return "unknown"

        if elevation < 400:
            return "lowland"

        if elevation < 800:
            return "central"

        return "andean"

    def _soil_type(self, ecosystem):
        soils = {
            "bajial": "Hydromorphic",
            "pastizales": "Degraded agricultural soil",
            "purma": "Secondary forest soil",
            "bosque_alto": "Mature forest soil",
        }

        return soils.get(ecosystem, "Unknown")

    def _slope_class(self, slope):
        if slope is None:
            return "unknown"

        if slope < 5:
            return "flat"

        if slope < 15:
            return "moderate"

        if slope < 30:
            return "steep"

        return "very_steep"

    def _water_presence(self, ndwi):
        return ndwi is not None and ndwi > 0.2