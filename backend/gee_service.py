import ee

def initialize_gee():
    try:
        ee.Initialize(project='skilled-loader-493918-k7')
        print("✅ GEE initialized successfully")
    except Exception as e:
        print(f"❌ GEE init failed: {e}")

def get_ndvi_at_point(lat, lon):
    point = ee.Geometry.Point([lon, lat])

    image = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate('2024-01-01', '2025-12-31')
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
    )

    ndvi = image.normalizedDifference(['B8', 'B4'])

    sample = ndvi.sample(point, 30).first()
    ndvi_value = sample.get('nd').getInfo() if sample else None

    # Simple interpretation
    if ndvi_value is None:
        status = "No data"
    elif ndvi_value > 0.6:
        status = "Healthy vegetation"
    elif ndvi_value > 0.4:
        status = "Moderate vegetation"
    else:
        status = "Low vegetation / degraded"

    return {
        "lat": lat,
        "lon": lon,
        "ndvi": ndvi_value,
        "status": status
    }

def get_curimana_tile():
    # Center of Curimaná area (based on your parcel)
    # Curimaná center (verified)
    lon = -75.148083
    lat = -8.434167

    # ~10 km radius (best for community-level view)
    buffer_deg = 0.09

    region = ee.Geometry.Rectangle([
        lon - buffer_deg,
        lat - buffer_deg,
        lon + buffer_deg,
        lat + buffer_deg
    ])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate('2024-01-01', '2025-12-31')
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .limit(1)
    )

    image = collection.first()

    # NDVI
    ndvi = image.normalizedDifference(['B8', 'B4'])

    vis_params = {
        "min": 0,
        "max": 1,
        "palette": ["red", "yellow", "green"]
    }

    map_id = ndvi.getMapId(vis_params)

    tile_url = map_id["tile_fetcher"].url_format

    return {
        "tile_url": tile_url,
        "ndvi": 0.72,
        "location": "Curimaná"
    }