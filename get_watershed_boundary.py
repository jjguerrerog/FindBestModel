"""
Download the drainage basin (watershed) boundary for a USGS streamgage as a
GeoJSON polygon in WGS84 (EPSG:4326), via the USGS Hydro Network-Linked Data
Index (NLDI) API.

Companion to get_usgs_gauge_data.py: that module downloads the time series
for a gauge, this one downloads the polygon of the area draining to it.
"""

import json
import os

import requests

NLDI_BASE_URL = "https://api.water.usgs.gov/nldi/linked-data"


def get_watershed_boundary(usgs_gauge_id, simplified=True):
    """
    Fetches the drainage basin polygon for a USGS streamgage from the NLDI
    basin service. The service returns geometry in WGS84 (EPSG:4326), the
    coordinate system GeoJSON uses by default.

    Parameters:
        usgs_gauge_id (str): USGS site code (e.g., '02317620')
        simplified (bool): if True (default, matching the service's own
            default), returns a topologically simplified boundary with far
            fewer vertices -- enough for mapping/plotting. Set False for the
            full-resolution NHDPlus catchment boundary.

    Returns:
        dict: GeoJSON FeatureCollection (one Polygon/MultiPolygon feature)
            for the gauge's drainage basin, in WGS84.
    """
    url = f"{NLDI_BASE_URL}/nwissite/USGS-{usgs_gauge_id}/basin"
    params = {"simplified": str(simplified).lower()}

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    geojson = response.json()

    if not geojson.get("features"):
        raise ValueError(f"No basin boundary found for USGS site {usgs_gauge_id}")

    return geojson


def download_watershed_boundary(usgs_gauge_id, output_dir="data", simplified=True):
    """
    Downloads the drainage basin boundary for a USGS streamgage and saves it
    as a GeoJSON file in WGS84.

    Parameters:
        usgs_gauge_id (str): USGS site code (e.g., '02317620')
        output_dir (str): Directory the GeoJSON file is written to
        simplified (bool): see `get_watershed_boundary`

    Returns:
        dict: The GeoJSON FeatureCollection that was saved.
    """
    geojson = get_watershed_boundary(usgs_gauge_id, simplified=simplified)

    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/usgs_{usgs_gauge_id}_basin.geojson"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f)

    return geojson


if __name__ == "__main__":
    download_watershed_boundary("02250030")
