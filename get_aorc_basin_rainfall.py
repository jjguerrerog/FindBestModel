"""
Compute the basin-averaged (areal average) hourly rainfall for a USGS
streamgage from NOAA's Analysis of Record for Calibration (AORC v1.1) and save
the time series as a parquet file.

The drainage basin polygon is downloaded with get_watershed_boundary.py (USGS
NLDI) and used as a "cookie cutter" over the AORC grid. AORC is read directly
from its public cloud-optimized Zarr store on AWS (s3://noaa-nws-aorc-v1-1-1km,
one store per year, anonymous access). Only the small window of the grid that
covers the basin's bounding box is read, block by block; each block is reduced
to one number per hour in memory and discarded, so no AORC rasters are ever
written to disk.

Usage:
    python get_aorc_basin_rainfall.py                 # prompts for everything
    python get_aorc_basin_rainfall.py 02250030 2020-01-01 2020-12-31
"""

import os
import sys

import numpy as np
import pandas as pd
import s3fs
import shapely
import xarray as xr

from get_watershed_boundary import get_watershed_boundary

AORC_BUCKET = "noaa-nws-aorc-v1-1-1km"
AORC_FIRST_TIME = pd.Timestamp("1979-02-01 00:00")
AORC_RES_DEG = 1 / 120  # AORC grid spacing (~800 m)
PRECIP_VAR = "APCP_surface"  # kg/m^2 per hour == mm per hour

# Hours read per request. AORC's Zarr chunks hold 144 hours, so a multiple of
# 144 keeps every request aligned with whole chunks.
HOURS_PER_BLOCK = 144 * 5


# ---------------------------------------------------------------------------
# Basin mask (cookie cutter)
# ---------------------------------------------------------------------------

def basin_polygon(usgs_gauge_id):
    """
    Downloads the drainage basin of a USGS gauge and returns it as a single
    shapely (Multi)Polygon in WGS84 (EPSG:4326, same CRS as AORC).
    """
    geojson = get_watershed_boundary(usgs_gauge_id, simplified=False)
    geometries = [shapely.geometry.shape(f["geometry"]) for f in geojson["features"]]
    return shapely.union_all(geometries)


def coverage_weights(polygon, latitudes, longitudes, subsamples=10):
    """
    Computes the area weight of every AORC cell in a lat/lon window: the
    fraction of the cell covered by the basin (estimated by testing an
    `subsamples` x `subsamples` grid of points inside each cell) times
    cos(latitude), to account for cells shrinking toward the pole.

    Using fractional coverage instead of "cell center inside polygon" keeps the
    average accurate for small basins that only span a handful of cells.

    Returns:
        np.ndarray: weights with shape (len(latitudes), len(longitudes))
    """
    offsets = (np.arange(subsamples) + 0.5) / subsamples - 0.5  # in cell units
    sub_lat = (latitudes[:, None] + offsets[None, :] * AORC_RES_DEG).ravel()
    sub_lon = (longitudes[:, None] + offsets[None, :] * AORC_RES_DEG).ravel()

    lon_grid, lat_grid = np.meshgrid(sub_lon, sub_lat)
    shapely.prepare(polygon)
    inside = shapely.contains_xy(polygon, lon_grid, lat_grid)

    # Collapse each subsamples x subsamples block back to one cell
    n_lat, n_lon = len(latitudes), len(longitudes)
    fraction = inside.reshape(n_lat, subsamples, n_lon, subsamples).mean(axis=(1, 3))

    return fraction * np.cos(np.deg2rad(latitudes))[:, None]


# ---------------------------------------------------------------------------
# AORC access
# ---------------------------------------------------------------------------

def open_aorc_year(fs, year):
    """Lazily opens one yearly AORC Zarr store (only metadata is downloaded)."""
    store = s3fs.S3Map(f"{AORC_BUCKET}/{year}.zarr", s3=fs)
    return xr.open_zarr(store, consolidated=True, chunks=None)[[PRECIP_VAR]]


def basin_window(ds, polygon):
    """
    Returns index slices of the smallest AORC lat/lon window that contains the
    basin (padded by one cell so partially covered edge cells are included).
    """
    lon_min, lat_min, lon_max, lat_max = polygon.bounds
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    i0 = max(np.searchsorted(lat, lat_min - AORC_RES_DEG) - 1, 0)
    i1 = min(np.searchsorted(lat, lat_max + AORC_RES_DEG) + 1, len(lat))
    j0 = max(np.searchsorted(lon, lon_min - AORC_RES_DEG) - 1, 0)
    j1 = min(np.searchsorted(lon, lon_max + AORC_RES_DEG) + 1, len(lon))

    if i0 >= i1 or j0 >= j1:
        raise ValueError("Basin lies outside the AORC domain (CONUS, 20-55N, 130-60W)")

    return slice(i0, i1), slice(j0, j1)


def areal_average(precip, weights):
    """
    Weighted spatial mean of a (time, lat, lon) block. Cells with missing data
    at a given hour are dropped from that hour's average (weights renormalized);
    hours with no valid cell in the basin come out as NaN.
    """
    valid = np.isfinite(precip)
    w = np.where(valid, weights[None, :, :], 0.0)
    total_w = w.sum(axis=(1, 2))
    total = (np.where(valid, precip, 0.0) * w).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total_w > 0, total / total_w, np.nan)


def get_aorc_basin_rainfall(usgs_gauge_id, start_date, end_date):
    """
    Computes the hourly basin-averaged AORC precipitation for a USGS gauge.

    Parameters:
        usgs_gauge_id (str): USGS site code (e.g., '02250030')
        start_date (str): first day, 'YYYY-MM-DD' (from 00:00 UTC)
        end_date (str): last day, 'YYYY-MM-DD' (through 23:00 UTC, inclusive)

    Returns:
        pd.DataFrame: indexed by UTC time with column 'precip_mm' (mm/hour,
            AORC APCP_surface averaged over the drainage basin).
    """
    start = pd.Timestamp(start_date).floor("D")
    end = pd.Timestamp(end_date).floor("D") + pd.Timedelta(hours=23)
    if end < start:
        raise ValueError("End date must be on or after the start date")
    if start < AORC_FIRST_TIME:
        raise ValueError(f"AORC starts on {AORC_FIRST_TIME:%Y-%m-%d}")

    print(f"Downloading basin boundary for USGS {usgs_gauge_id} ...")
    polygon = basin_polygon(usgs_gauge_id)

    fs = s3fs.S3FileSystem(anon=True)
    available_years = {
        int(os.path.basename(p).split(".")[0])
        for p in fs.ls(AORC_BUCKET)
        if p.endswith(".zarr")
    }

    weights = None
    series = []
    for year in range(start.year, end.year + 1):
        if year not in available_years:
            print(f"  {year}: not available in AORC, skipped")
            continue

        ds = open_aorc_year(fs, year)
        if weights is None:
            # The grid is identical in every yearly store, so the window and
            # weights are computed only once.
            lat_slice, lon_slice = basin_window(ds, polygon)
            weights = coverage_weights(
                polygon,
                ds["latitude"].values[lat_slice],
                ds["longitude"].values[lon_slice],
            )
            if weights.sum() == 0:
                raise ValueError("Basin does not overlap any AORC grid cell")
            print(f"  basin window: {weights.shape[0]} x {weights.shape[1]} AORC cells, "
                  f"{(weights > 0).sum()} touched by the basin")

        precip = ds[PRECIP_VAR].isel(latitude=lat_slice, longitude=lon_slice)
        precip = precip.sel(time=slice(max(start, pd.Timestamp(year, 1, 1)),
                                       min(end, pd.Timestamp(year, 12, 31, 23))))
        n_hours = precip.sizes["time"]

        for k in range(0, n_hours, HOURS_PER_BLOCK):
            block = precip.isel(time=slice(k, k + HOURS_PER_BLOCK))
            values = block.values.astype("float64")  # read into memory only
            series.append(pd.Series(areal_average(values, weights),
                                    index=block["time"].values))
            print(f"  {year}: {min(k + HOURS_PER_BLOCK, n_hours)}/{n_hours} hours",
                  end="\r")
        print()

    if not series:
        raise ValueError("No AORC data found for the requested period")

    result = pd.concat(series).sort_index().to_frame("precip_mm")
    result.index.name = "time_utc"
    return result


def download_aorc_basin_rainfall(usgs_gauge_id, start_date, end_date, output_dir="data"):
    """
    Computes the hourly basin-averaged AORC precipitation for a USGS gauge and
    saves it as data/usgs_<id>_aorc_hourly.parquet.

    Returns:
        pd.DataFrame: the saved time series (see `get_aorc_basin_rainfall`).
    """
    df = get_aorc_basin_rainfall(usgs_gauge_id, start_date, end_date)

    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/usgs_{usgs_gauge_id}_aorc_hourly.parquet"
    df.to_parquet(output_path)
    print(f"Saved {len(df)} hourly values to {output_path}")

    return df


def ask_date(prompt):
    """Prompts until the user enters a valid YYYY-MM-DD date."""
    while True:
        text = input(prompt).strip()
        try:
            return pd.Timestamp(text).strftime("%Y-%m-%d")
        except ValueError:
            print("  Invalid date, please use YYYY-MM-DD")


if __name__ == "__main__":
    args = sys.argv[1:]
    gauge_id = args[0] if len(args) > 0 else input("USGS gauge ID (e.g., 02250030): ").strip()
    start_date = args[1] if len(args) > 1 else ask_date("Initial date (YYYY-MM-DD): ")
    end_date = args[2] if len(args) > 2 else ask_date("Ending date (YYYY-MM-DD): ")

    download_aorc_basin_rainfall(gauge_id, start_date, end_date)
