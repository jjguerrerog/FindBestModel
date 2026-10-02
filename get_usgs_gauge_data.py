"""
Download daily streamflow, meteorological forcings, and evapotranspiration
estimates for a USGS streamgage and save the combined table as a parquet file.

Adapted from the exploratory notebook at
C:\\Users\\nicol\\Downloads\\Get_streamflow_and_forcings (NASA POWER +
USGS NWIS download functions, and the FAO-56 Penman-Monteith / Hargreaves-Samani /
Blaney-Criddle / Priestley-Taylor / Turc ET estimators).
"""

import io
import math
import os
import warnings

import pandas as pd
import requests

SQ_MILES_TO_SQ_KM = 2.58999

# ---------------------------------------------------------------------------
# USGS site metadata and streamflow
# ---------------------------------------------------------------------------

def get_usgs_site_info(site_code):
    """
    Looks up the decimal latitude/longitude and drainage area of a USGS site.

    Parameters:
        site_code (str): USGS site code (e.g., '02317620')

    Returns:
        tuple[float, float, float]: (latitude, longitude, drainage_area_km2).
            drainage_area_km2 is NaN if the site file has no drainage area on record.
    """
    url = "https://waterservices.usgs.gov/nwis/site/"
    params = {"format": "rdb", "sites": site_code, "siteOutput": "expanded"}

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    lines = [line for line in response.text.splitlines() if not line.startswith("#")]
    if len(lines) < 3:
        raise ValueError(f"No site metadata found for USGS site {site_code}")

    # lines[0] is the header, lines[1] is the RDB format-spec row, lines[2:] is data
    site_info = pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])), sep="\t")

    latitude = float(site_info.loc[0, "dec_lat_va"])
    longitude = float(site_info.loc[0, "dec_long_va"])

    # drain_area_va is the total drainage area; contrib_drain_area_va (contributing
    # area) is used as a fallback for sites where the total area isn't on file.
    # Both are reported by USGS in square miles.
    area_sq_mi = site_info.loc[0, "drain_area_va"]
    if pd.isna(area_sq_mi):
        area_sq_mi = site_info.loc[0, "contrib_drain_area_va"]

    if pd.isna(area_sq_mi):
        warnings.warn(f"No drainage area on record for USGS site {site_code}")
        drainage_area_km2 = float("nan")
    else:
        drainage_area_km2 = round(float(area_sq_mi) * SQ_MILES_TO_SQ_KM, 2)

    return latitude, longitude, drainage_area_km2


def fetch_usgs_daily_streamflow(site_code, start_date, end_date):
    """
    Fetches USGS mean daily streamflow for a given site and date range.

    Parameters:
        site_code (str): USGS site code (e.g., '02317620')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: DataFrame indexed by date with a 'Flow(cms)' column
    """
    url = "https://waterservices.usgs.gov/nwis/dv/"
    params = {
        "format": "json",
        "sites": site_code,
        "startDT": start_date,
        "endDT": end_date,
        "parameterCd": "00060",  # discharge
        "statCd": "00003",  # daily mean
        "siteStatus": "all",
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    timeseries = data.get("value", {}).get("timeSeries", [])

    records = []
    for series in timeseries:
        for entry in series["values"][0]["value"]:
            timestamp = entry["dateTime"]
            value = float(entry["value"]) if entry["value"] != "" else None
            if value is not None:
                value *= 0.0283168  # cfs -> cms
            records.append([timestamp, value])

    df = pd.DataFrame(records, columns=["Date", "Flow(cms)"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize().dt.tz_localize(None)
    df.set_index("Date", inplace=True)
    return df


# ---------------------------------------------------------------------------
# NASA POWER meteorological forcings
# ---------------------------------------------------------------------------

def download_nasa_power_data(latitude, longitude, start_date, end_date):
    """
    Downloads daily meteorological data from the NASA POWER API.

    Parameters:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_date (str): Start date in 'YYYYMMDD' format.
        end_date (str): End date in 'YYYYMMDD' format.

    Returns:
        pd.DataFrame: DataFrame indexed by date with the required variables.
    """
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    parameters = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN", "PS", "PRECTOTCORR"]

    params = {
        "parameters": ",".join(parameters),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }

    response = requests.get(base_url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    parameter_data = data["properties"]["parameter"]
    dates = parameter_data["T2M_MAX"].keys()
    df = pd.DataFrame({
        "Date": pd.to_datetime(list(dates)),
        "T2M_MAX (°C)": parameter_data["T2M_MAX"].values(),
        "T2M_MIN (°C)": parameter_data["T2M_MIN"].values(),
        "RH2M (%)": parameter_data["RH2M"].values(),
        "WS2M (m/s)": parameter_data["WS2M"].values(),
        "Solar Radiation (MJ/m²/day)": parameter_data["ALLSKY_SFC_SW_DWN"].values(),
        "Pressure (kPa)": parameter_data["PS"].values(),
        "Precipitation (mm/day)": parameter_data["PRECTOTCORR"].values(),
    })
    df.set_index("Date", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Evapotranspiration estimators
# ---------------------------------------------------------------------------

ALBEDO = 0.23  # Albedo for reference crop


def saturation_vapor_pressure(T):
    """Saturation vapor pressure (es, kPa) from temperature (°C)."""
    return 0.6108 * math.exp((17.27 * T) / (T + 237.3))


def delta_svp(T):
    """Slope of the saturation vapor pressure curve (delta)."""
    return (4098 * saturation_vapor_pressure(T)) / ((T + 237.3) ** 2)


def psychrometric_constant(pressure_kPa):
    """Psychrometric constant (gamma) from pressure (kPa)."""
    return 0.000665 * pressure_kPa


def net_radiation(solar_rad, albedo=ALBEDO):
    """Net radiation (Rn) from solar radiation and albedo."""
    return (1 - albedo) * solar_rad


def potential_et_penman_monteith(row):
    """FAO-56 Penman-Monteith reference ET (mm/day)."""
    t_max = row["T2M_MAX (°C)"]
    t_min = row["T2M_MIN (°C)"]
    rh = row["RH2M (%)"]
    wind_speed = row["WS2M (m/s)"]
    solar_rad = row["Solar Radiation (MJ/m²/day)"]
    pressure = row["Pressure (kPa)"]

    t_mean = (t_max + t_min) / 2
    delta = delta_svp(t_mean)
    gamma = psychrometric_constant(pressure)
    es = (saturation_vapor_pressure(t_max) + saturation_vapor_pressure(t_min)) / 2
    ea = es * (rh / 100)
    rn = net_radiation(solar_rad)

    numerator = (0.408 * delta * rn) + (gamma * (900 / (t_mean + 273)) * wind_speed * (es - ea))
    denominator = delta + gamma * (1 + 0.34 * wind_speed)
    return round(numerator / denominator, 2)


def hargreaves_samani(row):
    """Hargreaves-Samani reference ET (mm/day)."""
    t_max, t_min = row["T2M_MAX (°C)"], row["T2M_MIN (°C)"]
    t_mean, solar_rad = (t_max + t_min) / 2, row["Solar Radiation (MJ/m²/day)"]
    et = 0.0023 * (t_mean + 17.8) * ((t_max - t_min) ** 0.5) * solar_rad
    return round(et, 2)


def blaney_criddle(row):
    """Blaney-Criddle reference ET (mm/day)."""
    t_mean = (row["T2M_MAX (°C)"] + row["T2M_MIN (°C)"]) / 2
    day_of_year = row.name.timetuple().tm_yday
    daylight_hours = 12 + 4 * math.sin((2 * math.pi * (day_of_year - 80)) / 365)  # Approximation
    et = 0.0173 * daylight_hours * (0.46 * t_mean + 8)
    return round(et, 2)


def priestley_taylor(row, alpha_pt=1.26, latent_heat=2.45):
    """Priestley-Taylor reference ET (mm/day)."""
    t_mean = (row["T2M_MAX (°C)"] + row["T2M_MIN (°C)"]) / 2
    delta = delta_svp(t_mean)
    gamma = psychrometric_constant(row["Pressure (kPa)"])
    rn = net_radiation(row["Solar Radiation (MJ/m²/day)"])
    G = 0  # Soil heat flux ~ 0 at daily scale

    et_mj = (alpha_pt * delta * (rn - G)) / (delta + gamma)
    et_mm = et_mj / latent_heat
    return max(round(et_mm, 2), 0)


def turc_method(row):
    """Turc reference ET (mm/day)."""
    t_mean = (row["T2M_MAX (°C)"] + row["T2M_MIN (°C)"]) / 2
    rh = row["RH2M (%)"]
    solar_rad_mj = row["Solar Radiation (MJ/m²/day)"]

    if t_mean <= 0:
        return 0  # No ET under freezing conditions

    solar_rad_cal = solar_rad_mj * 23.9  # MJ/m²/day -> cal/cm²/day
    et = (0.013 * t_mean) / (t_mean + 15) * (solar_rad_cal + 50)

    if rh < 50:
        et *= 1 + ((50 - rh) / 70)

    return round(et, 2)


ET_METHODS = ["Penman", "Hargreaves-Samani", "Blaney-Criddle", "Priestley-Taylor", "Turc"]


def min_max_total_et(etp, precipitation):
    """
    Total ET (ETt) from potential ET (ETp) via the min-max method: on a given
    day, ET is limited by whichever is smaller, the atmospheric demand (ETp)
    or the available water (precipitation) -- i.e. ETt = min(ETp, P).
    """
    return min(etp, precipitation)


def _add_et_columns(met_df):
    met_df = met_df.copy()
    met_df["ET Penman (mm/day)"] = met_df.apply(potential_et_penman_monteith, axis=1)
    met_df["ET Hargreaves-Samani (mm/day)"] = met_df.apply(hargreaves_samani, axis=1)
    met_df["ET Blaney-Criddle (mm/day)"] = met_df.apply(blaney_criddle, axis=1)
    met_df["ET Priestley-Taylor (mm/day)"] = met_df.apply(priestley_taylor, axis=1)
    met_df["ET Turc (mm/day)"] = met_df.apply(turc_method, axis=1)

    precipitation = met_df["Precipitation (mm/day)"]
    for method in ET_METHODS:
        met_df[f"ETt {method} (mm/day)"] = met_df[f"ET {method} (mm/day)"].combine(
            precipitation, min_max_total_et
        )

    return met_df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def download_usgs_gauge_data(usgs_gauge_id, start_date, end_date, output_dir="data"):
    """
    Downloads and saves a parquet file with daily streamflow, meteorological
    forcings, and reference ET estimates for a USGS gauge.

    Parameters:
        usgs_gauge_id (str): USGS site code (e.g., '02317620')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        output_dir (str): Directory the parquet file is written to

    Returns:
        pd.DataFrame: The daily table that was saved to parquet, indexed by date,
            with columns for watershed area, precipitation, streamflow, potential
            ET (ETp) and total ET (ETt, via the min-max method: ETt = min(ETp, P))
            for each of Penman-Monteith, Hargreaves-Samani, Blaney-Criddle,
            Priestley-Taylor, and Turc.
    """
    latitude, longitude, drainage_area_km2 = get_usgs_site_info(usgs_gauge_id)

    streamflow_df = fetch_usgs_daily_streamflow(usgs_gauge_id, start_date, end_date)

    met_df = download_nasa_power_data(
        latitude, longitude, start_date.replace("-", ""), end_date.replace("-", "")
    )
    met_df = _add_et_columns(met_df)

    combined_df = met_df.join(streamflow_df, how="inner")
    combined_df["Drainage Area (km2)"] = drainage_area_km2
    combined_df = combined_df[[
        "Drainage Area (km2)",
        "Precipitation (mm/day)",
        "Flow(cms)",
        "ET Penman (mm/day)",
        "ET Hargreaves-Samani (mm/day)",
        "ET Blaney-Criddle (mm/day)",
        "ET Priestley-Taylor (mm/day)",
        "ET Turc (mm/day)",
        "ETt Penman (mm/day)",
        "ETt Hargreaves-Samani (mm/day)",
        "ETt Blaney-Criddle (mm/day)",
        "ETt Priestley-Taylor (mm/day)",
        "ETt Turc (mm/day)",
    ]]

    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/usgs_{usgs_gauge_id}_daily.parquet"
    combined_df.to_parquet(output_path)

    return combined_df


if __name__ == "__main__":
    download_usgs_gauge_data("02317620", "2020-01-01", "2023-12-31")
