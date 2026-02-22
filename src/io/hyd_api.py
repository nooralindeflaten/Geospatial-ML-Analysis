import os
import sys
import json
from pathlib import Path
import requests
import pandas as pd

def get_api_key():
    api_key = os.getenv("HYDAPI_KEY")
    if not api_key:
        raise SystemExit(
            "Missing HYDAPI_KEY environment variable.\n\n"
            "Set it in your shell first, e.g.:\n"
            "  export HYDAPI_KEY='your_real_key_here'\n"
        )
    return api_key

def fetch_parameters(
):
    headers = {
        "Accept": "application/json",
        "X-API-Key": get_api_key(),
    }

    rows = []
    base_url = "https://hydapi.nve.no/api/v1/Parameters"
    print("Fetching station params ...")
    resp = requests.get(base_url, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    for series in data.get("data", []):
        rows.append({
                **series})

    df = pd.DataFrame(rows)
    return df

def fetch_observations(
    station_ids,
    parameter="1001",        # 1001 = Vannføring
    resolution_time=1440,    # 1440 = daily
    reference_years=40,      # last 40 years
    active="OnlyActive",
):
    """
    Fetch daily time series for one or more stations from HydAPI.

    Returns a pandas DataFrame with columns:
      station_id, parameter, unit, time, value, quality_code, resolution_time
    """
    base_url = "https://hydapi.nve.no/api/v1/Observations"
    api_key = 'VcA81jMjx0WSbYdNaUu4sw=='

    headers = {
        "Accept": "application/json",
        "X-API-Key": api_key,
    }

    reference_time = f"P{reference_years}Y/"  # e.g. "P40Y/"

    rows = []

    for sid in station_ids:
        params = {
            "StationId": sid,
            "Parameter": parameter,
            "ResolutionTime": resolution_time,
            "ReferenceTime": reference_time,
        }

        print(f"Fetching station {sid} ...", file=sys.stderr)
        resp = requests.get(base_url, headers=headers, params=params, timeout=60)
        print(resp)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            print(f"  ERROR for station {sid}: {e}", file=sys.stderr)
            continue

        try:
            data = resp.json()
        except json.JSONDecodeError:
            print(f"  ERROR decoding JSON for station {sid}", file=sys.stderr)
            continue

        for series in data.get("data", []):
            station_id = series.get("StationId", sid)
            param = series.get("Parameter")
            unit = series.get("Unit")

            for obs in series.get("Observations", []):
                rows.append(
                    {
                        "station_id": station_id,
                        "parameter": param,
                        "unit": unit,
                        "time": obs.get("Time"),
                        "value": obs.get("Value"),
                        "quality_code": obs.get("QualityCode"),
                        "resolution_time": resolution_time,
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(["station_id", "time"]).reset_index(drop=True)
    return df