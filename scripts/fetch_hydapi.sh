export HYDAPI_KEY = 

curl -s -G "https://hydapi.nve.no/api/v1/Stations" \
-H "accept: application/json" -H "X-API-Key: $HYDAPI_KEY" \
--data-urlencode "Active=0" \
--data-urlencode f"Polygon={WKT}" \
-o geodata_analysis_final_version/data/raw/vector/stations/nve_inactive_stations.json

# This calls active stations
curl -s -G "https://hydapi.nve.no/api/v1/Stations" \
-H "accept: application/json" -H "X-API-Key: $HYDAPI_KEY" \
--data-urlencode "Active=1" \
--data-urlencode f"Polygon={WKT}" \
-o geodata_analysis_final_version/data/raw/vector/stations/nve_stations_active.json

python - << 'PY'
import json, os, subprocess, csv, datetime as dt
with open("geodata_analysis_final_version/data/raw/vector/stations/nve_stations_active.json") as f:
    d = json.load(f)

os.makedirs("geodata_analysis_final_version/data/raw/timeseries/flow", exist_ok=True)
os.makedirs("geodata_analysis_final_version/data/raw/timeseries/stage", exist_ok=True)

ids = [s["stationId"] for s in d.get("data", []) if s.get("stationId")]
FROM=os.environ.get("FROM","1994-01-01T00:00")
TO=os.environ.get("TO", "2026-01-31T23:59")
api_key=os.environ.get("HYDAPI_KEY")

def pull_series(stdid):
    out = f"geodata_analysis_final_version/data/raw/timeseries/flow/{stdid.replace(':','_')}_NVE.csv"
    url="https://hydapi.nve.no/api/v1/Observations"
    headers = {
        "Accept": "application/json",
        "X-API-Key": api_key,
    }

    params=[
        "--data-urlencode",f"stationId={stdid}",
        "--data-urlencode",f"Parameter=1000,1001",      
        "--data-urlencode","ResolutionTime=1440",
        "--data-urlencode",f"ReferenceTime={FROM}/{TO}"
    ]
    cmd = ["curl","-s","-G",f"{url}","-H","accept: application/json","-H",f"X-API-Key: {api_key}"] + params
    with open(out,"wb") as fo:
        subprocess.run(cmd, check=False, stdout=fo)
    print("saved", out)

PY