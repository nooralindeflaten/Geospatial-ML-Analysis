python - << 'PY'
import geopandas as gpd, json
kolbotn = gpd.read_file("geodata_analysis_final_version/data/raw/vector/admin/kolbotn_area.gpkg").to_crs(4326)
minx, miny, maxx, maxy = kolbotn.total_bounds
print(f"{minx:.5f},{miny:.5f},{maxx:.5f},{maxy:.5f}")
PY

# output from python code above, which is the bounding box of the area of interest (Kolbotn)
BBOX_WGS84="...,...,...,..." # replace with actual values
FROST_CLIENT_ID="...."
IFS=',' read XMIN YMIN XMAX YMAX <<< "$BBOX_WGS84"
WKT="POLYGON(($XMIN $YMIN,$XMAX $YMIN,$XMAX $YMAX,$XMIN $YMAX,$XMIN $YMIN))"
curl -s -G "https://frost.met.no/sources/v0.jsonld" \
--data-urlencode "geometry=$WKT" \
--user "${FROST_CLIENT_ID}:" \
-o geodata_analysis_final_version/data/raw/vector/stations/frost_stations.json

export FROST_CLIENT_ID="...." # replace with actual client ID
python - << 'PY'
import json, os, subprocess, csv, datetime as dt
infile = "geodata_analysis_final_version/data/raw/vector/stations/frost_stations.json"
with open(infile) as f:
    d = json.load(f)

ids = [s["id"] for s in d.get("data", []) if s.get("id")]
print(ids)
ids18 = ['SN17895', 'SN17820', 'SN76914', 'SN17875']
ids = [s for s in ids if s not in ids18]
print(ids)
client=os.environ.get("FROST_CLIENT_ID")
FROM="2005-01-01T00:00"
TO="2026-12-31T23:59"

FROM_OLD="2018-01-01T00:00"
TO_OLD="2026-01-01T23:59"

def pull(stid):
    out = f"geodata_analysis_final_version/data/raw/timeseries/rain/{stid.replace(':','_')}_precip_daily.csv"
    url = "https://frost.met.no/observations/v0.csv"
    params = [
        "--data-urlencode", f"sources={stid}",
        "--data-urlencode", "elements=sum(precipitation_amount P1D)",
        "--data-urlencode", f"referencetime={FROM}/{TO}",
    ]
    cmd = ["curl","-s","-G",f"{url}","--user",f"{client}:"] + params
    with open(out,"wb") as fo:
        subprocess.run(cmd, check=False, stdout=fo)
    print("saved", out)

def pull_rest(stid):
    out = f"geodata_analysis_final_version/data/raw/timeseries/rain/{stid.replace(':','_')}_precip_daily.csv"
    url = "https://frost.met.no/observations/v0.csv"
    params = [
        "--data-urlencode", f"sources={stid}",
        "--data-urlencode", "elements=sum(precipitation_amount PT1H)",
        "--data-urlencode", f"referencetime={FROM_OLD}/{TO_OLD}",
    ]
    cmd = ["curl","-s","-G",f"{url}","--user",f"{client}:"] + params
    with open(out,"wb") as fo:
        subprocess.run(cmd, check=False, stdout=fo)
    print("saved", out)

for st_new in ids:
    pull(st_new)
for st in ids18:
    pull_rest(st)
PY