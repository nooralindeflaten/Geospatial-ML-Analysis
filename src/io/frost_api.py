
def fetch_parameters(
):
    headers = {
        "Accept": "application/json",
        "X-API-Key": HYDAPI_KEY,
    }

    rows = []

    print("Fetching station params ...")
    resp = requests.get(BASE_URL, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    for series in data.get("data", []):
        rows.append({
                **series})

    df = pd.DataFrame(rows)
    return df