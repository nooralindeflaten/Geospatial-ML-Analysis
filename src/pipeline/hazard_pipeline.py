from pathlib import Path
from src.io.arcgis_fetch import fetch_layer_for_region
from src.io.examples_builder import build_examples_for_region
HAZARD_LAYERS = [
    {
        "url": "https://nve.geodataonline.no/arcgis/rest/services/Flomsoner2/MapServer",
        "layers": [
            {"num": 0, "name": "flomsone_analyseomrade", "geometry_type": "MULTIPOLYGON"},
            {"num": 13, "name": "flomsone_10", "geometry_type": "MULTIPOLYGON"},
            {"num": 14, "name": "flomsone_20", "geometry_type": "MULTIPOLYGON"},
            {"num": 15, "name": "flomsone_50", "geometry_type": "MULTIPOLYGON"},
            {"num": 16, "name": "flomsone_100", "geometry_type": "MULTIPOLYGON"},
            {"num": 17, "name": "flomsone_200", "geometry_type": "MULTIPOLYGON"},
            {"num": 18, "name": "flomsone_500", "geometry_type": "MULTIPOLYGON"},
            {"num": 19, "name": "flomsone_1000", "geometry_type": "MULTIPOLYGON"},
            {"num": 20, "name": "flomsone_20_klima", "geometry_type": "MULTIPOLYGON"},
            {"num": 21, "name": "flomsone_200_klima", "geometry_type": "MULTIPOLYGON"},
            {"num": 22, "name": "flomsone_1000_klima", "geometry_type": "MULTIPOLYGON"},
            ]
    },
    {
        "url": "https://nve.geodataonline.no/arcgis/rest/services/SkredHendelser1/MapServer",
        "layers": [
            {"num": 0, "name": "landslides", "geometry_type": "POINT"},
            {"num": 1, "name": "landslide_trigger_points", "geometry_type": "POINT"},
            {"num": 2, "name": "landslide_runout_points", "geometry_type": "POINT"},
            {"num": 3, "name": "landslide_source_areas", "geometry_type": "MULTIPOLYGON"},
            {"num": 4, "name": "landslide_runout_areas", "geometry_type": "MULTIPOLYGON"},
        ]
    }
]

def build_all_region_hazard(bbox,region_name,region_gpkg, hazard_gpkg):
    out_gpkg = Path(hazard_gpkg)

    if out_gpkg.exists():
        print(f"[{region_name}] Removing existing {out_gpkg}")
        out_gpkg.unlink()
    for hazard_source in HAZARD_LAYERS:
        base_url = hazard_source["url"]
        for layer_cfg in hazard_source["layers"]:
            fetch_layer_for_region(
                bbox,
                region_gpkg,
                out_gpkg,
                layer_name=layer_cfg["name"],
                layer_url=base_url + f"/{layer_cfg['num']}/",
                geometry_type=layer_cfg["geometry_type"],
            )
        
            # print(f"{layer_cfg['name']} done") <- debug line
    print(f"✓ Added hazard layers for {region_name},  at {out_gpkg} \n")
