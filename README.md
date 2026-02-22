
# Folder structure command:

alias tree="find . -print | sed -e 's;[^/]*/;|____;g;s;____|; |;g'"

tree

# Admin
**source**: `data/raw/vector/admin/admin.gpkg`
This was made using python code. Usually I use bash commands so I'm a bit skeptical about the python code.
I could make a shell script with the commands. 

```bash
mkdir -p geodata_analysis_final_version/data/raw/vector/admin

ogrinfo "WFS:https://wfs.geonorge.no/skwms1/wfs.administrative_enheter?service=WFS&acceptversions=2.0.0&request=GetCapabilities"

# I know this prints kommune, fylke and grense

OUT="geodata_analysis_final_version/data/raw/vector/admin/admin.gpkg"
rm -f "$OUT"

ogr2ogr -f GPKG "$OUT" \
  "WFS:https://wfs.geonorge.no/skwms1/wfs.administrative_enheter?service=WFS&acceptversions=2.0.0&request=GetCapabilities" \
  "app:Kommune"
``` 
```bash
geodata_analysis_final_version/
|____.DS_Store
|____imgs
|____docs
|____README.md
|____folder_tree.txt
|____examples
| |____hydro_layers 
| | |______region_catchments.csv # These are csv files for each hydro layer. (catchments, vannforing_stasjoner, lakes and rivers as raw data from query-response -> gpkg -> read head())
| |____hazard_layers
| |____building_layers
|____configs
|____scripts
|____data
| |____.DS_Store
| |____sql_version
| | |____db # location for the sqlite database 
| | |____points_files # The files created in this version (response from api etc.)
| |____processed
| | |____links # Links between objects (e.g. house + hydro links)
| | |____ml # model inputs (e.g. landslide + static terrain, graph..)
| | |____vector
| | | |____exposure # Not sure if this is necessary, but maybe geopackages with saved predictions from ML model
| | | |____hazards # I'll get back to these
| | | |____hydro
| | |____rasters # terrain rasters (twi, flow_acc_d8, slope_deg, aspect_deg, curvature, dtm_filled)
| |____downloads # direct downloads
| | |____.DS_Store
| | |____gml_files # folder for gml files since there's not many
| | | |____Basisdata_4643_Ardal_25833_MatrikkelenBygning_GML.gml
| | |____Nedlastingspakke # folder with all dtm zips and tifs opened  
| | | |____Basisdata_6800-2_Celle_25833_DTM10UTM33_TIFF.zip
| | | |__...
| |____raw 
| | |____timeseries # from Hyd and Frost we'll get back to it
| | |____hyd_frost_API # data from other endpoints like "avalibletimeseries", "parameters" etc. 
| | |____vector
| | | |____stations # Stations from hyd_api and frostAPI. Maybe after finishing initial commit I'll add the stations from the hydro data package
| | | |____exposure # house geopackage retrieved from wfs and downloaded gml. 
| | | |____admin # main aoi (initially retrieved by bash command, but this one was python code. )
| | | |   |_______admin.gpkg
| | | |   |_______{region}_area.gpkg
| | | |____hazards
| | | |____hydro
| | |____rasters
| | | |____wcs_dtms # dtm files generated from calling wcs
| | | |____{region}_dtms # dtm created fromtiles downloaded
| |____qa # map html files generated using folium
|____notebooks
| |____01_debug_terrain.ipynb
| |____03_model_training.ipynb
| |____02_debug_links.ipynb
| |____00_quick_view.ipynb
|____src
| |____pipeline
| |____processing
| | |____schemas.py
| |____datasets
| |____io
| |____utils
| |____models

```


# Scripts list:

datasets        io              models          pipeline        processing      utils

## geodata_analysis_final_version/src/datasets:
tabular_dataset.py
terrain_patches.py

## geodata_analysis_final_version/src/io:
arcgis_fetch.py wcs_dtm.py

## geodata_analysis_final_version/src/models:

## geodata_analysis_final_version/src/pipeline:
hazard_pipeline.py      hydro_pipeline.py       run_region.py

## geodata_analysis_final_version/src/processing:
aoi             graphs          links           ml              schemas.py      terrain

### geodata_analysis_final_version/src/processing/aoi:
admin_aoi.py            exposure_houses.py      hydro_hazard.py

### geodata_analysis_final_version/src/processing/graphs:
knn_graph.py

### geodata_analysis_final_version/src/processing/links:
flood_labels.py         hydro_links.py          landslide_links.py

### geodata_analysis_final_version/src/processing/ml:
assemble_tabular.py     channel_stats.py

### geodata_analysis_final_version/src/processing/terrain:
derivatives.py          dtm_from_tiles.py

### geodata_analysis_final_version/src/utils:
paths.py