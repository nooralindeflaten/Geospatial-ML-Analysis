mkdir -p geodata_analysis_final_version/data/raw/vector/admin

ogrinfo "WFS:https://wfs.geonorge.no/skwms1/wfs.administrative_enheter?service=WFS&acceptversions=2.0.0&request=GetCapabilities"

# I know this prints kommune, fylke and grense

OUT="geodata_analysis_final_version/data/raw/vector/admin/admin.gpkg"
rm -f "$OUT"

ogr2ogr -f GPKG "$OUT" \
  "WFS:https://wfs.geonorge.no/skwms1/wfs.administrative_enheter?service=WFS&acceptversions=2.0.0&request=GetCapabilities" \
  "app:Kommune"