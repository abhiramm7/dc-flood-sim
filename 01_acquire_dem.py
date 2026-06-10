#!/usr/bin/env python3
"""
Milestone 1 — Acquire & clip a DEM for the Potomac at Key Bridge, Washington DC.

This is step [1] of the flood-modeling pipeline. It produces a clipped DEM
GeoTIFF (in meters, projected) plus a PNG quicklook so you can eyeball it.

TWO ACQUISITION PATHS
---------------------
A) AUTOMATED (default): USGS 3DEP via the `py3dep` package. This is *land-only*
   elevation, but it runs end-to-end with no manual download, so you can build
   and test the entire solver pipeline today. The riverbed will read as ~flat
   water surface until you swap in topobathy data.

B) TOPOBATHY LIDAR (recommended for the real run): the 2021 USGS Topobathy
   Lidar for the Potomac includes the submerged riverbed. It isn't available
   through a simple Python API, so download it manually (instructions below),
   drop the .tif in this folder, and run this script with --local-dem.

USAGE
-----
    pip install py3dep rioxarray rasterio matplotlib numpy --break-system-packages

    # Path A — automated land DEM:
    python 01_acquire_dem.py

    # Path B — use a topobathy GeoTIFF you downloaded:
    python 01_acquire_dem.py --local-dem potomac_topobathy.tif

DOWNLOADING TOPOBATHY (Path B)
------------------------------
1. Go to NOAA Digital Coast Data Access Viewer: https://coast.noaa.gov/dataviewer/
2. Search the area around 38.902 N, -77.069 W (Key Bridge).
3. Select "2021 USGS Topobathy Lidar: Potomac River" (DEM product).
4. Define a bounding box covering the box below, request GeoTIFF, download.
5. Save it next to this script and pass it with --local-dem.

Outputs: keybridge_dem_clipped.tif  +  keybridge_dem_preview.png
"""

import argparse
import sys
import numpy as np

# ----------------------------------------------------------------------------
# Study area: bounding box around Francis Scott Key Bridge, Washington DC.
# Coordinates in WGS84 (lon/lat). ~3 km x ~2 km, covering the river reach
# from a bit upstream (toward Three Sisters) to downstream past the bridge.
# Adjust freely.
# ----------------------------------------------------------------------------
KEY_BRIDGE = dict(lat=38.9024, lon=-77.0694)

BBOX = dict(
    west=-77.0850,   # lon min
    south=38.8950,   # lat min
    east=-77.0540,   # lon max
    north=38.9100,   # lat max
)

# A projected CRS in meters is required for hydraulic modeling.
# UTM Zone 18N (EPSG:32618) covers Washington DC.
TARGET_CRS = "EPSG:32618"
TARGET_RES = 3.0   # meters; resample target for the solver grid

OUT_TIF = "keybridge_dem_clipped.tif"
OUT_PNG = "keybridge_dem_preview.png"


def fetch_py3dep(bbox):
    """Path A: pull a 1m 3DEP DEM via py3dep (land-only)."""
    try:
        import py3dep
    except ImportError:
        sys.exit(
            "py3dep not installed. Run:\n"
            "  pip install py3dep rioxarray rasterio matplotlib numpy --break-system-packages\n"
            "Or use Path B with --local-dem <file.tif>."
        )
    geom = (bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    print("Requesting 3DEP 1m DEM from USGS (this hits the network)...")
    # resolution=1 -> 1 meter; crs of the input bbox is WGS84 (4326)
    dem = py3dep.get_map("DEM", geom, resolution=1, geo_crs="EPSG:4326", crs="EPSG:4326")
    print(f"  received array shape={dem.shape}")
    return dem  # xarray DataArray


def load_local(path):
    """Path B: load a DEM GeoTIFF you downloaded (e.g. topobathy)."""
    import rioxarray  # noqa
    import xarray as xr
    print(f"Loading local DEM: {path}")
    da = xr.open_dataarray(path, engine="rasterio").squeeze()
    return da


def clip_reproject(da, bbox):
    """Clip to bbox, reproject to a meters CRS, resample to TARGET_RES."""
    import rioxarray  # registers .rio accessor  # noqa

    # Ensure the source has a CRS; if missing assume 4326.
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326")

    # Clip in the source CRS (bbox is WGS84). reproject bbox if source isn't 4326.
    if da.rio.crs.to_epsg() != 4326:
        # clip in projected space by first reprojecting a tiny envelope is overkill;
        # simplest: reproject whole array to 4326, clip, then to target.
        da = da.rio.reproject("EPSG:4326")

    clipped = da.rio.clip_box(
        minx=bbox["west"], miny=bbox["south"],
        maxx=bbox["east"], maxy=bbox["north"],
    )
    print("  clipped to bounding box")

    proj = clipped.rio.reproject(TARGET_CRS, resolution=TARGET_RES)
    print(f"  reprojected to {TARGET_CRS} @ {TARGET_RES} m  shape={proj.shape}")
    return proj


def save_outputs(da):
    da.rio.to_raster(OUT_TIF)
    print(f"Wrote {OUT_TIF}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = da.values.astype("float32")
    # mask nodata
    nodata = da.rio.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, cmap="terrain")
    ax.set_title("Key Bridge DEM (elevation, m)")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    fig.colorbar(im, ax=ax, label="elevation (m)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print(f"Wrote {OUT_PNG}")

    valid = arr[np.isfinite(arr)]
    if valid.size:
        print(f"  elevation range: {valid.min():.2f} .. {valid.max():.2f} m  "
              f"(n={valid.size} valid cells)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-dem", help="path to a downloaded DEM GeoTIFF (Path B)")
    args = ap.parse_args()

    print(f"Study area: Key Bridge ({KEY_BRIDGE['lat']}, {KEY_BRIDGE['lon']})")
    print(f"BBox: {BBOX}\n")

    da = load_local(args.local_dem) if args.local_dem else fetch_py3dep(BBOX)
    proj = clip_reproject(da, BBOX)
    save_outputs(proj)
    print("\nDone. Next: 02_extract_channel.py (river extraction).")


if __name__ == "__main__":
    main()
