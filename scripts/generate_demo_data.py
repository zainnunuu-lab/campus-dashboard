"""Generate clearly labelled SYNTHETIC demo data for the dashboard.

    python scripts/generate_demo_data.py            # (re)create data/demo/*

Nothing produced here is a drone survey result. The layout is an invented grid of
blocks placed near the configured map centre so that the pages have something to
show before real survey outputs exist. Every file gets:

* a ``.meta.json`` sidecar with ``is_demo: true``;
* a ``data_status = "SYNTHETIC DEMO"`` attribute on every vector feature;
* a ``DATA_STATUS`` tag in the GeoTIFF;
* and it lives under ``data/demo/``, which the dashboard always treats as demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import ColorInterp
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from campus_dashboard import config  # noqa: E402
from campus_dashboard.metadata import DatasetMetadata, save_metadata  # noqa: E402

UTM = "EPSG:32642"
STATUS = "SYNTHETIC DEMO"
SEED = 42
HALF_W, HALF_H = 180.0, 150.0  # metres, extent of the invented study area
ROAD_HALF = 6.0
PIXEL = 0.5  # metres, synthetic raster resolution

CLASS_RGB = {
    "building": (176, 160, 150),
    "road": (70, 70, 72),
    "parking": (128, 128, 125),
    "paved_open": (196, 186, 165),
    "bare_soil": (170, 130, 90),
    "lawn_grass": (110, 160, 70),
    "trees_shrubs": (45, 95, 45),
    "sports_field": (95, 150, 80),
    "water": (60, 100, 140),
    "other": (150, 140, 130),
}
CLASS_WEIGHTS = {
    "building": 0.28, "parking": 0.10, "paved_open": 0.14, "bare_soil": 0.08,
    "lawn_grass": 0.18, "trees_shrubs": 0.16, "sports_field": 0.04, "other": 0.02,
}


def _origin():
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    lat, lon = config.CAMPUS_CENTER_LATLON
    return to_utm.transform(lon, lat)


def build_layers(rng: np.random.Generator):
    ox, oy = _origin()
    area = box(ox - HALF_W, oy - HALF_H, ox + HALF_W, oy + HALF_H)
    roads = unary_union([
        box(ox - HALF_W, oy - ROAD_HALF, ox + HALF_W, oy + ROAD_HALF),
        box(ox - ROAD_HALF, oy - HALF_H, ox + ROAD_HALF, oy + HALF_H),
    ])

    classes = list(CLASS_WEIGHTS)
    probs = np.array(list(CLASS_WEIGHTS.values()))
    probs = probs / probs.sum()
    lu_rows, bld_rows = [], []
    cell = 30.0
    xs = np.arange(ox - HALF_W, ox + HALF_W, cell)
    ys = np.arange(oy - HALF_H, oy + HALF_H, cell)
    block_no = 0
    for x in xs:
        for y in ys:
            geom = box(x, y, x + cell, y + cell).difference(roads)
            if geom.is_empty or geom.area < 1:
                continue
            cls = str(rng.choice(classes, p=probs))
            lu_rows.append({"class": cls, "geometry": geom})
            if cls == "building":
                block_no += 1
                fp = geom.buffer(-3, join_style="mitre")
                if not fp.is_empty:
                    bld_rows.append({
                        "name": f"Demo Block {block_no:02d}",
                        "height_m": float(rng.choice([7.5, 11, 14.5, 18, 21.5])),
                        "floors": None,
                        "geometry": fp,
                    })
    lu_rows.append({"class": "road", "geometry": roads})
    # A small synthetic pond so the water class appears in the legend.
    pond = Point(ox + 120, oy - 100).buffer(9)
    lu = gpd.GeoDataFrame(lu_rows, crs=UTM)
    lu["geometry"] = lu.geometry.difference(pond)
    lu = pd.concat([lu, gpd.GeoDataFrame([{"class": "water", "geometry": pond}], crs=UTM)], ignore_index=True)
    lu = lu[~lu.geometry.is_empty].reset_index(drop=True)
    lu.insert(0, "id", range(1, len(lu) + 1))

    blds = gpd.GeoDataFrame(bld_rows, crs=UTM)
    blds["floors"] = (blds["height_m"] / 3.5).round().astype(int)

    paths = gpd.GeoDataFrame(
        [
            {"name": "Demo Road E-W", "type": "road", "geometry": LineString([(ox - HALF_W, oy), (ox + HALF_W, oy)])},
            {"name": "Demo Road N-S", "type": "road", "geometry": LineString([(ox, oy - HALF_H), (ox, oy + HALF_H)])},
            {"name": "Demo Footpath 1", "type": "footpath",
             "geometry": LineString([(ox - 150, oy + 90), (ox - 60, oy + 40), (ox - 6, oy + 40)])},
            {"name": "Demo Footpath 2", "type": "footpath",
             "geometry": LineString([(ox + 6, oy - 60), (ox + 90, oy - 60), (ox + 150, oy - 120)])},
        ],
        crs=UTM,
    )

    green = lu[lu["class"].isin(["trees_shrubs", "lawn_grass"])]
    tree_rows = []
    for _, row in green.iterrows():
        n = 6 if row["class"] == "trees_shrubs" else 1
        minx, miny, maxx, maxy = row.geometry.bounds
        placed = 0
        while placed < n:
            p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if row.geometry.contains(p):
                tree_rows.append({"species": "unspecified (demo)", "canopy_m": round(float(rng.uniform(3, 9)), 1),
                                  "geometry": p})
                placed += 1
    trees = gpd.GeoDataFrame(tree_rows, crs=UTM)
    boundary = gpd.GeoDataFrame([{"name": "Demo study area", "geometry": area}], crs=UTM)
    return {"boundary": boundary, "landuse": lu, "buildings": blds, "paths": paths, "trees": trees}


def build_raster(layers, rng):
    ox, oy = _origin()
    width, height = int(2 * HALF_W / PIXEL), int(2 * HALF_H / PIXEL)
    transform = from_origin(ox - HALF_W, oy + HALF_H, PIXEL, PIXEL)
    codes = {c: i + 1 for i, c in enumerate(CLASS_RGB)}
    lu = layers["landuse"]
    class_img = rasterize(((g, codes[c]) for g, c in zip(lu.geometry, lu["class"])),
                          out_shape=(height, width), transform=transform, fill=0, dtype="uint8")
    rgb = np.zeros((3, height, width), dtype="float32")
    for c, code in codes.items():
        sel = class_img == code
        for b in range(3):
            rgb[b][sel] = CLASS_RGB[c][b]
    roofs = rasterize(((g, 1) for g in layers["buildings"].geometry), out_shape=(height, width),
                      transform=transform, fill=0, dtype="uint8")
    rgb[:, roofs == 1] = np.array([205, 200, 195], dtype="float32")[:, None]
    # tree crowns
    yy, xx = np.mgrid[0:height, 0:width]
    for p, r in zip(layers["trees"].geometry, layers["trees"]["canopy_m"]):
        col, row = ~transform * (p.x, p.y)
        rad = r / 2 / PIXEL
        y0, y1 = int(max(row - rad, 0)), int(min(row + rad + 1, height))
        x0, x1 = int(max(col - rad, 0)), int(min(col + rad + 1, width))
        sub = (yy[y0:y1, x0:x1] - row) ** 2 + (xx[y0:y1, x0:x1] - col) ** 2 <= rad**2
        for b, v in enumerate((38, 82, 38)):
            rgb[b, y0:y1, x0:x1][sub] = v
    rgb += rng.normal(0, 9, size=rgb.shape).astype("float32")
    rgb = np.clip(rgb, 0, 255).astype("uint8")
    alpha = np.where(class_img > 0, 255, 0).astype("uint8")
    return np.concatenate([rgb, alpha[None]]), transform, class_img, codes


def write_raster(path: Path, data, transform):
    profile = {
        "driver": "GTiff", "width": data.shape[2], "height": data.shape[1], "count": 4, "dtype": "uint8",
        "crs": UTM, "transform": transform, "compress": "deflate", "tiled": True, "blockxsize": 256,
        "blockysize": 256, "photometric": "RGB", "alpha": "YES",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
        dst.update_tags(DATA_STATUS=STATUS, DESCRIPTION=config.DEMO_DISCLAIMER)


def build_reference_points(class_img, codes, transform, rng, n=120):
    """Synthetic 'ground truth' read from the synthetic class map - only to demo the workflow."""
    green_codes = {codes[c] for c, info in config.LANDUSE_CLASSES.items() if info.is_green and c in codes}
    rows = []
    h, w = class_img.shape
    while len(rows) < n:
        r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
        if class_img[r, c] == 0:
            continue
        x, y = transform * (c + 0.5, r + 0.5)
        rows.append({"is_green": int(class_img[r, c] in green_codes), "data_status": STATUS,
                     "geometry": Point(x, y)})
    return gpd.GeoDataFrame(rows, crs=UTM)


def build_glb(buildings: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> bytes:
    import trimesh

    ox, oy = boundary.geometry.iloc[0].centroid.coords[0]
    meshes = []
    ground = trimesh.creation.box(extents=(2 * HALF_W, 2 * HALF_H, 0.5))
    ground.apply_translation((0, 0, -0.25))
    ground.visual.vertex_colors = [200, 205, 195, 255]
    meshes.append(ground)
    for geom, h in zip(buildings.geometry, buildings["height_m"]):
        minx, miny, maxx, maxy = geom.bounds
        b = trimesh.creation.box(extents=(maxx - minx, maxy - miny, h))
        b.apply_translation(((minx + maxx) / 2 - ox, (miny + maxy) / 2 - oy, h / 2))
        b.visual.vertex_colors = [208, 59, 59, 255]
        meshes.append(b)
    scene = trimesh.Scene(meshes)
    # glTF is Y-up; rotate so Z (height) points up in the viewer.
    scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return scene.export(file_type="glb")


def build_ply(raster, transform, buildings, rng, n=25_000) -> bytes:
    import trimesh

    _, h, w = raster.shape
    rows = rng.integers(0, h, n)
    cols = rng.integers(0, w, n)
    keep = raster[3, rows, cols] > 0
    rows, cols = rows[keep], cols[keep]
    xs, ys = transform * (cols + 0.5, rows + 0.5)
    xs, ys = np.asarray(xs), np.asarray(ys)
    zs = rng.normal(0, 0.05, len(xs))
    roof_h = rasterize(((g, float(hh)) for g, hh in zip(buildings.geometry, buildings["height_m"])),
                       out_shape=(h, w), transform=transform, fill=0, dtype="float32")
    zs += roof_h[rows, cols]
    colors = np.vstack([raster[0, rows, cols], raster[1, rows, cols], raster[2, rows, cols],
                        np.full(len(xs), 255)]).T.astype("uint8")
    cloud = trimesh.PointCloud(np.vstack([xs - xs.mean(), ys - ys.mean(), zs]).T, colors=colors)
    return cloud.export(file_type="ply")


def meta(title, kind, category, description, **extra) -> DatasetMetadata:
    return DatasetMetadata(
        title=title, kind=kind, category=category, is_demo=True, description=description,
        source="Synthetic - generated by scripts/generate_demo_data.py (no drone flight)",
        capture_date="", processing_software="generate_demo_data.py",
        accuracy_statement="Not applicable - synthetic data with no relation to ground truth.",
        notes=config.DEMO_DISCLAIMER, **extra,
    )


def main(out_dir: Path = config.DEMO_DIR) -> None:
    rng = np.random.default_rng(SEED)
    vec_dir, ras_dir, d3_dir = (out_dir / k for k in ("vector", "raster", "3d"))
    for d in (vec_dir, ras_dir, d3_dir):
        d.mkdir(parents=True, exist_ok=True)

    layers = build_layers(rng)
    descriptions = {
        "boundary": ("Demo study-area boundary", "Rectangle around the invented demo layout."),
        "landuse": ("Demo land-use polygons", "Randomly assigned classes on a 30 m grid."),
        "buildings": ("Demo building footprints", "Invented footprints with invented heights (height_m)."),
        "paths": ("Demo roads and footpaths", "Invented centre-lines."),
        "trees": ("Demo tree points", "Random points inside demo vegetation polygons."),
    }
    for cat, gdf in layers.items():
        gdf = gdf.copy()
        gdf["data_status"] = STATUS
        path = vec_dir / f"demo_{cat}.geojson"
        gdf.to_crs("EPSG:4326").to_file(path, driver="GeoJSON", engine="pyogrio")
        title, desc = descriptions[cat]
        save_metadata(path, meta(title, "vector", cat, desc, crs="EPSG:4326", original_filename=path.name))

    raster, transform, class_img, codes = build_raster(layers, rng)
    tif = ras_dir / "demo_orthomosaic.tif"
    write_raster(tif, raster, transform)
    save_metadata(tif, meta("Demo synthetic orthomosaic", "raster", "orthomosaic",
                            "Colours painted from the demo land-use grid plus noise. Not imagery.",
                            crs=UTM, gsd_cm=PIXEL * 100, original_filename=tif.name))

    ref = build_reference_points(class_img, codes, transform, rng)
    ref_path = vec_dir / "demo_reference_points.geojson"
    ref.to_crs("EPSG:4326").to_file(ref_path, driver="GeoJSON", engine="pyogrio")
    save_metadata(ref_path, meta("Demo green-cover reference points", "vector", "other",
                                 "Labels read from the synthetic class map - demonstrates the validation "
                                 "workflow only; any accuracy computed from them is meaningless.",
                                 crs="EPSG:4326", original_filename=ref_path.name))

    glb = d3_dir / "demo_campus_blocks.glb"
    glb.write_bytes(build_glb(layers["buildings"], layers["boundary"]))
    save_metadata(glb, meta("Demo block model", "3d", "model_3d",
                            "Boxes extruded from the demo footprints. Not a photogrammetric mesh.",
                            original_filename=glb.name))
    ply = d3_dir / "demo_point_cloud.ply"
    ply.write_bytes(build_ply(raster, transform, layers["buildings"], rng))
    save_metadata(ply, meta("Demo point cloud", "3d", "point_cloud",
                            "Points sampled from the synthetic raster with demo roof heights.",
                            original_filename=ply.name))
    print(f"Demo data written to {out_dir}")


if __name__ == "__main__":
    main()
