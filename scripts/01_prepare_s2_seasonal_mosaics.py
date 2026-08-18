#!/usr/bin/env python3
"""
Prepare seasonal Sentinel-2 mosaics for the ECHO optical-SAR fusion workflow.

This script mosaics Google Earth Engine Sentinel-2 feature-export tiles for
12 seasonal windows spanning three annual cycles:

    2017-2018
    2022-2023
    2025-2026

Each seasonal Sentinel-2 product contains 18 predictors:

    - 10 surface-reflectance bands
    - 8 spectral indices

The script preserves the exported predictor values, creates a compressed
full-study-area seasonal GeoTIFF, assigns canonical band descriptions, and
audits the resulting raster metadata.

Expected input layout
---------------------
<data_root>/
    2017-2018/
        Winter 2017/
            Original/S2/
                S2_WPE_FEATURES_WINTER_2017-*.tif
        ...
    2022-2023/
        ...
    2025-2026/
        ...

Outputs are written to:

<data_root>/<period>/<season>/Mosaic/S2/
    S2_WPE_FEATURES_<SEASON>_<YEAR>_MOSAIC.tif

Notes
-----
- Input tiles are expected to have been exported at 10 m in EPSG:3577.
- Sentinel-2 preprocessing and feature generation are performed in GEE.
- This script only mosaics the exported feature tiles.
- No P1-P5 wetland clipping is performed here.
- No common validity mask is imposed across seasons or sensors.
- GDAL command-line tools `gdalbuildvrt` and `gdal_translate` must be
  available on the system PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import rasterio


# =============================================================================
# Workflow configuration
# =============================================================================

EXPECTED_BANDS = 18
EXPECTED_EPSG = 3577
EXPECTED_RESOLUTION = (10.0, 10.0)

# The original Earth Engine export commonly produced three tiles.
# This is diagnostic only; processing does not require exactly three.
EXPECTED_TILE_COUNT = 3


SEASONS = [
    ("2017-2018", "Winter 2017", "WINTER", "2017"),
    ("2017-2018", "Spring 2017", "SPRING", "2017"),
    ("2017-2018", "Summer 2018", "SUMMER", "2018"),
    ("2017-2018", "Autumn 2018", "AUTUMN", "2018"),

    ("2022-2023", "Winter 2022", "WINTER", "2022"),
    ("2022-2023", "Spring 2022", "SPRING", "2022"),
    ("2022-2023", "Summer 2023", "SUMMER", "2023"),
    ("2022-2023", "Autumn 2023", "AUTUMN", "2023"),

    ("2025-2026", "Winter 2025", "WINTER", "2025"),
    ("2025-2026", "Spring 2025", "SPRING", "2025"),
    ("2025-2026", "Summer 2026", "SUMMER", "2026"),
    ("2025-2026", "Autumn 2026", "AUTUMN", "2026"),
]


# Canonical order produced by gee/06_s2_feature_extraction.js.
S2_BAND_NAMES = [
    "S2_BLUE",
    "S2_GREEN",
    "S2_RED",
    "S2_RE1",
    "S2_RE2",
    "S2_RE3",
    "S2_NIR",
    "S2_NIR_NARROW",
    "S2_SWIR1",
    "S2_SWIR2",
    "S2_NDVI",
    "S2_EVI",
    "S2_NDWI",
    "S2_LSWI",
    "S2_MSAVI2",
    "S2_DBSI",
    "S2_GNDVI",
    "S2_TVI",
]


# =============================================================================
# Utility functions
# =============================================================================

def run_command(command: list[str]) -> None:
    """
    Run an external command and stop immediately if it fails.
    """

    print("Running:")
    print(" ".join(map(str, command)))

    subprocess.run(
        command,
        check=True,
    )


def require_gdal_tools() -> None:
    """
    Confirm that required GDAL command-line tools are available.
    """

    required_tools = [
        "gdalbuildvrt",
        "gdal_translate",
    ]

    missing = [
        tool
        for tool in required_tools
        if shutil.which(tool) is None
    ]

    if missing:
        raise RuntimeError(
            "Missing required GDAL command-line tool(s): "
            + ", ".join(missing)
            + ". Install GDAL and ensure the tools are available on PATH."
        )


def validate_input_tiles(tiles: list[Path]) -> None:
    """
    Validate Sentinel-2 feature-export tiles before mosaicking.

    Checks:
    - 18 predictor bands
    - EPSG:3577
    - 10 m spatial resolution
    """

    problems = []

    for tile in tiles:

        with rasterio.open(tile) as src:

            if src.count != EXPECTED_BANDS:
                problems.append(
                    f"{tile.name}: {src.count} bands; "
                    f"expected {EXPECTED_BANDS}."
                )

            epsg = (
                src.crs.to_epsg()
                if src.crs is not None
                else None
            )

            if epsg != EXPECTED_EPSG:
                problems.append(
                    f"{tile.name}: CRS is {src.crs}; "
                    f"expected EPSG:{EXPECTED_EPSG}."
                )

            if tuple(src.res) != EXPECTED_RESOLUTION:
                problems.append(
                    f"{tile.name}: resolution is {src.res}; "
                    f"expected {EXPECTED_RESOLUTION}."
                )

    if problems:
        raise ValueError(
            "Input-tile audit failed:\n  - "
            + "\n  - ".join(problems)
        )


def set_band_descriptions(path: Path) -> None:
    """
    Assign canonical Sentinel-2 predictor names to a seasonal mosaic.
    """

    with rasterio.open(path, "r+") as dst:

        if dst.count != len(S2_BAND_NAMES):
            raise ValueError(
                f"{path.name}: {dst.count} bands; "
                f"expected {len(S2_BAND_NAMES)}."
            )

        dst.descriptions = tuple(
            S2_BAND_NAMES
        )


def audit_mosaic(path: Path) -> dict:
    """
    Audit metadata for one completed Sentinel-2 seasonal mosaic.
    """

    with rasterio.open(path) as src:

        descriptions = tuple(
            src.descriptions or ()
        )

        epsg = (
            src.crs.to_epsg()
            if src.crs is not None
            else None
        )

        checks = {
            "bands_18":
                src.count == EXPECTED_BANDS,

            "crs_epsg_3577":
                epsg == EXPECTED_EPSG,

            "resolution_10m":
                tuple(src.res) == EXPECTED_RESOLUTION,

            "band_names_complete":
                (
                    len(descriptions) == EXPECTED_BANDS
                    and descriptions == tuple(S2_BAND_NAMES)
                ),
        }

        return {
            "path": str(path),
            "band_count": src.count,
            "crs": str(src.crs),
            "resolution": tuple(src.res),
            "width": src.width,
            "height": src.height,
            "bounds": tuple(src.bounds),
            "nodata": src.nodata,
            "checks": checks,
            "status":
                "PASS"
                if all(checks.values())
                else "CHECK",
        }


# =============================================================================
# Seasonal processing
# =============================================================================

def process_season(
    data_root: Path,
    period: str,
    season_folder: str,
    season_upper: str,
    year: str,
    overwrite: bool,
) -> dict:
    """
    Create and audit one full seasonal Sentinel-2 mosaic.
    """

    base = (
        data_root
        / period
        / season_folder
    )

    original_dir = (
        base
        / "Original"
        / "S2"
    )

    mosaic_dir = (
        base
        / "Mosaic"
        / "S2"
    )

    mosaic_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -------------------------------------------------------------------------
    # Find exported GEE tiles
    # -------------------------------------------------------------------------

    tile_pattern = (
        f"S2_WPE_FEATURES_"
        f"{season_upper}_{year}-*.tif"
    )

    tiles = sorted(
        original_dir.glob(
            tile_pattern
        )
    )


    output = (
        mosaic_dir
        / (
            f"S2_WPE_FEATURES_"
            f"{season_upper}_{year}_MOSAIC.tif"
        )
    )


    print("\n" + "=" * 88)

    print(
        f"Sentinel-2 seasonal mosaic: "
        f"{season_folder} ({period})"
    )

    print("=" * 88)


    print(
        "Input directory:",
        original_dir,
    )

    print(
        "Output:",
        output,
    )

    print(
        "Tiles found:",
        len(tiles),
    )


    # -------------------------------------------------------------------------
    # Input checks
    # -------------------------------------------------------------------------

    if not original_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{original_dir}"
        )


    if not tiles:
        raise FileNotFoundError(
            f"No Sentinel-2 tiles matched: "
            f"{original_dir / tile_pattern}"
        )


    for tile in tiles:
        print(
            "  -",
            tile.name,
        )


    if len(tiles) != EXPECTED_TILE_COUNT:

        print(
            f"WARNING: expected approximately "
            f"{EXPECTED_TILE_COUNT} exported tiles "
            f"from the original workflow, but found "
            f"{len(tiles)}."
        )

        print(
            "Continuing because Earth Engine export "
            "tiling may vary."
        )


    validate_input_tiles(
        tiles
    )


    # -------------------------------------------------------------------------
    # Existing output
    # -------------------------------------------------------------------------

    if output.exists() and not overwrite:

        print(
            "Output already exists; "
            "keeping existing mosaic."
        )

        # Ensure canonical predictor descriptions exist.
        set_band_descriptions(
            output
        )

        result = audit_mosaic(
            output
        )

        result.update({
            "period": period,
            "season": season_folder,
            "tiles": len(tiles),
        })

        return result


    if output.exists():
        output.unlink()


    # -------------------------------------------------------------------------
    # Build seasonal mosaic
    # -------------------------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="echo_s2_"
    ) as temp_dir:

        vrt = (
            Path(temp_dir)
            / f"S2_{season_upper}_{year}.vrt"
        )


        # Build a virtual mosaic from the Earth Engine tiles.
        run_command(
            [
                "gdalbuildvrt",
                str(vrt),
            ]
            + [
                str(tile)
                for tile in tiles
            ]
        )


        # Convert the VRT to a compressed GeoTIFF.
        run_command(
            [
                "gdal_translate",

                str(vrt),
                str(output),

                "-ot",
                "Float32",

                "-co",
                "COMPRESS=LZW",

                "-co",
                "TILED=YES",

                "-co",
                "BIGTIFF=YES",
            ]
        )


    # -------------------------------------------------------------------------
    # Add predictor descriptions
    # -------------------------------------------------------------------------

    set_band_descriptions(
        output
    )


    # -------------------------------------------------------------------------
    # Final metadata audit
    # -------------------------------------------------------------------------

    result = audit_mosaic(
        output
    )

    result.update({
        "period": period,
        "season": season_folder,
        "tiles": len(tiles),
    })

    return result


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    repo_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    default_data_root = (
        repo_root
        / "data"
        / "images"
    )


    parser = argparse.ArgumentParser(
        description=(
            "Mosaic seasonal Sentinel-2 "
            "Google Earth Engine feature exports."
        )
    )


    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help=(
            "Root directory containing the "
            "period/season image folders. "
            f"Default: {default_data_root}"
        ),
    )


    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Recreate seasonal mosaics even "
            "when outputs already exist."
        ),
    )


    return parser.parse_args()


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    """
    Run the complete Sentinel-2 seasonal mosaic workflow.
    """

    args = parse_args()

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )


    require_gdal_tools()


    print(
        "ECHO Sentinel-2 seasonal mosaic preparation"
    )

    print(
        "Data root:",
        data_root,
    )

    print(
        "Expected predictors per season:",
        EXPECTED_BANDS,
    )

    print(
        "Expected grid: EPSG:3577 at 10 m"
    )


    results = []
    failures = []


    # -------------------------------------------------------------------------
    # Process all 12 seasonal products
    # -------------------------------------------------------------------------

    for (
        period,
        season_folder,
        season_upper,
        year,
    ) in SEASONS:

        try:

            result = process_season(
                data_root=data_root,
                period=period,
                season_folder=season_folder,
                season_upper=season_upper,
                year=year,
                overwrite=args.overwrite,
            )


            results.append(
                result
            )


            print(
                "Audit status:",
                result["status"],
            )

            print(
                "Band count:",
                result["band_count"],
            )

            print(
                "CRS:",
                result["crs"],
            )

            print(
                "Resolution:",
                result["resolution"],
            )


            if result["status"] != "PASS":

                print(
                    "Checks:",
                    result["checks"],
                )


        except Exception as exc:

            failures.append(
                (
                    season_folder,
                    str(exc),
                )
            )

            print(
                f"ERROR [{season_folder}]: "
                f"{exc}"
            )


    # -------------------------------------------------------------------------
    # Final audit summary
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 88
    )

    print(
        "FINAL SENTINEL-2 SEASONAL MOSAIC AUDIT"
    )

    print(
        "=" * 88
    )


    for result in results:

        print(
            f"{result['season']:<15} | "
            f"tiles={result['tiles']:<2} | "
            f"bands={result['band_count']:<3} | "
            f"{result['status']}"
        )


    # -------------------------------------------------------------------------
    # Fail when processing or audit problems occur
    # -------------------------------------------------------------------------

    if failures:

        print(
            "\nFailures:"
        )

        for season, message in failures:

            print(
                f"  - {season}: "
                f"{message}"
            )

        raise SystemExit(1)


    if any(
        result["status"] != "PASS"
        for result in results
    ):

        raise SystemExit(
            "One or more seasonal Sentinel-2 mosaics "
            "failed the metadata audit."
        )


    print(
        "\nAll 12 Sentinel-2 seasonal "
        "mosaics passed the audit."
    )


if __name__ == "__main__":
    main()