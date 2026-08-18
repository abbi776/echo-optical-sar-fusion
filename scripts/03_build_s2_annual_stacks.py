#!/usr/bin/env python3
"""
Build annual Sentinel-2 predictor stacks for the ECHO workflow.

The script combines four seasonal Sentinel-2 mosaics into one annual
72-band stack for each analysis period:

    2017-2018
    2022-2023
    2025-2026

Each seasonal mosaic contains 18 Sentinel-2 predictors:

    10 surface-reflectance bands
    8 spectral-index predictors

Annual band order
-----------------
Bands  1-18 : Winter
Bands 19-36 : Spring
Bands 37-54 : Summer
Bands 55-72 : Autumn

Predictor names encode season, acquisition year, sensor, and feature identity.

Examples:
    WINTER_2025_S2_BLUE
    SPRING_2025_S2_NDVI
    SUMMER_2026_S2_SWIR1

Important
---------
- Seasonal rasters must share the same CRS, transform, dimensions,
  resolution, and spatial bounds.
- Expected grid: EPSG:3577 at 10 m.
- No common validity mask is applied across seasons.
- Invalid cells are written as -9999.
- Processing is block-wise to avoid loading the complete annual stack
  into memory.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio


# =============================================================================
# Configuration
# =============================================================================

NODATA = -9999.0

EXPECTED_SEASONAL_BANDS = 18
EXPECTED_ANNUAL_BANDS = 72

EXPECTED_EPSG = 3577
EXPECTED_RESOLUTION = (10.0, 10.0)


PERIODS = [
    (
        "2017-2018",
        [
            ("Winter 2017", "WINTER", "2017"),
            ("Spring 2017", "SPRING", "2017"),
            ("Summer 2018", "SUMMER", "2018"),
            ("Autumn 2018", "AUTUMN", "2018"),
        ],
    ),
    (
        "2022-2023",
        [
            ("Winter 2022", "WINTER", "2022"),
            ("Spring 2022", "SPRING", "2022"),
            ("Summer 2023", "SUMMER", "2023"),
            ("Autumn 2023", "AUTUMN", "2023"),
        ],
    ),
    (
        "2025-2026",
        [
            ("Winter 2025", "WINTER", "2025"),
            ("Spring 2025", "SPRING", "2025"),
            ("Summer 2026", "SUMMER", "2026"),
            ("Autumn 2026", "AUTUMN", "2026"),
        ],
    ),
]


# Canonical per-season predictor order produced by
# gee/06_s2_feature_extraction.js.
S2_FEATURE_NAMES = [
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
# Path helpers
# =============================================================================

def seasonal_mosaic_path(
    data_root: Path,
    period: str,
    season_folder: str,
    season_upper: str,
    year: str,
) -> Path:
    """Return the expected path to one seasonal Sentinel-2 mosaic."""

    return (
        data_root
        / period
        / season_folder
        / "Mosaic"
        / "S2"
        / f"S2_WPE_FEATURES_{season_upper}_{year}_MOSAIC.tif"
    )


def annual_stack_path(
    data_root: Path,
    period: str,
) -> Path:
    """Return the output path for one annual Sentinel-2 stack."""

    period_tag = period.replace("-", "_")

    return (
        data_root
        / period
        / "Annual_Stack"
        / "S2_full_mosaic"
        / f"S2_{period_tag}_FULL_ANNUAL_STACK.tif"
    )


# =============================================================================
# Metadata and naming helpers
# =============================================================================

def clean_band_name(name: str) -> str:
    """Convert a raster-band description to a clean predictor identifier."""

    name = str(name).strip()

    for character in [
        " ",
        "-",
        "/",
        "\\",
        "(",
        ")",
        ".",
        ",",
        ":",
        ";",
    ]:
        name = name.replace(character, "_")

    while "__" in name:
        name = name.replace("__", "_")

    return name.strip("_")


def get_feature_names(
    dataset: rasterio.io.DatasetReader,
) -> list[str]:
    """
    Obtain the 18 Sentinel-2 feature identities from a seasonal raster.

    Existing band descriptions are used when available. If descriptions
    are missing, the canonical Sentinel-2 feature order is used.
    """

    if dataset.count != EXPECTED_SEASONAL_BANDS:
        raise ValueError(
            f"Expected {EXPECTED_SEASONAL_BANDS} seasonal bands, "
            f"found {dataset.count}."
        )

    descriptions = dataset.descriptions

    names = []

    for index in range(dataset.count):

        description = (
            descriptions[index]
            if descriptions
            else None
        )

        if (
            description is not None
            and str(description).strip()
            and str(description).strip().lower() != "none"
        ):
            name = clean_band_name(
                description
            )

        else:
            name = S2_FEATURE_NAMES[index]

        if not name.startswith("S2_"):
            name = f"S2_{name}"

        names.append(
            name
        )

    return names


def make_annual_band_names(
    datasets: list[rasterio.io.DatasetReader],
    season_list: list[tuple[str, str, str]],
) -> tuple[str, ...]:
    """
    Build all 72 annual predictor names.

    Format:
        <SEASON>_<YEAR>_<SENSOR>_<FEATURE>

    Example:
        WINTER_2025_S2_NDVI
    """

    annual_names = []

    for dataset, (
        _season_folder,
        season_upper,
        year,
    ) in zip(
        datasets,
        season_list,
    ):

        feature_names = get_feature_names(
            dataset
        )

        for feature_name in feature_names:

            annual_names.append(
                f"{season_upper}_{year}_{feature_name}"
            )

    if len(annual_names) != EXPECTED_ANNUAL_BANDS:
        raise RuntimeError(
            f"Generated {len(annual_names)} annual band names; "
            f"expected {EXPECTED_ANNUAL_BANDS}."
        )

    if len(set(annual_names)) != EXPECTED_ANNUAL_BANDS:
        raise RuntimeError(
            "Duplicate annual Sentinel-2 predictor names detected."
        )

    return tuple(
        annual_names
    )


# =============================================================================
# Alignment audit
# =============================================================================

def audit_seasonal_alignment(
    datasets: list[rasterio.io.DatasetReader],
    paths: list[Path],
) -> None:
    """
    Require all four seasonal Sentinel-2 mosaics to share the same grid.
    """

    reference = datasets[0]

    problems = []

    for path, dataset in zip(
        paths,
        datasets,
    ):

        checks = {
            "18_bands":
                dataset.count == EXPECTED_SEASONAL_BANDS,

            "EPSG_3577":
                (
                    dataset.crs is not None
                    and dataset.crs.to_epsg() == EXPECTED_EPSG
                ),

            "10_m_resolution":
                tuple(dataset.res) == EXPECTED_RESOLUTION,

            "same_CRS":
                dataset.crs == reference.crs,

            "same_transform":
                dataset.transform == reference.transform,

            "same_dimensions":
                (
                    dataset.width == reference.width
                    and dataset.height == reference.height
                ),

            "same_resolution":
                dataset.res == reference.res,

            "same_bounds":
                dataset.bounds == reference.bounds,
        }

        failed = [
            key
            for key, passed in checks.items()
            if not passed
        ]

        if failed:
            problems.append(
                (
                    path,
                    failed,
                )
            )

    if problems:

        lines = [
            "Seasonal Sentinel-2 alignment audit failed."
        ]

        for path, failed in problems:
            lines.append(
                f"{path}: "
                + ", ".join(failed)
            )

        raise ValueError(
            "\n".join(lines)
        )


# =============================================================================
# Annual stack creation
# =============================================================================

def build_annual_stack(
    data_root: Path,
    period: str,
    season_list: list[tuple[str, str, str]],
    overwrite: bool,
) -> dict:
    """
    Build one 72-band Sentinel-2 annual stack.
    """

    print("\n" + "=" * 88)

    print(
        f"Building Sentinel-2 annual stack: {period}"
    )

    print("=" * 88)


    # -------------------------------------------------------------------------
    # Locate seasonal inputs
    # -------------------------------------------------------------------------

    input_paths = [
        seasonal_mosaic_path(
            data_root,
            period,
            season_folder,
            season_upper,
            year,
        )
        for (
            season_folder,
            season_upper,
            year,
        ) in season_list
    ]


    for path in input_paths:
        print(
            "Input:",
            path,
        )


    missing = [
        path
        for path in input_paths
        if not path.exists()
    ]


    if missing:

        raise FileNotFoundError(
            "Missing seasonal Sentinel-2 mosaic(s):\n  - "
            + "\n  - ".join(
                map(str, missing)
            )
        )


    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    output = annual_stack_path(
        data_root,
        period,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    if output.exists() and not overwrite:

        print(
            "Output already exists; auditing existing stack:"
        )

        print(
            output
        )

        audit = audit_annual_stack(
            output
        )

        if audit["status"] != "PASS":
            raise RuntimeError(
                "Existing annual Sentinel-2 stack failed audit. "
                "Run with --overwrite to rebuild it."
            )

        return audit


    # -------------------------------------------------------------------------
    # Open seasonal mosaics
    # -------------------------------------------------------------------------

    datasets = [
        rasterio.open(
            path
        )
        for path in input_paths
    ]


    try:

        # ---------------------------------------------------------------------
        # Spatial alignment audit
        # ---------------------------------------------------------------------

        audit_seasonal_alignment(
            datasets,
            input_paths,
        )

        print(
            "Seasonal spatial-alignment audit: PASS"
        )


        # ---------------------------------------------------------------------
        # Predictor-name audit
        # ---------------------------------------------------------------------

        annual_band_names = make_annual_band_names(
            datasets,
            season_list,
        )

        print(
            "Annual predictor-name audit: PASS"
        )

        print(
            "First predictor:",
            annual_band_names[0],
        )

        print(
            "Last predictor:",
            annual_band_names[-1],
        )


        # ---------------------------------------------------------------------
        # Output profile
        # ---------------------------------------------------------------------

        reference = datasets[0]

        profile = reference.profile.copy()

        profile.update(
            driver="GTiff",
            count=EXPECTED_ANNUAL_BANDS,
            dtype="float32",
            nodata=NODATA,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="YES",
        )


        # ---------------------------------------------------------------------
        # Temporary output
        # ---------------------------------------------------------------------

        with tempfile.NamedTemporaryFile(
            prefix="echo_s2_annual_",
            suffix=".tif",
            dir=output.parent,
            delete=False,
        ) as tmp:

            temp_path = Path(
                tmp.name
            )


        try:

            with rasterio.open(
                temp_path,
                "w",
                **profile,
            ) as dst:

                dst.descriptions = annual_band_names


                # -------------------------------------------------------------
                # Dataset metadata
                # -------------------------------------------------------------

                dst.update_tags(
                    sensor="Sentinel-2",
                    period=period,
                    stack_type="S2_FULL_MOSAIC_ANNUAL_STACK",
                    seasons="Winter,Spring,Summer,Autumn",
                    seasonal_predictors="18",
                    annual_predictors="72",
                    common_valid_mask="False",
                    nodata=str(NODATA),
                    band_order=(
                        "1-18 Winter; "
                        "19-36 Spring; "
                        "37-54 Summer; "
                        "55-72 Autumn"
                    ),
                )


                # -------------------------------------------------------------
                # Block-wise annual stacking
                # -------------------------------------------------------------

                windows = list(
                    reference.block_windows(1)
                )

                total_windows = len(
                    windows
                )


                for window_number, (
                    _block_index,
                    window,
                ) in enumerate(
                    windows,
                    start=1,
                ):

                    destination_band = 1


                    for dataset in datasets:

                        data = dataset.read(
                            window=window,
                            out_dtype="float32",
                        )


                        # -----------------------------------------------------
                        # Preserve each season's validity independently
                        # -----------------------------------------------------

                        invalid = ~np.isfinite(
                            data
                        )


                        if dataset.nodata is not None:

                            invalid |= np.isclose(
                                data,
                                dataset.nodata,
                            )


                        data[
                            invalid
                        ] = NODATA


                        first_band = destination_band

                        last_band = (
                            first_band
                            + dataset.count
                            - 1
                        )


                        dst.write(
                            data,
                            indexes=list(
                                range(
                                    first_band,
                                    last_band + 1,
                                )
                            ),
                            window=window,
                        )


                        destination_band = (
                            last_band + 1
                        )


                    if (
                        window_number % 100 == 0
                        or window_number == total_windows
                    ):

                        print(
                            f"Written blocks: "
                            f"{window_number}/"
                            f"{total_windows}"
                        )


            # -----------------------------------------------------------------
            # Replace final output only after successful write
            # -----------------------------------------------------------------

            if output.exists():
                output.unlink()

            os.replace(
                temp_path,
                output,
            )


        except Exception:

            if temp_path.exists():
                temp_path.unlink()

            raise


    finally:

        for dataset in datasets:
            dataset.close()


    # -------------------------------------------------------------------------
    # Final audit
    # -------------------------------------------------------------------------

    audit = audit_annual_stack(
        output
    )


    if audit["status"] != "PASS":

        raise RuntimeError(
            f"Created Sentinel-2 stack failed final audit: "
            f"{audit['checks']}"
        )


    print(
        "Annual stack audit: PASS"
    )

    print(
        "Output:",
        output,
    )


    return audit


# =============================================================================
# Annual-stack audit
# =============================================================================

def audit_annual_stack(
    path: Path,
) -> dict:
    """
    Audit one completed 72-band Sentinel-2 annual stack.
    """

    if not path.exists():

        return {
            "path": str(path),
            "status": "MISSING",
            "checks": {},
        }


    with rasterio.open(
        path
    ) as src:

        descriptions = tuple(
            src.descriptions or ()
        )


        checks = {
            "bands_72":
                src.count == EXPECTED_ANNUAL_BANDS,

            "EPSG_3577":
                (
                    src.crs is not None
                    and src.crs.to_epsg() == EXPECTED_EPSG
                ),

            "resolution_10m":
                tuple(src.res) == EXPECTED_RESOLUTION,

            "nodata_-9999":
                src.nodata == NODATA,

            "72_band_descriptions":
                (
                    len(descriptions) == EXPECTED_ANNUAL_BANDS
                    and all(
                        description is not None
                        and str(description).strip()
                        for description in descriptions
                    )
                ),

            "unique_band_names":
                (
                    len(set(descriptions))
                    == EXPECTED_ANNUAL_BANDS
                ),
        }


        return {
            "path": str(path),
            "band_count": src.count,
            "crs": str(src.crs),
            "resolution": tuple(src.res),
            "nodata": src.nodata,
            "width": src.width,
            "height": src.height,
            "bounds": tuple(src.bounds),

            "first_band":
                (
                    descriptions[0]
                    if descriptions
                    else None
                ),

            "last_band":
                (
                    descriptions[-1]
                    if descriptions
                    else None
                ),

            "checks": checks,

            "status":
                (
                    "PASS"
                    if all(checks.values())
                    else "CHECK"
                ),
        }


# =============================================================================
# Command-line interface
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
            "Build 72-band annual Sentinel-2 "
            "stacks from four seasonal mosaics."
        )
    )


    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help=(
            "Root directory containing seasonal "
            "Sentinel-2 mosaics. "
            f"Default: {default_data_root}"
        ),
    )


    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rebuild annual stacks even when "
            "outputs already exist."
        ),
    )


    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Build and audit annual Sentinel-2 stacks.
    """

    args = parse_args()

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )


    print(
        "ECHO Sentinel-2 annual stack builder"
    )

    print(
        "Data root:",
        data_root,
    )

    print(
        "Seasonal predictors:",
        EXPECTED_SEASONAL_BANDS,
    )

    print(
        "Annual predictors:",
        EXPECTED_ANNUAL_BANDS,
    )

    print(
        "Season order: "
        "Winter -> Spring -> Summer -> Autumn"
    )


    results = []
    failures = []


    for (
        period,
        season_list,
    ) in PERIODS:

        try:

            result = build_annual_stack(
                data_root=data_root,
                period=period,
                season_list=season_list,
                overwrite=args.overwrite,
            )

            results.append(
                (
                    period,
                    result,
                )
            )


        except Exception as exc:

            failures.append(
                (
                    period,
                    str(exc),
                )
            )

            print(
                f"\nERROR [{period}]: "
                f"{exc}"
            )


    # =========================================================================
    # Final summary
    # =========================================================================

    print(
        "\n" + "=" * 88
    )

    print(
        "FINAL SENTINEL-2 ANNUAL STACK AUDIT"
    )

    print(
        "=" * 88
    )


    for (
        period,
        result,
    ) in results:

        print(
            f"{period:<10} | "
            f"bands={result['band_count']:<2} | "
            f"{result['status']} | "
            f"{result['first_band']} -> "
            f"{result['last_band']}"
        )


    if failures:

        print(
            "\nFailures:"
        )

        for (
            period,
            message,
        ) in failures:

            print(
                f"  - {period}: "
                f"{message}"
            )

        raise SystemExit(1)


    if len(results) != len(PERIODS):

        raise SystemExit(
            "Not all annual Sentinel-2 stacks were created."
        )


    if any(
        result["status"] != "PASS"
        for _period, result in results
    ):

        raise SystemExit(
            "One or more Sentinel-2 annual stacks "
            "failed the metadata audit."
        )


    print(
        "\nAll three Sentinel-2 annual stacks "
        "passed the audit."
    )


if __name__ == "__main__":
    main()