#!/usr/bin/env python3
"""
Build annual Sentinel-1 predictor stacks for the ECHO workflow.

The script combines four seasonal Sentinel-1 mosaics into one annual
104-band stack for each analysis period:

    2017-2018
    2022-2023
    2025-2026

Each seasonal mosaic contains 26 Sentinel-1 predictors:

    2 backscatter predictors
    10 polarisation/algebraic predictors
    14 GLCM texture predictors

Annual band order
-----------------
Bands   1-26  : Winter
Bands  27-52  : Spring
Bands  53-78  : Summer
Bands 79-104  : Autumn

Predictor names encode season, acquisition year, sensor, and feature identity.

Example:
    WINTER_2025_S1_VV_dB
    SPRING_2025_S1_RVI
    SUMMER_2026_S1_GLCM_ENTROPY_VH

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

EXPECTED_SEASONAL_BANDS = 26
EXPECTED_ANNUAL_BANDS = 104

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


# Canonical per-season predictor order produced by the cleaned
# gee/04_s1_feature_extraction.js workflow.
S1_FEATURE_NAMES = [
    "S1_VV_dB",
    "S1_VH_dB",
    "S1_NDPI",
    "S1_NRPB",
    "S1_VV_VH_RATIO",
    "S1_VH_VV_RATIO",
    "S1_RVI",
    "S1_SUM",
    "S1_DIFF",
    "S1_PROD",
    "S1_VDDI",
    "S1_LOG_RATIO",

    "S1_GLCM_ASM_VV",
    "S1_GLCM_CORRELATION_VV",
    "S1_GLCM_VARIANCE_VV",
    "S1_GLCM_IDM_VV",
    "S1_GLCM_SUMAVE_VV",
    "S1_GLCM_ENTROPY_VV",
    "S1_GLCM_CONTRAST_VV",

    "S1_GLCM_ASM_VH",
    "S1_GLCM_CORRELATION_VH",
    "S1_GLCM_VARIANCE_VH",
    "S1_GLCM_IDM_VH",
    "S1_GLCM_SUMAVE_VH",
    "S1_GLCM_ENTROPY_VH",
    "S1_GLCM_CONTRAST_VH",
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
    """Return the expected path to one seasonal Sentinel-1 mosaic."""

    return (
        data_root
        / period
        / season_folder
        / "Mosaic"
        / "S1"
        / f"S1_WPE_FEATURES_{season_upper}_{year}_MOSAIC.tif"
    )


def annual_stack_path(
    data_root: Path,
    period: str,
) -> Path:
    """Return the output path for one annual Sentinel-1 stack."""

    period_tag = period.replace("-", "_")

    return (
        data_root
        / period
        / "Annual_Stack"
        / "S1_full_mosaic"
        / f"S1_{period_tag}_FULL_ANNUAL_STACK.tif"
    )


# =============================================================================
# Metadata and naming helpers
# =============================================================================

def clean_band_name(name: str) -> str:
    """Convert a raster-band description to a filesystem-safe identifier."""

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


def get_feature_names(dataset: rasterio.io.DatasetReader) -> list[str]:
    """
    Obtain the 26 Sentinel-1 feature identities from a seasonal raster.

    Existing band descriptions are used when present. If descriptions are
    missing, the canonical Sentinel-1 predictor order is used.
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
            name = S1_FEATURE_NAMES[index]

        # Ensure all names explicitly identify Sentinel-1.
        if not name.startswith("S1_"):
            name = f"S1_{name}"

        names.append(name)

    return names


def make_annual_band_names(
    datasets: list[rasterio.io.DatasetReader],
    season_list: list[tuple[str, str, str]],
) -> tuple[str, ...]:
    """
    Build all 104 annual predictor names.

    Format:
        <SEASON>_<YEAR>_<SENSOR>_<FEATURE>

    Example:
        WINTER_2025_S1_VV_dB
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
            "Duplicate annual Sentinel-1 predictor names detected."
        )

    return tuple(annual_names)


# =============================================================================
# Alignment audit
# =============================================================================

def audit_seasonal_alignment(
    datasets: list[rasterio.io.DatasetReader],
    paths: list[Path],
) -> None:
    """
    Require all four seasonal mosaics to use exactly the same spatial grid.
    """

    reference = datasets[0]

    problems = []

    for path, dataset in zip(
        paths,
        datasets,
    ):

        checks = {
            "26_bands":
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
                    checks,
                )
            )

    if problems:

        lines = [
            "Seasonal Sentinel-1 alignment audit failed."
        ]

        for path, failed, _checks in problems:

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
    Build one 104-band Sentinel-1 annual stack.
    """

    print("\n" + "=" * 88)
    print(
        f"Building Sentinel-1 annual stack: {period}"
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
            "Missing seasonal Sentinel-1 mosaic(s):\n  - "
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
                "Existing annual stack failed audit. "
                "Run with --overwrite to rebuild it."
            )

        return audit


    # -------------------------------------------------------------------------
    # Open seasonal mosaics
    # -------------------------------------------------------------------------

    datasets = [
        rasterio.open(path)
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
        # Predictor naming audit
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
            prefix="echo_s1_annual_",
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
                # Dataset-level metadata
                # -------------------------------------------------------------

                dst.update_tags(
                    sensor="Sentinel-1",
                    period=period,
                    stack_type="S1_FULL_MOSAIC_ANNUAL_STACK",
                    seasons="Winter,Spring,Summer,Autumn",
                    seasonal_predictors="26",
                    annual_predictors="104",
                    common_valid_mask="False",
                    nodata=str(NODATA),
                    band_order=(
                        "1-26 Winter; "
                        "27-52 Spring; "
                        "53-78 Summer; "
                        "79-104 Autumn"
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
                        # Preserve per-season validity independently
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
            f"Created stack failed final audit: "
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
    Audit one completed Sentinel-1 annual stack.
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
            "bands_104":
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

            "104_band_descriptions":
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
    """Parse command-line arguments."""

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
            "Build 104-band annual Sentinel-1 "
            "stacks from four seasonal mosaics."
        )
    )


    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help=(
            "Root directory containing seasonal "
            "Sentinel-1 mosaics. "
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
    """Build and audit annual Sentinel-1 stacks."""

    args = parse_args()

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )


    print(
        "ECHO Sentinel-1 annual stack builder"
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
    # Summary
    # =========================================================================

    print(
        "\n" + "=" * 88
    )

    print(
        "FINAL SENTINEL-1 ANNUAL STACK AUDIT"
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
            f"bands={result['band_count']:<3} | "
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
            "Not all annual Sentinel-1 stacks were created."
        )


    if any(
        result["status"] != "PASS"
        for _period, result in results
    ):

        raise SystemExit(
            "One or more Sentinel-1 annual stacks "
            "failed the metadata audit."
        )


    print(
        "\nAll three Sentinel-1 annual stacks "
        "passed the audit."
    )


if __name__ == "__main__":
    main()