#!/usr/bin/env python3
"""
Build annual Sentinel-1 + Sentinel-2 fusion stacks for the ECHO workflow.

The script combines:

    104-band annual Sentinel-1 stack
    72-band annual Sentinel-2 stack

into one 176-band annual fusion stack for each analysis period:

    2017-2018
    2022-2023
    2025-2026

Fusion band order
-----------------
Within each season:

    26 Sentinel-1 predictors
    followed by
    18 Sentinel-2 predictors

Season order:

    Winter
    Spring
    Summer
    Autumn

Therefore:

    Bands   1-44   : Winter
        1-26       : Winter Sentinel-1
        27-44      : Winter Sentinel-2

    Bands  45-88   : Spring
        45-70      : Spring Sentinel-1
        71-88      : Spring Sentinel-2

    Bands  89-132  : Summer
        89-114     : Summer Sentinel-1
        115-132    : Summer Sentinel-2

    Bands 133-176  : Autumn
        133-158    : Autumn Sentinel-1
        159-176    : Autumn Sentinel-2

Important
---------
- S1 and S2 annual rasters must be on exactly the same spatial grid.
- Expected grid: EPSG:3577 at 10 m.
- No common validity mask is imposed between sensors or seasons.
- Invalid cells remain -9999 independently for each predictor.
- Predictor identity and order are audited before and after fusion.
- Processing is block-wise to avoid loading the complete 176-band
  raster into memory.
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

S1_SEASONAL_BANDS = 26
S2_SEASONAL_BANDS = 18

S1_ANNUAL_BANDS = 104
S2_ANNUAL_BANDS = 72

FUSION_SEASONAL_BANDS = 44
FUSION_ANNUAL_BANDS = 176

EXPECTED_EPSG = 3577
EXPECTED_RESOLUTION = (10.0, 10.0)

SEASONS = [
    "WINTER",
    "SPRING",
    "SUMMER",
    "AUTUMN",
]

PERIODS = [
    "2017-2018",
    "2022-2023",
    "2025-2026",
]


# =============================================================================
# Path helpers
# =============================================================================

def s1_annual_path(
    data_root: Path,
    period: str,
) -> Path:
    """Return the annual Sentinel-1 stack path."""

    period_tag = period.replace("-", "_")

    return (
        data_root
        / period
        / "Annual_Stack"
        / "S1_full_mosaic"
        / f"S1_{period_tag}_FULL_ANNUAL_STACK.tif"
    )


def s2_annual_path(
    data_root: Path,
    period: str,
) -> Path:
    """Return the annual Sentinel-2 stack path."""

    period_tag = period.replace("-", "_")

    return (
        data_root
        / period
        / "Annual_Stack"
        / "S2_full_mosaic"
        / f"S2_{period_tag}_FULL_ANNUAL_STACK.tif"
    )


def fusion_annual_path(
    data_root: Path,
    period: str,
) -> Path:
    """Return the final 176-band fusion-stack path."""

    period_tag = period.replace("-", "_")

    return (
        data_root
        / period
        / "Annual_Stack"
        / "Fusion_full_mosaic"
        / f"S1S2_{period_tag}_FULL_FUSION_STACK.tif"
    )


# =============================================================================
# Grid-alignment audit
# =============================================================================

def audit_sensor_alignment(
    s1: rasterio.io.DatasetReader,
    s2: rasterio.io.DatasetReader,
) -> None:
    """
    Require annual Sentinel-1 and Sentinel-2 stacks to use exactly
    the same spatial grid.
    """

    checks = {
        "S1_band_count":
            s1.count == S1_ANNUAL_BANDS,

        "S2_band_count":
            s2.count == S2_ANNUAL_BANDS,

        "S1_EPSG_3577":
            (
                s1.crs is not None
                and s1.crs.to_epsg() == EXPECTED_EPSG
            ),

        "S2_EPSG_3577":
            (
                s2.crs is not None
                and s2.crs.to_epsg() == EXPECTED_EPSG
            ),

        "S1_resolution_10m":
            tuple(s1.res) == EXPECTED_RESOLUTION,

        "S2_resolution_10m":
            tuple(s2.res) == EXPECTED_RESOLUTION,

        "same_CRS":
            s1.crs == s2.crs,

        "same_transform":
            s1.transform == s2.transform,

        "same_dimensions":
            (
                s1.width == s2.width
                and s1.height == s2.height
            ),

        "same_resolution":
            s1.res == s2.res,

        "same_bounds":
            s1.bounds == s2.bounds,
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise ValueError(
            "S1/S2 annual-stack alignment audit failed:\n  - "
            + "\n  - ".join(failed)
        )


# =============================================================================
# Predictor-name audit
# =============================================================================

def get_band_names(
    dataset: rasterio.io.DatasetReader,
    expected_count: int,
    sensor: str,
) -> tuple[str, ...]:
    """
    Read and validate annual predictor names from raster descriptions.
    """

    if dataset.count != expected_count:
        raise ValueError(
            f"{sensor}: found {dataset.count} bands; "
            f"expected {expected_count}."
        )

    descriptions = tuple(
        dataset.descriptions or ()
    )

    if len(descriptions) != expected_count:
        raise ValueError(
            f"{sensor}: missing band descriptions."
        )

    if any(
        description is None
        or not str(description).strip()
        for description in descriptions
    ):
        raise ValueError(
            f"{sensor}: one or more annual predictor names are missing."
        )

    if len(set(descriptions)) != expected_count:
        raise ValueError(
            f"{sensor}: duplicate annual predictor names detected."
        )

    return descriptions


def validate_seasonal_blocks(
    names: tuple[str, ...],
    seasonal_band_count: int,
    sensor_token: str,
) -> None:
    """
    Verify that annual predictor names are grouped by season in the
    expected Winter -> Spring -> Summer -> Autumn order.
    """

    if len(names) != seasonal_band_count * 4:
        raise ValueError(
            "Unexpected annual predictor count."
        )

    for season_index, season in enumerate(SEASONS):

        start = season_index * seasonal_band_count
        end = start + seasonal_band_count

        block = names[start:end]

        if not all(
            name.startswith(f"{season}_")
            for name in block
        ):
            raise ValueError(
                f"{sensor_token}: predictor block for {season} "
                "does not contain the expected season prefix."
            )

        if not all(
            f"_{sensor_token}_" in name
            for name in block
        ):
            raise ValueError(
                f"{sensor_token}: predictor block for {season} "
                f"does not consistently identify {sensor_token}."
            )


def build_fusion_band_names(
    s1_names: tuple[str, ...],
    s2_names: tuple[str, ...],
) -> tuple[str, ...]:
    """
    Build the final 176-band fusion order.

    Per season:
        26 S1 predictors
        followed by
        18 S2 predictors.
    """

    validate_seasonal_blocks(
        s1_names,
        S1_SEASONAL_BANDS,
        "S1",
    )

    validate_seasonal_blocks(
        s2_names,
        S2_SEASONAL_BANDS,
        "S2",
    )

    fusion_names = []

    for season_index in range(4):

        s1_start = (
            season_index
            * S1_SEASONAL_BANDS
        )

        s1_end = (
            s1_start
            + S1_SEASONAL_BANDS
        )


        s2_start = (
            season_index
            * S2_SEASONAL_BANDS
        )

        s2_end = (
            s2_start
            + S2_SEASONAL_BANDS
        )


        fusion_names.extend(
            s1_names[
                s1_start:s1_end
            ]
        )

        fusion_names.extend(
            s2_names[
                s2_start:s2_end
            ]
        )


    if len(fusion_names) != FUSION_ANNUAL_BANDS:
        raise RuntimeError(
            f"Generated {len(fusion_names)} fusion predictor names; "
            f"expected {FUSION_ANNUAL_BANDS}."
        )


    if len(set(fusion_names)) != FUSION_ANNUAL_BANDS:
        raise RuntimeError(
            "Duplicate predictor names detected in fusion stack."
        )


    return tuple(
        fusion_names
    )


# =============================================================================
# Data helpers
# =============================================================================

def normalize_invalid_values(
    data: np.ndarray,
    source_nodata: float | None,
) -> np.ndarray:
    """
    Convert non-finite and source-NoData values to the common -9999
    sentinel value without imposing any additional validity mask.
    """

    invalid = ~np.isfinite(
        data
    )

    if source_nodata is not None:

        invalid |= np.isclose(
            data,
            source_nodata,
        )

    data[
        invalid
    ] = NODATA

    return data


# =============================================================================
# Final stack audit
# =============================================================================

def audit_fusion_stack(
    path: Path,
) -> dict:
    """
    Audit a completed 176-band annual fusion stack.
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


        # ---------------------------------------------------------------------
        # General checks
        # ---------------------------------------------------------------------

        checks = {
            "bands_176":
                src.count == FUSION_ANNUAL_BANDS,

            "EPSG_3577":
                (
                    src.crs is not None
                    and src.crs.to_epsg() == EXPECTED_EPSG
                ),

            "resolution_10m":
                tuple(src.res) == EXPECTED_RESOLUTION,

            "nodata_-9999":
                src.nodata == NODATA,

            "176_band_descriptions":
                (
                    len(descriptions)
                    == FUSION_ANNUAL_BANDS
                    and all(
                        description is not None
                        and str(description).strip()
                        for description in descriptions
                    )
                ),

            "unique_predictor_names":
                (
                    len(set(descriptions))
                    == FUSION_ANNUAL_BANDS
                ),
        }


        # ---------------------------------------------------------------------
        # Seasonal ordering checks
        # ---------------------------------------------------------------------

        if len(descriptions) == FUSION_ANNUAL_BANDS:

            for season_index, season in enumerate(
                SEASONS
            ):

                block_start = (
                    season_index
                    * FUSION_SEASONAL_BANDS
                )

                block_end = (
                    block_start
                    + FUSION_SEASONAL_BANDS
                )

                season_block = descriptions[
                    block_start:block_end
                ]


                s1_block = season_block[
                    :S1_SEASONAL_BANDS
                ]

                s2_block = season_block[
                    S1_SEASONAL_BANDS:
                ]


                checks[
                    f"{season}_season_prefix"
                ] = all(
                    name.startswith(
                        f"{season}_"
                    )
                    for name in season_block
                )


                checks[
                    f"{season}_S1_first"
                ] = all(
                    "_S1_" in name
                    for name in s1_block
                )


                checks[
                    f"{season}_S2_second"
                ] = all(
                    "_S2_" in name
                    for name in s2_block
                )


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
                    if checks
                    and all(
                        checks.values()
                    )
                    else "CHECK"
                ),
        }


# =============================================================================
# Fusion-stack creation
# =============================================================================

def build_fusion_stack(
    data_root: Path,
    period: str,
    overwrite: bool,
) -> dict:
    """
    Build one annual 176-band Sentinel-1 + Sentinel-2 fusion stack.
    """

    print(
        "\n" + "=" * 88
    )

    print(
        f"Building annual fusion stack: {period}"
    )

    print(
        "=" * 88
    )


    # -------------------------------------------------------------------------
    # Input paths
    # -------------------------------------------------------------------------

    s1_path = s1_annual_path(
        data_root,
        period,
    )

    s2_path = s2_annual_path(
        data_root,
        period,
    )

    output = fusion_annual_path(
        data_root,
        period,
    )


    print(
        "Sentinel-1 input:",
        s1_path,
    )

    print(
        "Sentinel-2 input:",
        s2_path,
    )

    print(
        "Fusion output:",
        output,
    )


    # -------------------------------------------------------------------------
    # Existence checks
    # -------------------------------------------------------------------------

    missing = [
        path
        for path in [
            s1_path,
            s2_path,
        ]
        if not path.exists()
    ]


    if missing:

        raise FileNotFoundError(
            "Missing annual input stack(s):\n  - "
            + "\n  - ".join(
                map(str, missing)
            )
        )


    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -------------------------------------------------------------------------
    # Existing output
    # -------------------------------------------------------------------------

    if output.exists() and not overwrite:

        print(
            "Output already exists; auditing existing fusion stack."
        )

        audit = audit_fusion_stack(
            output
        )

        if audit["status"] != "PASS":

            raise RuntimeError(
                "Existing fusion stack failed audit. "
                "Run with --overwrite to rebuild it."
            )

        return audit


    # -------------------------------------------------------------------------
    # Open annual sensor stacks
    # -------------------------------------------------------------------------

    with rasterio.open(
        s1_path
    ) as s1, rasterio.open(
        s2_path
    ) as s2:


        # ---------------------------------------------------------------------
        # Grid audit
        # ---------------------------------------------------------------------

        audit_sensor_alignment(
            s1,
            s2,
        )

        print(
            "S1/S2 spatial-alignment audit: PASS"
        )


        # ---------------------------------------------------------------------
        # Predictor identity and ordering audit
        # ---------------------------------------------------------------------

        s1_names = get_band_names(
            s1,
            S1_ANNUAL_BANDS,
            "Sentinel-1",
        )

        s2_names = get_band_names(
            s2,
            S2_ANNUAL_BANDS,
            "Sentinel-2",
        )


        fusion_names = build_fusion_band_names(
            s1_names,
            s2_names,
        )


        print(
            "Predictor identity/order audit: PASS"
        )

        print(
            "Fusion predictor count:",
            len(fusion_names),
        )

        print(
            "First predictor:",
            fusion_names[0],
        )

        print(
            "Last predictor:",
            fusion_names[-1],
        )


        # ---------------------------------------------------------------------
        # Output raster profile
        # ---------------------------------------------------------------------

        profile = s1.profile.copy()

        profile.update(
            driver="GTiff",
            count=FUSION_ANNUAL_BANDS,
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
            prefix="echo_fusion_",
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


                # -------------------------------------------------------------
                # Band descriptions
                # -------------------------------------------------------------

                dst.descriptions = (
                    fusion_names
                )


                # -------------------------------------------------------------
                # Dataset metadata
                # -------------------------------------------------------------

                dst.update_tags(
                    sensors="Sentinel-1,Sentinel-2",
                    period=period,
                    stack_type="FULL_OPTICAL_SAR_FUSION",
                    seasons="Winter,Spring,Summer,Autumn",
                    S1_predictors="104",
                    S2_predictors="72",
                    fusion_predictors="176",
                    predictors_per_season="44",
                    S1_predictors_per_season="26",
                    S2_predictors_per_season="18",
                    common_valid_mask="False",
                    nodata=str(NODATA),
                    band_order=(
                        "Winter S1(26)+S2(18); "
                        "Spring S1(26)+S2(18); "
                        "Summer S1(26)+S2(18); "
                        "Autumn S1(26)+S2(18)"
                    ),
                )


                # -------------------------------------------------------------
                # Block-wise fusion
                # -------------------------------------------------------------

                windows = list(
                    s1.block_windows(1)
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


                    # =========================================================
                    # Process one seasonal block at a time
                    # =========================================================

                    for season_index in range(4):


                        # -----------------------------------------------------
                        # Sentinel-1 seasonal block
                        # -----------------------------------------------------

                        s1_start = (
                            season_index
                            * S1_SEASONAL_BANDS
                            + 1
                        )

                        s1_end = (
                            s1_start
                            + S1_SEASONAL_BANDS
                            - 1
                        )

                        s1_indexes = list(
                            range(
                                s1_start,
                                s1_end + 1,
                            )
                        )


                        s1_data = s1.read(
                            indexes=s1_indexes,
                            window=window,
                            out_dtype="float32",
                        )


                        s1_data = normalize_invalid_values(
                            s1_data,
                            s1.nodata,
                        )


                        destination_indexes = list(
                            range(
                                destination_band,
                                destination_band
                                + S1_SEASONAL_BANDS,
                            )
                        )


                        dst.write(
                            s1_data,
                            indexes=destination_indexes,
                            window=window,
                        )


                        destination_band += (
                            S1_SEASONAL_BANDS
                        )


                        # -----------------------------------------------------
                        # Sentinel-2 seasonal block
                        # -----------------------------------------------------

                        s2_start = (
                            season_index
                            * S2_SEASONAL_BANDS
                            + 1
                        )

                        s2_end = (
                            s2_start
                            + S2_SEASONAL_BANDS
                            - 1
                        )


                        s2_indexes = list(
                            range(
                                s2_start,
                                s2_end + 1,
                            )
                        )


                        s2_data = s2.read(
                            indexes=s2_indexes,
                            window=window,
                            out_dtype="float32",
                        )


                        s2_data = normalize_invalid_values(
                            s2_data,
                            s2.nodata,
                        )


                        destination_indexes = list(
                            range(
                                destination_band,
                                destination_band
                                + S2_SEASONAL_BANDS,
                            )
                        )


                        dst.write(
                            s2_data,
                            indexes=destination_indexes,
                            window=window,
                        )


                        destination_band += (
                            S2_SEASONAL_BANDS
                        )


                    # ---------------------------------------------------------
                    # Progress
                    # ---------------------------------------------------------

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


    # -------------------------------------------------------------------------
    # Final audit
    # -------------------------------------------------------------------------

    audit = audit_fusion_stack(
        output
    )


    if audit["status"] != "PASS":

        raise RuntimeError(
            "Created fusion stack failed final audit:\n"
            f"{audit['checks']}"
        )


    print(
        "Fusion stack audit: PASS"
    )

    print(
        "Output:",
        output,
    )


    return audit


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
            "Build 176-band annual Sentinel-1 + Sentinel-2 "
            "fusion stacks."
        )
    )


    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help=(
            "Root directory containing annual "
            "Sentinel-1 and Sentinel-2 stacks. "
            f"Default: {default_data_root}"
        ),
    )


    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rebuild fusion stacks even when "
            "outputs already exist."
        ),
    )


    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Build and audit all three annual fusion stacks."""

    args = parse_args()

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )


    print(
        "ECHO annual optical-SAR fusion stack builder"
    )

    print(
        "Data root:",
        data_root,
    )

    print(
        "Sentinel-1 predictors:",
        S1_ANNUAL_BANDS,
    )

    print(
        "Sentinel-2 predictors:",
        S2_ANNUAL_BANDS,
    )

    print(
        "Fusion predictors:",
        FUSION_ANNUAL_BANDS,
    )

    print(
        "Ordering: "
        "Winter S1 -> Winter S2 -> "
        "Spring S1 -> Spring S2 -> "
        "Summer S1 -> Summer S2 -> "
        "Autumn S1 -> Autumn S2"
    )


    results = []
    failures = []


    for period in PERIODS:

        try:

            result = build_fusion_stack(
                data_root=data_root,
                period=period,
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
        "FINAL OPTICAL-SAR FUSION STACK AUDIT"
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
            "Not all annual fusion stacks were created."
        )


    if any(
        result["status"] != "PASS"
        for _period, result in results
    ):

        raise SystemExit(
            "One or more annual fusion stacks "
            "failed the audit."
        )


    print(
        "\nAll three 176-band fusion stacks "
        "passed the audit."
    )


if __name__ == "__main__":
    main()