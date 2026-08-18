#!/usr/bin/env python3
"""
Prepare polygon-level reference samples for the ECHO workflow.

This script extracts predictor values from the 2025-2026 annual
Sentinel-1, Sentinel-2, and optical-SAR fusion stacks using the
525 labelled reference polygons.

For each polygon and predictor:

    feature value = mean of valid raster pixels intersecting the polygon

The reference design used by the study contains:

    525 polygons total
    5 spatial regions
    105 polygons per region
    35 polygons per class per region

Classes:

    RRG
    NFFP
    Water

Inputs
------
1. Label polygons
2. 2025-2026 104-band Sentinel-1 annual stack
3. 2025-2026 72-band Sentinel-2 annual stack
4. 2025-2026 176-band Sentinel-1/Sentinel-2 fusion stack

Outputs
-------
For each predictor configuration source:

    raw/
        polygon means with NaN retained when a predictor has no valid
        intersecting raster pixels

    model_ready/
        modelling table in which missing predictor values are encoded
        as -9999

    audits/
        extraction and quality-control summaries

Important
---------
- Polygon geometry and labels are identical across S1, S2 and fusion
  extraction.
- Raster NoData values are excluded from polygon means.
- NoData values are never included in a mean calculation.
- `all_touched=True` is used so raster pixels intersecting the polygon
  are considered during extraction.
- The script requires all 525 polygons to be represented in every
  model-ready table.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from shapely.geometry import mapping


# =============================================================================
# Configuration
# =============================================================================

PERIOD = "2025-2026"
PERIOD_TAG = "2025_2026"

NODATA = -9999.0

EXPECTED_POLYGONS = 525

EXPECTED_CLASSES = (
    "RRG",
    "NFFP",
    "Water",
)

EXPECTED_REGIONS = (
    "SR1",
    "SR2",
    "SR3",
    "SR4",
    "SR5",
)

EXPECTED_PER_REGION = 105
EXPECTED_PER_CLASS_REGION = 35

EXPECTED_FEATURE_COUNTS = {
    "S1": 104,
    "S2": 72,
    "S1S2": 176,
}

EXPECTED_EPSG = 3577
EXPECTED_RESOLUTION = (10.0, 10.0)

ALL_TOUCHED = True


# Metadata fields that must never be supplied to a classifier.
META_COLUMNS = [
    "unique_id",
    "class_raw",
    "class_label",
    "class_id",
    "region",
    "stack",
    "period",
    "valid_bands",
    "total_bands",
    "valid_band_fraction",
    "min_valid_pixels_per_band",
    "max_valid_pixels_per_band",
]


# =============================================================================
# Class and region parsing
# =============================================================================

def find_id_column(gdf: gpd.GeoDataFrame) -> str:
    """
    Identify the polygon identifier field.

    Expected identifiers resemble:

        Redgum_P1_1
        Water_P4_360
        Non-Forest_Floodplain_P1_18
    """

    candidates = [
        "unique_id",
        "Unique_ID",
        "UNIQUE_ID",
        "uid",
        "UID",
        "sample_id",
        "Sample_ID",
        "id",
        "ID",
    ]

    for column in candidates:

        if column in gdf.columns:
            return column


    # Fallback: detect a text field containing the P1-P5 pattern.
    for column in gdf.columns:

        if column == "geometry":
            continue

        values = (
            gdf[column]
            .astype(str)
            .head(50)
            .tolist()
        )

        if any(
            re.search(
                r"_P[1-5]_",
                value,
            )
            for value in values
        ):
            return column


    raise ValueError(
        "Could not determine polygon ID column. "
        f"Available columns: {list(gdf.columns)}"
    )


def parse_region_from_uid(
    uid: str,
) -> str | None:
    """
    Convert the region encoded in a polygon ID from P1-P5 to SR1-SR5.

    Example:
        Redgum_P3_17 -> SR3
    """

    match = re.search(
        r"_P([1-5])_",
        str(uid),
    )

    if match:
        return f"SR{match.group(1)}"

    return None


def parse_class_from_uid(
    uid: str,
) -> tuple[str, str]:
    """
    Parse and standardize the reference class encoded in a polygon ID.
    """

    uid = str(
        uid
    )

    class_raw = re.sub(
        r"_P[1-5]_\d+$",
        "",
        uid,
    ).strip()


    class_map = {
        "Redgum": "RRG",
        "River_Red_Gum": "RRG",
        "River-Red-Gum": "RRG",
        "RRG": "RRG",

        "Non-Forest_Floodplain": "NFFP",
        "Non_Forest_Floodplain": "NFFP",
        "Non-Forest-Floodplain": "NFFP",
        "NFFP": "NFFP",

        "Water": "Water",
        "WATER": "Water",
    }


    class_label = class_map.get(
        class_raw,
        class_raw,
    )


    return (
        class_raw,
        class_label,
    )


def class_to_id(
    class_label: str,
) -> int:
    """
    Numeric class identifiers used for audit and modelling metadata.
    """

    mapping_dict = {
        "RRG": 1,
        "NFFP": 2,
        "Water": 3,
    }

    return mapping_dict.get(
        class_label,
        -1,
    )


# =============================================================================
# Predictor-name helpers
# =============================================================================

def clean_feature_name(
    name: str,
) -> str:
    """Convert a raster-band description to a clean column name."""

    value = str(
        name
    ).strip()

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
        value = value.replace(
            character,
            "_",
        )

    while "__" in value:
        value = value.replace(
            "__",
            "_",
        )

    return value.strip(
        "_"
    )


def get_band_feature_names(
    src: rasterio.io.DatasetReader,
    stack_name: str,
) -> list[str]:
    """
    Read predictor identities from raster band descriptions.

    Annual stacks created by the preceding scripts are expected to contain
    complete and unique descriptions.
    """

    descriptions = tuple(
        src.descriptions or ()
    )


    if len(descriptions) != src.count:

        raise ValueError(
            f"{stack_name}: raster does not contain "
            "one description for every band."
        )


    names = []


    for band_index, description in enumerate(
        descriptions,
        start=1,
    ):

        if (
            description is None
            or not str(description).strip()
        ):

            raise ValueError(
                f"{stack_name}: missing description for "
                f"band {band_index}."
            )


        name = clean_feature_name(
            description
        )


        if name in names:

            raise ValueError(
                f"{stack_name}: duplicate predictor "
                f"name detected: {name}"
            )


        names.append(
            name
        )


    return names


# =============================================================================
# Raster metadata audit
# =============================================================================

def audit_stack_metadata(
    stack_name: str,
    path: Path,
    expected_bands: int,
) -> dict:
    """
    Audit one annual raster before reference-sample extraction.
    """

    if not path.exists():

        return {
            "stack": stack_name,
            "path": str(path),
            "status": "MISSING",
        }


    with rasterio.open(
        path
    ) as src:

        descriptions = tuple(
            src.descriptions or ()
        )


        checks = {
            "band_count":
                src.count == expected_bands,

            "EPSG_3577":
                (
                    src.crs is not None
                    and src.crs.to_epsg()
                    == EXPECTED_EPSG
                ),

            "resolution_10m":
                tuple(src.res)
                == EXPECTED_RESOLUTION,

            "nodata_-9999":
                src.nodata == NODATA,

            "band_descriptions":
                (
                    len(descriptions)
                    == expected_bands
                    and all(
                        description is not None
                        and str(description).strip()
                        for description in descriptions
                    )
                ),

            "unique_band_names":
                (
                    len(set(descriptions))
                    == expected_bands
                ),
        }


        return {
            "stack": stack_name,
            "path": str(path),
            "bands": src.count,
            "crs": str(src.crs),
            "resolution_x": src.res[0],
            "resolution_y": src.res[1],
            "nodata": src.nodata,
            "width": src.width,
            "height": src.height,
            "first_predictor":
                descriptions[0]
                if descriptions
                else None,
            "last_predictor":
                descriptions[-1]
                if descriptions
                else None,
            **checks,
            "status":
                (
                    "PASS"
                    if all(checks.values())
                    else "CHECK"
                ),
        }


# =============================================================================
# Reference polygon preparation
# =============================================================================

def load_and_prepare_labels(
    labels_path: Path,
    target_crs,
) -> gpd.GeoDataFrame:
    """
    Load, validate, standardize and audit the reference polygons.
    """

    if not labels_path.exists():

        raise FileNotFoundError(
            f"Reference labels not found: {labels_path}"
        )


    labels = gpd.read_file(
        labels_path
    )


    print(
        "Loaded reference polygons:",
        len(labels),
    )

    print(
        "Input CRS:",
        labels.crs,
    )


    if labels.crs is None:

        raise ValueError(
            "Reference polygon file has no CRS."
        )


    # -------------------------------------------------------------------------
    # Geometry checks
    # -------------------------------------------------------------------------

    labels = labels[
        labels.geometry.notnull()
    ].copy()


    empty_count = int(
        labels.geometry.is_empty.sum()
    )

    if empty_count:

        raise ValueError(
            f"{empty_count} reference geometries are empty."
        )


    invalid_count = int(
        (~labels.geometry.is_valid).sum()
    )


    if invalid_count:

        print(
            f"Repairing {invalid_count} invalid geometries."
        )

        labels["geometry"] = (
            labels.geometry.buffer(0)
        )


    remaining_invalid = int(
        (~labels.geometry.is_valid).sum()
    )


    if remaining_invalid:

        raise ValueError(
            f"{remaining_invalid} geometries remain invalid "
            "after repair."
        )


    # -------------------------------------------------------------------------
    # Parse polygon identifiers
    # -------------------------------------------------------------------------

    id_column = find_id_column(
        labels
    )


    print(
        "Detected polygon ID field:",
        id_column,
    )


    labels["unique_id"] = (
        labels[id_column]
        .astype(str)
    )


    parsed = labels[
        "unique_id"
    ].apply(
        parse_class_from_uid
    )


    labels["class_raw"] = parsed.apply(
        lambda item: item[0]
    )

    labels["class_label"] = parsed.apply(
        lambda item: item[1]
    )

    labels["class_id"] = labels[
        "class_label"
    ].apply(
        class_to_id
    )

    labels["region"] = labels[
        "unique_id"
    ].apply(
        parse_region_from_uid
    )


    # -------------------------------------------------------------------------
    # Reproject to raster grid CRS
    # -------------------------------------------------------------------------

    if labels.crs != target_crs:

        print(
            "Reprojecting labels:"
        )

        print(
            "  From:",
            labels.crs,
        )

        print(
            "  To:",
            target_crs,
        )


        labels = labels.to_crs(
            target_crs
        )


    # -------------------------------------------------------------------------
    # Strict reference-design audit
    # -------------------------------------------------------------------------

    if len(labels) != EXPECTED_POLYGONS:

        raise ValueError(
            f"Expected {EXPECTED_POLYGONS} polygons; "
            f"found {len(labels)}."
        )


    duplicate_count = int(
        labels["unique_id"]
        .duplicated()
        .sum()
    )


    if duplicate_count:

        raise ValueError(
            f"Detected {duplicate_count} duplicate polygon IDs."
        )


    unknown_classes = sorted(
        set(labels["class_label"])
        - set(EXPECTED_CLASSES)
    )


    if unknown_classes:

        raise ValueError(
            "Unexpected reference class(es): "
            + ", ".join(unknown_classes)
        )


    missing_regions = int(
        labels["region"]
        .isna()
        .sum()
    )


    if missing_regions:

        raise ValueError(
            f"{missing_regions} polygons have no parsed region."
        )


    unknown_regions = sorted(
        set(labels["region"])
        - set(EXPECTED_REGIONS)
    )


    if unknown_regions:

        raise ValueError(
            "Unexpected spatial region(s): "
            + ", ".join(unknown_regions)
        )


    class_counts = (
        labels["class_label"]
        .value_counts()
    )


    expected_class_total = (
        EXPECTED_POLYGONS
        // len(EXPECTED_CLASSES)
    )


    for class_name in EXPECTED_CLASSES:

        observed = int(
            class_counts.get(
                class_name,
                0,
            )
        )

        if observed != expected_class_total:

            raise ValueError(
                f"Class {class_name}: expected "
                f"{expected_class_total}, found {observed}."
            )


    region_counts = (
        labels["region"]
        .value_counts()
    )


    for region in EXPECTED_REGIONS:

        observed = int(
            region_counts.get(
                region,
                0,
            )
        )

        if observed != EXPECTED_PER_REGION:

            raise ValueError(
                f"{region}: expected "
                f"{EXPECTED_PER_REGION} polygons, "
                f"found {observed}."
            )


    class_region = pd.crosstab(
        labels["region"],
        labels["class_label"],
    )


    for region in EXPECTED_REGIONS:

        for class_name in EXPECTED_CLASSES:

            observed = int(
                class_region.loc[
                    region,
                    class_name,
                ]
            )

            if observed != EXPECTED_PER_CLASS_REGION:

                raise ValueError(
                    f"{region} / {class_name}: expected "
                    f"{EXPECTED_PER_CLASS_REGION}, "
                    f"found {observed}."
                )


    print(
        "Reference-design audit: PASS"
    )

    print(
        "\nClass × region:"
    )

    print(
        class_region
    )


    return labels


# =============================================================================
# Polygon extraction
# =============================================================================

def extract_polygon_mean_features(
    gdf: gpd.GeoDataFrame,
    raster_path: Path,
    stack_name: str,
    expected_bands: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract polygon-mean predictor values from one annual raster.

    One output row corresponds to one reference polygon.

    For each predictor:
    - select raster pixels intersecting the polygon;
    - exclude raster NoData and non-finite values;
    - calculate the arithmetic mean of valid pixels.

    A predictor with no valid pixels for a polygon remains NaN in the raw
    table and is converted to -9999 only in the model-ready table.
    """

    print(
        "\n" + "=" * 88
    )

    print(
        f"Extracting {stack_name} polygon means"
    )

    print(
        "=" * 88
    )

    print(
        "Raster:",
        raster_path,
    )


    feature_rows = []
    audit_rows = []


    with rasterio.open(
        raster_path
    ) as src:


        if src.count != expected_bands:

            raise ValueError(
                f"{stack_name}: expected {expected_bands} bands; "
                f"found {src.count}."
            )


        feature_names = get_band_feature_names(
            src,
            stack_name,
        )


        raster_nodata = (
            src.nodata
            if src.nodata is not None
            else NODATA
        )


        print(
            "Predictors:",
            len(feature_names),
        )

        print(
            "CRS:",
            src.crs,
        )

        print(
            "NoData:",
            raster_nodata,
        )


        # ---------------------------------------------------------------------
        # Extract every reference polygon
        # ---------------------------------------------------------------------

        for polygon_number, row in enumerate(
            gdf.itertuples(
                index=False
            ),
            start=1,
        ):


            base_info = {
                "unique_id": row.unique_id,
                "class_raw": row.class_raw,
                "class_label": row.class_label,
                "class_id": row.class_id,
                "region": row.region,
            }


            geometry = row.geometry


            if (
                geometry is None
                or geometry.is_empty
            ):

                raise ValueError(
                    f"{stack_name}: empty geometry for "
                    f"{row.unique_id}."
                )


            try:

                out, _transform = rasterio.mask.mask(
                    src,
                    [
                        mapping(
                            geometry
                        )
                    ],
                    crop=True,
                    filled=True,
                    nodata=NODATA,
                    all_touched=ALL_TOUCHED,
                )


            except ValueError as exc:

                raise ValueError(
                    f"{stack_name}: polygon {row.unique_id} "
                    "does not overlap the raster."
                ) from exc


            values = out.astype(
                np.float64,
                copy=False,
            )


            # -----------------------------------------------------------------
            # Identify invalid cells
            # -----------------------------------------------------------------

            invalid = ~np.isfinite(
                values
            )


            if raster_nodata is not None:

                invalid |= np.isclose(
                    values,
                    raster_nodata,
                )


            invalid |= np.isclose(
                values,
                NODATA,
            )


            values = values.copy()

            values[
                invalid
            ] = np.nan


            # -----------------------------------------------------------------
            # Predictor-level valid pixel counts
            # -----------------------------------------------------------------

            valid_counts = np.sum(
                np.isfinite(
                    values
                ),
                axis=(
                    1,
                    2,
                ),
            )


            valid_bands = int(
                np.sum(
                    valid_counts > 0
                )
            )

            total_bands = int(
                src.count
            )

            valid_band_fraction = (
                valid_bands
                / total_bands
            )


            min_valid_pixels = int(
                valid_counts.min()
            )

            max_valid_pixels = int(
                valid_counts.max()
            )


            if valid_bands == 0:

                raise ValueError(
                    f"{stack_name}: polygon {row.unique_id} "
                    "contains no valid raster pixels."
                )


            # -----------------------------------------------------------------
            # Polygon means
            # -----------------------------------------------------------------

            with np.errstate(
                invalid="ignore",
                divide="ignore",
            ):

                band_means = np.nanmean(
                    values,
                    axis=(
                        1,
                        2,
                    ),
                )


            feature_record = {
                **base_info,
                "stack": stack_name,
                "period": PERIOD,
                "valid_bands": valid_bands,
                "total_bands": total_bands,
                "valid_band_fraction": valid_band_fraction,
                "min_valid_pixels_per_band": min_valid_pixels,
                "max_valid_pixels_per_band": max_valid_pixels,
            }


            for (
                feature_name,
                value,
            ) in zip(
                feature_names,
                band_means,
            ):

                feature_record[
                    feature_name
                ] = value


            feature_rows.append(
                feature_record
            )


            audit_rows.append({
                **base_info,
                "stack": stack_name,
                "status": "OK",
                "valid_bands": valid_bands,
                "total_bands": total_bands,
                "valid_band_fraction": valid_band_fraction,
                "min_valid_pixels_per_band": min_valid_pixels,
                "max_valid_pixels_per_band": max_valid_pixels,
            })


            if (
                polygon_number % 50 == 0
                or polygon_number
                == len(gdf)
            ):

                print(
                    f"Processed polygons: "
                    f"{polygon_number}/"
                    f"{len(gdf)}"
                )


    features_df = pd.DataFrame(
        feature_rows
    )

    audit_df = pd.DataFrame(
        audit_rows
    )


    return (
        features_df,
        audit_df,
    )


# =============================================================================
# Model-ready QC
# =============================================================================

def feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return predictor columns while excluding all metadata/QC fields."""

    return [
        column
        for column in dataframe.columns
        if column not in META_COLUMNS
    ]


def audit_model_ready_table(
    dataframe: pd.DataFrame,
    stack_name: str,
    expected_features: int,
) -> dict:
    """
    Apply strict quality checks before a table is released to modelling.
    """

    predictors = feature_columns(
        dataframe
    )


    class_counts = (
        dataframe["class_label"]
        .value_counts()
    )

    region_counts = (
        dataframe["region"]
        .value_counts()
    )

    class_region = pd.crosstab(
        dataframe["region"],
        dataframe["class_label"],
    )


    checks = {
        "rows_525":
            len(dataframe)
            == EXPECTED_POLYGONS,

        "expected_predictors":
            len(predictors)
            == expected_features,

        "unique_polygon_ids":
            dataframe[
                "unique_id"
            ].nunique()
            == EXPECTED_POLYGONS,

        "no_duplicate_polygon_ids":
            not dataframe[
                "unique_id"
            ].duplicated().any(),

        "no_nan_predictors":
            not dataframe[
                predictors
            ].isna().any().any(),

        "classes_correct":
            set(
                dataframe["class_label"]
            )
            == set(EXPECTED_CLASSES),

        "regions_correct":
            set(
                dataframe["region"]
            )
            == set(EXPECTED_REGIONS),

        "175_per_class":
            all(
                int(
                    class_counts.get(
                        class_name,
                        0,
                    )
                )
                == 175
                for class_name in EXPECTED_CLASSES
            ),

        "105_per_region":
            all(
                int(
                    region_counts.get(
                        region,
                        0,
                    )
                )
                == EXPECTED_PER_REGION
                for region in EXPECTED_REGIONS
            ),

        "35_per_class_per_region":
            all(
                int(
                    class_region.loc[
                        region,
                        class_name,
                    ]
                )
                == EXPECTED_PER_CLASS_REGION
                for region in EXPECTED_REGIONS
                for class_name in EXPECTED_CLASSES
            ),
    }


    return {
        "stack": stack_name,
        "rows": len(dataframe),
        "predictor_columns": len(
            predictors
        ),
        "duplicate_unique_id": int(
            dataframe[
                "unique_id"
            ].duplicated().sum()
        ),
        "nan_predictor_values": int(
            dataframe[
                predictors
            ].isna().sum().sum()
        ),
        "min_valid_band_fraction": float(
            dataframe[
                "valid_band_fraction"
            ].min()
        ),
        "mean_valid_band_fraction": float(
            dataframe[
                "valid_band_fraction"
            ].mean()
        ),
        **checks,
        "status":
            (
                "PASS"
                if all(
                    checks.values()
                )
                else "CHECK"
            ),
    }


# =============================================================================
# Output helpers
# =============================================================================

def save_stack_outputs(
    features_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    stack_name: str,
    expected_features: int,
    output_root: Path,
) -> dict:
    """
    Save raw polygon means, model-ready samples, and extraction audit.
    """

    raw_dir = (
        output_root
        / "raw"
    )

    model_dir = (
        output_root
        / "model_ready"
    )

    audit_dir = (
        output_root
        / "audits"
    )


    for directory in [
        raw_dir,
        model_dir,
        audit_dir,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


    # -------------------------------------------------------------------------
    # Raw means
    # -------------------------------------------------------------------------

    raw_path = (
        raw_dir
        / (
            f"reference_samples_"
            f"{PERIOD_TAG}_"
            f"{stack_name}_polygon_means.csv"
        )
    )


    features_df.to_csv(
        raw_path,
        index=False,
    )


    # -------------------------------------------------------------------------
    # Model-ready table
    # -------------------------------------------------------------------------

    model_df = features_df.copy()

    predictors = feature_columns(
        model_df
    )


    # Predictors that had no valid pixels for a specific polygon are encoded
    # with the same NoData sentinel used by the annual rasters.
    model_df[
        predictors
    ] = model_df[
        predictors
    ].fillna(
        NODATA
    )


    model_path = (
        model_dir
        / (
            f"reference_samples_"
            f"{PERIOD_TAG}_"
            f"{stack_name}_MODEL_READY.csv"
        )
    )


    model_df.to_csv(
        model_path,
        index=False,
    )


    # -------------------------------------------------------------------------
    # Extraction audit
    # -------------------------------------------------------------------------

    audit_path = (
        audit_dir
        / (
            f"reference_samples_"
            f"{PERIOD_TAG}_"
            f"{stack_name}_extraction_audit.csv"
        )
    )


    audit_df.to_csv(
        audit_path,
        index=False,
    )


    # -------------------------------------------------------------------------
    # Model-ready QC
    # -------------------------------------------------------------------------

    qc = audit_model_ready_table(
        model_df,
        stack_name,
        expected_features,
    )


    if qc["status"] != "PASS":

        raise RuntimeError(
            f"{stack_name} model-ready table "
            f"failed QC:\n{qc}"
        )


    print(
        f"{stack_name} model-ready QC: PASS"
    )

    print(
        "Raw means:",
        raw_path,
    )

    print(
        "Model-ready:",
        model_path,
    )

    print(
        "Audit:",
        audit_path,
    )


    return qc


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line configuration.
    """

    repo_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )


    default_data_root = (
        repo_root
        / "data"
    )


    parser = argparse.ArgumentParser(
        description=(
            "Extract 2025-2026 polygon-mean "
            "reference samples for ECHO."
        )
    )


    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help=(
            "Root data directory. "
            f"Default: {default_data_root}"
        ),
    )


    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=(
            "Reference polygon file. "
            "Default: <data-root>/labels/"
            "ROI_labels_525_samples_uniqueIDs.shp"
        ),
    )


    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. "
            "Default: <data-root>/reference_samples"
        ),
    )


    return parser.parse_args()


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    """
    Prepare and audit all reference-sample tables.
    """

    args = parse_args()


    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )


    labels_path = (
        args.labels
        .expanduser()
        .resolve()
        if args.labels is not None
        else (
            data_root
            / "labels"
            / "ROI_labels_525_samples_uniqueIDs.shp"
        )
    )


    output_root = (
        args.output_dir
        .expanduser()
        .resolve()
        if args.output_dir is not None
        else (
            data_root
            / "reference_samples"
        )
    )


    image_root = (
        data_root
        / "images"
    )


    # =========================================================================
    # Annual-stack paths
    # =========================================================================

    s1_stack = (
        image_root
        / PERIOD
        / "Annual_Stack"
        / "S1_full_mosaic"
        / "S1_2025_2026_FULL_ANNUAL_STACK.tif"
    )


    s2_stack = (
        image_root
        / PERIOD
        / "Annual_Stack"
        / "S2_full_mosaic"
        / "S2_2025_2026_FULL_ANNUAL_STACK.tif"
    )


    fusion_stack = (
        image_root
        / PERIOD
        / "Annual_Stack"
        / "Fusion_full_mosaic"
        / "S1S2_2025_2026_FULL_FUSION_STACK.tif"
    )


    stacks = {
        "S1": {
            "path": s1_stack,
            "expected_bands": 104,
        },

        "S2": {
            "path": s2_stack,
            "expected_bands": 72,
        },

        "S1S2": {
            "path": fusion_stack,
            "expected_bands": 176,
        },
    }


    print(
        "ECHO reference-sample preparation"
    )

    print(
        "Labels:",
        labels_path,
    )

    print(
        "Output:",
        output_root,
    )


    # =========================================================================
    # Input raster audit
    # =========================================================================

    stack_audits = []


    for stack_name, configuration in stacks.items():

        audit = audit_stack_metadata(
            stack_name=stack_name,
            path=configuration["path"],
            expected_bands=configuration[
                "expected_bands"
            ],
        )


        stack_audits.append(
            audit
        )


        print(
            f"{stack_name} raster audit:",
            audit["status"],
        )


        if audit["status"] != "PASS":

            raise RuntimeError(
                f"{stack_name} annual stack "
                f"failed metadata audit:\n{audit}"
            )


    audit_dir = (
        output_root
        / "audits"
    )

    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    pd.DataFrame(
        stack_audits
    ).to_csv(
        audit_dir
        / "input_stack_metadata_audit.csv",
        index=False,
    )


    # =========================================================================
    # Load labels using fusion CRS as reference
    # =========================================================================

    with rasterio.open(
        fusion_stack
    ) as reference:

        target_crs = reference.crs


    labels = load_and_prepare_labels(
        labels_path,
        target_crs,
    )


    # =========================================================================
    # Extract all three predictor tables
    # =========================================================================

    qc_rows = []
    extraction_audits = []


    for stack_name, configuration in stacks.items():

        features_df, audit_df = (
            extract_polygon_mean_features(
                gdf=labels,
                raster_path=configuration["path"],
                stack_name=stack_name,
                expected_bands=configuration[
                    "expected_bands"
                ],
            )
        )


        if len(features_df) != EXPECTED_POLYGONS:

            raise RuntimeError(
                f"{stack_name}: expected "
                f"{EXPECTED_POLYGONS} extracted polygons, "
                f"found {len(features_df)}."
            )


        qc = save_stack_outputs(
            features_df=features_df,
            audit_df=audit_df,
            stack_name=stack_name,
            expected_features=configuration[
                "expected_bands"
            ],
            output_root=output_root,
        )


        qc_rows.append(
            qc
        )


        extraction_audits.append(
            audit_df
        )


    # =========================================================================
    # Combined extraction audit
    # =========================================================================

    combined_audit = pd.concat(
        extraction_audits,
        ignore_index=True,
    )


    combined_audit.to_csv(
        audit_dir
        / "combined_extraction_audit.csv",
        index=False,
    )


    # =========================================================================
    # Final model-ready QC summary
    # =========================================================================

    qc_df = pd.DataFrame(
        qc_rows
    )


    qc_df.to_csv(
        audit_dir
        / "reference_samples_model_ready_QC.csv",
        index=False,
    )


    print(
        "\n" + "=" * 88
    )

    print(
        "FINAL REFERENCE-SAMPLE AUDIT"
    )

    print(
        "=" * 88
    )


    for row in qc_rows:

        print(
            f"{row['stack']:<4} | "
            f"rows={row['rows']:<3} | "
            f"predictors={row['predictor_columns']:<3} | "
            f"{row['status']}"
        )


    if not all(
        row["status"] == "PASS"
        for row in qc_rows
    ):

        raise SystemExit(
            "One or more reference-sample tables "
            "failed final QC."
        )


    print(
        "\nReference-sample preparation complete."
    )

    print(
        "All three modelling tables contain "
        "525 balanced polygons."
    )


if __name__ == "__main__":
    main()