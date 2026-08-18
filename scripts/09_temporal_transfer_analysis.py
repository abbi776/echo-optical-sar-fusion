#!/usr/bin/env python3
"""
Temporal transfer and reliability diagnostics for the ECHO framework.

This script applies the final 25-feature Compact Fusion XGBoost model to:

    2017-2018  lower-flow period
    2022-2023  high-flow period
    2025-2026  post-flood reference period

The historical outputs are treated as model-consistent temporal transfers
and reliability diagnostics. They are NOT treated as independently
validated classifications or conventional ecological change-detection
products.

Workflow
--------
1. Load the final Compact XGBoost model.
2. Read the exact 25 training predictors in their original order.
3. Match those predictors to equivalent bands in each annual fusion stack
   by season, sensor and feature identity, independent of acquisition year.
4. Require every predictor to occur exactly once.
5. Apply the model block-wise only where all 25 predictors are valid.
6. Save:
       - predicted class map
       - maximum predicted class probability (pmax)
7. Rasterize SR1-SR5 ANAE wetland extents using pixel-centre inclusion.
8. Calculate:
       - class area by period and sub-region
       - combined SR1-SR5 class area
       - class proportions of valid extent
       - pmax summaries by period and predicted class
       - full 3 x 3 class-state transition matrices
9. Save explicit RRG-classified gain/loss summaries without interpreting
   them as recruitment, mortality, encroachment or retreat.

Class-map coding
----------------
    0 = NoData
    1 = RRG
    2 = NFFP
    3 = Water

Probability-map NoData
----------------------
    -9999

Spatial grid
------------
    EPSG:3577
    10 m pixels
    0.01 ha per pixel
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio

from rasterio.features import rasterize
from rasterio.windows import Window


# =============================================================================
# Core configuration
# =============================================================================

EXPECTED_FULL_FUSION_BANDS = 176
EXPECTED_COMPACT_FEATURES = 25

EXPECTED_EPSG = 3577
EXPECTED_RESOLUTION = (10.0, 10.0)

CLASS_MAP_NODATA = 0
PROBABILITY_NODATA = -9999.0

PIXEL_SIZE_M = 10.0
PIXEL_AREA_M2 = PIXEL_SIZE_M ** 2
PIXEL_AREA_HA = PIXEL_AREA_M2 / 10000.0

DEFAULT_BLOCK_SIZE = 512


CLASS_LABELS = {
    1: "RRG",
    2: "NFFP",
    3: "Water",
}

CLASS_IDS = tuple(
    CLASS_LABELS.keys()
)

CLASS_NAME_TO_ID = {
    value: key
    for key, value in CLASS_LABELS.items()
}


REGION_ID_MAP = {
    "SR1": 1,
    "SR2": 2,
    "SR3": 3,
    "SR4": 4,
    "SR5": 5,
}

REGION_LABEL_MAP = {
    value: key
    for key, value in REGION_ID_MAP.items()
}


SEASONS = (
    "WINTER",
    "SPRING",
    "SUMMER",
    "AUTUMN",
)


PERIODS = OrderedDict({
    "2017_2018": {
        "label": "2017-2018",
        "condition": "lower-flow",
        "season_years": {
            "WINTER": 2017,
            "SPRING": 2017,
            "SUMMER": 2018,
            "AUTUMN": 2018,
        },
    },

    "2022_2023": {
        "label": "2022-2023",
        "condition": "high-flow",
        "season_years": {
            "WINTER": 2022,
            "SPRING": 2022,
            "SUMMER": 2023,
            "AUTUMN": 2023,
        },
    },

    "2025_2026": {
        "label": "2025-2026",
        "condition": "post-flood reference",
        "season_years": {
            "WINTER": 2025,
            "SPRING": 2025,
            "SUMMER": 2026,
            "AUTUMN": 2026,
        },
    },
})


TRANSITION_PAIRS = (
    ("2017_2018", "2022_2023"),
    ("2022_2023", "2025_2026"),
    ("2017_2018", "2025_2026"),
)


# =============================================================================
# Feature-name normalisation
# =============================================================================

# These aliases make the temporal-transfer script tolerant of both the
# descriptive names used by the cleaned repository and legacy names used
# during development.

S1_ALIASES = {
    "PR": "VV_VH_RATIO",
    "XPR": "VH_VV_RATIO",
    "VDDPI": "VDDI",

    "GLCM_CORR_VV": "GLCM_CORRELATION_VV",
    "GLCM_CORR_VH": "GLCM_CORRELATION_VH",

    "GLCM_VAR_VV": "GLCM_VARIANCE_VV",
    "GLCM_VAR_VH": "GLCM_VARIANCE_VH",

    "GLCM_SAVAVE_VV": "GLCM_SUMAVE_VV",
    "GLCM_SAVAVE_VH": "GLCM_SUMAVE_VH",
}


S2_FEATURE_ORDER = [
    "BLUE",
    "GREEN",
    "RED",
    "RE1",
    "RE2",
    "RE3",
    "NIR",
    "NIR_NARROW",
    "SWIR1",
    "SWIR2",
    "NDVI",
    "EVI",
    "NDWI",
    "LSWI",
    "MSAVI2",
    "DBSI",
    "GNDVI",
    "TVI",
]


S2_BAND_NUMBER_TO_NAME = {
    index: name
    for index, name in enumerate(
        S2_FEATURE_ORDER,
        start=1,
    )
}


def normalise_tokens(
    name: str,
) -> list[str]:
    """
    Convert a feature name into consistent uppercase tokens.
    """

    value = str(
        name
    ).strip().upper()

    for character in (
        "-",
        " ",
        "/",
        "\\",
        ":",
        ".",
        ",",
        "(",
        ")",
        ";",
    ):
        value = value.replace(
            character,
            "_",
        )

    value = re.sub(
        r"_+",
        "_",
        value,
    ).strip("_")

    return (
        value.split("_")
        if value
        else []
    )


def canonical_s1_identity(
    tokens: list[str],
) -> str:
    """
    Convert an S1 feature tail to the cleaned canonical identity.
    """

    identity = "_".join(
        tokens
    ).upper()

    return S1_ALIASES.get(
        identity,
        identity,
    )


def canonical_s2_identity(
    tokens: list[str],
) -> str:
    """
    Convert S2 descriptive or BAND_XX identities to one canonical name.
    """

    identity = "_".join(
        tokens
    ).upper()


    # BAND_01, BAND_1, BAND01
    match = re.fullmatch(
        r"BAND_?(\d{1,2})",
        identity,
    )

    if match:

        number = int(
            match.group(1)
        )

        if number in S2_BAND_NUMBER_TO_NAME:

            return S2_BAND_NUMBER_TO_NAME[
                number
            ]


    # B01 / B1
    match = re.fullmatch(
        r"B(\d{1,2})",
        identity,
    )

    if match:

        number = int(
            match.group(1)
        )

        if number in S2_BAND_NUMBER_TO_NAME:

            return S2_BAND_NUMBER_TO_NAME[
                number
            ]


    return identity


def feature_name_to_key(
    name: str,
) -> str:
    """
    Convert a year-specific predictor name to a year-independent key.

    Examples
    --------
    SPRING_2025_S1_VH_dB
        -> SPRING_S1_VH_DB

    SPRING_2017_S1_VH_dB
        -> SPRING_S1_VH_DB

    SUMMER_2026_S2_NDVI
        -> SUMMER_S2_NDVI

    SUMMER_2018_S2_NDVI
        -> SUMMER_S2_NDVI

    Legacy:
    SUMMER_2018_S2_BAND_11
        -> SUMMER_S2_NDVI
    """

    tokens = normalise_tokens(
        name
    )

    # Remove year tokens.
    tokens = [
        token
        for token in tokens
        if not re.fullmatch(
            r"\d{4}",
            token,
        )
    ]


    seasons = [
        token
        for token in tokens
        if token in SEASONS
    ]


    sensors = [
        token
        for token in tokens
        if token in {
            "S1",
            "S2",
        }
    ]


    if len(seasons) != 1:

        raise ValueError(
            f"Expected exactly one season in predictor name: {name}"
        )


    if len(sensors) != 1:

        raise ValueError(
            f"Expected exactly one sensor in predictor name: {name}"
        )


    season = seasons[0]
    sensor = sensors[0]


    tail_tokens = [
        token
        for token in tokens
        if token not in SEASONS
        and token not in {
            "S1",
            "S2",
        }
    ]


    if not tail_tokens:

        raise ValueError(
            f"Could not identify feature identity: {name}"
        )


    if sensor == "S1":

        identity = canonical_s1_identity(
            tail_tokens
        )

    else:

        identity = canonical_s2_identity(
            tail_tokens
        )


    return (
        f"{season}_"
        f"{sensor}_"
        f"{identity}"
    )


# =============================================================================
# Raster paths
# =============================================================================

def fusion_stack_path(
    data_root: Path,
    period_key: str,
) -> Path:
    """
    Return the annual 176-band fusion stack for one period.
    """

    period_label = (
        PERIODS[
            period_key
        ][
            "label"
        ]
    )

    period_tag = period_key


    return (
        data_root
        / "images"
        / period_label
        / "Annual_Stack"
        / "Fusion_full_mosaic"
        / f"S1S2_{period_tag}_FULL_FUSION_STACK.tif"
    )


# =============================================================================
# Model loading
# =============================================================================

def load_operational_model(
    model_path: Path,
) -> tuple:
    """
    Load the final Compact XGBoost package.

    Returns
    -------
    package
    model
    compact_features
    label_encoder
    """

    if not model_path.exists():

        raise FileNotFoundError(
            f"Operational ECHO model not found: "
            f"{model_path}"
        )


    package = joblib.load(
        model_path
    )


    if not isinstance(
        package,
        dict,
    ):

        raise TypeError(
            "Expected the operational model file to contain "
            "a model-package dictionary."
        )


    required = {
        "model",
        "feature_cols",
        "label_encoder",
    }


    missing = (
        required
        - set(
            package.keys()
        )
    )


    if missing:

        raise ValueError(
            "Operational model package is missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


    model = package[
        "model"
    ]


    compact_features = list(
        package[
            "feature_cols"
        ]
    )


    label_encoder = package[
        "label_encoder"
    ]


    if len(
        compact_features
    ) != EXPECTED_COMPACT_FEATURES:

        raise ValueError(
            f"Expected {EXPECTED_COMPACT_FEATURES} operational "
            f"predictors; found {len(compact_features)}."
        )


    if len(
        set(
            compact_features
        )
    ) != EXPECTED_COMPACT_FEATURES:

        raise ValueError(
            "Duplicate predictors detected in operational model package."
        )


    if label_encoder is None:

        raise ValueError(
            "Operational XGBoost model package has no label encoder."
        )


    if not hasattr(
        label_encoder,
        "classes_",
    ):

        raise ValueError(
            "Saved label encoder has no classes_."
        )


    expected_classes = {
        "RRG",
        "NFFP",
        "Water",
    }


    if set(
        label_encoder.classes_
    ) != expected_classes:

        raise ValueError(
            "Unexpected label-encoder classes: "
            f"{list(label_encoder.classes_)}"
        )


    return (
        package,
        model,
        compact_features,
        label_encoder,
    )


# =============================================================================
# Raster inventory and feature matching
# =============================================================================

def read_band_inventory(
    stack_path: Path,
) -> pd.DataFrame:
    """
    Read all 176 raster band descriptions and derive feature keys.
    """

    rows = []


    with rasterio.open(
        stack_path
    ) as src:


        if src.count != EXPECTED_FULL_FUSION_BANDS:

            raise ValueError(
                f"{stack_path.name}: expected "
                f"{EXPECTED_FULL_FUSION_BANDS} bands; "
                f"found {src.count}."
            )


        if (
            src.crs is None
            or src.crs.to_epsg()
            != EXPECTED_EPSG
        ):

            raise ValueError(
                f"{stack_path.name}: expected EPSG:{EXPECTED_EPSG}; "
                f"found {src.crs}."
            )


        if tuple(
            src.res
        ) != EXPECTED_RESOLUTION:

            raise ValueError(
                f"{stack_path.name}: expected 10 m resolution; "
                f"found {src.res}."
            )


        descriptions = tuple(
            src.descriptions or ()
        )


        if (
            len(
                descriptions
            )
            != EXPECTED_FULL_FUSION_BANDS
        ):

            raise ValueError(
                f"{stack_path.name}: incomplete band descriptions."
            )


        for band_index, description in enumerate(
            descriptions,
            start=1,
        ):


            if (
                description is None
                or not str(
                    description
                ).strip()
            ):

                raise ValueError(
                    f"{stack_path.name}: band {band_index} "
                    "has no predictor description."
                )


            feature_key = feature_name_to_key(
                description
            )


            rows.append({
                "band_index":
                    band_index,

                "band_description":
                    str(
                        description
                    ).strip(),

                "feature_key":
                    feature_key,
            })


    inventory = pd.DataFrame(
        rows
    )


    duplicate_keys = (
        inventory[
            "feature_key"
        ]
        .value_counts()
    )


    duplicate_keys = duplicate_keys[
        duplicate_keys > 1
    ]


    if not duplicate_keys.empty:

        raise ValueError(
            f"Duplicate feature identities in {stack_path}:\n"
            f"{duplicate_keys}"
        )


    return inventory


def match_compact_features(
    stack_paths: dict[str, Path],
    compact_features: list[str],
) -> tuple[dict[str, list[int]], pd.DataFrame]:
    """
    Match the 25 model predictors to each annual fusion stack.

    Matching is based on:
        season + sensor + feature identity

    Acquisition year is deliberately ignored.

    The resulting indexes remain in original model-training order.
    """

    required_keys = [
        feature_name_to_key(
            feature
        )
        for feature in compact_features
    ]


    if len(
        set(
            required_keys
        )
    ) != EXPECTED_COMPACT_FEATURES:

        raise ValueError(
            "Compact feature list produces duplicate "
            "year-independent feature keys."
        )


    matched_indexes = {}
    audit_rows = []


    for period_key, stack_path in stack_paths.items():

        inventory = read_band_inventory(
            stack_path
        )


        key_to_row = {
            row.feature_key: row
            for row in inventory.itertuples(
                index=False
            )
        }


        indexes = []


        for model_order, (
            model_feature,
            required_key,
        ) in enumerate(
            zip(
                compact_features,
                required_keys,
            ),
            start=1,
        ):


            matching_rows = inventory[
                inventory[
                    "feature_key"
                ]
                == required_key
            ]


            if len(
                matching_rows
            ) != 1:

                raise ValueError(
                    f"{period_key}: predictor {model_feature} "
                    f"({required_key}) matched "
                    f"{len(matching_rows)} raster bands; "
                    "expected exactly one."
                )


            row = matching_rows.iloc[
                0
            ]


            band_index = int(
                row[
                    "band_index"
                ]
            )


            indexes.append(
                band_index
            )


            audit_rows.append({
                "model_feature_order":
                    model_order,

                "training_feature":
                    model_feature,

                "feature_key":
                    required_key,

                "period":
                    period_key,

                "period_label":
                    PERIODS[
                        period_key
                    ][
                        "label"
                    ],

                "matched_stack_band":
                    row[
                        "band_description"
                    ],

                "matched_band_index":
                    band_index,

                "matched_exactly_once":
                    True,

                "stack_path":
                    str(
                        stack_path
                    ),
            })


        if len(
            indexes
        ) != EXPECTED_COMPACT_FEATURES:

            raise RuntimeError(
                f"{period_key}: expected "
                f"{EXPECTED_COMPACT_FEATURES} matched indexes; "
                f"found {len(indexes)}."
            )


        if len(
            set(
                indexes
            )
        ) != EXPECTED_COMPACT_FEATURES:

            raise RuntimeError(
                f"{period_key}: duplicate matched raster indexes."
            )


        matched_indexes[
            period_key
        ] = indexes


    return (
        matched_indexes,
        pd.DataFrame(
            audit_rows
        ),
    )


# =============================================================================
# Cross-period grid audit
# =============================================================================

def audit_stack_alignment(
    stack_paths: dict[str, Path],
) -> None:
    """
    Require all three annual fusion stacks to occupy the same grid.
    """

    paths = list(
        stack_paths.values()
    )


    with rasterio.open(
        paths[0]
    ) as reference:


        reference_signature = {
            "crs":
                reference.crs,

            "transform":
                reference.transform,

            "width":
                reference.width,

            "height":
                reference.height,

            "resolution":
                reference.res,

            "bounds":
                reference.bounds,
        }


    problems = []


    for path in paths[1:]:

        with rasterio.open(
            path
        ) as src:


            checks = {
                "crs":
                    src.crs
                    == reference_signature[
                        "crs"
                    ],

                "transform":
                    src.transform
                    == reference_signature[
                        "transform"
                    ],

                "dimensions":
                    (
                        src.width
                        == reference_signature[
                            "width"
                        ]
                        and
                        src.height
                        == reference_signature[
                            "height"
                        ]
                    ),

                "resolution":
                    src.res
                    == reference_signature[
                        "resolution"
                    ],

                "bounds":
                    src.bounds
                    == reference_signature[
                        "bounds"
                    ],
            }


            failed = [
                key
                for key, passed
                in checks.items()
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

        message = [
            "Annual fusion stacks are not exactly aligned."
        ]


        for path, failed in problems:

            message.append(
                f"{path}: "
                + ", ".join(
                    failed
                )
            )


        raise ValueError(
            "\n".join(
                message
            )
        )


# =============================================================================
# Prediction helpers
# =============================================================================

def generate_windows(
    width: int,
    height: int,
    block_size: int,
):
    """
    Yield fixed-size windows for memory-safe raster prediction.
    """

    for row_offset in range(
        0,
        height,
        block_size,
    ):


        height_now = min(
            block_size,
            height - row_offset,
        )


        for column_offset in range(
            0,
            width,
            block_size,
        ):


            width_now = min(
                block_size,
                width - column_offset,
            )


            yield Window(
                col_off=column_offset,
                row_off=row_offset,
                width=width_now,
                height=height_now,
            )


def encoded_predictions_to_class_ids(
    encoded_predictions: np.ndarray,
    label_encoder,
) -> np.ndarray:
    """
    Convert XGBoost encoded predictions into project class IDs.

    Mapping is derived from the saved LabelEncoder rather than hard-coded.
    """

    encoded_predictions = np.asarray(
        encoded_predictions
    ).astype(int)


    class_names = (
        label_encoder
        .inverse_transform(
            encoded_predictions
        )
    )


    class_ids = np.empty(
        class_names.shape[
            0
        ],
        dtype="uint8",
    )


    for class_name, class_id in (
        CLASS_NAME_TO_ID.items()
    ):

        class_ids[
            class_names
            == class_name
        ] = class_id


    unexpected = (
        set(
            class_names
        )
        - set(
            CLASS_NAME_TO_ID
        )
    )


    if unexpected:

        raise ValueError(
            "Unexpected model class(es): "
            f"{sorted(unexpected)}"
        )


    return class_ids


def predict_period(
    period_key: str,
    stack_path: Path,
    band_indexes: list[int],
    model,
    compact_features: list[str],
    label_encoder,
    output_dir: Path,
    block_size: int,
    overwrite: bool,
) -> tuple[Path, Path, dict]:
    """
    Apply the Compact XGBoost model to one annual fusion stack.

    Prediction occurs only where all 25 required predictors are valid.
    """

    maps_dir = (
        output_dir
        / "maps"
    )

    maps_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    class_output = (
        maps_dir
        / f"class_map_{period_key}_compact_xgboost.tif"
    )


    probability_output = (
        maps_dir
        / f"pmax_{period_key}_compact_xgboost.tif"
    )


    if (
        class_output.exists()
        and probability_output.exists()
        and not overwrite
    ):

        print(
            f"{period_key}: existing maps retained."
        )

        return (
            class_output,
            probability_output,
            {
                "period":
                    period_key,

                "status":
                    "existing_output",
            },
        )


    audit = {
        "period":
            period_key,

        "period_label":
            PERIODS[
                period_key
            ][
                "label"
            ],

        "condition":
            PERIODS[
                period_key
            ][
                "condition"
            ],

        "input_stack":
            str(
                stack_path
            ),

        "selected_band_indexes":
            [
                int(
                    value
                )
                for value in band_indexes
            ],

        "total_pixels":
            0,

        "valid_prediction_pixels":
            0,

        "invalid_prediction_pixels":
            0,

        "windows_processed":
            0,

        "class_pixel_counts":
            {
                str(
                    class_id
                ): 0
                for class_id
                in CLASS_IDS
            },
    }


    with rasterio.open(
        stack_path
    ) as src:


        if src.count != EXPECTED_FULL_FUSION_BANDS:

            raise ValueError(
                f"{period_key}: fusion stack does not "
                "contain 176 predictors."
            )


        nodata_values = (
            src.nodatavals
        )


        class_profile = (
            src.profile.copy()
        )

        class_profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=CLASS_MAP_NODATA,
            compress="DEFLATE",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )


        probability_profile = (
            src.profile.copy()
        )

        probability_profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=PROBABILITY_NODATA,
            compress="DEFLATE",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )


        windows = list(
            generate_windows(
                width=src.width,
                height=src.height,
                block_size=block_size,
            )
        )


        with rasterio.open(
            class_output,
            "w",
            **class_profile,
        ) as class_dst, rasterio.open(
            probability_output,
            "w",
            **probability_profile,
        ) as probability_dst:


            class_dst.set_band_description(
                1,
                f"{period_key}_compact_xgboost_class_id",
            )


            probability_dst.set_band_description(
                1,
                f"{period_key}_compact_xgboost_pmax",
            )


            class_dst.update_tags(
                interpretation=(
                    "Temporal-transfer class state; "
                    "not independently validated historical change"
                ),
                class_1="RRG",
                class_2="NFFP",
                class_3="Water",
            )


            probability_dst.update_tags(
                interpretation=(
                    "Maximum predicted class probability used as "
                    "relative prediction decisiveness; "
                    "not calibrated uncertainty"
                ),
            )


            for window_number, window in enumerate(
                windows,
                start=1,
            ):


                data = src.read(
                    band_indexes,
                    window=window,
                    out_dtype="float32",
                )


                n_bands, rows, columns = (
                    data.shape
                )


                n_pixels = (
                    rows
                    * columns
                )


                X = (
                    data
                    .reshape(
                        n_bands,
                        n_pixels,
                    )
                    .T
                )


                # -------------------------------------------------------------
                # All 25 predictors must be valid
                # -------------------------------------------------------------

                valid = np.all(
                    np.isfinite(
                        X
                    ),
                    axis=1,
                )


                for feature_position, band_index in enumerate(
                    band_indexes
                ):


                    nodata = (
                        nodata_values[
                            band_index - 1
                        ]
                    )


                    if (
                        nodata is not None
                        and np.isfinite(
                            nodata
                        )
                    ):

                        valid &= (
                            X[
                                :,
                                feature_position
                            ]
                            != nodata
                        )


                valid_count = int(
                    valid.sum()
                )


                class_block = np.full(
                    n_pixels,
                    CLASS_MAP_NODATA,
                    dtype="uint8",
                )


                probability_block = np.full(
                    n_pixels,
                    PROBABILITY_NODATA,
                    dtype="float32",
                )


                if valid_count > 0:


                    X_valid = pd.DataFrame(
                        X[
                            valid
                        ],
                        columns=compact_features,
                    )


                    encoded_predictions = (
                        model.predict(
                            X_valid
                        )
                        .astype(int)
                    )


                    probabilities = (
                        model.predict_proba(
                            X_valid
                        )
                    )


                    pmax = np.max(
                        probabilities,
                        axis=1,
                    ).astype(
                        "float32"
                    )


                    class_ids = (
                        encoded_predictions_to_class_ids(
                            encoded_predictions,
                            label_encoder,
                        )
                    )


                    class_block[
                        valid
                    ] = class_ids


                    probability_block[
                        valid
                    ] = pmax


                    unique_classes, counts = (
                        np.unique(
                            class_ids,
                            return_counts=True,
                        )
                    )


                    for class_id, count in zip(
                        unique_classes,
                        counts,
                    ):

                        audit[
                            "class_pixel_counts"
                        ][
                            str(
                                int(
                                    class_id
                                )
                            )
                        ] += int(
                            count
                        )


                class_dst.write(
                    class_block.reshape(
                        rows,
                        columns,
                    ),
                    1,
                    window=window,
                )


                probability_dst.write(
                    probability_block.reshape(
                        rows,
                        columns,
                    ),
                    1,
                    window=window,
                )


                audit[
                    "total_pixels"
                ] += n_pixels


                audit[
                    "valid_prediction_pixels"
                ] += valid_count


                audit[
                    "invalid_prediction_pixels"
                ] += (
                    n_pixels
                    - valid_count
                )


                audit[
                    "windows_processed"
                ] += 1


                if (
                    window_number == 1
                    or
                    window_number % 50 == 0
                    or
                    window_number
                    == len(
                        windows
                    )
                ):

                    print(
                        f"{period_key}: "
                        f"window "
                        f"{window_number}/"
                        f"{len(windows)} | "
                        f"valid pixels="
                        f"{audit['valid_prediction_pixels']:,}"
                    )


    return (
        class_output,
        probability_output,
        audit,
    )


# =============================================================================
# ANAE wetland regions
# =============================================================================

def find_region_file(
    wetlands_dir: Path,
    region_number: int,
) -> Path:
    """
    Find an ANAE wetland polygon file for one sub-region.

    Both cleaned SR naming and the original P1-P5 filenames are supported.
    """

    candidates = [
        wetlands_dir
        / f"SR{region_number}ANAEWetlands.shp",

        wetlands_dir
        / f"SR{region_number}_ANAE_Wetlands.shp",

        wetlands_dir
        / f"P{region_number}ANAEWetlands.shp",

        wetlands_dir
        / f"P{region_number}_ANAE_Wetlands.shp",
    ]


    for path in candidates:

        if path.exists():

            return path


    raise FileNotFoundError(
        f"Could not locate ANAE wetland polygon file "
        f"for SR{region_number} in {wetlands_dir}."
    )


def load_anae_regions(
    wetlands_dir: Path,
    target_crs,
) -> gpd.GeoDataFrame:
    """
    Load SR1-SR5 ANAE wetland polygons and dissolve by sub-region.
    """

    region_frames = []


    for region_number in range(
        1,
        6,
    ):


        path = find_region_file(
            wetlands_dir,
            region_number,
        )


        gdf = gpd.read_file(
            path
        )


        if gdf.crs is None:

            raise ValueError(
                f"ANAE file has no CRS: {path}"
            )


        gdf = gdf[
            gdf.geometry.notnull()
            & ~gdf.geometry.is_empty
        ].copy()


        if gdf.empty:

            raise ValueError(
                f"ANAE file contains no usable geometry: {path}"
            )


        gdf[
            "region"
        ] = (
            f"SR{region_number}"
        )


        region_frames.append(
            gdf[
                [
                    "region",
                    "geometry",
                ]
            ]
        )


    combined = pd.concat(
        region_frames,
        ignore_index=True,
    )


    combined = gpd.GeoDataFrame(
        combined,
        geometry="geometry",
        crs=region_frames[
            0
        ].crs,
    )


    combined = combined.to_crs(
        target_crs
    )


    dissolved = (
        combined
        .dissolve(
            by="region"
        )
        .reset_index()
    )


    observed_regions = set(
        dissolved[
            "region"
        ]
    )


    if observed_regions != set(
        REGION_ID_MAP
    ):

        raise ValueError(
            "ANAE region set does not match SR1-SR5."
        )


    return dissolved


def create_region_id_raster(
    regions: gpd.GeoDataFrame,
    reference_raster: Path,
    output_path: Path,
    overwrite: bool,
) -> pd.DataFrame:
    """
    Rasterize SR1-SR5 ANAE wetlands to the prediction grid.

    all_touched=False implements pixel-centre inclusion.
    """

    if (
        output_path.exists()
        and not overwrite
    ):

        print(
            "Existing ANAE region-ID raster retained."
        )


    else:

        with rasterio.open(
            reference_raster
        ) as ref:


            profile = (
                ref.profile.copy()
            )


            regions_now = regions.to_crs(
                ref.crs
            )


            shapes = [
                (
                    row.geometry,
                    REGION_ID_MAP[
                        row.region
                    ],
                )
                for row
                in regions_now.itertuples(
                    index=False
                )
            ]


            region_array = rasterize(
                shapes=shapes,
                out_shape=(
                    ref.height,
                    ref.width,
                ),
                transform=ref.transform,
                fill=0,
                dtype="uint8",

                # Manuscript pixel-centre inclusion rule.
                all_touched=False,
            )


            profile.update(
                driver="GTiff",
                count=1,
                dtype="uint8",
                nodata=0,
                compress="DEFLATE",
                predictor=2,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                BIGTIFF="IF_SAFER",
            )


            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )


            with rasterio.open(
                output_path,
                "w",
                **profile,
            ) as dst:


                dst.write(
                    region_array,
                    1,
                )


                dst.set_band_description(
                    1,
                    "SR1_to_SR5_ANAE_wetland_region_id",
                )


                dst.update_tags(
                    region_1="SR1",
                    region_2="SR2",
                    region_3="SR3",
                    region_4="SR4",
                    region_5="SR5",
                    rasterization_rule="pixel-centre inclusion",
                )


    # =========================================================================
    # Audit rasterized areas
    # =========================================================================

    with rasterio.open(
        output_path
    ) as src:


        array = src.read(
            1
        )


    rows = []


    for region_id, region_label in (
        REGION_LABEL_MAP.items()
    ):


        count = int(
            np.sum(
                array
                == region_id
            )
        )


        rows.append({
            "region_id":
                region_id,

            "region":
                region_label,

            "pixel_count":
                count,

            "area_ha":
                count
                * PIXEL_AREA_HA,
        })


    return pd.DataFrame(
        rows
    )


# =============================================================================
# Raster alignment helper
# =============================================================================

def require_aligned(
    first_path: Path,
    second_path: Path,
) -> None:
    """
    Require two rasters to share dimensions, CRS and transform.
    """

    with rasterio.open(
        first_path
    ) as first, rasterio.open(
        second_path
    ) as second:


        checks = {
            "dimensions":
                (
                    first.width
                    == second.width
                    and
                    first.height
                    == second.height
                ),

            "crs":
                first.crs
                == second.crs,

            "transform":
                first.transform
                == second.transform,
        }


    failed = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]


    if failed:

        raise ValueError(
            f"Raster alignment failure between:\n"
            f"{first_path}\n"
            f"{second_path}\n"
            f"Failed: {', '.join(failed)}"
        )


# =============================================================================
# Area summaries
# =============================================================================

def summarize_class_area_by_region(
    period_key: str,
    class_map: Path,
    region_raster: Path,
) -> pd.DataFrame:
    """
    Calculate predicted class areas within each SR ANAE wetland extent.
    """

    require_aligned(
        class_map,
        region_raster,
    )


    class_counts = {
        region_id: {
            class_id: 0
            for class_id
            in CLASS_IDS
        }
        for region_id
        in REGION_LABEL_MAP
    }


    valid_counts = {
        region_id: 0
        for region_id
        in REGION_LABEL_MAP
    }


    mask_counts = {
        region_id: 0
        for region_id
        in REGION_LABEL_MAP
    }


    with rasterio.open(
        class_map
    ) as class_src, rasterio.open(
        region_raster
    ) as region_src:


        for _, window in class_src.block_windows(
            1
        ):


            classes = class_src.read(
                1,
                window=window,
            )


            regions = region_src.read(
                1,
                window=window,
            )


            for region_id in REGION_LABEL_MAP:


                inside = (
                    regions
                    == region_id
                )


                mask_counts[
                    region_id
                ] += int(
                    inside.sum()
                )


                valid = (
                    inside
                    & (
                        classes
                        != CLASS_MAP_NODATA
                    )
                )


                valid_counts[
                    region_id
                ] += int(
                    valid.sum()
                )


                for class_id in CLASS_IDS:


                    class_counts[
                        region_id
                    ][
                        class_id
                    ] += int(
                        (
                            valid
                            & (
                                classes
                                == class_id
                            )
                        ).sum()
                    )


    rows = []


    for region_id, region_label in (
        REGION_LABEL_MAP.items()
    ):


        valid_pixels = (
            valid_counts[
                region_id
            ]
        )


        for class_id in CLASS_IDS:


            pixels = (
                class_counts[
                    region_id
                ][
                    class_id
                ]
            )


            rows.append({
                "period":
                    period_key,

                "period_label":
                    PERIODS[
                        period_key
                    ][
                        "label"
                    ],

                "hydrological_condition":
                    PERIODS[
                        period_key
                    ][
                        "condition"
                    ],

                "region_id":
                    region_id,

                "region":
                    region_label,

                "class_id":
                    class_id,

                "class_label":
                    CLASS_LABELS[
                        class_id
                    ],

                "pixel_count":
                    pixels,

                "area_ha":
                    pixels
                    * PIXEL_AREA_HA,

                "percent_of_valid_extent":
                    (
                        100.0
                        * pixels
                        / valid_pixels
                        if valid_pixels > 0
                        else np.nan
                    ),

                "valid_region_pixels":
                    valid_pixels,

                "valid_region_area_ha":
                    valid_pixels
                    * PIXEL_AREA_HA,

                "anae_region_pixels":
                    mask_counts[
                        region_id
                    ],

                "anae_region_area_ha":
                    mask_counts[
                        region_id
                    ]
                    * PIXEL_AREA_HA,
            })


    return pd.DataFrame(
        rows
    )


def make_combined_area_summary(
    regional_area: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate SR1-SR5 into the combined ANAE wetland extent.
    """

    rows = []


    for period_key in PERIODS:


        subset = regional_area[
            regional_area[
                "period"
            ]
            == period_key
        ]


        valid_pixels = int(
            subset[
                [
                    "region",
                    "valid_region_pixels",
                ]
            ]
            .drop_duplicates()[
                "valid_region_pixels"
            ]
            .sum()
        )


        anae_pixels = int(
            subset[
                [
                    "region",
                    "anae_region_pixels",
                ]
            ]
            .drop_duplicates()[
                "anae_region_pixels"
            ]
            .sum()
        )


        for class_id in CLASS_IDS:


            class_subset = subset[
                subset[
                    "class_id"
                ]
                == class_id
            ]


            pixels = int(
                class_subset[
                    "pixel_count"
                ].sum()
            )


            rows.append({
                "period":
                    period_key,

                "period_label":
                    PERIODS[
                        period_key
                    ][
                        "label"
                    ],

                "hydrological_condition":
                    PERIODS[
                        period_key
                    ][
                        "condition"
                    ],

                "class_id":
                    class_id,

                "class_label":
                    CLASS_LABELS[
                        class_id
                    ],

                "pixel_count":
                    pixels,

                "area_ha":
                    pixels
                    * PIXEL_AREA_HA,

                "percent_of_valid_extent":
                    (
                        100.0
                        * pixels
                        / valid_pixels
                        if valid_pixels > 0
                        else np.nan
                    ),

                "valid_extent_pixels":
                    valid_pixels,

                "valid_extent_area_ha":
                    valid_pixels
                    * PIXEL_AREA_HA,

                "anae_extent_pixels":
                    anae_pixels,

                "anae_extent_area_ha":
                    anae_pixels
                    * PIXEL_AREA_HA,
            })


    return pd.DataFrame(
        rows
    )


# =============================================================================
# pmax summaries
# =============================================================================

def summarize_pmax_by_class(
    period_key: str,
    class_map: Path,
    pmax_map: Path,
    region_raster: Path,
) -> pd.DataFrame:
    """
    Summarize pmax within the combined SR1-SR5 ANAE extent by predicted class.
    """

    require_aligned(
        class_map,
        pmax_map,
    )

    require_aligned(
        class_map,
        region_raster,
    )


    values_by_class = {
        class_id: []
        for class_id
        in CLASS_IDS
    }


    with rasterio.open(
        class_map
    ) as class_src, rasterio.open(
        pmax_map
    ) as probability_src, rasterio.open(
        region_raster
    ) as region_src:


        for _, window in class_src.block_windows(
            1
        ):


            classes = class_src.read(
                1,
                window=window,
            )


            probabilities = (
                probability_src.read(
                    1,
                    window=window,
                )
            )


            regions = region_src.read(
                1,
                window=window,
            )


            inside_anae = (
                regions > 0
            )


            probability_valid = (
                np.isfinite(
                    probabilities
                )
                & (
                    probabilities
                    != PROBABILITY_NODATA
                )
            )


            for class_id in CLASS_IDS:


                mask = (
                    inside_anae
                    & probability_valid
                    & (
                        classes
                        == class_id
                    )
                )


                if np.any(
                    mask
                ):

                    values_by_class[
                        class_id
                    ].append(
                        probabilities[
                            mask
                        ].astype(
                            "float32"
                        )
                    )


    rows = []


    for class_id in CLASS_IDS:


        arrays = values_by_class[
            class_id
        ]


        if arrays:

            values = np.concatenate(
                arrays
            )


            rows.append({
                "period":
                    period_key,

                "period_label":
                    PERIODS[
                        period_key
                    ][
                        "label"
                    ],

                "hydrological_condition":
                    PERIODS[
                        period_key
                    ][
                        "condition"
                    ],

                "class_id":
                    class_id,

                "class_label":
                    CLASS_LABELS[
                        class_id
                    ],

                "n_pixels":
                    int(
                        values.size
                    ),

                "pmax_mean":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "pmax_median":
                    float(
                        np.median(
                            values
                        )
                    ),

                "pmax_p05":
                    float(
                        np.percentile(
                            values,
                            5,
                        )
                    ),

                "pmax_p25":
                    float(
                        np.percentile(
                            values,
                            25,
                        )
                    ),

                "pmax_p75":
                    float(
                        np.percentile(
                            values,
                            75,
                        )
                    ),

                "pmax_p95":
                    float(
                        np.percentile(
                            values,
                            95,
                        )
                    ),
            })


        else:

            rows.append({
                "period":
                    period_key,

                "period_label":
                    PERIODS[
                        period_key
                    ][
                        "label"
                    ],

                "hydrological_condition":
                    PERIODS[
                        period_key
                    ][
                        "condition"
                    ],

                "class_id":
                    class_id,

                "class_label":
                    CLASS_LABELS[
                        class_id
                    ],

                "n_pixels":
                    0,

                "pmax_mean":
                    np.nan,

                "pmax_median":
                    np.nan,

                "pmax_p05":
                    np.nan,

                "pmax_p25":
                    np.nan,

                "pmax_p75":
                    np.nan,

                "pmax_p95":
                    np.nan,
            })


    return pd.DataFrame(
        rows
    )


# =============================================================================
# Transition matrices
# =============================================================================

def transition_matrix_by_region(
    from_period: str,
    to_period: str,
    from_map: Path,
    to_map: Path,
    region_raster: Path,
) -> pd.DataFrame:
    """
    Calculate the complete 3 x 3 class-state transition matrix by region.

    Only pixels valid in both periods are included.
    """

    require_aligned(
        from_map,
        to_map,
    )

    require_aligned(
        from_map,
        region_raster,
    )


    counts = {
        region_id: {
            (
                from_class,
                to_class,
            ): 0

            for from_class
            in CLASS_IDS

            for to_class
            in CLASS_IDS
        }

        for region_id
        in REGION_LABEL_MAP
    }


    valid_counts = {
        region_id: 0
        for region_id
        in REGION_LABEL_MAP
    }


    with rasterio.open(
        from_map
    ) as first_src, rasterio.open(
        to_map
    ) as second_src, rasterio.open(
        region_raster
    ) as region_src:


        for _, window in first_src.block_windows(
            1
        ):


            first = first_src.read(
                1,
                window=window,
            )


            second = second_src.read(
                1,
                window=window,
            )


            regions = region_src.read(
                1,
                window=window,
            )


            base_valid = (
                (first != CLASS_MAP_NODATA)
                & (second != CLASS_MAP_NODATA)
            )


            for region_id in REGION_LABEL_MAP:


                valid = (
                    base_valid
                    & (
                        regions
                        == region_id
                    )
                )


                valid_counts[
                    region_id
                ] += int(
                    valid.sum()
                )


                for from_class in CLASS_IDS:

                    for to_class in CLASS_IDS:


                        counts[
                            region_id
                        ][
                            (
                                from_class,
                                to_class,
                            )
                        ] += int(
                            (
                                valid
                                & (
                                    first
                                    == from_class
                                )
                                & (
                                    second
                                    == to_class
                                )
                            ).sum()
                        )


    rows = []


    for region_id, region_label in (
        REGION_LABEL_MAP.items()
    ):


        valid_pixels = (
            valid_counts[
                region_id
            ]
        )


        for from_class in CLASS_IDS:

            for to_class in CLASS_IDS:


                pixels = (
                    counts[
                        region_id
                    ][
                        (
                            from_class,
                            to_class,
                        )
                    ]
                )


                rows.append({
                    "from_period":
                        from_period,

                    "from_period_label":
                        PERIODS[
                            from_period
                        ][
                            "label"
                        ],

                    "to_period":
                        to_period,

                    "to_period_label":
                        PERIODS[
                            to_period
                        ][
                            "label"
                        ],

                    "region_id":
                        region_id,

                    "region":
                        region_label,

                    "from_class_id":
                        from_class,

                    "from_class_label":
                        CLASS_LABELS[
                            from_class
                        ],

                    "to_class_id":
                        to_class,

                    "to_class_label":
                        CLASS_LABELS[
                            to_class
                        ],

                    "pixel_count":
                        pixels,

                    "area_ha":
                        pixels
                        * PIXEL_AREA_HA,

                    "percent_of_valid_transition_extent":
                        (
                            100.0
                            * pixels
                            / valid_pixels
                            if valid_pixels > 0
                            else np.nan
                        ),

                    "valid_transition_pixels":
                        valid_pixels,

                    "valid_transition_area_ha":
                        valid_pixels
                        * PIXEL_AREA_HA,
                })


    return pd.DataFrame(
        rows
    )


def make_combined_transition_summary(
    regional_transitions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate the regional transition matrices across SR1-SR5.
    """

    group_columns = [
        "from_period",
        "from_period_label",
        "to_period",
        "to_period_label",
        "from_class_id",
        "from_class_label",
        "to_class_id",
        "to_class_label",
    ]


    combined = (
        regional_transitions
        .groupby(
            group_columns,
            as_index=False,
        )[
            "pixel_count"
        ]
        .sum()
    )


    valid_lookup = (
        regional_transitions[
            [
                "from_period",
                "to_period",
                "region",
                "valid_transition_pixels",
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "from_period",
                "to_period",
            ],
            as_index=False,
        )[
            "valid_transition_pixels"
        ]
        .sum()
    )


    combined = combined.merge(
        valid_lookup,
        on=[
            "from_period",
            "to_period",
        ],
        how="left",
    )


    combined[
        "area_ha"
    ] = (
        combined[
            "pixel_count"
        ]
        * PIXEL_AREA_HA
    )


    combined[
        "valid_transition_area_ha"
    ] = (
        combined[
            "valid_transition_pixels"
        ]
        * PIXEL_AREA_HA
    )


    combined[
        "percent_of_valid_transition_extent"
    ] = (
        100.0
        * combined[
            "pixel_count"
        ]
        / combined[
            "valid_transition_pixels"
        ]
    )


    return combined


# =============================================================================
# RRG-NFFP class-state summaries
# =============================================================================

def summarize_rrg_nffp_exchange(
    transition_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract the RRG-NFFP transitions emphasized in the manuscript.

    Terminology:
        NFFP -> RRG = RRG-classified gain
        RRG -> NFFP = RRG-classified loss

    These describe modelled class-state transitions only.
    """

    rows = []


    pair_columns = [
        "from_period",
        "from_period_label",
        "to_period",
        "to_period_label",
    ]


    has_region = (
        "region"
        in transition_table.columns
    )


    if has_region:

        pair_columns.extend(
            [
                "region_id",
                "region",
            ]
        )


    grouped = transition_table.groupby(
        pair_columns,
        dropna=False,
    )


    for keys, subset in grouped:


        if not isinstance(
            keys,
            tuple,
        ):

            keys = (
                keys,
            )


        metadata = dict(
            zip(
                pair_columns,
                keys,
            )
        )


        def area_for(
            from_label: str,
            to_label: str,
        ) -> float:

            selected = subset[
                (
                    subset[
                        "from_class_label"
                    ]
                    == from_label
                )
                &
                (
                    subset[
                        "to_class_label"
                    ]
                    == to_label
                )
            ]


            if selected.empty:

                return 0.0


            return float(
                selected[
                    "area_ha"
                ].sum()
            )


        rrg_gain = area_for(
            "NFFP",
            "RRG",
        )


        rrg_loss = area_for(
            "RRG",
            "NFFP",
        )


        water_involved = float(
            subset[
                (
                    subset[
                        "from_class_label"
                    ]
                    == "Water"
                )
                |
                (
                    subset[
                        "to_class_label"
                    ]
                    == "Water"
                )
            ][
                "area_ha"
            ].sum()
        )


        rows.append({
            **metadata,

            "nffp_to_rrg_classified_gain_ha":
                rrg_gain,

            "rrg_to_nffp_classified_loss_ha":
                rrg_loss,

            "net_rrg_classified_exchange_ha":
                rrg_gain
                - rrg_loss,

            "water_involved_transition_area_ha":
                water_involved,

            "interpretation":
                (
                    "Modelled class-state transitions only; "
                    "not recruitment, mortality, encroachment or retreat."
                ),
        })


    return pd.DataFrame(
        rows
    )


# =============================================================================
# Command-line configuration
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


    parser = argparse.ArgumentParser(
        description=(
            "Apply the final ECHO Compact XGBoost model "
            "across three hydrological periods and produce "
            "temporal-transfer reliability diagnostics."
        )
    )


    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            repo_root
            / "data"
        ),
        help=(
            "Root data directory containing images/ and wetlands/. "
            "Default: <repo>/data"
        ),
    )


    parser.add_argument(
        "--model",
        type=Path,
        default=(
            repo_root
            / "models"
            / "compact_fusion"
            / "ECHO_OPERATIONAL_COMPACT_XGBOOST.joblib"
        ),
        help=(
            "Operational Compact XGBoost model produced "
            "by script 08."
        ),
    )


    parser.add_argument(
        "--wetlands-dir",
        type=Path,
        default=None,
        help=(
            "Folder containing the five ANAE wetland "
            "polygon files. Default: <data-root>/wetlands"
        ),
    )


    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repo_root
            / "results"
            / "temporal_transfer"
        ),
        help=(
            "Directory for temporal-transfer outputs."
        ),
    )


    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=(
            f"Prediction window size. "
            f"Default: {DEFAULT_BLOCK_SIZE}"
        ),
    )


    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Recreate prediction maps and region raster "
            "when outputs already exist."
        ),
    )


    return parser.parse_args()


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    """
    Run temporal transfer and reliability diagnostics.
    """

    args = parse_args()


    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )


    model_path = (
        args.model
        .expanduser()
        .resolve()
    )


    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )


    wetlands_dir = (
        args.wetlands_dir
        .expanduser()
        .resolve()
        if args.wetlands_dir is not None
        else (
            data_root
            / "wetlands"
        )
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    audit_dir = (
        output_dir
        / "audits"
    )


    summary_dir = (
        output_dir
        / "summaries"
    )


    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # =========================================================================
    # Load operational model
    # =========================================================================

    (
        model_package,
        model,
        compact_features,
        label_encoder,
    ) = load_operational_model(
        model_path
    )


    print(
        "ECHO temporal-transfer analysis"
    )

    print(
        "Operational model:",
        model_path,
    )

    print(
        "Compact predictors:",
        len(
            compact_features
        ),
    )

    print(
        "Encoded XGBoost class order:",
        list(
            label_encoder.classes_
        ),
    )


    print(
        "\nImportant interpretation:"
    )

    print(
        "Historical maps are temporal-transfer reliability "
        "diagnostics, not independently validated ecological change."
    )


    # =========================================================================
    # Locate all three annual fusion stacks
    # =========================================================================

    stack_paths = {
        period_key:
            fusion_stack_path(
                data_root,
                period_key,
            )

        for period_key
        in PERIODS
    }


    for period_key, path in (
        stack_paths.items()
    ):


        if not path.exists():

            raise FileNotFoundError(
                f"Fusion stack not found for "
                f"{period_key}: {path}"
            )


        print(
            f"{period_key}: {path}"
        )


    audit_stack_alignment(
        stack_paths
    )


    print(
        "Cross-period raster alignment audit: PASS"
    )


    # =========================================================================
    # Match all 25 predictors by feature identity
    # =========================================================================

    (
        matched_band_indexes,
        feature_audit,
    ) = match_compact_features(
        stack_paths=stack_paths,
        compact_features=compact_features,
    )


    feature_audit_path = (
        audit_dir
        / "temporal_feature_matching_audit.csv"
    )


    feature_audit.to_csv(
        feature_audit_path,
        index=False,
    )


    print(
        "Temporal feature matching audit: PASS"
    )

    print(
        "Saved:",
        feature_audit_path,
    )


    # =========================================================================
    # Generate class and pmax maps
    # =========================================================================

    class_maps = {}
    pmax_maps = {}
    prediction_audits = {}


    for period_key in PERIODS:


        print(
            "\n" + "=" * 88
        )

        print(
            f"Temporal transfer: "
            f"{PERIODS[period_key]['label']}"
        )

        print(
            "=" * 88
        )


        (
            class_map,
            pmax_map,
            prediction_audit,
        ) = predict_period(
            period_key=period_key,
            stack_path=stack_paths[
                period_key
            ],
            band_indexes=matched_band_indexes[
                period_key
            ],
            model=model,
            compact_features=compact_features,
            label_encoder=label_encoder,
            output_dir=output_dir,
            block_size=args.block_size,
            overwrite=args.overwrite,
        )


        class_maps[
            period_key
        ] = class_map


        pmax_maps[
            period_key
        ] = pmax_map


        prediction_audits[
            period_key
        ] = prediction_audit


    with open(
        audit_dir
        / "prediction_audit.json",
        "w",
        encoding="utf-8",
    ) as file:


        json.dump(
            prediction_audits,
            file,
            indent=2,
        )


    # =========================================================================
    # Load and rasterize ANAE wetland regions
    # =========================================================================

    with rasterio.open(
        class_maps[
            "2025_2026"
        ]
    ) as reference:


        target_crs = reference.crs


    regions = load_anae_regions(
        wetlands_dir,
        target_crs,
    )


    regions_path = (
        audit_dir
        / "SR1_SR5_ANAE_wetland_extents.gpkg"
    )


    regions.to_file(
        regions_path,
        driver="GPKG",
    )


    region_raster = (
        audit_dir
        / "SR1_SR5_ANAE_region_id.tif"
    )


    region_area_audit = create_region_id_raster(
        regions=regions,
        reference_raster=class_maps[
            "2025_2026"
        ],
        output_path=region_raster,
        overwrite=args.overwrite,
    )


    region_area_audit.to_csv(
        audit_dir
        / "anae_region_raster_area_audit.csv",
        index=False,
    )


    print(
        "ANAE pixel-centre rasterization: PASS"
    )


    # =========================================================================
    # Class area summaries
    # =========================================================================

    regional_area_tables = []


    for period_key in PERIODS:


        regional_area_tables.append(
            summarize_class_area_by_region(
                period_key=period_key,
                class_map=class_maps[
                    period_key
                ],
                region_raster=region_raster,
            )
        )


    regional_area = pd.concat(
        regional_area_tables,
        ignore_index=True,
    )


    combined_area = (
        make_combined_area_summary(
            regional_area
        )
    )


    regional_area.to_csv(
        summary_dir
        / "class_area_by_period_region.csv",
        index=False,
    )


    combined_area.to_csv(
        summary_dir
        / "class_area_by_period_combined_SR1_SR5.csv",
        index=False,
    )


    # =========================================================================
    # pmax reliability diagnostics
    # =========================================================================

    pmax_tables = []


    for period_key in PERIODS:


        pmax_tables.append(
            summarize_pmax_by_class(
                period_key=period_key,
                class_map=class_maps[
                    period_key
                ],
                pmax_map=pmax_maps[
                    period_key
                ],
                region_raster=region_raster,
            )
        )


    pmax_summary = pd.concat(
        pmax_tables,
        ignore_index=True,
    )


    pmax_summary.to_csv(
        summary_dir
        / "pmax_by_period_and_predicted_class.csv",
        index=False,
    )


    # =========================================================================
    # Full 3 x 3 class-state transitions
    # =========================================================================

    regional_transition_tables = []


    for from_period, to_period in (
        TRANSITION_PAIRS
    ):


        table = transition_matrix_by_region(
            from_period=from_period,
            to_period=to_period,
            from_map=class_maps[
                from_period
            ],
            to_map=class_maps[
                to_period
            ],
            region_raster=region_raster,
        )


        regional_transition_tables.append(
            table
        )


    regional_transitions = pd.concat(
        regional_transition_tables,
        ignore_index=True,
    )


    combined_transitions = (
        make_combined_transition_summary(
            regional_transitions
        )
    )


    regional_transitions.to_csv(
        summary_dir
        / "class_state_transitions_by_region.csv",
        index=False,
    )


    combined_transitions.to_csv(
        summary_dir
        / "class_state_transitions_combined_SR1_SR5.csv",
        index=False,
    )


    # =========================================================================
    # RRG-NFFP exchange summaries
    # =========================================================================

    regional_exchange = (
        summarize_rrg_nffp_exchange(
            regional_transitions
        )
    )


    combined_exchange = (
        summarize_rrg_nffp_exchange(
            combined_transitions
        )
    )


    regional_exchange.to_csv(
        summary_dir
        / "rrg_nffp_classified_exchange_by_region.csv",
        index=False,
    )


    combined_exchange.to_csv(
        summary_dir
        / "rrg_nffp_classified_exchange_combined_SR1_SR5.csv",
        index=False,
    )


    # =========================================================================
    # Save run metadata
    # =========================================================================

    run_metadata = {
        "framework":
            "ECHO",

        "model":
            str(
                model_path
            ),

        "operational_classifier":
            "Compact XGBoost",

        "n_features":
            len(
                compact_features
            ),

        "periods":
            PERIODS,

        "transition_pairs":
            [
                list(
                    pair
                )
                for pair
                in TRANSITION_PAIRS
            ],

        "pixel_size_m":
            PIXEL_SIZE_M,

        "pixel_area_ha":
            PIXEL_AREA_HA,

        "anae_rasterization":
            "pixel-centre inclusion (all_touched=False)",

        "prediction_validity_rule":
            (
                "A pixel is predicted only when all 25 "
                "required predictors are valid."
            ),

        "pmax_interpretation":
            (
                "Maximum predicted class probability used as "
                "relative prediction decisiveness; "
                "not calibrated uncertainty."
            ),

        "historical_interpretation":
            (
                "Earlier maps are model-consistent temporal transfers "
                "used as reliability and condition-sensitivity diagnostics; "
                "they are not independently validated classifications or "
                "conventional ecological change-detection products."
            ),

        "transition_terminology":
            (
                "NFFP->RRG = RRG-classified gain; "
                "RRG->NFFP = RRG-classified loss. "
                "These terms do not imply recruitment, mortality, "
                "encroachment or retreat."
            ),
    }


    with open(
        output_dir
        / "temporal_transfer_run_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:


        json.dump(
            run_metadata,
            file,
            indent=2,
        )


    # =========================================================================
    # Final console summary
    # =========================================================================

    print(
        "\n" + "=" * 96
    )

    print(
        "TEMPORAL TRANSFER COMPLETE"
    )

    print(
        "=" * 96
    )


    print(
        "\nCombined SR1-SR5 class areas:"
    )


    print(
        combined_area[
            [
                "period_label",
                "class_label",
                "area_ha",
                "percent_of_valid_extent",
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nPrediction decisiveness (median pmax):"
    )


    print(
        pmax_summary[
            [
                "period_label",
                "class_label",
                "pmax_median",
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nRRG-NFFP class-state exchange:"
    )


    print(
        combined_exchange[
            [
                "from_period_label",
                "to_period_label",
                "nffp_to_rrg_classified_gain_ha",
                "rrg_to_nffp_classified_loss_ha",
                "net_rrg_classified_exchange_ha",
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nOutputs:",
        output_dir,
    )


    print(
        "\nReminder:"
    )

    print(
        "These outputs characterize temporal-transfer behaviour "
        "and hydrologically conditioned class separability. "
        "They are not independently validated historical change."
    )


if __name__ == "__main__":
    main()