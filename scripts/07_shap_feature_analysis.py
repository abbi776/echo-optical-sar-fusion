#!/usr/bin/env python3
"""
SHAP interpretation and compact-feature selection for ECHO.

This script performs two deliberately separate SHAP analyses using the
176-predictor XGBoost Full S1-S2 Fusion model.

PART A — Descriptive all-sample SHAP
------------------------------------
A Full Fusion XGBoost model fitted to all 525 reference polygons is
interpreted descriptively.

Outputs include:
    - overall mean absolute SHAP importance
    - class-wise mean absolute SHAP importance
    - sensor contribution overall and by class
    - feature-group contribution
    - seasonal contribution
    - top individual predictors
    - signed class-specific SHAP values for later figure generation

These results are for interpretation only and are NOT used to select
the compact model.

PART B — Fold-wise training-only SHAP
-------------------------------------
For each leave-one-region-out fold:

    1. hold out one spatial region;
    2. fit the 176-feature XGBoost model using the other four regions;
    3. calculate SHAP values using training observations only;
    4. compute mean absolute SHAP by feature and class.

The held-out region is excluded from both model fitting and SHAP
attribution estimation.

Fold-specific importance values are then:

    averaged by feature and class,
    averaged across classes,
    averaged across folds,
    ranked,
    converted to relative attribution,
    accumulated in descending order.

The smallest subset reaching at least 80% cumulative attribution is
retained as the compact feature set.

No feature is forced into the compact subset on the basis of sensor,
class, season, or physical interpretation.

No compact-model accuracy is evaluated in this script. That evaluation
is performed later in 08_train_compact_fusion_model.py.

Expected design
---------------
Samples:          525
Regions:          SR1-SR5
Classes:          RRG, NFFP, Water
Fusion features:  176
Classifier:       XGBoost
Selection rule:   >= 80% cumulative fold-wise training-only SHAP
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# =============================================================================
# Experimental design
# =============================================================================

RANDOM_STATE = 42

EXPECTED_ROWS = 525
EXPECTED_FEATURES = 176

TARGET_COL = "class_label"
GROUP_COL = "region"

CLASS_ORDER = [
    "RRG",
    "NFFP",
    "Water",
]

REGION_ORDER = [
    "SR1",
    "SR2",
    "SR3",
    "SR4",
    "SR5",
]

EXPECTED_TRAIN_PER_FOLD = 420
EXPECTED_TEST_PER_FOLD = 105

CUMULATIVE_THRESHOLD = 0.80


# =============================================================================
# Feature definitions
# =============================================================================

SEASON_ORDER = [
    "WINTER",
    "SPRING",
    "SUMMER",
    "AUTUMN",
]


S2_BANDS = {
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
}


S2_INDICES = {
    "NDVI",
    "EVI",
    "NDWI",
    "LSWI",
    "MSAVI2",
    "DBSI",
    "GNDVI",
    "TVI",
}


S2_POSITIONAL_NAMES = {
    1: "BLUE",
    2: "GREEN",
    3: "RED",
    4: "RE1",
    5: "RE2",
    6: "RE3",
    7: "NIR",
    8: "NIR_NARROW",
    9: "SWIR1",
    10: "SWIR2",
    11: "NDVI",
    12: "EVI",
    13: "NDWI",
    14: "LSWI",
    15: "MSAVI2",
    16: "DBSI",
    17: "GNDVI",
    18: "TVI",
}


S1_BACKSCATTER = {
    "VV_DB",
    "VH_DB",
}


S1_POLARISATION = {
    "NDPI",
    "NRPB",

    # Clean repository names
    "VV_VH_RATIO",
    "VH_VV_RATIO",
    "VDDI",

    # Legacy names from the original analysis
    "PR",
    "XPR",
    "VDDPI",

    "RVI",
    "SUM",
    "DIFF",
    "PROD",
    "LOG_RATIO",
}


# =============================================================================
# Metadata fields
# =============================================================================

METADATA_COLUMNS = {
    "unique_id",
    "sample_id",
    "roi_id",
    "polygon_id",
    "label_id",

    "class_raw",
    "class_label",
    "class_id",
    "class_name",
    "class",

    "region",
    "site",
    "period",
    "year",
    "stack",
    "sensor",

    "geometry",
    "x",
    "y",
    "area",
    "area_m2",

    "valid_bands",
    "total_bands",
    "valid_band_fraction",
    "min_valid_pixels_per_band",
    "max_valid_pixels_per_band",
    "mean_valid_pixels_per_band",
    "median_valid_pixels_per_band",
    "valid_pixel_count",
    "n_pixels",
    "pixel_count",
}


# =============================================================================
# General feature helpers
# =============================================================================

def normalise_name(
    value: str,
) -> str:
    """
    Convert a predictor name to a consistent uppercase underscore form.
    """

    value = str(
        value
    ).strip().upper()

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


def feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return numeric predictor columns excluding metadata.
    """

    numeric_columns = (
        dataframe
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .columns
        .tolist()
    )

    return [
        column
        for column in numeric_columns
        if column not in METADATA_COLUMNS
    ]


def parse_season(
    feature: str,
) -> str:
    """
    Determine the season encoded in a predictor name.
    """

    value = normalise_name(
        feature
    )

    for season in SEASON_ORDER:

        if value.startswith(
            season + "_"
        ):

            return season.title()

    return "Unknown"


def parse_sensor(
    feature: str,
) -> str:
    """
    Determine whether a predictor belongs to Sentinel-1 or Sentinel-2.
    """

    value = normalise_name(
        feature
    )

    if "_S1_" in value:
        return "S1"

    if "_S2_" in value:
        return "S2"

    raise ValueError(
        f"Could not determine sensor from predictor: {feature}"
    )


def feature_suffix(
    feature: str,
    sensor: str,
) -> str:
    """
    Return predictor identity after the S1 or S2 token.
    """

    value = normalise_name(
        feature
    )

    token = (
        f"_{sensor.upper()}_"
    )

    if token not in value:

        raise ValueError(
            f"Predictor {feature} does not contain {token}."
        )

    return value.split(
        token,
        1,
    )[1]


def parse_generic_band_number(
    feature: str,
    sensor: str,
) -> int | None:
    """
    Recognise legacy generic names such as S2_BAND_04.
    """

    value = normalise_name(
        feature
    )

    match = re.search(
        rf"{sensor.upper()}_BAND_(\d+)",
        value,
    )

    if match:

        return int(
            match.group(1)
        )

    return None


def parse_feature_identity(
    feature: str,
) -> str:
    """
    Return the physical feature identity independent of season/year.

    Examples
    --------
    SPRING_2025_S1_VH_dB
        -> VH_DB

    SUMMER_2026_S2_NDVI
        -> NDVI

    Legacy:
    WINTER_2025_S2_BAND_11
        -> NDVI
    """

    sensor = parse_sensor(
        feature
    )

    positional_number = (
        parse_generic_band_number(
            feature,
            sensor,
        )
    )

    if (
        sensor == "S2"
        and positional_number is not None
    ):

        if positional_number not in S2_POSITIONAL_NAMES:

            raise ValueError(
                f"Invalid S2 positional band number "
                f"in {feature}."
            )

        return S2_POSITIONAL_NAMES[
            positional_number
        ]

    return feature_suffix(
        feature,
        sensor,
    )


def parse_feature_group(
    feature: str,
) -> str:
    """
    Assign a predictor to the five groups used in the manuscript:

        S2 bands
        S2 indices
        S1 backscatter
        S1 polarisation
        S1 texture
    """

    sensor = parse_sensor(
        feature
    )

    identity = parse_feature_identity(
        feature
    )


    if sensor == "S2":

        if identity in S2_BANDS:
            return "S2 bands"

        if identity in S2_INDICES:
            return "S2 indices"


    if sensor == "S1":

        generic_number = (
            parse_generic_band_number(
                feature,
                "S1",
            )
        )


        if generic_number is not None:

            if 1 <= generic_number <= 2:
                return "S1 backscatter"

            if 3 <= generic_number <= 12:
                return "S1 polarisation"

            if 13 <= generic_number <= 26:
                return "S1 texture"


        if identity in S1_BACKSCATTER:
            return "S1 backscatter"

        if identity in S1_POLARISATION:
            return "S1 polarisation"

        if identity.startswith(
            "GLCM_"
        ):
            return "S1 texture"


    raise ValueError(
        f"Could not assign feature group: {feature}"
    )


def build_feature_metadata(
    features: list[str],
) -> pd.DataFrame:
    """
    Construct reusable metadata for all 176 predictors.
    """

    rows = []

    for position, feature in enumerate(
        features,
        start=1,
    ):

        rows.append({
            "feature_position":
                position,

            "feature":
                feature,

            "sensor":
                parse_sensor(
                    feature
                ),

            "season":
                parse_season(
                    feature
                ),

            "feature_group":
                parse_feature_group(
                    feature
                ),

            "feature_identity":
                parse_feature_identity(
                    feature
                ),
        })


    metadata = pd.DataFrame(
        rows
    )


    if len(metadata) != EXPECTED_FEATURES:

        raise ValueError(
            f"Expected metadata for {EXPECTED_FEATURES} predictors; "
            f"generated {len(metadata)}."
        )


    if metadata["feature"].duplicated().any():

        raise ValueError(
            "Duplicate feature names detected."
        )


    return metadata


# =============================================================================
# Input audit
# =============================================================================

def audit_samples(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Strictly validate the Full Fusion reference-sample table.
    """

    required = {
        "unique_id",
        TARGET_COL,
        GROUP_COL,
    }


    missing = (
        required
        - set(
            dataframe.columns
        )
    )


    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


    if len(dataframe) != EXPECTED_ROWS:

        raise ValueError(
            f"Expected {EXPECTED_ROWS} samples; "
            f"found {len(dataframe)}."
        )


    features = feature_columns(
        dataframe
    )


    if len(features) != EXPECTED_FEATURES:

        raise ValueError(
            f"Expected {EXPECTED_FEATURES} predictors; "
            f"found {len(features)}."
        )


    if dataframe[
        "unique_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate polygon IDs detected."
        )


    if dataframe[
        features
    ].isna().any().any():

        raise ValueError(
            "NaN predictor values detected."
        )


    if not np.isfinite(
        dataframe[
            features
        ].to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            "Non-finite predictor values detected."
        )


    if set(
        dataframe[TARGET_COL]
        .astype(str)
        .unique()
    ) != set(CLASS_ORDER):

        raise ValueError(
            "Unexpected class labels detected."
        )


    if set(
        dataframe[GROUP_COL]
        .astype(str)
        .unique()
    ) != set(REGION_ORDER):

        raise ValueError(
            "Unexpected spatial regions detected."
        )


    for region in REGION_ORDER:

        subset = dataframe[
            dataframe[
                GROUP_COL
            ].astype(str)
            == region
        ]


        if len(subset) != EXPECTED_TEST_PER_FOLD:

            raise ValueError(
                f"{region}: expected "
                f"{EXPECTED_TEST_PER_FOLD} polygons, "
                f"found {len(subset)}."
            )


        counts = (
            subset[
                TARGET_COL
            ]
            .astype(str)
            .value_counts()
        )


        if any(
            int(
                counts.get(
                    class_name,
                    0,
                )
            ) != 35
            for class_name in CLASS_ORDER
        ):

            raise ValueError(
                f"{region}: expected 35 samples "
                "per class."
            )


    return features


# =============================================================================
# Model factory
# =============================================================================

def make_xgboost(
    num_classes: int,
) -> XGBClassifier:
    """
    Full Fusion XGBoost configuration used in the study.
    """

    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# =============================================================================
# SHAP-format handling
# =============================================================================

def normalise_multiclass_shap(
    shap_values,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    """
    Convert common SHAP multiclass output formats to:

        (samples, features, classes)

    Handles:

        list[class] of (samples, features)

    and arrays shaped as:

        (samples, features, classes)
        (classes, samples, features)
        (samples, classes, features)

    A strict shape check prevents silent class/feature misalignment.
    """

    # -------------------------------------------------------------------------
    # Older SHAP behaviour: one array per class
    # -------------------------------------------------------------------------

    if isinstance(
        shap_values,
        list,
    ):

        if len(shap_values) != n_classes:

            raise ValueError(
                "Unexpected number of SHAP class arrays: "
                f"{len(shap_values)}."
            )


        arrays = [
            np.asarray(
                values
            )
            for values in shap_values
        ]


        for class_index, values in enumerate(
            arrays
        ):

            if values.shape != (
                n_samples,
                n_features,
            ):

                raise ValueError(
                    f"Class {class_index} SHAP array "
                    f"has unexpected shape {values.shape}."
                )


        return np.stack(
            arrays,
            axis=2,
        )


    # -------------------------------------------------------------------------
    # Newer SHAP behaviour: one 3-D array
    # -------------------------------------------------------------------------

    values = np.asarray(
        shap_values
    )


    expected_shapes = {
        (
            n_samples,
            n_features,
            n_classes,
        ):
            "samples_features_classes",

        (
            n_classes,
            n_samples,
            n_features,
        ):
            "classes_samples_features",

        (
            n_samples,
            n_classes,
            n_features,
        ):
            "samples_classes_features",
    }


    if values.shape not in expected_shapes:

        raise ValueError(
            "Unexpected multiclass SHAP shape: "
            f"{values.shape}. "
            f"Expected one of {list(expected_shapes)}."
        )


    orientation = (
        expected_shapes[
            values.shape
        ]
    )


    if orientation == "samples_features_classes":

        return values


    if orientation == "classes_samples_features":

        return np.transpose(
            values,
            (
                1,
                2,
                0,
            ),
        )


    if orientation == "samples_classes_features":

        return np.transpose(
            values,
            (
                0,
                2,
                1,
            ),
        )


    raise RuntimeError(
        "Could not normalise SHAP output."
    )


def compute_shap_values(
    model,
    X: pd.DataFrame,
    n_classes: int,
) -> np.ndarray:
    """
    Compute TreeSHAP values and return a standard
    samples x features x classes array.
    """

    explainer = shap.TreeExplainer(
        model
    )


    raw_values = explainer.shap_values(
        X
    )


    return normalise_multiclass_shap(
        shap_values=raw_values,
        n_samples=len(X),
        n_features=X.shape[1],
        n_classes=n_classes,
    )


# =============================================================================
# PART A — All-sample descriptive SHAP
# =============================================================================

def run_descriptive_shap(
    dataframe: pd.DataFrame,
    features: list[str],
    feature_metadata: pd.DataFrame,
    model_package_path: Path,
    output_dir: Path,
) -> None:
    """
    Interpret the XGBoost Full Fusion model fitted to all 525 polygons.

    These results are descriptive only and are not used for compact
    feature selection.
    """

    print(
        "\n" + "=" * 96
    )

    print(
        "PART A — DESCRIPTIVE ALL-SAMPLE SHAP"
    )

    print(
        "=" * 96
    )


    # =========================================================================
    # Load the final all-sample model generated by script 06
    # =========================================================================

    if not model_package_path.exists():

        raise FileNotFoundError(
            "Full Fusion XGBoost model not found: "
            f"{model_package_path}"
        )


    package = joblib.load(
        model_package_path
    )


    model = package[
        "model"
    ]


    model_features = package[
        "feature_cols"
    ]


    if list(
        model_features
    ) != list(
        features
    ):

        raise ValueError(
            "Feature order in saved Full Fusion model does not "
            "match the model-ready fusion table."
        )


    label_encoder = package.get(
        "label_encoder"
    )


    if label_encoder is None:

        raise ValueError(
            "Saved XGBoost model package does not contain "
            "the label encoder."
        )


    model_classes = list(
        label_encoder.classes_
    )


    if set(
        model_classes
    ) != set(
        CLASS_ORDER
    ):

        raise ValueError(
            f"Unexpected XGBoost class order: "
            f"{model_classes}"
        )


    print(
        "Model:",
        package.get(
            "experiment",
            "FULL_S1S2_FUSION",
        ),
    )

    print(
        "Training samples:",
        len(dataframe),
    )

    print(
        "Predictors:",
        len(features),
    )

    print(
        "XGBoost class order:",
        model_classes,
    )


    X = (
        dataframe[
            features
        ]
        .astype(float)
    )


    # =========================================================================
    # SHAP
    # =========================================================================

    shap_values = compute_shap_values(
        model=model,
        X=X,
        n_classes=len(
            model_classes
        ),
    )


    print(
        "SHAP tensor:",
        shap_values.shape,
    )


    # Shape:
    # samples x features x classes
    abs_values = np.abs(
        shap_values
    )


    # =========================================================================
    # Class-wise feature importance
    # =========================================================================

    class_importance_rows = []


    for class_index, class_name in enumerate(
        model_classes
    ):

        mean_abs = np.mean(
            abs_values[
                :,
                :,
                class_index,
            ],
            axis=0,
        )


        for feature, importance in zip(
            features,
            mean_abs,
        ):

            class_importance_rows.append({
                "class":
                    class_name,

                "feature":
                    feature,

                "mean_abs_shap":
                    float(
                        importance
                    ),
            })


    class_importance = pd.DataFrame(
        class_importance_rows
    )


    class_importance = class_importance.merge(
        feature_metadata,
        on="feature",
        how="left",
    )


    class_importance[
        "rank_within_class"
    ] = (
        class_importance
        .groupby(
            "class"
        )[
            "mean_abs_shap"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )


    # =========================================================================
    # Overall feature importance
    #
    # Manuscript:
    # absolute SHAP averaged across samples for each predictor and class,
    # then across classes.
    # =========================================================================

    overall_importance = (
        class_importance
        .groupby(
            [
                "feature",
                "feature_position",
                "sensor",
                "season",
                "feature_group",
                "feature_identity",
            ],
            as_index=False,
        )[
            "mean_abs_shap"
        ]
        .mean()
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    overall_importance[
        "rank"
    ] = np.arange(
        1,
        len(
            overall_importance
        ) + 1,
    )


    overall_total = float(
        overall_importance[
            "mean_abs_shap"
        ].sum()
    )


    overall_importance[
        "relative_contribution"
    ] = (
        overall_importance[
            "mean_abs_shap"
        ]
        / overall_total
    )


    overall_importance[
        "relative_contribution_pct"
    ] = (
        100.0
        * overall_importance[
            "relative_contribution"
        ]
    )


    # =========================================================================
    # Overall contribution by sensor
    # =========================================================================

    sensor_overall = (
        overall_importance
        .groupby(
            "sensor",
            as_index=False,
        )[
            "mean_abs_shap"
        ]
        .sum()
    )


    sensor_overall[
        "relative_contribution_pct"
    ] = (
        100.0
        * sensor_overall[
            "mean_abs_shap"
        ]
        / sensor_overall[
            "mean_abs_shap"
        ].sum()
    )


    # =========================================================================
    # Contribution by feature group
    # =========================================================================

    feature_group_summary = (
        overall_importance
        .groupby(
            "feature_group",
            as_index=False,
        )[
            "mean_abs_shap"
        ]
        .sum()
    )


    feature_group_summary[
        "relative_contribution_pct"
    ] = (
        100.0
        * feature_group_summary[
            "mean_abs_shap"
        ]
        / feature_group_summary[
            "mean_abs_shap"
        ].sum()
    )


    # =========================================================================
    # Contribution by season
    # =========================================================================

    season_summary = (
        overall_importance
        .groupby(
            "season",
            as_index=False,
        )[
            "mean_abs_shap"
        ]
        .sum()
    )


    season_summary[
        "relative_contribution_pct"
    ] = (
        100.0
        * season_summary[
            "mean_abs_shap"
        ]
        / season_summary[
            "mean_abs_shap"
        ].sum()
    )


    # =========================================================================
    # Class-specific sensor contribution
    # =========================================================================

    class_sensor = (
        class_importance
        .groupby(
            [
                "class",
                "sensor",
            ],
            as_index=False,
        )[
            "mean_abs_shap"
        ]
        .sum()
    )


    class_sensor[
        "relative_contribution_pct"
    ] = (
        class_sensor
        .groupby(
            "class"
        )[
            "mean_abs_shap"
        ]
        .transform(
            lambda values:
                100.0
                * values
                / values.sum()
        )
    )


    # =========================================================================
    # Class-specific feature-group contribution
    # =========================================================================

    class_feature_group = (
        class_importance
        .groupby(
            [
                "class",
                "feature_group",
            ],
            as_index=False,
        )[
            "mean_abs_shap"
        ]
        .sum()
    )


    class_feature_group[
        "relative_contribution_pct"
    ] = (
        class_feature_group
        .groupby(
            "class"
        )[
            "mean_abs_shap"
        ]
        .transform(
            lambda values:
                100.0
                * values
                / values.sum()
        )
    )


    # =========================================================================
    # Top predictors
    # =========================================================================

    top_15_overall = (
        overall_importance
        .head(15)
        .copy()
    )


    top_20_overall = (
        overall_importance
        .head(20)
        .copy()
    )


    top_20_by_class = (
        class_importance[
            class_importance[
                "rank_within_class"
            ] <= 20
        ]
        .sort_values(
            [
                "class",
                "rank_within_class",
            ]
        )
        .copy()
    )


    # =========================================================================
    # Signed SHAP long table
    #
    # This is retained so figure-generation scripts can create the
    # class-specific signed SHAP plots without recalculating SHAP.
    # =========================================================================

    signed_rows = []


    sample_ids = (
        dataframe[
            "unique_id"
        ]
        .astype(str)
        .tolist()
    )


    feature_values = (
        X.to_numpy(
            dtype=float
        )
    )


    for class_index, class_name in enumerate(
        model_classes
    ):

        for sample_index, sample_id in enumerate(
            sample_ids
        ):

            for feature_index, feature in enumerate(
                features
            ):

                signed_rows.append({
                    "unique_id":
                        sample_id,

                    "class":
                        class_name,

                    "feature":
                        feature,

                    "feature_value":
                        float(
                            feature_values[
                                sample_index,
                                feature_index,
                            ]
                        ),

                    "shap_value":
                        float(
                            shap_values[
                                sample_index,
                                feature_index,
                                class_index,
                            ]
                        ),
                })


    signed_shap = pd.DataFrame(
        signed_rows
    )


    signed_shap = signed_shap.merge(
        feature_metadata,
        on="feature",
        how="left",
    )


    # =========================================================================
    # Save Part A outputs
    # =========================================================================

    descriptive_dir = (
        output_dir
        / "descriptive_all_sample"
    )

    descriptive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    outputs = {
        "overall_feature_importance":
            descriptive_dir
            / "shap_overall_feature_importance.csv",

        "class_feature_importance":
            descriptive_dir
            / "shap_class_feature_importance.csv",

        "sensor_overall":
            descriptive_dir
            / "shap_sensor_contribution_overall.csv",

        "sensor_by_class":
            descriptive_dir
            / "shap_sensor_contribution_by_class.csv",

        "feature_group_overall":
            descriptive_dir
            / "shap_feature_group_contribution.csv",

        "feature_group_by_class":
            descriptive_dir
            / "shap_feature_group_contribution_by_class.csv",

        "season_overall":
            descriptive_dir
            / "shap_season_contribution.csv",

        "top_15_overall":
            descriptive_dir
            / "shap_top15_predictors_overall.csv",

        "top_20_overall":
            descriptive_dir
            / "shap_top20_predictors_overall.csv",

        "top_20_by_class":
            descriptive_dir
            / "shap_top20_predictors_by_class.csv",

        "signed_shap":
            descriptive_dir
            / "shap_signed_values_all_samples.csv",
    }


    overall_importance.to_csv(
        outputs[
            "overall_feature_importance"
        ],
        index=False,
    )


    class_importance.to_csv(
        outputs[
            "class_feature_importance"
        ],
        index=False,
    )


    sensor_overall.to_csv(
        outputs[
            "sensor_overall"
        ],
        index=False,
    )


    class_sensor.to_csv(
        outputs[
            "sensor_by_class"
        ],
        index=False,
    )


    feature_group_summary.to_csv(
        outputs[
            "feature_group_overall"
        ],
        index=False,
    )


    class_feature_group.to_csv(
        outputs[
            "feature_group_by_class"
        ],
        index=False,
    )


    season_summary.to_csv(
        outputs[
            "season_overall"
        ],
        index=False,
    )


    top_15_overall.to_csv(
        outputs[
            "top_15_overall"
        ],
        index=False,
    )


    top_20_overall.to_csv(
        outputs[
            "top_20_overall"
        ],
        index=False,
    )


    top_20_by_class.to_csv(
        outputs[
            "top_20_by_class"
        ],
        index=False,
    )


    signed_shap.to_csv(
        outputs[
            "signed_shap"
        ],
        index=False,
    )


    # =========================================================================
    # Console summary
    # =========================================================================

    print(
        "\nOverall sensor contribution:"
    )

    print(
        sensor_overall[
            [
                "sensor",
                "relative_contribution_pct",
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nClass-specific sensor contribution:"
    )

    print(
        class_sensor[
            [
                "class",
                "sensor",
                "relative_contribution_pct",
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nSeason contribution:"
    )

    print(
        season_summary[
            [
                "season",
                "relative_contribution_pct",
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nTop 15 predictors:"
    )

    print(
        top_15_overall[
            [
                "rank",
                "feature",
                "sensor",
                "feature_group",
                "season",
                "relative_contribution_pct",
            ]
        ].to_string(
            index=False
        )
    )


# =============================================================================
# PART B — Fold-wise training-only SHAP
# =============================================================================

def run_foldwise_training_shap(
    dataframe: pd.DataFrame,
    features: list[str],
    feature_metadata: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    """
    Compute fold-wise training-only SHAP and derive the compact subset.
    """

    print(
        "\n" + "=" * 96
    )

    print(
        "PART B — FOLD-WISE TRAINING-ONLY SHAP"
    )

    print(
        "=" * 96
    )


    # =========================================================================
    # Stable label encoding
    # =========================================================================

    label_encoder = LabelEncoder()

    label_encoder.fit(
        CLASS_ORDER
    )


    model_classes = list(
        label_encoder.classes_
    )


    fold_dir = (
        output_dir
        / "foldwise_training_only"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    fold_feature_rows = []
    fold_class_rows = []


    # =========================================================================
    # Five spatial folds
    # =========================================================================

    for fold_id, held_out_region in enumerate(
        REGION_ORDER,
        start=1,
    ):

        print(
            "\n" + "-" * 88
        )

        print(
            f"Fold {fold_id}: hold out {held_out_region}"
        )

        print(
            "-" * 88
        )


        training = dataframe[
            dataframe[
                GROUP_COL
            ].astype(str)
            != held_out_region
        ].copy()


        held_out = dataframe[
            dataframe[
                GROUP_COL
            ].astype(str)
            == held_out_region
        ].copy()


        if len(training) != EXPECTED_TRAIN_PER_FOLD:

            raise ValueError(
                f"{held_out_region}: expected "
                f"{EXPECTED_TRAIN_PER_FOLD} training rows; "
                f"found {len(training)}."
            )


        if len(held_out) != EXPECTED_TEST_PER_FOLD:

            raise ValueError(
                f"{held_out_region}: expected "
                f"{EXPECTED_TEST_PER_FOLD} held-out rows; "
                f"found {len(held_out)}."
            )


        X_train = (
            training[
                features
            ]
            .astype(float)
        )


        y_train = (
            label_encoder.transform(
                training[
                    TARGET_COL
                ].astype(str)
            )
        )


        # =====================================================================
        # Fit Full Fusion model using training regions only
        # =====================================================================

        model = make_xgboost(
            num_classes=len(
                CLASS_ORDER
            )
        )


        model.fit(
            X_train,
            y_train,
        )


        # =====================================================================
        # SHAP calculated ONLY on the training observations
        # =====================================================================

        shap_values = compute_shap_values(
            model=model,
            X=X_train,
            n_classes=len(
                model_classes
            ),
        )


        abs_values = np.abs(
            shap_values
        )


        # =====================================================================
        # Class-wise mean absolute SHAP
        # =====================================================================

        fold_class_importance = []


        for class_index, class_name in enumerate(
            model_classes
        ):

            values = np.mean(
                abs_values[
                    :,
                    :,
                    class_index,
                ],
                axis=0,
            )


            for feature, importance in zip(
                features,
                values,
            ):

                row = {
                    "fold":
                        fold_id,

                    "held_out_region":
                        held_out_region,

                    "class":
                        class_name,

                    "feature":
                        feature,

                    "mean_abs_shap":
                        float(
                            importance
                        ),
                }


                fold_class_rows.append(
                    row
                )


                fold_class_importance.append(
                    row
                )


        fold_class_df = pd.DataFrame(
            fold_class_importance
        )


        fold_class_df = fold_class_df.merge(
            feature_metadata,
            on="feature",
            how="left",
        )


        # =====================================================================
        # Average across classes within this fold
        # =====================================================================

        fold_feature = (
            fold_class_df
            .groupby(
                [
                    "fold",
                    "held_out_region",
                    "feature",
                    "feature_position",
                    "sensor",
                    "season",
                    "feature_group",
                    "feature_identity",
                ],
                as_index=False,
            )[
                "mean_abs_shap"
            ]
            .mean()
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )


        fold_feature[
            "rank"
        ] = np.arange(
            1,
            len(
                fold_feature
            ) + 1,
        )


        fold_total = float(
            fold_feature[
                "mean_abs_shap"
            ].sum()
        )


        fold_feature[
            "relative_contribution"
        ] = (
            fold_feature[
                "mean_abs_shap"
            ]
            / fold_total
        )


        fold_feature[
            "cumulative_contribution"
        ] = (
            fold_feature[
                "relative_contribution"
            ]
            .cumsum()
        )


        fold_feature[
            "appears_in_top20"
        ] = (
            fold_feature[
                "rank"
            ] <= 20
        )


        fold_feature_rows.extend(
            fold_feature.to_dict(
                orient="records"
            )
        )


        # =====================================================================
        # Save individual fold tables
        # =====================================================================

        fold_feature.to_csv(
            fold_dir
            / (
                f"fold_{fold_id}_"
                f"holdout_{held_out_region}_"
                f"feature_importance.csv"
            ),
            index=False,
        )


        fold_class_df.to_csv(
            fold_dir
            / (
                f"fold_{fold_id}_"
                f"holdout_{held_out_region}_"
                f"class_feature_importance.csv"
            ),
            index=False,
        )


        print(
            "Training observations:",
            len(
                training
            ),
        )

        print(
            "Held-out observations excluded from SHAP:",
            len(
                held_out
            ),
        )

        print(
            "Top predictor:",
            fold_feature.iloc[
                0
            ][
                "feature"
            ],
        )


    # =========================================================================
    # Aggregate all fold outputs
    # =========================================================================

    fold_feature_df = pd.DataFrame(
        fold_feature_rows
    )


    fold_class_df = pd.DataFrame(
        fold_class_rows
    )


    fold_class_df = fold_class_df.merge(
        feature_metadata,
        on="feature",
        how="left",
    )


    # =========================================================================
    # Manuscript aggregation:
    #
    # fold-specific mean absolute SHAP values were averaged by feature
    # and class, then across classes and folds.
    #
    # Here we preserve both intermediate levels.
    # =========================================================================

    class_crossfold = (
        fold_class_df
        .groupby(
            [
                "feature",
                "feature_position",
                "sensor",
                "season",
                "feature_group",
                "feature_identity",
                "class",
            ],
            as_index=False,
        )
        .agg(
            mean_abs_shap=(
                "mean_abs_shap",
                "mean",
            ),

            shap_std_across_folds=(
                "mean_abs_shap",
                "std",
            ),
        )
    )


    # Average across classes after fold aggregation.
    crossfold_importance = (
        class_crossfold
        .groupby(
            [
                "feature",
                "feature_position",
                "sensor",
                "season",
                "feature_group",
                "feature_identity",
            ],
            as_index=False,
        )[
            "mean_abs_shap"
        ]
        .mean()
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    # Add rank-stability diagnostics.
    fold_rank_summary = (
        fold_feature_df
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            mean_fold_rank=(
                "rank",
                "mean",
            ),

            min_fold_rank=(
                "rank",
                "min",
            ),

            max_fold_rank=(
                "rank",
                "max",
            ),

            folds_in_top20=(
                "appears_in_top20",
                "sum",
            ),

            fold_mean_abs_shap_std=(
                "mean_abs_shap",
                "std",
            ),
        )
    )


    crossfold_importance = (
        crossfold_importance
        .merge(
            fold_rank_summary,
            on="feature",
            how="left",
        )
    )


    crossfold_importance[
        "rank"
    ] = np.arange(
        1,
        len(
            crossfold_importance
        ) + 1,
    )


    total_importance = float(
        crossfold_importance[
            "mean_abs_shap"
        ].sum()
    )


    crossfold_importance[
        "relative_contribution"
    ] = (
        crossfold_importance[
            "mean_abs_shap"
        ]
        / total_importance
    )


    crossfold_importance[
        "relative_contribution_pct"
    ] = (
        100.0
        * crossfold_importance[
            "relative_contribution"
        ]
    )


    crossfold_importance[
        "cumulative_contribution"
    ] = (
        crossfold_importance[
            "relative_contribution"
        ]
        .cumsum()
    )


    crossfold_importance[
        "cumulative_contribution_pct"
    ] = (
        100.0
        * crossfold_importance[
            "cumulative_contribution"
        ]
    )


    # =========================================================================
    # >=80% cumulative rule
    #
    # Smallest prefix for which cumulative attribution reaches threshold.
    # =========================================================================

    threshold_hits = np.flatnonzero(
        crossfold_importance[
            "cumulative_contribution"
        ].to_numpy()
        >= CUMULATIVE_THRESHOLD
    )


    if len(
        threshold_hits
    ) == 0:

        raise RuntimeError(
            "Cumulative SHAP importance never reached "
            f"{CUMULATIVE_THRESHOLD:.0%}."
        )


    selected_count = int(
        threshold_hits[
            0
        ]
        + 1
    )


    compact = (
        crossfold_importance
        .iloc[
            :selected_count
        ]
        .copy()
    )


    compact[
        "selected"
    ] = True


    compact[
        "selection_rule"
    ] = (
        "smallest subset reaching >=80% cumulative "
        "fold-wise training-only mean absolute SHAP"
    )


    compact_features = (
        compact[
            "feature"
        ]
        .tolist()
    )


    # =========================================================================
    # Compact-set composition
    # =========================================================================

    compact_sensor = (
        compact[
            "sensor"
        ]
        .value_counts()
        .rename_axis(
            "sensor"
        )
        .reset_index(
            name="n_features"
        )
    )


    compact_season = (
        compact[
            "season"
        ]
        .value_counts()
        .rename_axis(
            "season"
        )
        .reset_index(
            name="n_features"
        )
    )


    compact_group = (
        compact[
            "feature_group"
        ]
        .value_counts()
        .rename_axis(
            "feature_group"
        )
        .reset_index(
            name="n_features"
        )
    )


    # =========================================================================
    # Save aggregated fold-wise outputs
    # =========================================================================

    fold_feature_df.to_csv(
        fold_dir
        / "foldwise_feature_importance_all_folds.csv",
        index=False,
    )


    fold_class_df.to_csv(
        fold_dir
        / "foldwise_class_feature_importance_all_folds.csv",
        index=False,
    )


    class_crossfold.to_csv(
        fold_dir
        / "crossfold_class_feature_importance.csv",
        index=False,
    )


    crossfold_importance.to_csv(
        fold_dir
        / "crossfold_feature_ranking.csv",
        index=False,
    )


    compact.to_csv(
        fold_dir
        / "compact_feature_selection.csv",
        index=False,
    )


    compact_sensor.to_csv(
        fold_dir
        / "compact_feature_sensor_composition.csv",
        index=False,
    )


    compact_season.to_csv(
        fold_dir
        / "compact_feature_season_composition.csv",
        index=False,
    )


    compact_group.to_csv(
        fold_dir
        / "compact_feature_group_composition.csv",
        index=False,
    )


    # =========================================================================
    # Save machine-readable compact feature list
    # =========================================================================

    compact_json = {
        "selection_method":
            (
                "fold-wise training-only mean absolute SHAP; "
                "averaged by feature and class, then across "
                "classes and folds"
            ),

        "cumulative_threshold":
            CUMULATIVE_THRESHOLD,

        "n_full_features":
            EXPECTED_FEATURES,

        "n_selected_features":
            selected_count,

        "feature_reduction_pct":
            100.0
            * (
                1.0
                - selected_count
                / EXPECTED_FEATURES
            ),

        "selected_features":
            compact_features,
    }


    with open(
        fold_dir
        / "compact_feature_selection.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            compact_json,
            file,
            indent=2,
        )


    # =========================================================================
    # Console summary
    # =========================================================================

    print(
        "\n" + "=" * 96
    )

    print(
        "COMPACT FEATURE SELECTION"
    )

    print(
        "=" * 96
    )


    print(
        "Full feature space:",
        EXPECTED_FEATURES,
    )

    print(
        "Cumulative threshold:",
        f"{CUMULATIVE_THRESHOLD:.0%}",
    )

    print(
        "Selected predictors:",
        selected_count,
    )

    print(
        "Feature reduction:",
        f"{100.0 * (1 - selected_count / EXPECTED_FEATURES):.1f}%",
    )


    print(
        "\nSensor composition:"
    )

    print(
        compact_sensor.to_string(
            index=False
        )
    )


    print(
        "\nSeason composition:"
    )

    print(
        compact_season.to_string(
            index=False
        )
    )


    print(
        "\nSelected predictors:"
    )

    print(
        compact[
            [
                "rank",
                "feature",
                "sensor",
                "season",
                "feature_group",
                "mean_abs_shap",
                "relative_contribution_pct",
                "cumulative_contribution_pct",
            ]
        ].to_string(
            index=False
        )
    )


    return compact_features


# =============================================================================
# Save feature metadata
# =============================================================================

def save_feature_metadata(
    metadata: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save the complete 176-feature inventory used by both SHAP analyses.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    metadata.to_csv(
        output_dir
        / "full_fusion_feature_metadata.csv",
        index=False,
    )


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


    default_samples = (
        repo_root
        / "data"
        / "reference_samples"
        / "model_ready"
        / "reference_samples_2025_2026_S1S2_MODEL_READY.csv"
    )


    default_model = (
        repo_root
        / "models"
        / "loro_ablation"
        / "XGBOOST_FULL_S1S2_FUSION_FINAL_ALL_SAMPLES.joblib"
    )


    default_output = (
        repo_root
        / "results"
        / "shap_analysis"
    )


    parser = argparse.ArgumentParser(
        description=(
            "Run descriptive and fold-wise training-only "
            "SHAP analyses for ECHO Full Fusion XGBoost."
        )
    )


    parser.add_argument(
        "--samples",
        type=Path,
        default=default_samples,
        help=(
            "2025-2026 model-ready 176-feature "
            "fusion reference-sample CSV."
        ),
    )


    parser.add_argument(
        "--full-model",
        type=Path,
        default=default_model,
        help=(
            "Final all-sample XGBoost Full Fusion "
            "model generated by script 06."
        ),
    )


    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=(
            "Directory for SHAP tables and "
            "compact-feature selection."
        ),
    )


    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Run both SHAP workflows.
    """

    args = parse_args()


    samples_path = (
        args.samples
        .expanduser()
        .resolve()
    )


    full_model_path = (
        args.full_model
        .expanduser()
        .resolve()
    )


    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )


    if not samples_path.exists():

        raise FileNotFoundError(
            f"Fusion sample table not found: "
            f"{samples_path}"
        )


    dataframe = pd.read_csv(
        samples_path
    )


    print(
        "ECHO SHAP feature analysis"
    )

    print(
        "Samples:",
        samples_path,
    )

    print(
        "Shape:",
        dataframe.shape,
    )


    # =========================================================================
    # Input and feature audit
    # =========================================================================

    features = audit_samples(
        dataframe
    )


    metadata = build_feature_metadata(
        features
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    save_feature_metadata(
        metadata,
        output_dir,
    )


    print(
        "Input audit: PASS"
    )

    print(
        "Predictors:",
        len(features),
    )


    print(
        "\nFeature groups:"
    )

    print(
        metadata[
            "feature_group"
        ]
        .value_counts()
        .to_string()
    )


    # =========================================================================
    # PART A
    # =========================================================================

    run_descriptive_shap(
        dataframe=dataframe,
        features=features,
        feature_metadata=metadata,
        model_package_path=full_model_path,
        output_dir=output_dir,
    )


    # =========================================================================
    # PART B
    # =========================================================================

    compact_features = (
        run_foldwise_training_shap(
            dataframe=dataframe,
            features=features,
            feature_metadata=metadata,
            output_dir=output_dir,
        )
    )


    # =========================================================================
    # Final integrity check
    # =========================================================================

    if not compact_features:

        raise RuntimeError(
            "Compact feature selection returned no predictors."
        )


    print(
        "\n" + "=" * 96
    )

    print(
        "SHAP ANALYSIS COMPLETE"
    )

    print(
        "=" * 96
    )


    print(
        "Descriptive SHAP:"
    )

    print(
        output_dir
        / "descriptive_all_sample"
    )


    print(
        "\nTraining-only selection:"
    )

    print(
        output_dir
        / "foldwise_training_only"
    )


    print(
        "\nSelected predictors:",
        len(
            compact_features
        ),
    )


if __name__ == "__main__":
    main()