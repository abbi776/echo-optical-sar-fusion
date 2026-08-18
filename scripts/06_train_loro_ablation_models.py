#!/usr/bin/env python3
"""
Run the baseline Leave-One-Region-Out (LORO) modelling experiments
for the ECHO optical-SAR fusion workflow.

Eight baseline feature configurations are evaluated:

    1. S2 bands only                  40 predictors
    2. S2 indices only                32 predictors
    3. S1 backscatter only             8 predictors
    4. S1 polarisation features       40 predictors
    5. S1 texture only                56 predictors
    6. Full S2                        72 predictors
    7. Full S1                       104 predictors
    8. Full S1-S2 fusion             176 predictors

Each configuration is evaluated with:

    Random Forest
    XGBoost

Validation uses five-fold leave-one-region-out cross-validation:

    SR1
    SR2
    SR3
    SR4
    SR5

Each fold contains:

    420 training polygons
    105 held-out polygons
    35 polygons per class in the held-out region

The same samples, folds and fixed model hyperparameters are used for
every feature configuration.

Outputs include:
    - pooled LORO metrics
    - fold-level metrics
    - pooled and fold confusion matrices
    - class-specific precision, recall and F1
    - held-out predictions and probabilities
    - feature importances
    - feature-group inventory
    - experiment configuration
    - final models fitted to all 525 polygons

No figures are produced here. Figure-generation code is kept separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


# =============================================================================
# Reproducibility and experimental design
# =============================================================================

RANDOM_STATE = 42

EXPECTED_ROWS = 525

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

EXPECTED_CLASS_COUNT = 175
EXPECTED_REGION_COUNT = 105
EXPECTED_CLASS_REGION_COUNT = 35

EXPECTED_TRAIN_PER_FOLD = 420
EXPECTED_TEST_PER_FOLD = 105


# =============================================================================
# Feature definitions
# =============================================================================

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

S1_BACKSCATTER = {
    "VV_DB",
    "VH_DB",
}

# Both the cleaned descriptive names and the legacy names from the
# original analysis notebooks are recognised.
S1_POLARISATION = {
    "NDPI",
    "NRPB",

    # Cleaned repository names
    "VV_VH_RATIO",
    "VH_VV_RATIO",
    "VDDI",

    # Legacy notebook names
    "PR",
    "XPR",
    "VDDPI",

    "RVI",
    "SUM",
    "DIFF",
    "PROD",
    "LOG_RATIO",
}


EXPECTED_FEATURE_COUNTS = {
    "S2_BANDS_ONLY": 40,
    "S2_INDICES_ONLY": 32,
    "S1_BACKSCATTER_ONLY": 8,
    "S1_POLARISATION_FEATURES": 40,
    "S1_TEXTURE_ONLY": 56,
    "FULL_S2": 72,
    "FULL_S1": 104,
    "FULL_S1S2_FUSION": 176,
}


# =============================================================================
# Metadata columns excluded from modelling
# =============================================================================

METADATA_COLUMNS = {
    # identifiers
    "unique_id",
    "sample_id",
    "roi_id",
    "polygon_id",
    "label_id",
    "OBJECTID",
    "objectid",

    # class information
    "class_raw",
    "class_label",
    "class_id",
    "class_name",
    "class",
    "classname",
    "classvalue",

    # grouping/source information
    "region",
    "site",
    "period",
    "year",
    "stack",
    "source_file",
    "source_stack",
    "sensor",

    # spatial metadata
    "geometry",
    "x",
    "y",
    "centroid_x",
    "centroid_y",
    "area",
    "area_m2",

    # extraction/QC fields
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
# General helpers
# =============================================================================

def normalise_name(name: str) -> str:
    """Return an uppercase underscore-normalised predictor name."""

    value = str(name).strip().upper()

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
        value = value.replace(character, "_")

    while "__" in value:
        value = value.replace("__", "_")

    return value.strip("_")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return numeric predictor columns while excluding labels,
    identifiers and QC fields.
    """

    numeric_columns = (
        df.select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    return [
        column
        for column in numeric_columns
        if column not in METADATA_COLUMNS
    ]


def feature_suffix(
    column: str,
    sensor: str,
) -> str | None:
    """
    Extract the feature identity after the sensor token.

    Examples
    --------
    WINTER_2025_S1_VV_dB
        -> VV_DB

    SPRING_2025_S1_VV_VH_RATIO
        -> VV_VH_RATIO

    SUMMER_2026_S2_NDVI
        -> NDVI
    """

    value = normalise_name(
        column
    )

    token = f"_{sensor.upper()}_"

    if token not in value:
        return None

    return value.split(
        token,
        1,
    )[1]


# =============================================================================
# Legacy generic-band support
# =============================================================================

def generic_s1_band_number(
    column: str,
) -> int | None:
    """
    Recognise legacy S1_BAND_XX predictor names.
    """

    import re

    match = re.search(
        r"S1_BAND_(\d+)",
        normalise_name(column),
    )

    if match:
        return int(
            match.group(1)
        )

    return None


def generic_s2_band_number(
    column: str,
) -> int | None:
    """
    Recognise legacy S2_BAND_XX predictor names.
    """

    import re

    match = re.search(
        r"S2_BAND_(\d+)",
        normalise_name(column),
    )

    if match:
        return int(
            match.group(1)
        )

    return None


# =============================================================================
# Feature-group classification
# =============================================================================

def is_s2_band(
    column: str,
) -> bool:
    """Return True for one of the ten Sentinel-2 reflectance bands."""

    band_number = generic_s2_band_number(
        column
    )

    if band_number is not None:
        return 1 <= band_number <= 10

    suffix = feature_suffix(
        column,
        "S2",
    )

    return suffix in S2_BANDS


def is_s2_index(
    column: str,
) -> bool:
    """Return True for one of the eight Sentinel-2 indices."""

    band_number = generic_s2_band_number(
        column
    )

    if band_number is not None:
        return 11 <= band_number <= 18

    suffix = feature_suffix(
        column,
        "S2",
    )

    return suffix in S2_INDICES


def is_s1_backscatter(
    column: str,
) -> bool:
    """Return True for VV or VH gamma0 backscatter."""

    band_number = generic_s1_band_number(
        column
    )

    if band_number is not None:
        return 1 <= band_number <= 2

    suffix = feature_suffix(
        column,
        "S1",
    )

    return suffix in S1_BACKSCATTER


def is_s1_polarisation(
    column: str,
) -> bool:
    """
    Return True for one of the ten Sentinel-1
    polarisation/algebraic predictors.
    """

    band_number = generic_s1_band_number(
        column
    )

    if band_number is not None:
        return 3 <= band_number <= 12

    suffix = feature_suffix(
        column,
        "S1",
    )

    return suffix in S1_POLARISATION


def is_s1_texture(
    column: str,
) -> bool:
    """Return True for one of the fourteen Sentinel-1 GLCM predictors."""

    band_number = generic_s1_band_number(
        column
    )

    if band_number is not None:
        return 13 <= band_number <= 26

    suffix = feature_suffix(
        column,
        "S1",
    )

    if suffix is None:
        return False

    return suffix.startswith(
        "GLCM_"
    )


# =============================================================================
# Input QC
# =============================================================================

def audit_reference_table(
    df: pd.DataFrame,
    name: str,
    expected_features: int,
) -> list[str]:
    """
    Strictly audit one model-ready reference-sample table.
    """

    required = {
        "unique_id",
        TARGET_COL,
        GROUP_COL,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{name}: missing required column(s): "
            + ", ".join(sorted(missing))
        )


    predictors = feature_columns(
        df
    )


    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"{name}: expected {EXPECTED_ROWS} rows; "
            f"found {len(df)}."
        )


    if len(predictors) != expected_features:
        raise ValueError(
            f"{name}: expected {expected_features} predictors; "
            f"found {len(predictors)}."
        )


    if df["unique_id"].duplicated().any():
        raise ValueError(
            f"{name}: duplicate polygon IDs detected."
        )


    if df[predictors].isna().any().any():
        raise ValueError(
            f"{name}: NaN predictor values detected."
        )


    if not np.isfinite(
        df[predictors].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            f"{name}: non-finite predictor values detected."
        )


    observed_classes = set(
        df[TARGET_COL]
        .astype(str)
        .unique()
    )

    if observed_classes != set(CLASS_ORDER):
        raise ValueError(
            f"{name}: unexpected classes. "
            f"Observed: {sorted(observed_classes)}"
        )


    observed_regions = set(
        df[GROUP_COL]
        .astype(str)
        .unique()
    )

    if observed_regions != set(REGION_ORDER):
        raise ValueError(
            f"{name}: unexpected regions. "
            f"Observed: {sorted(observed_regions)}"
        )


    class_counts = (
        df[TARGET_COL]
        .astype(str)
        .value_counts()
    )

    for class_name in CLASS_ORDER:

        if int(
            class_counts.get(
                class_name,
                0,
            )
        ) != EXPECTED_CLASS_COUNT:

            raise ValueError(
                f"{name}: class {class_name} does not "
                f"contain {EXPECTED_CLASS_COUNT} polygons."
            )


    region_counts = (
        df[GROUP_COL]
        .astype(str)
        .value_counts()
    )

    for region in REGION_ORDER:

        if int(
            region_counts.get(
                region,
                0,
            )
        ) != EXPECTED_REGION_COUNT:

            raise ValueError(
                f"{name}: {region} does not contain "
                f"{EXPECTED_REGION_COUNT} polygons."
            )


    cross_table = pd.crosstab(
        df[GROUP_COL].astype(str),
        df[TARGET_COL].astype(str),
    )


    for region in REGION_ORDER:

        for class_name in CLASS_ORDER:

            if int(
                cross_table.loc[
                    region,
                    class_name,
                ]
            ) != EXPECTED_CLASS_REGION_COUNT:

                raise ValueError(
                    f"{name}: {region}/{class_name} "
                    f"does not contain "
                    f"{EXPECTED_CLASS_REGION_COUNT} polygons."
                )


    return predictors


def audit_identical_samples(
    tables: dict[str, pd.DataFrame],
) -> None:
    """
    Confirm S1, S2 and fusion tables contain exactly the same
    polygons, labels and region assignments.
    """

    reference_name = "S1"

    reference = (
        tables[reference_name][
            [
                "unique_id",
                TARGET_COL,
                GROUP_COL,
            ]
        ]
        .sort_values("unique_id")
        .reset_index(drop=True)
    )


    for name, dataframe in tables.items():

        current = (
            dataframe[
                [
                    "unique_id",
                    TARGET_COL,
                    GROUP_COL,
                ]
            ]
            .sort_values("unique_id")
            .reset_index(drop=True)
        )


        if not reference.equals(
            current
        ):
            raise ValueError(
                f"{name}: polygon IDs, classes or regions "
                "do not exactly match the S1 reference table."
            )


# =============================================================================
# Feature configuration
# =============================================================================

def build_feature_groups(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_fusion: pd.DataFrame,
    s1_features: list[str],
    s2_features: list[str],
    fusion_features: list[str],
) -> dict:
    """
    Construct the eight baseline configurations reported in the study.
    """

    groups = {
        "S2_BANDS_ONLY": {
            "dataset": "S2",
            "dataframe": df_s2,
            "features": [
                column
                for column in s2_features
                if is_s2_band(column)
            ],
        },

        "S2_INDICES_ONLY": {
            "dataset": "S2",
            "dataframe": df_s2,
            "features": [
                column
                for column in s2_features
                if is_s2_index(column)
            ],
        },

        "S1_BACKSCATTER_ONLY": {
            "dataset": "S1",
            "dataframe": df_s1,
            "features": [
                column
                for column in s1_features
                if is_s1_backscatter(column)
            ],
        },

        "S1_POLARISATION_FEATURES": {
            "dataset": "S1",
            "dataframe": df_s1,
            "features": [
                column
                for column in s1_features
                if is_s1_polarisation(column)
            ],
        },

        "S1_TEXTURE_ONLY": {
            "dataset": "S1",
            "dataframe": df_s1,
            "features": [
                column
                for column in s1_features
                if is_s1_texture(column)
            ],
        },

        "FULL_S2": {
            "dataset": "S2",
            "dataframe": df_s2,
            "features": list(
                s2_features
            ),
        },

        "FULL_S1": {
            "dataset": "S1",
            "dataframe": df_s1,
            "features": list(
                s1_features
            ),
        },

        "FULL_S1S2_FUSION": {
            "dataset": "S1S2",
            "dataframe": df_fusion,
            "features": list(
                fusion_features
            ),
        },
    }


    # -------------------------------------------------------------------------
    # Strict feature-count audit
    # -------------------------------------------------------------------------

    for experiment_name, configuration in groups.items():

        observed = len(
            configuration["features"]
        )

        expected = EXPECTED_FEATURE_COUNTS[
            experiment_name
        ]


        if observed != expected:

            raise ValueError(
                f"{experiment_name}: expected {expected} predictors, "
                f"found {observed}.\n"
                "Check predictor names and feature-group parsing."
            )


    return groups


# =============================================================================
# Model factories
# =============================================================================

def make_random_forest() -> RandomForestClassifier:
    """
    Random Forest configuration used for all experiments.
    """

    return RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_xgboost(
    num_classes: int,
) -> XGBClassifier:
    """
    XGBoost configuration used for all experiments.
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


def make_model(
    algorithm: str,
    num_classes: int,
):
    """Create a fresh classifier for one experiment/fold."""

    if algorithm == "RF":
        return make_random_forest()

    if algorithm == "XGBOOST":
        return make_xgboost(
            num_classes
        )

    raise ValueError(
        f"Unknown algorithm: {algorithm}"
    )


MODEL_CONFIGS = {
    "RF": {
        "algorithm_name": "Random Forest",
        "encoded_labels": False,
    },

    "XGBOOST": {
        "algorithm_name": "XGBoost",
        "encoded_labels": True,
    },
}


# =============================================================================
# Metrics
# =============================================================================

def calculate_metrics(
    y_true,
    y_pred,
) -> dict:
    """Calculate pooled or fold-level classification metrics."""

    return {
        "overall_accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
            ),

        "weighted_f1":
            f1_score(
                y_true,
                y_pred,
                average="weighted",
            ),

        "cohens_kappa":
            cohen_kappa_score(
                y_true,
                y_pred,
            ),
    }


# =============================================================================
# Experiment runner
# =============================================================================

def run_experiments(
    feature_groups: dict,
    output_dir: Path,
    model_dir: Path,
    save_fold_models: bool,
    source_files: dict[str, Path],
) -> None:
    """
    Run all eight configurations with RF and XGBoost under five-fold LORO.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -------------------------------------------------------------------------
    # XGBoost label encoding
    #
    # LabelEncoder behaviour is retained from the original analysis notebook.
    # Its fitted class order is stored with every XGBoost model package.
    # -------------------------------------------------------------------------

    label_encoder = LabelEncoder()

    label_encoder.fit(
        CLASS_ORDER
    )


    fold_results = []
    pooled_results = []
    confusion_rows = []
    report_rows = []
    feature_importance_rows = []
    prediction_rows = []
    saved_model_rows = []
    feature_inventory_rows = []


    # =========================================================================
    # Eight feature configurations
    # =========================================================================

    for experiment_id, (
        experiment_name,
        configuration,
    ) in enumerate(
        feature_groups.items(),
        start=1,
    ):


        dataframe = configuration[
            "dataframe"
        ].copy()

        dataset_name = configuration[
            "dataset"
        ]

        predictors = configuration[
            "features"
        ]

        n_features = len(
            predictors
        )


        # ---------------------------------------------------------------------
        # Record exact feature inventory
        # ---------------------------------------------------------------------

        for feature_position, feature in enumerate(
            predictors,
            start=1,
        ):

            feature_inventory_rows.append({
                "experiment_id": experiment_id,
                "experiment": experiment_name,
                "dataset": dataset_name,
                "feature_position": feature_position,
                "feature": feature,
            })


        # =====================================================================
        # Two classifiers
        # =====================================================================

        for (
            algorithm,
            algorithm_configuration,
        ) in MODEL_CONFIGS.items():


            algorithm_name = algorithm_configuration[
                "algorithm_name"
            ]

            encoded_labels = algorithm_configuration[
                "encoded_labels"
            ]


            print(
                "\n" + "=" * 96
            )

            print(
                f"Experiment {experiment_id}: "
                f"{experiment_name}"
            )

            print(
                f"Classifier: {algorithm_name}"
            )

            print(
                f"Predictors: {n_features}"
            )

            print(
                "=" * 96
            )


            pooled_true = []
            pooled_predicted = []


            # =================================================================
            # Five LORO folds
            # =================================================================

            for fold_id, held_out_region in enumerate(
                REGION_ORDER,
                start=1,
            ):


                train_df = dataframe[
                    dataframe[GROUP_COL].astype(str)
                    != held_out_region
                ].copy()


                test_df = dataframe[
                    dataframe[GROUP_COL].astype(str)
                    == held_out_region
                ].copy()


                if len(train_df) != EXPECTED_TRAIN_PER_FOLD:

                    raise ValueError(
                        f"{experiment_name}/{held_out_region}: "
                        f"expected {EXPECTED_TRAIN_PER_FOLD} "
                        f"training samples; found {len(train_df)}."
                    )


                if len(test_df) != EXPECTED_TEST_PER_FOLD:

                    raise ValueError(
                        f"{experiment_name}/{held_out_region}: "
                        f"expected {EXPECTED_TEST_PER_FOLD} "
                        f"test samples; found {len(test_df)}."
                    )


                test_class_counts = (
                    test_df[TARGET_COL]
                    .astype(str)
                    .value_counts()
                )


                if any(
                    int(
                        test_class_counts.get(
                            class_name,
                            0,
                        )
                    )
                    != EXPECTED_CLASS_REGION_COUNT
                    for class_name in CLASS_ORDER
                ):

                    raise ValueError(
                        f"{held_out_region}: held-out class "
                        "composition is not 35/35/35."
                    )


                print(
                    f"Fold {fold_id}: "
                    f"hold out {held_out_region} "
                    f"(train={len(train_df)}, "
                    f"test={len(test_df)})"
                )


                X_train = (
                    train_df[predictors]
                    .astype(float)
                )

                X_test = (
                    test_df[predictors]
                    .astype(float)
                )

                y_train = (
                    train_df[TARGET_COL]
                    .astype(str)
                )

                y_test = (
                    test_df[TARGET_COL]
                    .astype(str)
                )


                model = make_model(
                    algorithm=algorithm,
                    num_classes=len(
                        CLASS_ORDER
                    ),
                )


                # -------------------------------------------------------------
                # Fit and predict
                # -------------------------------------------------------------

                if encoded_labels:

                    y_train_encoded = (
                        label_encoder.transform(
                            y_train
                        )
                    )

                    model.fit(
                        X_train,
                        y_train_encoded,
                    )


                    prediction_encoded = (
                        model.predict(
                            X_test
                        )
                        .astype(int)
                    )


                    y_pred = (
                        label_encoder.inverse_transform(
                            prediction_encoded
                        )
                    )


                    probabilities = (
                        model.predict_proba(
                            X_test
                        )
                    )

                    probability_classes = (
                        label_encoder.classes_
                    )


                else:

                    model.fit(
                        X_train,
                        y_train,
                    )


                    y_pred = model.predict(
                        X_test
                    )


                    probabilities = (
                        model.predict_proba(
                            X_test
                        )
                    )

                    probability_classes = (
                        model.classes_
                    )


                # -------------------------------------------------------------
                # Fold metrics
                # -------------------------------------------------------------

                metrics = calculate_metrics(
                    y_test,
                    y_pred,
                )


                fold_results.append({
                    "experiment_id": experiment_id,
                    "experiment": experiment_name,
                    "algorithm": algorithm,
                    "algorithm_name": algorithm_name,
                    "dataset": dataset_name,
                    "fold": fold_id,
                    "held_out_region": held_out_region,
                    "n_features": n_features,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    **metrics,
                })


                # -------------------------------------------------------------
                # Fold confusion matrix
                # -------------------------------------------------------------

                cm = confusion_matrix(
                    y_test,
                    y_pred,
                    labels=CLASS_ORDER,
                )


                for true_index, true_label in enumerate(
                    CLASS_ORDER
                ):

                    for predicted_index, predicted_label in enumerate(
                        CLASS_ORDER
                    ):

                        confusion_rows.append({
                            "experiment_id": experiment_id,
                            "experiment": experiment_name,
                            "algorithm": algorithm,
                            "algorithm_name": algorithm_name,
                            "dataset": dataset_name,
                            "fold": fold_id,
                            "held_out_region": held_out_region,
                            "n_features": n_features,
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "count": int(
                                cm[
                                    true_index,
                                    predicted_index,
                                ]
                            ),
                        })


                # -------------------------------------------------------------
                # Class-specific fold metrics
                # -------------------------------------------------------------

                report = classification_report(
                    y_test,
                    y_pred,
                    labels=CLASS_ORDER,
                    output_dict=True,
                    zero_division=0,
                )


                for label in CLASS_ORDER:

                    values = report[
                        label
                    ]

                    report_rows.append({
                        "experiment_id": experiment_id,
                        "experiment": experiment_name,
                        "algorithm": algorithm,
                        "algorithm_name": algorithm_name,
                        "dataset": dataset_name,
                        "fold": fold_id,
                        "held_out_region": held_out_region,
                        "n_features": n_features,
                        "label": label,
                        "precision": values[
                            "precision"
                        ],
                        "recall": values[
                            "recall"
                        ],
                        "f1_score": values[
                            "f1-score"
                        ],
                        "support": values[
                            "support"
                        ],
                    })


                # -------------------------------------------------------------
                # Tree feature importance
                # -------------------------------------------------------------

                if hasattr(
                    model,
                    "feature_importances_",
                ):

                    for feature, importance in zip(
                        predictors,
                        model.feature_importances_,
                    ):

                        feature_importance_rows.append({
                            "experiment_id": experiment_id,
                            "experiment": experiment_name,
                            "algorithm": algorithm,
                            "algorithm_name": algorithm_name,
                            "dataset": dataset_name,
                            "fold": fold_id,
                            "held_out_region": held_out_region,
                            "feature": feature,
                            "importance": float(
                                importance
                            ),
                        })


                # -------------------------------------------------------------
                # Held-out predictions
                # -------------------------------------------------------------

                for row_position, (
                    row_index,
                    sample,
                ) in enumerate(
                    test_df.iterrows()
                ):


                    prediction_row = {
                        "experiment_id": experiment_id,
                        "experiment": experiment_name,
                        "algorithm": algorithm,
                        "algorithm_name": algorithm_name,
                        "dataset": dataset_name,
                        "fold": fold_id,
                        "held_out_region": held_out_region,
                        "n_features": n_features,
                        "unique_id": sample[
                            "unique_id"
                        ],
                        "region": sample[
                            GROUP_COL
                        ],
                        "true_label": y_test.loc[
                            row_index
                        ],
                        "predicted_label": y_pred[
                            row_position
                        ],
                        "correct": bool(
                            y_test.loc[row_index]
                            == y_pred[row_position]
                        ),
                    }


                    if "class_id" in test_df.columns:

                        prediction_row[
                            "class_id"
                        ] = sample[
                            "class_id"
                        ]


                    for class_index, class_name in enumerate(
                        probability_classes
                    ):

                        prediction_row[
                            f"prob_{class_name}"
                        ] = float(
                            probabilities[
                                row_position,
                                class_index,
                            ]
                        )


                    prediction_rows.append(
                        prediction_row
                    )


                # -------------------------------------------------------------
                # Optional fold-model persistence
                # -------------------------------------------------------------

                if save_fold_models:

                    model_path = (
                        model_dir
                        / (
                            f"{algorithm}_"
                            f"{experiment_name}_"
                            f"LORO_HOLDOUT_"
                            f"{held_out_region}.joblib"
                        )
                    )


                    package = {
                        "model": model,
                        "algorithm": algorithm,
                        "algorithm_name": algorithm_name,
                        "experiment_id": experiment_id,
                        "experiment": experiment_name,
                        "dataset": dataset_name,
                        "feature_cols": predictors,
                        "n_features": n_features,
                        "target_col": TARGET_COL,
                        "group_col": GROUP_COL,
                        "class_order": CLASS_ORDER,
                        "region_order": REGION_ORDER,
                        "held_out_region": held_out_region,
                        "label_encoder":
                            label_encoder
                            if encoded_labels
                            else None,
                        "training_type":
                            "LORO_FOLD_MODEL",
                        "random_state":
                            RANDOM_STATE,
                    }


                    joblib.dump(
                        package,
                        model_path,
                    )


                    saved_model_rows.append({
                        "experiment_id": experiment_id,
                        "experiment": experiment_name,
                        "algorithm": algorithm,
                        "algorithm_name": algorithm_name,
                        "dataset": dataset_name,
                        "training_type": "LORO_FOLD_MODEL",
                        "held_out_region": held_out_region,
                        "n_features": n_features,
                        "model_path": str(
                            model_path
                        ),
                    })


                pooled_true.extend(
                    y_test.tolist()
                )

                pooled_predicted.extend(
                    list(y_pred)
                )


            # =================================================================
            # Pooled LORO results across all five held-out regions
            # =================================================================

            pooled_metrics = calculate_metrics(
                pooled_true,
                pooled_predicted,
            )


            pooled_report = classification_report(
                pooled_true,
                pooled_predicted,
                labels=CLASS_ORDER,
                output_dict=True,
                zero_division=0,
            )


            pooled_result = {
                "experiment_id": experiment_id,
                "experiment": experiment_name,
                "algorithm": algorithm,
                "algorithm_name": algorithm_name,
                "dataset": dataset_name,
                "n_features": n_features,
                "n_samples": len(
                    pooled_true
                ),
                **pooled_metrics,
            }


            for class_name in CLASS_ORDER:

                safe_name = class_name.lower()

                pooled_result[
                    f"{safe_name}_precision"
                ] = pooled_report[
                    class_name
                ]["precision"]

                pooled_result[
                    f"{safe_name}_recall"
                ] = pooled_report[
                    class_name
                ]["recall"]

                pooled_result[
                    f"{safe_name}_f1"
                ] = pooled_report[
                    class_name
                ]["f1-score"]


            pooled_results.append(
                pooled_result
            )


            # =================================================================
            # Final all-sample model
            # =================================================================

            print(
                f"Training all-sample model: "
                f"{algorithm_name} | {experiment_name}"
            )


            X_full = (
                dataframe[predictors]
                .astype(float)
            )

            y_full = (
                dataframe[TARGET_COL]
                .astype(str)
            )


            final_model = make_model(
                algorithm=algorithm,
                num_classes=len(
                    CLASS_ORDER
                ),
            )


            if encoded_labels:

                final_model.fit(
                    X_full,
                    label_encoder.transform(
                        y_full
                    ),
                )

            else:

                final_model.fit(
                    X_full,
                    y_full,
                )


            final_model_path = (
                model_dir
                / (
                    f"{algorithm}_"
                    f"{experiment_name}_"
                    f"FINAL_ALL_SAMPLES.joblib"
                )
            )


            final_package = {
                "model": final_model,
                "algorithm": algorithm,
                "algorithm_name": algorithm_name,
                "experiment_id": experiment_id,
                "experiment": experiment_name,
                "dataset": dataset_name,
                "feature_cols": predictors,
                "n_features": n_features,
                "target_col": TARGET_COL,
                "group_col": GROUP_COL,
                "class_order": CLASS_ORDER,
                "region_order": REGION_ORDER,
                "label_encoder":
                    label_encoder
                    if encoded_labels
                    else None,
                "training_type":
                    "FINAL_ALL_SAMPLES_MODEL",
                "n_training_samples":
                    len(dataframe),
                "random_state":
                    RANDOM_STATE,
                "source_samples": {
                    name: str(path)
                    for name, path
                    in source_files.items()
                },
            }


            joblib.dump(
                final_package,
                final_model_path,
            )


            saved_model_rows.append({
                "experiment_id": experiment_id,
                "experiment": experiment_name,
                "algorithm": algorithm,
                "algorithm_name": algorithm_name,
                "dataset": dataset_name,
                "training_type":
                    "FINAL_ALL_SAMPLES_MODEL",
                "held_out_region": "",
                "n_features": n_features,
                "model_path": str(
                    final_model_path
                ),
            })


    # =========================================================================
    # Convert collected outputs
    # =========================================================================

    fold_df = pd.DataFrame(
        fold_results
    )

    pooled_df = pd.DataFrame(
        pooled_results
    )

    confusion_df = pd.DataFrame(
        confusion_rows
    )

    report_df = pd.DataFrame(
        report_rows
    )

    importance_df = pd.DataFrame(
        feature_importance_rows
    )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    model_index_df = pd.DataFrame(
        saved_model_rows
    )

    feature_inventory_df = pd.DataFrame(
        feature_inventory_rows
    )


    # =========================================================================
    # Fold mean and SD
    # =========================================================================

    fold_summary = (
        fold_df
        .groupby(
            [
                "experiment_id",
                "experiment",
                "algorithm",
                "algorithm_name",
                "dataset",
                "n_features",
            ]
        )
        .agg(
            fold_accuracy_mean=(
                "overall_accuracy",
                "mean",
            ),
            fold_accuracy_std=(
                "overall_accuracy",
                "std",
            ),
            fold_macro_f1_mean=(
                "macro_f1",
                "mean",
            ),
            fold_macro_f1_std=(
                "macro_f1",
                "std",
            ),
            fold_kappa_mean=(
                "cohens_kappa",
                "mean",
            ),
            fold_kappa_std=(
                "cohens_kappa",
                "std",
            ),
        )
        .reset_index()
    )


    pooled_df = pooled_df.merge(
        fold_summary,
        on=[
            "experiment_id",
            "experiment",
            "algorithm",
            "algorithm_name",
            "dataset",
            "n_features",
        ],
        how="left",
    )


    # =========================================================================
    # Pooled confusion matrices
    # =========================================================================

    pooled_confusion = (
        confusion_df
        .groupby(
            [
                "experiment_id",
                "experiment",
                "algorithm",
                "algorithm_name",
                "dataset",
                "n_features",
                "true_label",
                "predicted_label",
            ],
            as_index=False,
        )["count"]
        .sum()
    )


    # =========================================================================
    # Mean tree importance over LORO folds
    # =========================================================================

    importance_summary = (
        importance_df
        .groupby(
            [
                "experiment_id",
                "experiment",
                "algorithm",
                "algorithm_name",
                "dataset",
                "feature",
            ]
        )
        .agg(
            importance_mean=(
                "importance",
                "mean",
            ),
            importance_std=(
                "importance",
                "std",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "experiment_id",
                "algorithm",
                "importance_mean",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
    )


    # =========================================================================
    # Ranking
    # =========================================================================

    ranking = (
        pooled_df
        .sort_values(
            [
                "macro_f1",
                "cohens_kappa",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    # =========================================================================
    # Save CSV outputs
    # =========================================================================

    outputs = {
        "pooled_results":
            output_dir
            / "loro_ablation_pooled_results.csv",

        "fold_results":
            output_dir
            / "loro_ablation_fold_results.csv",

        "fold_confusion_matrices":
            output_dir
            / "loro_ablation_confusion_by_fold.csv",

        "pooled_confusion_matrices":
            output_dir
            / "loro_ablation_confusion_pooled.csv",

        "classification_reports":
            output_dir
            / "loro_ablation_class_metrics_by_fold.csv",

        "predictions":
            output_dir
            / "loro_ablation_predictions.csv",

        "feature_importance_by_fold":
            output_dir
            / "loro_ablation_feature_importance_by_fold.csv",

        "feature_importance_summary":
            output_dir
            / "loro_ablation_feature_importance_summary.csv",

        "feature_inventory":
            output_dir
            / "loro_ablation_feature_inventory.csv",

        "saved_models":
            output_dir
            / "loro_ablation_saved_model_index.csv",

        "ranking":
            output_dir
            / "loro_ablation_ranking.csv",
    }


    pooled_df.to_csv(
        outputs["pooled_results"],
        index=False,
    )

    fold_df.to_csv(
        outputs["fold_results"],
        index=False,
    )

    confusion_df.to_csv(
        outputs["fold_confusion_matrices"],
        index=False,
    )

    pooled_confusion.to_csv(
        outputs["pooled_confusion_matrices"],
        index=False,
    )

    report_df.to_csv(
        outputs["classification_reports"],
        index=False,
    )

    predictions_df.to_csv(
        outputs["predictions"],
        index=False,
    )

    importance_df.to_csv(
        outputs["feature_importance_by_fold"],
        index=False,
    )

    importance_summary.to_csv(
        outputs["feature_importance_summary"],
        index=False,
    )

    feature_inventory_df.to_csv(
        outputs["feature_inventory"],
        index=False,
    )

    model_index_df.to_csv(
        outputs["saved_models"],
        index=False,
    )

    ranking.to_csv(
        outputs["ranking"],
        index=False,
    )


    # =========================================================================
    # Save exact experiment configuration
    # =========================================================================

    configuration_json = []


    for experiment_id, (
        experiment_name,
        configuration,
    ) in enumerate(
        feature_groups.items(),
        start=1,
    ):

        configuration_json.append({
            "experiment_id":
                experiment_id,

            "experiment":
                experiment_name,

            "dataset":
                configuration["dataset"],

            "n_features":
                len(
                    configuration[
                        "features"
                    ]
                ),

            "features":
                configuration[
                    "features"
                ],
        })


    config_path = (
        output_dir
        / "loro_ablation_experiment_config.json"
    )


    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            configuration_json,
            file,
            indent=2,
        )


    # =========================================================================
    # Final integrity checks
    # =========================================================================

    expected_experiment_algorithm_pairs = (
        len(EXPECTED_FEATURE_COUNTS)
        * len(MODEL_CONFIGS)
    )


    if len(pooled_df) != expected_experiment_algorithm_pairs:

        raise RuntimeError(
            "Unexpected number of pooled experiment results."
        )


    expected_fold_rows = (
        expected_experiment_algorithm_pairs
        * len(REGION_ORDER)
    )


    if len(fold_df) != expected_fold_rows:

        raise RuntimeError(
            f"Expected {expected_fold_rows} fold results; "
            f"found {len(fold_df)}."
        )


    final_models = model_index_df[
        model_index_df["training_type"]
        == "FINAL_ALL_SAMPLES_MODEL"
    ]


    if len(final_models) != expected_experiment_algorithm_pairs:

        raise RuntimeError(
            f"Expected {expected_experiment_algorithm_pairs} "
            f"final all-sample models; "
            f"found {len(final_models)}."
        )


    # =========================================================================
    # Console summary
    # =========================================================================

    print(
        "\n" + "=" * 96
    )

    print(
        "LORO ABLATION MODELLING COMPLETE"
    )

    print(
        "=" * 96
    )


    summary_columns = [
        "experiment",
        "algorithm",
        "n_features",
        "macro_f1",
        "rrg_f1",
        "nffp_f1",
        "water_f1",
        "cohens_kappa",
        "fold_macro_f1_mean",
        "fold_macro_f1_std",
    ]


    print(
        ranking[
            summary_columns
        ].to_string(
            index=False
        )
    )


    print(
        "\nOutputs:"
    )

    for name, path in outputs.items():

        print(
            f"  {name:<32} {path}"
        )


    print(
        f"  {'experiment_config':<32} "
        f"{config_path}"
    )


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


    default_input_dir = (
        repo_root
        / "data"
        / "reference_samples"
        / "model_ready"
    )


    default_output_dir = (
        repo_root
        / "results"
        / "loro_ablation"
    )


    default_model_dir = (
        repo_root
        / "models"
        / "loro_ablation"
    )


    parser = argparse.ArgumentParser(
        description=(
            "Run the eight baseline RF/XGBoost "
            "LORO experiments for ECHO."
        )
    )


    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help=(
            "Directory containing model-ready "
            "reference-sample CSV files."
        ),
    )


    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=(
            "Directory for LORO result tables."
        ),
    )


    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help=(
            "Directory for fitted model packages."
        ),
    )


    parser.add_argument(
        "--save-fold-models",
        action="store_true",
        help=(
            "Also save the 80 individual LORO fold models. "
            "By default only the 16 all-sample baseline models "
            "are retained."
        ),
    )


    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Load samples, construct feature groups and run all experiments."""

    args = parse_args()


    input_dir = (
        args.input_dir
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    model_dir = (
        args.model_dir
        .expanduser()
        .resolve()
    )


    # =========================================================================
    # Model-ready input files produced by 05_prepare_reference_samples.py
    # =========================================================================

    s1_file = (
        input_dir
        / "reference_samples_2025_2026_S1_MODEL_READY.csv"
    )

    s2_file = (
        input_dir
        / "reference_samples_2025_2026_S2_MODEL_READY.csv"
    )

    fusion_file = (
        input_dir
        / "reference_samples_2025_2026_S1S2_MODEL_READY.csv"
    )


    source_files = {
        "S1": s1_file,
        "S2": s2_file,
        "S1S2": fusion_file,
    }


    for name, path in source_files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Missing {name} reference-sample file: "
                f"{path}"
            )


    # =========================================================================
    # Load data
    # =========================================================================

    df_s1 = pd.read_csv(
        s1_file
    )

    df_s2 = pd.read_csv(
        s2_file
    )

    df_fusion = pd.read_csv(
        fusion_file
    )


    print(
        "ECHO baseline LORO modelling"
    )

    print(
        f"S1 samples:    {df_s1.shape}"
    )

    print(
        f"S2 samples:    {df_s2.shape}"
    )

    print(
        f"Fusion samples: {df_fusion.shape}"
    )


    # =========================================================================
    # Input audit
    # =========================================================================

    s1_features = audit_reference_table(
        df_s1,
        "S1",
        104,
    )

    s2_features = audit_reference_table(
        df_s2,
        "S2",
        72,
    )

    fusion_features = audit_reference_table(
        df_fusion,
        "S1S2",
        176,
    )


    audit_identical_samples({
        "S1": df_s1,
        "S2": df_s2,
        "S1S2": df_fusion,
    })


    print(
        "Reference-sample audit: PASS"
    )


    # =========================================================================
    # Build and audit eight feature configurations
    # =========================================================================

    groups = build_feature_groups(
        df_s1=df_s1,
        df_s2=df_s2,
        df_fusion=df_fusion,
        s1_features=s1_features,
        s2_features=s2_features,
        fusion_features=fusion_features,
    )


    print(
        "\nFeature configurations:"
    )


    for name, configuration in groups.items():

        print(
            f"  {name:<30} "
            f"{len(configuration['features']):>3} predictors"
        )


    # =========================================================================
    # Run LORO experiments
    # =========================================================================

    run_experiments(
        feature_groups=groups,
        output_dir=output_dir,
        model_dir=model_dir,
        save_fold_models=args.save_fold_models,
        source_files=source_files,
    )


if __name__ == "__main__":
    main()