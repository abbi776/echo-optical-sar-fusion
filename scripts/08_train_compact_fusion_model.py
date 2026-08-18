#!/usr/bin/env python3
"""
Train and evaluate the SHAP-guided Compact Fusion model for ECHO.

This script implements the ninth feature configuration:

    Compact S1-S2 Fusion

The compact feature subset is read directly from the fold-wise,
training-only SHAP selection produced by:

    07_shap_feature_analysis.py

The feature-selection rule is:

    smallest subset reaching at least 80% cumulative
    fold-wise training-only mean absolute SHAP attribution

The compact subset is NOT hard-coded in this script.

Both classifiers are evaluated:

    Random Forest
    XGBoost

using the same five-fold Leave-One-Region-Out (LORO) protocol and the
same fixed hyperparameters used for the eight baseline configurations.

The script then:

    1. evaluates Compact Fusion with RF and XGBoost;
    2. calculates pooled and fold-level metrics;
    3. calculates pooled and fold confusion matrices;
    4. calculates class-specific precision, recall and F1;
    5. records held-out predictions and class probabilities;
    6. compares Compact Fusion with selected baseline configurations;
    7. fits final Compact RF and XGBoost models to all 525 polygons;
    8. identifies Compact XGBoost as the operational ECHO classifier.

Important methodological note
-----------------------------
The common compact subset was derived by aggregating fold-wise,
training-only SHAP rankings before compact-model LORO evaluation.
Therefore, the compact-model assessment is not fully nested, matching
the limitation stated in the manuscript.

No figures are generated here.
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
# Experimental design
# =============================================================================

RANDOM_STATE = 42

EXPECTED_ROWS = 525
EXPECTED_FULL_FEATURES = 176
EXPECTED_COMPACT_FEATURES = 25

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
EXPECTED_CLASS_REGION_COUNT = 35

NODATA = -9999.0


# =============================================================================
# Metadata columns excluded from modelling
# =============================================================================

METADATA_COLUMNS = {
    "unique_id",
    "sample_id",
    "roi_id",
    "polygon_id",
    "label_id",
    "OBJECTID",
    "objectid",

    "class_raw",
    "class_label",
    "class_id",
    "class_name",
    "class",
    "classname",
    "classvalue",

    "region",
    "site",
    "period",
    "year",
    "stack",
    "source_file",
    "source_stack",
    "sensor",

    "geometry",
    "x",
    "y",
    "centroid_x",
    "centroid_y",
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
# Input helpers
# =============================================================================

def feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return numeric modelling predictors while excluding metadata fields.
    """

    numeric_columns = (
        dataframe
        .select_dtypes(
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


def audit_reference_table(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Strictly audit the 176-feature Full Fusion reference table.
    """

    required_columns = {
        "unique_id",
        TARGET_COL,
        GROUP_COL,
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:

        raise ValueError(
            "Missing required column(s): "
            + ", ".join(
                sorted(missing)
            )
        )


    if len(dataframe) != EXPECTED_ROWS:

        raise ValueError(
            f"Expected {EXPECTED_ROWS} reference polygons; "
            f"found {len(dataframe)}."
        )


    if dataframe[
        "unique_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate polygon identifiers detected."
        )


    predictors = feature_columns(
        dataframe
    )


    if len(predictors) != EXPECTED_FULL_FEATURES:

        raise ValueError(
            f"Expected {EXPECTED_FULL_FEATURES} Full Fusion predictors; "
            f"found {len(predictors)}."
        )


    if dataframe[
        predictors
    ].isna().any().any():

        raise ValueError(
            "NaN predictor values detected."
        )


    if not np.isfinite(
        dataframe[
            predictors
        ].to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            "Non-finite predictor values detected."
        )


    observed_classes = set(
        dataframe[
            TARGET_COL
        ]
        .astype(str)
        .unique()
    )


    if observed_classes != set(
        CLASS_ORDER
    ):

        raise ValueError(
            f"Unexpected class labels: "
            f"{sorted(observed_classes)}"
        )


    observed_regions = set(
        dataframe[
            GROUP_COL
        ]
        .astype(str)
        .unique()
    )


    if observed_regions != set(
        REGION_ORDER
    ):

        raise ValueError(
            f"Unexpected spatial regions: "
            f"{sorted(observed_regions)}"
        )


    for region in REGION_ORDER:

        region_df = dataframe[
            dataframe[
                GROUP_COL
            ].astype(str)
            == region
        ]


        if len(region_df) != EXPECTED_TEST_PER_FOLD:

            raise ValueError(
                f"{region}: expected "
                f"{EXPECTED_TEST_PER_FOLD} polygons; "
                f"found {len(region_df)}."
            )


        counts = (
            region_df[
                TARGET_COL
            ]
            .astype(str)
            .value_counts()
        )


        for class_name in CLASS_ORDER:

            if int(
                counts.get(
                    class_name,
                    0,
                )
            ) != EXPECTED_CLASS_REGION_COUNT:

                raise ValueError(
                    f"{region}/{class_name}: expected "
                    f"{EXPECTED_CLASS_REGION_COUNT} polygons."
                )


    return predictors


# =============================================================================
# Compact feature selection
# =============================================================================

def load_compact_features(
    selection_path: Path,
    full_features: list[str],
) -> tuple[list[str], dict]:
    """
    Read the SHAP-selected compact feature subset.

    Feature order is preserved exactly as produced by the SHAP ranking.
    """

    if not selection_path.exists():

        raise FileNotFoundError(
            "Compact feature-selection file not found: "
            f"{selection_path}"
        )


    with open(
        selection_path,
        "r",
        encoding="utf-8",
    ) as file:

        selection = json.load(
            file
        )


    compact_features = selection.get(
        "selected_features",
        []
    )


    if not compact_features:

        raise ValueError(
            "No selected_features found in SHAP selection file."
        )


    if len(
        compact_features
    ) != EXPECTED_COMPACT_FEATURES:

        raise ValueError(
            f"Expected {EXPECTED_COMPACT_FEATURES} compact predictors; "
            f"found {len(compact_features)}."
        )


    if len(
        set(compact_features)
    ) != len(
        compact_features
    ):

        raise ValueError(
            "Duplicate compact predictors detected."
        )


    missing = [
        feature
        for feature in compact_features
        if feature not in full_features
    ]


    if missing:

        raise ValueError(
            "Compact predictor(s) not present in the "
            "Full Fusion sample table:\n  - "
            + "\n  - ".join(
                missing
            )
        )


    threshold = selection.get(
        "cumulative_threshold"
    )


    if threshold is not None:

        if not np.isclose(
            float(threshold),
            0.80,
        ):

            raise ValueError(
                "Unexpected SHAP cumulative threshold: "
                f"{threshold}"
            )


    return (
        compact_features,
        selection,
    )


# =============================================================================
# Model factories
# =============================================================================

def make_random_forest() -> RandomForestClassifier:
    """
    Random Forest configuration used throughout the study.
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
    XGBoost configuration used throughout the study.
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


MODEL_CONFIGS = {
    "RF": {
        "algorithm_name": "Random Forest",
        "requires_encoding": False,
    },

    "XGBOOST": {
        "algorithm_name": "XGBoost",
        "requires_encoding": True,
    },
}


# =============================================================================
# Metrics
# =============================================================================

def calculate_metrics(
    y_true,
    y_pred,
) -> dict:
    """
    Calculate classification metrics used in the experiment.
    """

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


def count_rrg_nffp_errors(
    y_true,
    y_pred,
) -> dict:
    """
    Count the two directions of residual RRG-NFFP exchange.
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )


    rrg_to_nffp = int(
        np.sum(
            (y_true == "RRG")
            & (y_pred == "NFFP")
        )
    )


    nffp_to_rrg = int(
        np.sum(
            (y_true == "NFFP")
            & (y_pred == "RRG")
        )
    )


    return {
        "rrg_to_nffp_errors":
            rrg_to_nffp,

        "nffp_to_rrg_errors":
            nffp_to_rrg,

        "total_rrg_nffp_errors":
            (
                rrg_to_nffp
                + nffp_to_rrg
            ),
    }


# =============================================================================
# Compact LORO experiment
# =============================================================================

def run_compact_loro(
    dataframe: pd.DataFrame,
    compact_features: list[str],
    output_dir: Path,
) -> dict:
    """
    Evaluate Compact Fusion with RF and XGBoost under five-fold LORO.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -------------------------------------------------------------------------
    # Stable XGBoost label encoder
    #
    # LabelEncoder sorts alphabetically, producing:
    # NFFP, RRG, Water
    #
    # This matches the original compact-model experiment.
    # -------------------------------------------------------------------------

    label_encoder = LabelEncoder()

    label_encoder.fit(
        CLASS_ORDER
    )


    fold_rows = []
    prediction_rows = []
    confusion_rows = []
    class_metric_rows = []
    pooled_rows = []


    # =========================================================================
    # Classifier loop
    # =========================================================================

    for (
        algorithm,
        configuration,
    ) in MODEL_CONFIGS.items():


        algorithm_name = configuration[
            "algorithm_name"
        ]

        requires_encoding = configuration[
            "requires_encoding"
        ]


        print(
            "\n" + "=" * 96
        )

        print(
            f"Compact Fusion | {algorithm_name}"
        )

        print(
            f"Predictors: {len(compact_features)}"
        )

        print(
            "=" * 96
        )


        pooled_true = []
        pooled_pred = []


        # =====================================================================
        # Five LORO folds
        # =====================================================================

        for fold_number, held_out_region in enumerate(
            REGION_ORDER,
            start=1,
        ):


            training = dataframe[
                dataframe[
                    GROUP_COL
                ].astype(str)
                != held_out_region
            ].copy()


            testing = dataframe[
                dataframe[
                    GROUP_COL
                ].astype(str)
                == held_out_region
            ].copy()


            if len(training) != EXPECTED_TRAIN_PER_FOLD:

                raise ValueError(
                    f"{held_out_region}: expected "
                    f"{EXPECTED_TRAIN_PER_FOLD} training samples; "
                    f"found {len(training)}."
                )


            if len(testing) != EXPECTED_TEST_PER_FOLD:

                raise ValueError(
                    f"{held_out_region}: expected "
                    f"{EXPECTED_TEST_PER_FOLD} test samples; "
                    f"found {len(testing)}."
                )


            X_train = (
                training[
                    compact_features
                ]
                .astype(float)
            )


            X_test = (
                testing[
                    compact_features
                ]
                .astype(float)
            )


            y_train = (
                training[
                    TARGET_COL
                ]
                .astype(str)
            )


            y_test = (
                testing[
                    TARGET_COL
                ]
                .astype(str)
            )


            # -----------------------------------------------------------------
            # Fresh model for each fold
            # -----------------------------------------------------------------

            if algorithm == "RF":

                model = make_random_forest()


            elif algorithm == "XGBOOST":

                model = make_xgboost(
                    num_classes=len(
                        CLASS_ORDER
                    )
                )


            else:

                raise ValueError(
                    f"Unknown classifier: {algorithm}"
                )


            # -----------------------------------------------------------------
            # Fit
            # -----------------------------------------------------------------

            if requires_encoding:

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


            # -----------------------------------------------------------------
            # Fold metrics
            # -----------------------------------------------------------------

            metrics = calculate_metrics(
                y_test,
                y_pred,
            )


            boundary_errors = (
                count_rrg_nffp_errors(
                    y_test,
                    y_pred,
                )
            )


            fold_rows.append({
                "experiment":
                    "COMPACT_S1S2_FUSION",

                "algorithm":
                    algorithm,

                "algorithm_name":
                    algorithm_name,

                "fold":
                    fold_number,

                "held_out_region":
                    held_out_region,

                "n_features":
                    len(
                        compact_features
                    ),

                "n_train":
                    len(
                        training
                    ),

                "n_test":
                    len(
                        testing
                    ),

                **metrics,
                **boundary_errors,
            })


            print(
                f"Fold {fold_number} | "
                f"{held_out_region} | "
                f"macro F1 = "
                f"{metrics['macro_f1']:.6f}"
            )


            # -----------------------------------------------------------------
            # Fold confusion matrix
            # -----------------------------------------------------------------

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
                        "experiment":
                            "COMPACT_S1S2_FUSION",

                        "algorithm":
                            algorithm,

                        "algorithm_name":
                            algorithm_name,

                        "fold":
                            fold_number,

                        "held_out_region":
                            held_out_region,

                        "true_label":
                            true_label,

                        "predicted_label":
                            predicted_label,

                        "count":
                            int(
                                cm[
                                    true_index,
                                    predicted_index,
                                ]
                            ),
                    })


            # -----------------------------------------------------------------
            # Class-specific metrics
            # -----------------------------------------------------------------

            report = classification_report(
                y_test,
                y_pred,
                labels=CLASS_ORDER,
                output_dict=True,
                zero_division=0,
            )


            for class_name in CLASS_ORDER:

                class_values = report[
                    class_name
                ]


                class_metric_rows.append({
                    "experiment":
                        "COMPACT_S1S2_FUSION",

                    "algorithm":
                        algorithm,

                    "algorithm_name":
                        algorithm_name,

                    "fold":
                        fold_number,

                    "held_out_region":
                        held_out_region,

                    "class":
                        class_name,

                    "precision":
                        class_values[
                            "precision"
                        ],

                    "recall":
                        class_values[
                            "recall"
                        ],

                    "f1":
                        class_values[
                            "f1-score"
                        ],

                    "support":
                        class_values[
                            "support"
                        ],
                })


            # -----------------------------------------------------------------
            # Held-out predictions and probabilities
            # -----------------------------------------------------------------

            testing_indices = list(
                testing.index
            )


            for position, row_index in enumerate(
                testing_indices
            ):


                sample = dataframe.loc[
                    row_index
                ]


                prediction_record = {
                    "experiment":
                        "COMPACT_S1S2_FUSION",

                    "algorithm":
                        algorithm,

                    "algorithm_name":
                        algorithm_name,

                    "fold":
                        fold_number,

                    "held_out_region":
                        held_out_region,

                    "unique_id":
                        sample[
                            "unique_id"
                        ],

                    "region":
                        sample[
                            GROUP_COL
                        ],

                    "true_label":
                        y_test.loc[
                            row_index
                        ],

                    "predicted_label":
                        y_pred[
                            position
                        ],

                    "correct":
                        bool(
                            y_test.loc[
                                row_index
                            ]
                            == y_pred[
                                position
                            ]
                        ),
                }


                if "class_id" in dataframe.columns:

                    prediction_record[
                        "class_id"
                    ] = sample[
                        "class_id"
                    ]


                for class_index, class_name in enumerate(
                    probability_classes
                ):

                    prediction_record[
                        f"prob_{class_name}"
                    ] = float(
                        probabilities[
                            position,
                            class_index,
                        ]
                    )


                prediction_rows.append(
                    prediction_record
                )


            pooled_true.extend(
                y_test.tolist()
            )

            pooled_pred.extend(
                list(
                    y_pred
                )
            )


        # =====================================================================
        # Pooled results across all five held-out regions
        # =====================================================================

        pooled_metrics = calculate_metrics(
            pooled_true,
            pooled_pred,
        )


        pooled_boundary_errors = (
            count_rrg_nffp_errors(
                pooled_true,
                pooled_pred,
            )
        )


        pooled_report = (
            classification_report(
                pooled_true,
                pooled_pred,
                labels=CLASS_ORDER,
                output_dict=True,
                zero_division=0,
            )
        )


        pooled_row = {
            "experiment":
                "COMPACT_S1S2_FUSION",

            "algorithm":
                algorithm,

            "algorithm_name":
                algorithm_name,

            "n_features":
                len(
                    compact_features
                ),

            "n_samples":
                len(
                    pooled_true
                ),

            **pooled_metrics,
            **pooled_boundary_errors,
        }


        for class_name in CLASS_ORDER:

            safe_name = (
                class_name
                .lower()
            )


            pooled_row[
                f"{safe_name}_precision"
            ] = (
                pooled_report[
                    class_name
                ][
                    "precision"
                ]
            )


            pooled_row[
                f"{safe_name}_recall"
            ] = (
                pooled_report[
                    class_name
                ][
                    "recall"
                ]
            )


            pooled_row[
                f"{safe_name}_f1"
            ] = (
                pooled_report[
                    class_name
                ][
                    "f1-score"
                ]
            )


        pooled_rows.append(
            pooled_row
        )


    # =========================================================================
    # DataFrames
    # =========================================================================

    fold_df = pd.DataFrame(
        fold_rows
    )

    pooled_df = pd.DataFrame(
        pooled_rows
    )

    confusion_df = pd.DataFrame(
        confusion_rows
    )

    class_metrics_df = pd.DataFrame(
        class_metric_rows
    )

    predictions_df = pd.DataFrame(
        prediction_rows
    )


    # =========================================================================
    # Fold means and SDs
    # =========================================================================

    fold_summary = (
        fold_df
        .groupby(
            [
                "experiment",
                "algorithm",
                "algorithm_name",
                "n_features",
            ]
        )
        .agg(
            fold_macro_f1_mean=(
                "macro_f1",
                "mean",
            ),

            fold_macro_f1_std=(
                "macro_f1",
                "std",
            ),

            fold_accuracy_mean=(
                "overall_accuracy",
                "mean",
            ),

            fold_accuracy_std=(
                "overall_accuracy",
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
            "experiment",
            "algorithm",
            "algorithm_name",
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
                "experiment",
                "algorithm",
                "algorithm_name",
                "true_label",
                "predicted_label",
            ],
            as_index=False,
        )[
            "count"
        ]
        .sum()
    )


    # =========================================================================
    # Save compact LORO outputs
    # =========================================================================

    output_paths = {
        "pooled":
            output_dir
            / "compact_fusion_pooled_results.csv",

        "folds":
            output_dir
            / "compact_fusion_fold_results.csv",

        "predictions":
            output_dir
            / "compact_fusion_predictions.csv",

        "class_metrics":
            output_dir
            / "compact_fusion_class_metrics_by_fold.csv",

        "confusion_by_fold":
            output_dir
            / "compact_fusion_confusion_by_fold.csv",

        "confusion_pooled":
            output_dir
            / "compact_fusion_confusion_pooled.csv",
    }


    pooled_df.to_csv(
        output_paths[
            "pooled"
        ],
        index=False,
    )


    fold_df.to_csv(
        output_paths[
            "folds"
        ],
        index=False,
    )


    predictions_df.to_csv(
        output_paths[
            "predictions"
        ],
        index=False,
    )


    class_metrics_df.to_csv(
        output_paths[
            "class_metrics"
        ],
        index=False,
    )


    confusion_df.to_csv(
        output_paths[
            "confusion_by_fold"
        ],
        index=False,
    )


    pooled_confusion.to_csv(
        output_paths[
            "confusion_pooled"
        ],
        index=False,
    )


    return {
        "pooled":
            pooled_df,

        "folds":
            fold_df,

        "predictions":
            predictions_df,

        "pooled_confusion":
            pooled_confusion,

        "label_encoder":
            label_encoder,
    }


# =============================================================================
# Baseline comparison
# =============================================================================

def compare_with_baselines(
    compact_results: pd.DataFrame,
    baseline_results_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Compare Compact Fusion with the principal baseline configurations.
    """

    if not baseline_results_path.exists():

        raise FileNotFoundError(
            "Baseline LORO results not found: "
            f"{baseline_results_path}"
        )


    baseline = pd.read_csv(
        baseline_results_path
    )


    required_baselines = [
        "S1_TEXTURE_ONLY",
        "FULL_S2",
        "FULL_S1",
        "FULL_S1S2_FUSION",
    ]


    baseline_subset = baseline[
        baseline[
            "experiment"
        ].isin(
            required_baselines
        )
    ].copy()


    compact_copy = compact_results.copy()


    # Align columns before concatenation.
    all_columns = sorted(
        set(
            baseline_subset.columns
        )
        | set(
            compact_copy.columns
        )
    )


    for column in all_columns:

        if column not in baseline_subset.columns:
            baseline_subset[
                column
            ] = np.nan

        if column not in compact_copy.columns:
            compact_copy[
                column
            ] = np.nan


    comparison = pd.concat(
        [
            baseline_subset[
                all_columns
            ],
            compact_copy[
                all_columns
            ],
        ],
        ignore_index=True,
    )


    comparison = (
        comparison
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


    comparison[
        "performance_rank"
    ] = np.arange(
        1,
        len(
            comparison
        ) + 1,
    )


    comparison_path = (
        output_dir
        / "compact_vs_principal_baselines.csv"
    )


    comparison.to_csv(
        comparison_path,
        index=False,
    )


    print(
        "\nComparison with principal baselines:"
    )


    display_columns = [
        "performance_rank",
        "experiment",
        "algorithm",
        "n_features",
        "macro_f1",
        "rrg_f1",
        "nffp_f1",
        "water_f1",
        "cohens_kappa",
    ]


    available_columns = [
        column
        for column in display_columns
        if column in comparison.columns
    ]


    print(
        comparison[
            available_columns
        ].to_string(
            index=False
        )
    )


    return comparison


# =============================================================================
# Train final all-sample compact models
# =============================================================================

def train_final_compact_models(
    dataframe: pd.DataFrame,
    compact_features: list[str],
    selection_metadata: dict,
    model_dir: Path,
) -> dict:
    """
    Fit Compact RF and Compact XGBoost to all 525 reference polygons.

    The Compact XGBoost package is the operational ECHO model.
    """

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    X_full = (
        dataframe[
            compact_features
        ]
        .astype(float)
    )


    y_full = (
        dataframe[
            TARGET_COL
        ]
        .astype(str)
    )


    label_encoder = LabelEncoder()

    label_encoder.fit(
        CLASS_ORDER
    )


    packages = {}


    # =========================================================================
    # Random Forest
    # =========================================================================

    rf_model = make_random_forest()


    rf_model.fit(
        X_full,
        y_full,
    )


    rf_package = {
        "model":
            rf_model,

        "algorithm":
            "RF",

        "algorithm_name":
            "Random Forest",

        "experiment":
            "COMPACT_S1S2_FUSION",

        "framework":
            "ECHO",

        "feature_cols":
            compact_features,

        "n_features":
            len(
                compact_features
            ),

        "full_fusion_features":
            EXPECTED_FULL_FEATURES,

        "feature_reduction_percent":
            100.0
            * (
                1.0
                - len(
                    compact_features
                )
                / EXPECTED_FULL_FEATURES
            ),

        "target_col":
            TARGET_COL,

        "group_col":
            GROUP_COL,

        "class_order":
            CLASS_ORDER,

        "region_order":
            REGION_ORDER,

        "label_encoder":
            None,

        "training_type":
            "FINAL_ALL_SAMPLES_MODEL",

        "n_training_samples":
            len(
                dataframe
            ),

        "random_state":
            RANDOM_STATE,

        "feature_selection":
            selection_metadata,

        "operational_model":
            False,
    }


    rf_path = (
        model_dir
        / "RF_COMPACT_S1S2_FUSION_FINAL_ALL_SAMPLES.joblib"
    )


    joblib.dump(
        rf_package,
        rf_path,
    )


    packages[
        "RF"
    ] = {
        "path":
            rf_path,

        "package":
            rf_package,
    }


    # =========================================================================
    # XGBoost
    # =========================================================================

    xgb_model = make_xgboost(
        num_classes=len(
            CLASS_ORDER
        )
    )


    y_encoded = (
        label_encoder.transform(
            y_full
        )
    )


    xgb_model.fit(
        X_full,
        y_encoded,
    )


    xgb_package = {
        "model":
            xgb_model,

        "algorithm":
            "XGBOOST",

        "algorithm_name":
            "XGBoost",

        "experiment":
            "COMPACT_S1S2_FUSION",

        "framework":
            "ECHO",

        "feature_cols":
            compact_features,

        "n_features":
            len(
                compact_features
            ),

        "full_fusion_features":
            EXPECTED_FULL_FEATURES,

        "feature_reduction_percent":
            100.0
            * (
                1.0
                - len(
                    compact_features
                )
                / EXPECTED_FULL_FEATURES
            ),

        "target_col":
            TARGET_COL,

        "group_col":
            GROUP_COL,

        "class_order":
            CLASS_ORDER,

        "xgboost_encoded_class_order":
            list(
                label_encoder.classes_
            ),

        "region_order":
            REGION_ORDER,

        "label_encoder":
            label_encoder,

        "training_type":
            "FINAL_ALL_SAMPLES_MODEL",

        "n_training_samples":
            len(
                dataframe
            ),

        "random_state":
            RANDOM_STATE,

        "feature_selection":
            selection_metadata,

        # This is the model transferred to the three annual periods.
        "operational_model":
            True,

        "probability_note":
            (
                "predict_proba outputs are treated as relative "
                "prediction decisiveness scores rather than "
                "calibrated uncertainty estimates."
            ),
    }


    xgb_path = (
        model_dir
        / "XGBOOST_COMPACT_S1S2_FUSION_FINAL_ALL_SAMPLES.joblib"
    )


    joblib.dump(
        xgb_package,
        xgb_path,
    )


    packages[
        "XGBOOST"
    ] = {
        "path":
            xgb_path,

        "package":
            xgb_package,
    }


    # =========================================================================
    # Operational-model alias
    # =========================================================================

    operational_path = (
        model_dir
        / "ECHO_OPERATIONAL_COMPACT_XGBOOST.joblib"
    )


    joblib.dump(
        xgb_package,
        operational_path,
    )


    packages[
        "OPERATIONAL"
    ] = {
        "path":
            operational_path,

        "package":
            xgb_package,
    }


    return packages


# =============================================================================
# Final integrity checks
# =============================================================================

def audit_expected_reproduction(
    compact_results: pd.DataFrame,
    compact_features: list[str],
) -> None:
    """
    Check key structural properties of the completed experiment.

    This deliberately does not hard-code manuscript performance values
    as pass/fail thresholds; the repository should reproduce them from
    the supplied data rather than force them.
    """

    if len(
        compact_features
    ) != EXPECTED_COMPACT_FEATURES:

        raise RuntimeError(
            "Compact model does not contain 25 predictors."
        )


    expected_algorithms = {
        "RF",
        "XGBOOST",
    }


    observed_algorithms = set(
        compact_results[
            "algorithm"
        ]
    )


    if observed_algorithms != expected_algorithms:

        raise RuntimeError(
            "Both RF and XGBoost compact results "
            "were not produced."
        )


    if len(
        compact_results
    ) != 2:

        raise RuntimeError(
            "Expected exactly two pooled Compact Fusion results."
        )


# =============================================================================
# Save experiment configuration
# =============================================================================

def save_experiment_config(
    compact_features: list[str],
    selection_metadata: dict,
    compact_results: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Save a concise machine-readable record of experiment 9.
    """

    xgb_result = (
        compact_results[
            compact_results[
                "algorithm"
            ]
            == "XGBOOST"
        ]
        .iloc[0]
    )


    rf_result = (
        compact_results[
            compact_results[
                "algorithm"
            ]
            == "RF"
        ]
        .iloc[0]
    )


    configuration = {
        "experiment":
            "COMPACT_S1S2_FUSION",

        "framework":
            "ECHO",

        "feature_selection_rule":
            selection_metadata.get(
                "selection_method",
                (
                    "Smallest subset reaching at least 80% "
                    "cumulative fold-wise training-only SHAP."
                ),
            ),

        "cumulative_threshold":
            selection_metadata.get(
                "cumulative_threshold",
                0.80,
            ),

        "full_fusion_features":
            EXPECTED_FULL_FEATURES,

        "compact_features_count":
            len(
                compact_features
            ),

        "feature_reduction_percent":
            100.0
            * (
                1.0
                - len(
                    compact_features
                )
                / EXPECTED_FULL_FEATURES
            ),

        "compact_features":
            compact_features,

        "validation":
            {
                "method":
                    "five-fold leave-one-region-out",

                "regions":
                    REGION_ORDER,

                "training_samples_per_fold":
                    EXPECTED_TRAIN_PER_FOLD,

                "test_samples_per_fold":
                    EXPECTED_TEST_PER_FOLD,
            },

        "models":
            {
                "Random Forest":
                    {
                        "n_estimators":
                            500,

                        "max_features":
                            "sqrt",

                        "min_samples_leaf":
                            1,

                        "class_weight":
                            "balanced",

                        "random_state":
                            RANDOM_STATE,

                        "pooled_macro_f1":
                            float(
                                rf_result[
                                    "macro_f1"
                                ]
                            ),
                    },

                "XGBoost":
                    {
                        "n_estimators":
                            500,

                        "learning_rate":
                            0.03,

                        "max_depth":
                            4,

                        "subsample":
                            0.8,

                        "colsample_bytree":
                            0.8,

                        "tree_method":
                            "hist",

                        "objective":
                            "multi:softprob",

                        "eval_metric":
                            "mlogloss",

                        "random_state":
                            RANDOM_STATE,

                        "pooled_macro_f1":
                            float(
                                xgb_result[
                                    "macro_f1"
                                ]
                            ),

                        "operational_model":
                            True,
                    },
            },

        "methodological_note":
            (
                "The fold-wise SHAP rankings were aggregated into "
                "one common compact subset before LORO re-evaluation. "
                "The compact assessment is therefore not fully nested."
            ),
    }


    path = (
        output_dir
        / "compact_fusion_experiment_config.json"
    )


    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            configuration,
            file,
            indent=2,
        )


    return path


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


    default_selection = (
        repo_root
        / "results"
        / "shap_analysis"
        / "foldwise_training_only"
        / "compact_feature_selection.json"
    )


    default_baseline_results = (
        repo_root
        / "results"
        / "loro_ablation"
        / "loro_ablation_pooled_results.csv"
    )


    default_output_dir = (
        repo_root
        / "results"
        / "compact_fusion"
    )


    default_model_dir = (
        repo_root
        / "models"
        / "compact_fusion"
    )


    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the SHAP-selected Compact Fusion "
            "configuration and train the final ECHO model."
        )
    )


    parser.add_argument(
        "--samples",
        type=Path,
        default=default_samples,
        help=(
            "2025-2026 model-ready Full Fusion "
            "reference-sample CSV."
        ),
    )


    parser.add_argument(
        "--selection",
        type=Path,
        default=default_selection,
        help=(
            "Compact feature-selection JSON produced "
            "by 07_shap_feature_analysis.py."
        ),
    )


    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=default_baseline_results,
        help=(
            "Pooled baseline LORO results produced "
            "by 06_train_loro_ablation_models.py."
        ),
    )


    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=(
            "Directory for compact-model results."
        ),
    )


    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help=(
            "Directory for final Compact Fusion model packages."
        ),
    )


    return parser.parse_args()


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    """
    Run experiment 9 and train the final operational ECHO model.
    """

    args = parse_args()


    samples_path = (
        args.samples
        .expanduser()
        .resolve()
    )


    selection_path = (
        args.selection
        .expanduser()
        .resolve()
    )


    baseline_results_path = (
        args.baseline_results
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


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # =========================================================================
    # Load reference data
    # =========================================================================

    if not samples_path.exists():

        raise FileNotFoundError(
            f"Fusion reference-sample table not found: "
            f"{samples_path}"
        )


    dataframe = pd.read_csv(
        samples_path
    )


    print(
        "ECHO Compact Fusion modelling"
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
    # Audit Full Fusion samples
    # =========================================================================

    full_features = audit_reference_table(
        dataframe
    )


    print(
        "Full Fusion sample audit: PASS"
    )

    print(
        "Full predictors:",
        len(
            full_features
        ),
    )


    # =========================================================================
    # Read the SHAP-selected subset
    # =========================================================================

    compact_features, selection_metadata = (
        load_compact_features(
            selection_path=selection_path,
            full_features=full_features,
        )
    )


    print(
        "\nCompact feature-selection audit: PASS"
    )

    print(
        "Selected predictors:",
        len(
            compact_features
        ),
    )

    print(
        "Feature reduction:",
        f"{100.0 * (1 - len(compact_features) / EXPECTED_FULL_FEATURES):.1f}%",
    )


    print(
        "\nCompact predictors:"
    )


    for index, feature in enumerate(
        compact_features,
        start=1,
    ):

        print(
            f"{index:>2}. {feature}"
        )


    # =========================================================================
    # LORO evaluation
    # =========================================================================

    compact_outputs = run_compact_loro(
        dataframe=dataframe,
        compact_features=compact_features,
        output_dir=output_dir,
    )


    compact_results = compact_outputs[
        "pooled"
    ]


    audit_expected_reproduction(
        compact_results=compact_results,
        compact_features=compact_features,
    )


    # =========================================================================
    # Compare with experiments 1-8
    # =========================================================================

    comparison = compare_with_baselines(
        compact_results=compact_results,
        baseline_results_path=baseline_results_path,
        output_dir=output_dir,
    )


    # =========================================================================
    # Train final all-sample compact models
    # =========================================================================

    model_packages = train_final_compact_models(
        dataframe=dataframe,
        compact_features=compact_features,
        selection_metadata=selection_metadata,
        model_dir=model_dir,
    )


    # =========================================================================
    # Save experiment configuration
    # =========================================================================

    config_path = save_experiment_config(
        compact_features=compact_features,
        selection_metadata=selection_metadata,
        compact_results=compact_results,
        output_dir=output_dir,
    )


    # =========================================================================
    # Final summary
    # =========================================================================

    print(
        "\n" + "=" * 96
    )

    print(
        "COMPACT FUSION EXPERIMENT COMPLETE"
    )

    print(
        "=" * 96
    )


    summary_columns = [
        "algorithm_name",
        "n_features",
        "overall_accuracy",
        "macro_f1",
        "rrg_f1",
        "nffp_f1",
        "water_f1",
        "cohens_kappa",
        "total_rrg_nffp_errors",
        "fold_macro_f1_mean",
        "fold_macro_f1_std",
    ]


    print(
        compact_results[
            summary_columns
        ].to_string(
            index=False
        )
    )


    # =========================================================================
    # Operational model
    # =========================================================================

    xgb_result = (
        compact_results[
            compact_results[
                "algorithm"
            ]
            == "XGBOOST"
        ]
        .iloc[0]
    )


    print(
        "\nOperational ECHO classifier:"
    )

    print(
        "  XGBoost Compact S1-S2 Fusion"
    )

    print(
        "  Predictors:",
        len(
            compact_features
        ),
    )

    print(
        "  LORO macro F1:",
        f"{xgb_result['macro_f1']:.6f}",
    )

    print(
        "  Model package:",
        model_packages[
            "OPERATIONAL"
        ][
            "path"
        ],
    )


    print(
        "\nExperiment configuration:",
        config_path,
    )


    print(
        "\nImportant:"
    )

    print(
        "The final mapping model is fitted to all 525 "
        "2025-2026 reference polygons."
    )

    print(
        "Reported accuracy remains the spatially independent "
        "LORO estimate, not training accuracy."
    )


if __name__ == "__main__":
    main()