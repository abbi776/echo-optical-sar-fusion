#!/usr/bin/env python3
"""
Generate publication Figures 3, 4, 5 and 7 for the ECHO manuscript.

The figure script is deliberately separate from the analysis workflow. It
reads only outputs produced by the core analysis scripts and does not retrain
models or recalculate SHAP values.

Figures
-------
Figure 3
    Nine pooled XGBoost confusion matrices for the eight baseline
    configurations plus Compact Fusion.

Figure 4
    SHAP attribution summary:
        (a) sensor contribution overall and by class
        (b) feature-group contribution
        (c) seasonal contribution
        (d) top 15 individual predictors

Figure 5
    Signed class-specific SHAP effects for RRG, NFFP and Water.

Figure 7
    Regional RRG- and NFFP-classified area trajectories together with
    Darlington Point water-level and discharge context.

Usage
-----
Generate every figure:

    python figures/generate_publication_figures.py

Generate one figure only:

    python figures/generate_publication_figures.py --figure 3
    python figures/generate_publication_figures.py --figure 4
    python figures/generate_publication_figures.py --figure 5
    python figures/generate_publication_figures.py --figure 7

Outputs are written by default to:

    figures/output/

Each figure is exported as PDF, SVG and 600-dpi PNG.
"""

from __future__ import annotations

import argparse
import re
import warnings
from collections import OrderedDict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")


# =============================================================================
# Shared configuration
# =============================================================================

CLASS_ORDER = [
    "RRG",
    "NFFP",
    "Water",
]

S1_COLOUR = "#4C78A8"
S2_COLOUR = "#F58518"

SEASON_ORDER = [
    "Spring",
    "Winter",
    "Summer",
    "Autumn",
]

SHAP_COLOUR_MAP = getattr(
    shap.plots.colors,
    "red_blue",
    plt.get_cmap("coolwarm"),
)


# =============================================================================
# General helpers
# =============================================================================

def configure_matplotlib() -> None:
    """Set shared publication/export defaults."""

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Liberation Sans",
            "DejaVu Sans",
        ],
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(
    figure,
    output_base: Path,
    pad_inches: float = 0.03,
) -> None:
    """Export PDF, SVG and 600-dpi PNG versions of a figure."""

    output_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_base.with_suffix(".pdf"),
        format="pdf",
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white",
    )

    figure.savefig(
        output_base.with_suffix(".svg"),
        format="svg",
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white",
    )

    figure.savefig(
        output_base.with_suffix(".png"),
        format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white",
    )

    print("Saved:")
    print(" ", output_base.with_suffix(".pdf"))
    print(" ", output_base.with_suffix(".svg"))
    print(" ", output_base.with_suffix(".png"))


def clean_axis(
    axis,
    horizontal_grid: bool = False,
    linewidth: float = 0.8,
) -> None:
    """Apply the manuscript's clean publication-axis style."""

    axis.spines["left"].set_visible(True)
    axis.spines["bottom"].set_visible(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.spines["left"].set_color("#222222")
    axis.spines["bottom"].set_color("#222222")

    axis.spines["left"].set_linewidth(linewidth)
    axis.spines["bottom"].set_linewidth(linewidth)

    axis.tick_params(
        axis="both",
        which="major",
        length=3,
        width=linewidth,
        color="#222222",
        direction="out",
        top=False,
        right=False,
    )

    axis.grid(False)

    if horizontal_grid:
        axis.grid(
            axis="y",
            alpha=0.22,
            linewidth=0.6,
        )


def add_panel_label(
    axis,
    label: str,
    x: float = -0.16,
    y: float = 1.05,
    fontsize: float = 8,
    bold: bool = True,
) -> None:
    """Add a publication-style subplot label."""

    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        ha="left",
        va="bottom",
    )


def require_file(path: Path, label: str) -> Path:
    """Raise a clear error when a required input is absent."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}:\n{path}"
        )

    return path


# =============================================================================
# Figure 3 — nine pooled XGBoost confusion matrices
# =============================================================================

FIG3_PANEL_METADATA = [
    {
        "experiment": "S2_BANDS_ONLY",
        "display_name": "S2 bands (40)",
    },
    {
        "experiment": "S2_INDICES_ONLY",
        "display_name": "S2 indices (32)",
    },
    {
        "experiment": "FULL_S2",
        "display_name": "Full S2 (72)",
    },
    {
        "experiment": "S1_BACKSCATTER_ONLY",
        "display_name": "S1 backscatter (8)",
    },
    {
        "experiment": "S1_POLARISATION_FEATURES",
        "display_name": "S1 polarisation (40)",
    },
    {
        "experiment": "S1_TEXTURE_ONLY",
        "display_name": "S1 texture (56)",
    },
    {
        "experiment": "FULL_S1",
        "display_name": "Full S1 (104)",
    },
    {
        "experiment": "FULL_S1S2_FUSION",
        "display_name": "Full fusion (176)",
    },
    {
        "experiment": "COMPACT_S1S2_FUSION",
        "display_name": "Compact fusion (25)",
    },
]

FIG3_CONFUSION_NORM = mcolors.PowerNorm(
    gamma=0.45,
    vmin=0,
    vmax=100,
)


def _normalise_algorithm(value: str) -> str:
    value = str(value).strip().upper().replace("-", "_").replace(" ", "_")

    if value in {
        "XGBOOST",
        "XG_BOOST",
        "XGB",
    }:
        return "XGBOOST"

    if value in {
        "RF",
        "RANDOM_FOREST",
        "RANDOMFOREST",
    }:
        return "RF"

    return value


def _normalise_experiment(value: str) -> str:
    value = re.sub(
        r"[^A-Z0-9]+",
        "_",
        str(value).strip().upper(),
    ).strip("_")

    aliases = {
        "S1_POLARISATION_INDICES_ONLY": "S1_POLARISATION_FEATURES",
        "S1_POLARIZATION_INDICES_ONLY": "S1_POLARISATION_FEATURES",
        "S1_POLARISATION_ONLY": "S1_POLARISATION_FEATURES",
        "FULL_S1_S2_FUSION": "FULL_S1S2_FUSION",
        "SHAP_COMPACT_FUSION_25": "COMPACT_S1S2_FUSION",
        "COMPACT_FUSION": "COMPACT_S1S2_FUSION",
        "COMPACT_FUSION_25": "COMPACT_S1S2_FUSION",
    }

    return aliases.get(
        value,
        value,
    )


def _normalise_class(value: str) -> str:
    value = str(value).strip()

    aliases = {
        "RRG": "RRG",
        "Redgum": "RRG",
        "River Red Gum": "RRG",
        "NFFP": "NFFP",
        "Non-Forest Floodplain": "NFFP",
        "Non_Forest_Floodplain": "NFFP",
        "Water": "Water",
        "WATER": "Water",
        "water": "Water",
    }

    return aliases.get(
        value,
        value,
    )


def _load_figure3_predictions(
    baseline_file: Path,
    compact_file: Path,
) -> pd.DataFrame:
    """Combine held-out predictions from scripts 06 and 08."""

    baseline = pd.read_csv(
        require_file(
            baseline_file,
            "baseline LORO predictions",
        )
    )

    compact = pd.read_csv(
        require_file(
            compact_file,
            "Compact Fusion LORO predictions",
        )
    )

    predictions = pd.concat(
        [
            baseline,
            compact,
        ],
        ignore_index=True,
        sort=False,
    )

    required = {
        "experiment",
        "algorithm",
        "true_label",
        "predicted_label",
    }

    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            "Figure 3 predictions are missing columns: "
            + ", ".join(sorted(missing))
        )

    predictions["experiment"] = (
        predictions["experiment"]
        .apply(_normalise_experiment)
    )

    predictions["algorithm"] = (
        predictions["algorithm"]
        .apply(_normalise_algorithm)
    )

    predictions["true_label"] = (
        predictions["true_label"]
        .apply(_normalise_class)
    )

    predictions["predicted_label"] = (
        predictions["predicted_label"]
        .apply(_normalise_class)
    )

    return predictions[
        predictions["algorithm"] == "XGBOOST"
    ].copy()


def _draw_confusion_subplot(
    axis,
    counts: np.ndarray,
    percentages: np.ndarray,
    title: str,
):
    """Draw one pooled LORO confusion matrix."""

    image = axis.imshow(
        percentages,
        cmap="Blues",
        norm=FIG3_CONFUSION_NORM,
        interpolation="nearest",
        aspect="equal",
    )

    axis.set_xticks(
        np.arange(len(CLASS_ORDER))
    )
    axis.set_yticks(
        np.arange(len(CLASS_ORDER))
    )

    axis.set_xticklabels(
        CLASS_ORDER,
        fontsize=6,
    )
    axis.set_yticklabels(
        CLASS_ORDER,
        fontsize=6,
    )

    axis.tick_params(
        axis="both",
        which="major",
        length=0,
        pad=1.0,
    )

    axis.set_xticks(
        np.arange(
            -0.5,
            len(CLASS_ORDER),
            1,
        ),
        minor=True,
    )
    axis.set_yticks(
        np.arange(
            -0.5,
            len(CLASS_ORDER),
            1,
        ),
        minor=True,
    )

    axis.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=0.9,
    )

    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for true_i in range(len(CLASS_ORDER)):
        for predicted_i in range(len(CLASS_ORDER)):

            count = int(
                counts[
                    true_i,
                    predicted_i,
                ]
            )

            percentage = float(
                percentages[
                    true_i,
                    predicted_i,
                ]
            )

            text_colour = (
                "white"
                if FIG3_CONFUSION_NORM(
                    percentage
                ) > 0.55
                else "black"
            )

            axis.text(
                predicted_i,
                true_i,
                f"{count}",
                ha="center",
                va="center",
                fontsize=6,
                color=text_colour,
                fontweight="normal",
            )

    axis.set_title(
        title,
        fontsize=6.2,
        pad=1.0,
        fontweight="normal",
    )

    return image


def make_figure_3(
    baseline_predictions: Path,
    compact_predictions: Path,
    output_dir: Path,
    show: bool = False,
) -> None:
    """Generate final publication Figure 3."""

    print("\nGenerating Figure 3...")

    plt.rcParams.update({
        "font.size": 6.5,
        "axes.titlesize": 6.2,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.6,
    })

    predictions = _load_figure3_predictions(
        baseline_predictions,
        compact_predictions,
    )

    matrix_results = {}

    for metadata in FIG3_PANEL_METADATA:

        experiment = metadata["experiment"]

        subset = predictions[
            predictions["experiment"] == experiment
        ].copy()

        if len(subset) != 525:
            raise ValueError(
                f"Figure 3: {experiment} contains "
                f"{len(subset)} pooled XGBoost predictions; expected 525."
            )

        true_counts = (
            subset["true_label"]
            .value_counts()
            .reindex(
                CLASS_ORDER,
                fill_value=0,
            )
        )

        if not np.all(
            true_counts.to_numpy() == 175
        ):
            raise ValueError(
                f"Figure 3: {experiment} does not contain "
                "175 reference polygons per class."
            )

        counts = confusion_matrix(
            subset["true_label"],
            subset["predicted_label"],
            labels=CLASS_ORDER,
        )

        row_totals = counts.sum(
            axis=1,
            keepdims=True,
        )

        percentages = np.divide(
            counts,
            row_totals,
            out=np.zeros_like(
                counts,
                dtype=float,
            ),
            where=row_totals != 0,
        ) * 100.0

        if not np.all(
            counts.sum(axis=1) == 175
        ):
            raise ValueError(
                f"Figure 3: row totals failed for {experiment}."
            )

        matrix_results[
            experiment
        ] = {
            "counts": counts,
            "percentages": percentages,
        }

    figure, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(
            3.5,
            3.72,
        ),
        sharex=True,
        sharey=True,
    )

    shared_image = None

    for panel_index, metadata in enumerate(
        FIG3_PANEL_METADATA
    ):

        row_index = panel_index // 3
        column_index = panel_index % 3

        axis = axes[
            row_index,
            column_index,
        ]

        result = matrix_results[
            metadata["experiment"]
        ]

        shared_image = _draw_confusion_subplot(
            axis,
            result["counts"],
            result["percentages"],
            metadata["display_name"],
        )

    for axis in axes.flat:
        axis.label_outer()

    figure.supxlabel(
        "Predicted class",
        fontsize=7.5,
        x=0.48,
        y=0.03,
    )

    figure.supylabel(
        "Reference class",
        fontsize=7.5,
        x=0.03,
        y=0.51,
    )

    figure.subplots_adjust(
        left=0.13,
        right=0.84,
        bottom=0.12,
        top=0.95,
        wspace=0.16,
        hspace=0.20,
    )

    colourbar_axis = figure.add_axes([
        0.86,
        0.22,
        0.018,
        0.54,
    ])

    colourbar = figure.colorbar(
        shared_image,
        cax=colourbar_axis,
    )

    colourbar.set_label(
        "Reference-class percentage (%)",
        fontsize=6.5,
        labelpad=4,
    )

    colourbar.set_ticks([
        0,
        20,
        40,
        60,
        80,
        100,
    ])

    colourbar.ax.tick_params(
        labelsize=6,
        width=0.5,
        length=2,
    )

    output_base = (
        output_dir
        / "Figure_3_nine_XGBoost_confusion_matrices_final"
    )

    save_figure(
        figure,
        output_base,
        pad_inches=0.02,
    )

    if show:
        plt.show()

    plt.close(
        figure
    )


# =============================================================================
# Figures 4 and 5 — SHAP helpers
# =============================================================================

S2_PRETTY_NAMES = {
    "BLUE": "Blue",
    "GREEN": "Green",
    "RED": "Red",
    "RE1": "Red-edge 1",
    "RE2": "Red-edge 2",
    "RE3": "Red-edge 3",
    "NIR": "NIR",
    "NIR_NARROW": "Narrow NIR",
    "SWIR1": "SWIR1",
    "SWIR2": "SWIR2",
    "NDVI": "NDVI",
    "EVI": "EVI",
    "NDWI": "NDWI",
    "LSWI": "LSWI",
    "MSAVI2": "MSAVI2",
    "DBSI": "DBSI",
    "GNDVI": "GNDVI",
    "TVI": "TVI",
}

S1_PRETTY_NAMES = {
    "VV_DB": "VV backscatter",
    "VH_DB": "VH backscatter",
    "NDPI": "NDPI",
    "NRPB": "NRPB",
    "PR": "Polarisation ratio",
    "VV_VH_RATIO": "Polarisation ratio",
    "XPR": "Cross-polarisation ratio",
    "VH_VV_RATIO": "Cross-polarisation ratio",
    "RVI": "Radar Vegetation Index",
    "SUM": "Polarisation sum",
    "DIFF": "Polarisation difference",
    "PROD": "Polarisation product",
    "VDDPI": "VDDI",
    "VDDI": "VDDI",
    "LOG_RATIO": "Log polarisation ratio",
}

GLCM_PRETTY_NAMES = {
    "ASM": "GLCM ASM",
    "CORR": "GLCM correlation",
    "CORRELATION": "GLCM correlation",
    "VAR": "GLCM variance",
    "VARIANCE": "GLCM variance",
    "IDM": "GLCM IDM",
    "SUMAVE": "GLCM sum average",
    "ENTROPY": "GLCM entropy",
    "CONTRAST": "GLCM contrast",
}


def pretty_fusion_feature_name(
    feature: str,
) -> str:
    """Convert a raw annual predictor name into the manuscript label style."""

    original = str(feature)
    upper = original.upper()

    match = re.match(
        r"^(WINTER|SPRING|SUMMER|AUTUMN)_\d{4}_S2_(.+)$",
        upper,
    )

    if match:

        season = match.group(1).title()
        identity = match.group(2)

        positional_match = re.fullmatch(
            r"BAND_(\d+)",
            identity,
        )

        if positional_match:
            positional_order = [
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

            position = int(
                positional_match.group(1)
            )

            if 1 <= position <= len(positional_order):
                identity = positional_order[
                    position - 1
                ]

        feature_name = S2_PRETTY_NAMES.get(
            identity,
            identity.replace("_", " ").title(),
        )

        return (
            f"{season} – {feature_name} (S2)"
        )

    match = re.match(
        r"^(WINTER|SPRING|SUMMER|AUTUMN)_\d{4}_S1_(.+)$",
        upper,
    )

    if match:

        season = match.group(1).title()
        identity = match.group(2)

        if identity in S1_PRETTY_NAMES:
            feature_name = S1_PRETTY_NAMES[
                identity
            ]

        else:
            glcm_match = re.match(
                r"^GLCM_(ASM|CORR|CORRELATION|VAR|VARIANCE|IDM|SUMAVE|ENTROPY|CONTRAST)_(VV|VH)$",
                identity,
            )

            if glcm_match:
                metric = GLCM_PRETTY_NAMES[
                    glcm_match.group(1)
                ]
                polarisation = glcm_match.group(2)
                feature_name = (
                    f"{metric} ({polarisation})"
                )
            else:
                feature_name = (
                    identity
                    .replace("_", " ")
                    .title()
                )

        return (
            f"{season} – {feature_name} (S1)"
        )

    return (
        original
        .replace("_", " ")
        .strip()
    )


def _make_labels_unique(
    labels: list[str],
) -> list[str]:
    """Ensure abbreviated labels remain unique."""

    observed = {}
    output = []

    for label in labels:

        count = observed.get(
            label,
            0,
        ) + 1

        observed[
            label
        ] = count

        if count == 1:
            output.append(
                label
            )
        else:
            output.append(
                f"{label} [{count}]"
            )

    return output


def _load_shap_summary_tables(
    shap_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load descriptive SHAP outputs produced by script 07."""

    descriptive_dir = (
        shap_dir
        / "descriptive_all_sample"
    )

    paths = {
        "sensor_overall": (
            descriptive_dir
            / "shap_sensor_contribution_overall.csv"
        ),
        "sensor_by_class": (
            descriptive_dir
            / "shap_sensor_contribution_by_class.csv"
        ),
        "feature_group": (
            descriptive_dir
            / "shap_feature_group_contribution.csv"
        ),
        "season": (
            descriptive_dir
            / "shap_season_contribution.csv"
        ),
        "top15": (
            descriptive_dir
            / "shap_top15_predictors_overall.csv"
        ),
        "signed": (
            descriptive_dir
            / "shap_signed_values_all_samples.csv"
        ),
    }

    return {
        name: pd.read_csv(
            require_file(
                path,
                f"SHAP {name} table",
            )
        )
        for name, path in paths.items()
    }


# =============================================================================
# Figure 4 — four-panel SHAP attribution
# =============================================================================

FIG4_FEATURE_GROUP_COLOURS = {
    "S2 bands": S2_COLOUR,
    "S2 indices": S2_COLOUR,
    "S1 texture": S1_COLOUR,
    "S1 backscatter": S1_COLOUR,
    "S1 polarisation": S1_COLOUR,
}

FIG4_SEASON_COLOUR = "#7F7F7F"


def _prepare_sensor_by_class(
    sensor_overall: pd.DataFrame,
    sensor_by_class: pd.DataFrame,
) -> pd.DataFrame:
    """Create Overall/RRG/NFFP/Water S1/S2 contribution table."""

    rows = []

    overall_lookup = (
        sensor_overall
        .set_index("sensor")[
            "relative_contribution_pct"
        ]
        .to_dict()
    )

    rows.append({
        "Class": "Overall",
        "S1": float(
            overall_lookup.get(
                "S1",
                0.0,
            )
        ),
        "S2": float(
            overall_lookup.get(
                "S2",
                0.0,
            )
        ),
    })

    for class_name in CLASS_ORDER:

        subset = sensor_by_class[
            sensor_by_class["class"]
            == class_name
        ]

        lookup = (
            subset
            .set_index("sensor")[
                "relative_contribution_pct"
            ]
            .to_dict()
        )

        rows.append({
            "Class": class_name,
            "S1": float(
                lookup.get(
                    "S1",
                    0.0,
                )
            ),
            "S2": float(
                lookup.get(
                    "S2",
                    0.0,
                )
            ),
        })

    return pd.DataFrame(
        rows
    )


def _draw_figure4_panel_a(
    axis,
    plot_df: pd.DataFrame,
) -> None:
    """Sensor contribution overall and by class."""

    y = np.arange(
        len(plot_df)
    )

    s1_values = plot_df[
        "S1"
    ].to_numpy()

    s2_values = plot_df[
        "S2"
    ].to_numpy()

    axis.barh(
        y,
        s1_values,
        color=S1_COLOUR,
        edgecolor="white",
        linewidth=0.7,
    )

    axis.barh(
        y,
        s2_values,
        left=s1_values,
        color=S2_COLOUR,
        edgecolor="white",
        linewidth=0.7,
    )

    for row_index, (
        s1_value,
        s2_value,
    ) in enumerate(
        zip(
            s1_values,
            s2_values,
        )
    ):

        if s1_value >= 4:

            if s1_value < 12:
                axis.text(
                    s1_value / 2,
                    row_index,
                    f"{s1_value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="white",
                    fontweight="normal",
                    clip_on=False,
                )
            else:
                axis.text(
                    s1_value / 2,
                    row_index,
                    f"{s1_value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white",
                    fontweight="bold",
                )

        if s2_value >= 4:
            axis.text(
                s1_value + s2_value / 2,
                row_index,
                f"{s2_value:.1f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white",
                fontweight="bold",
            )

    axis.set_yticks(
        y
    )

    axis.set_yticklabels(
        plot_df["Class"],
        fontsize=9.2,
    )

    axis.invert_yaxis()
    axis.set_xlim(
        0,
        100,
    )
    axis.xaxis.set_major_locator(
        MultipleLocator(20)
    )

    axis.set_xlabel(
        "Relative contribution (%)",
        fontsize=10,
        labelpad=4,
    )

    axis.set_title(
        "Sensor contribution overall and by class",
        fontsize=10.5,
        pad=5,
    )

    clean_axis(
        axis
    )

    add_panel_label(
        axis,
        "(a)",
        x=-0.18,
        y=1.11,
        fontsize=11,
        bold=True,
    )


def _draw_figure4_panel_b(
    axis,
    feature_group: pd.DataFrame,
) -> None:
    """Feature-group contribution panel."""

    preferred_order = [
        "S2 bands",
        "S2 indices",
        "S1 texture",
        "S1 backscatter",
        "S1 polarisation",
    ]

    plot_df = (
        feature_group
        .set_index("feature_group")
        .reindex(preferred_order)
        .dropna()
        .reset_index()
    )

    values = plot_df[
        "relative_contribution_pct"
    ].to_numpy()

    labels = plot_df[
        "feature_group"
    ].tolist()

    colours = [
        FIG4_FEATURE_GROUP_COLOURS.get(
            label,
            "#A7A7A7",
        )
        for label in labels
    ]

    y = np.arange(
        len(plot_df)
    )

    axis.barh(
        y,
        values,
        color=colours,
        edgecolor="white",
        linewidth=0.7,
    )

    axis.set_yticks(
        y
    )
    axis.set_yticklabels(
        labels,
        fontsize=9.0,
    )
    axis.invert_yaxis()

    axis.set_xlim(
        0,
        max(
            60,
            float(np.max(values)) + 8,
        ),
    )

    axis.xaxis.set_major_locator(
        MultipleLocator(20)
    )

    axis.set_xlabel(
        "Relative contribution (%)",
        fontsize=10,
        labelpad=4,
    )

    axis.set_title(
        "Feature-group contribution",
        fontsize=10.5,
        pad=5,
    )

    for row_index, value in enumerate(
        values
    ):
        axis.text(
            value + 0.7,
            row_index,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=8.8,
        )

    clean_axis(
        axis
    )

    add_panel_label(
        axis,
        "(b)",
        x=-0.32,
        y=1.11,
        fontsize=11,
        bold=True,
    )


def _draw_figure4_panel_c(
    axis,
    season_summary: pd.DataFrame,
) -> None:
    """Seasonal SHAP contribution panel."""

    plot_df = (
        season_summary
        .set_index("season")
        .reindex(SEASON_ORDER)
        .reset_index()
    )

    values = plot_df[
        "relative_contribution_pct"
    ].fillna(0.0).to_numpy()

    x = np.arange(
        len(plot_df)
    )

    axis.bar(
        x,
        values,
        width=0.62,
        color=FIG4_SEASON_COLOUR,
        edgecolor="white",
        linewidth=0.7,
    )

    axis.set_xticks(
        x
    )
    axis.set_xticklabels(
        plot_df["season"],
        fontsize=8.8,
    )

    axis.set_ylim(
        0,
        max(
            45,
            float(np.max(values)) + 7,
        ),
    )

    axis.set_ylabel(
        "Relative contribution (%)",
        fontsize=10,
    )

    axis.set_title(
        "Seasonal contribution",
        fontsize=10.5,
        pad=5,
    )

    for x_index, value in zip(
        x,
        values,
    ):
        axis.text(
            x_index,
            value + 0.8,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
        )

    clean_axis(
        axis
    )

    add_panel_label(
        axis,
        "(c)",
        x=-0.16,
        y=1.08,
        fontsize=11,
        bold=True,
    )


def _draw_figure4_panel_d(
    axis,
    top15: pd.DataFrame,
) -> None:
    """Top-15 overall predictors by mean absolute SHAP."""

    plot_df = (
        top15
        .sort_values(
            "mean_abs_shap",
            ascending=True,
        )
        .copy()
    )

    values = plot_df[
        "mean_abs_shap"
    ].to_numpy()

    labels = _make_labels_unique([
        pretty_fusion_feature_name(
            feature
        )
        for feature in plot_df[
            "feature"
        ].tolist()
    ])

    sensors = plot_df[
        "sensor"
    ].tolist()

    colours = [
        S1_COLOUR
        if sensor == "S1"
        else S2_COLOUR
        for sensor in sensors
    ]

    y = np.arange(
        len(plot_df)
    )

    axis.barh(
        y,
        values,
        color=colours,
        edgecolor="white",
        linewidth=0.5,
        height=0.70,
    )

    axis.set_yticks(
        y
    )
    axis.set_yticklabels(
        labels,
        fontsize=7.8,
    )

    axis.set_xlabel(
        "Mean |SHAP| across class outputs",
        fontsize=10,
        labelpad=5,
    )

    axis.set_title(
        "Top 15 individual predictors",
        fontsize=10.5,
        pad=5,
    )

    clean_axis(
        axis
    )

    add_panel_label(
        axis,
        "(d)",
        x=-0.32,
        y=1.08,
        fontsize=11,
        bold=True,
    )


def make_figure_4(
    shap_dir: Path,
    output_dir: Path,
    show: bool = False,
) -> None:
    """Generate final publication Figure 4."""

    print("\nGenerating Figure 4...")

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
    })

    tables = _load_shap_summary_tables(
        shap_dir
    )

    sensor_plot = _prepare_sensor_by_class(
        tables["sensor_overall"],
        tables["sensor_by_class"],
    )

    figure = plt.figure(
        figsize=(
            7.16,
            7.25,
        )
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[
            0.82,
            1.40,
        ],
        width_ratios=[
            1.0,
            1.15,
        ],
        hspace=0.47,
        wspace=0.68,
    )

    axis_a = figure.add_subplot(
        grid[0, 0]
    )
    axis_b = figure.add_subplot(
        grid[0, 1]
    )
    axis_c = figure.add_subplot(
        grid[1, 0]
    )
    axis_d = figure.add_subplot(
        grid[1, 1]
    )

    _draw_figure4_panel_a(
        axis_a,
        sensor_plot,
    )

    _draw_figure4_panel_b(
        axis_b,
        tables["feature_group"],
    )

    _draw_figure4_panel_c(
        axis_c,
        tables["season"],
    )

    _draw_figure4_panel_d(
        axis_d,
        tables["top15"],
    )

    sensor_legend_handles = [
        Patch(
            facecolor=S1_COLOUR,
            edgecolor="none",
            label="Sentinel-1",
        ),
        Patch(
            facecolor=S2_COLOUR,
            edgecolor="none",
            label="Sentinel-2",
        ),
    ]

    figure.legend(
        handles=sensor_legend_handles,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            0.012,
        ),
        columnspacing=1.5,
        handlelength=1.4,
        handletextpad=0.5,
        fontsize=9.0,
    )

    figure.subplots_adjust(
        left=0.12,
        right=0.97,
        top=0.95,
        bottom=0.105,
    )

    output_base = (
        output_dir
        / "Figure_4_SHAP_attribution_final_bar"
    )

    save_figure(
        figure,
        output_base,
    )

    if show:
        plt.show()

    plt.close(
        figure
    )


# =============================================================================
# Figure 5 — signed class-specific SHAP effects
# =============================================================================

def _remove_vertical_zero_lines(
    axis,
) -> None:
    """Remove automatic SHAP zero lines before adding one controlled line."""

    for line in list(
        axis.lines
    ):

        x_data = np.asarray(
            line.get_xdata(),
            dtype=float,
        )

        if (
            x_data.size > 0
            and np.allclose(
                x_data,
                x_data[0],
            )
            and np.isclose(
                x_data[0],
                0.0,
            )
        ):
            line.remove()


def _prepare_figure5_data(
    signed_shap: pd.DataFrame,
    max_display: int = 10,
) -> tuple[dict, float]:
    """Rebuild the three class-specific SHAP matrices from saved long data."""

    required = {
        "unique_id",
        "class",
        "feature",
        "feature_value",
        "shap_value",
    }

    missing = required - set(
        signed_shap.columns
    )

    if missing:
        raise ValueError(
            "Figure 5 signed-SHAP table is missing: "
            + ", ".join(sorted(missing))
        )

    output = {}
    signed_maxima = []

    for class_name in CLASS_ORDER:

        class_df = signed_shap[
            signed_shap["class"]
            == class_name
        ].copy()

        if class_df.empty:
            raise ValueError(
                f"Figure 5: no signed SHAP data for {class_name}."
            )

        ranking = (
            class_df
            .assign(
                abs_shap=lambda frame:
                    frame["shap_value"].abs()
            )
            .groupby(
                "feature"
            )["abs_shap"]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        top_features = ranking.index[
            :max_display
        ].tolist()

        shap_matrix = (
            class_df[
                class_df["feature"].isin(
                    top_features
                )
            ]
            .pivot(
                index="unique_id",
                columns="feature",
                values="shap_value",
            )
            .reindex(
                columns=top_features
            )
        )

        feature_matrix = (
            class_df[
                class_df["feature"].isin(
                    top_features
                )
            ]
            .pivot(
                index="unique_id",
                columns="feature",
                values="feature_value",
            )
            .reindex(
                index=shap_matrix.index,
                columns=top_features,
            )
        )

        if (
            shap_matrix.isna().any().any()
            or feature_matrix.isna().any().any()
        ):
            raise ValueError(
                f"Figure 5: incomplete SHAP matrix for {class_name}."
            )

        readable_names = _make_labels_unique([
            pretty_fusion_feature_name(
                feature
            )
            for feature in top_features
        ])

        feature_matrix.columns = (
            readable_names
        )

        output[
            class_name
        ] = {
            "values": shap_matrix.to_numpy(
                dtype=float
            ),
            "X": feature_matrix,
        }

        signed_maxima.append(
            float(
                np.nanmax(
                    np.abs(
                        shap_matrix.to_numpy(
                            dtype=float
                        )
                    )
                )
            )
        )

    shared_limit = (
        max(
            signed_maxima
        ) * 1.06
    )

    return (
        output,
        shared_limit,
    )


def _draw_figure5_panel(
    axis,
    class_name: str,
    panel_label: str,
    plot_data: dict,
    shared_limit: float,
) -> None:
    """Draw one signed class-specific SHAP beeswarm panel."""

    explanation = shap.Explanation(
        values=np.asarray(
            plot_data["values"],
            dtype=float,
        ),
        data=np.asarray(
            plot_data["X"],
            dtype=float,
        ),
        feature_names=list(
            plot_data["X"].columns
        ),
    )

    shap.plots.beeswarm(
        explanation,
        max_display=plot_data[
            "X"
        ].shape[1],
        show=False,
        ax=axis,
        plot_size=None,
        color_bar=False,
        group_remaining_features=False,
        s=12,
    )

    axis.set_xlim(
        -shared_limit,
        shared_limit,
    )

    _remove_vertical_zero_lines(
        axis
    )

    axis.axvline(
        0,
        color="#666666",
        linewidth=0.75,
        linestyle="-",
        zorder=0,
    )

    axis.set_title(
        f"{class_name}-class SHAP feature effects",
        fontsize=10,
        fontweight="normal",
        pad=7,
    )

    axis.set_xlabel(
        f"SHAP value for the {class_name} model output",
        fontsize=9,
        labelpad=5,
    )

    axis.set_ylabel("")

    axis.tick_params(
        axis="x",
        labelsize=8,
    )

    axis.tick_params(
        axis="y",
        labelsize=8,
        pad=2.5,
    )

    add_panel_label(
        axis,
        panel_label,
        x=-0.10,
        y=1.05,
        fontsize=9,
        bold=False,
    )

    clean_axis(
        axis,
        horizontal_grid=False,
        linewidth=0.8,
    )


def make_figure_5(
    shap_dir: Path,
    output_dir: Path,
    show: bool = False,
) -> None:
    """Generate final publication Figure 5."""

    print("\nGenerating Figure 5...")

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
    })

    tables = _load_shap_summary_tables(
        shap_dir
    )

    class_data, shared_limit = (
        _prepare_figure5_data(
            tables["signed"],
            max_display=10,
        )
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(
            12.0,
            3.44,
        ),
    )

    for axis, class_name, panel_label in zip(
        axes,
        CLASS_ORDER,
        [
            "(a)",
            "(b)",
            "(c)",
        ],
    ):
        _draw_figure5_panel(
            axis,
            class_name,
            panel_label,
            class_data[
                class_name
            ],
            shared_limit,
        )

    figure.subplots_adjust(
        left=0.08,
        right=0.90,
        top=0.88,
        bottom=0.18,
        wspace=0.46,
    )

    panel_c_position = (
        axes[2].get_position()
    )

    axes[2].set_position([
        panel_c_position.x0 + 0.012,
        panel_c_position.y0,
        panel_c_position.width - 0.004,
        panel_c_position.height,
    ])

    colourbar_axis = figure.add_axes([
        0.935,
        0.18,
        0.012,
        0.66,
    ])

    scalar_mappable = ScalarMappable(
        norm=Normalize(
            vmin=0,
            vmax=1,
        ),
        cmap=SHAP_COLOUR_MAP,
    )

    scalar_mappable.set_array([])

    colourbar = figure.colorbar(
        scalar_mappable,
        cax=colourbar_axis,
    )

    colourbar.set_ticks([
        0,
        1,
    ])

    colourbar.set_ticklabels([
        "Low",
        "High",
    ])

    colourbar.set_label(
        "Feature value",
        fontsize=9,
        labelpad=7,
    )

    colourbar.ax.tick_params(
        labelsize=8,
        length=0,
    )

    for spine in colourbar_axis.spines.values():
        spine.set_visible(
            False
        )

    output_base = (
        output_dir
        / "Figure_5_class_specific_SHAP_oldstyle_12in_final"
    )

    save_figure(
        figure,
        output_base,
    )

    if show:
        plt.show()

    plt.close(
        figure
    )


# =============================================================================
# Figure 7 — temporal classified area and hydrological context
# =============================================================================

FIG7_PERIODS = OrderedDict([
    (
        "2017_2018",
        {
            "label": "2017–2018",
            "short_label": "2017–18",
            "start": pd.Timestamp("2017-07-01"),
            "end": pd.Timestamp("2018-06-30"),
        },
    ),
    (
        "2022_2023",
        {
            "label": "2022–2023",
            "short_label": "2022–23",
            "start": pd.Timestamp("2022-07-01"),
            "end": pd.Timestamp("2023-06-30"),
        },
    ),
    (
        "2025_2026",
        {
            "label": "2025–2026",
            "short_label": "2025–26",
            "start": pd.Timestamp("2025-07-01"),
            "end": pd.Timestamp("2026-06-30"),
        },
    ),
])

FIG7_REGION_ORDER = [
    "SR1",
    "SR2",
    "SR3",
    "SR4",
    "SR5",
]

FIG7_REGION_COLOURS = {
    "SR1": "#1B9E77",
    "SR2": "#D95F02",
    "SR3": "#7570B3",
    "SR4": "#E7298A",
    "SR5": "#66A61E",
}

FIG7_REGION_MARKERS = {
    "SR1": "o",
    "SR2": "s",
    "SR3": "^",
    "SR4": "D",
    "SR5": "v",
}

FIG7_WINDOW_COLOURS = {
    "2017_2018": "#FDE0C5",
    "2022_2023": "#D9EAF7",
    "2025_2026": "#DCEFD7",
}

FIG7_LEVEL_DAILY_COLOUR = "#9ECAE1"
FIG7_LEVEL_MEAN_COLOUR = "#08519C"
FIG7_DISCHARGE_DAILY_COLOUR = "#FCBBA1"
FIG7_DISCHARGE_MEAN_COLOUR = "#CB181D"


def _normalise_period(value: str) -> str:
    """Convert period variants to project keys."""

    text = str(value).strip()

    replacements = {
        "2017-2018": "2017_2018",
        "2017–2018": "2017_2018",
        "2017_2018": "2017_2018",
        "2022-2023": "2022_2023",
        "2022–2023": "2022_2023",
        "2022_2023": "2022_2023",
        "2025-2026": "2025_2026",
        "2025–2026": "2025_2026",
        "2025_2026": "2025_2026",
    }

    return replacements.get(
        text,
        text,
    )


def _load_figure7_area(
    area_file: Path,
) -> pd.DataFrame:
    """Load regional class-area output from script 09."""

    area = pd.read_csv(
        require_file(
            area_file,
            "regional temporal class-area table",
        )
    )

    required = {
        "region",
        "period",
        "class_label",
        "area_ha",
        "valid_region_area_ha",
    }

    missing = required - set(
        area.columns
    )

    if missing:
        raise ValueError(
            "Figure 7 regional area table is missing: "
            + ", ".join(sorted(missing))
        )

    area = area[
        [
            "region",
            "period",
            "class_label",
            "area_ha",
            "valid_region_area_ha",
        ]
    ].copy()

    area["region"] = (
        area["region"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(
            r"^P([1-5])$",
            r"SR\1",
            regex=True,
        )
    )

    area["period"] = (
        area["period"]
        .apply(
            _normalise_period
        )
    )

    area["class_label"] = (
        area["class_label"]
        .apply(
            _normalise_class
        )
    )

    area["area_ha"] = pd.to_numeric(
        area["area_ha"],
        errors="raise",
    )

    area["valid_region_area_ha"] = pd.to_numeric(
        area["valid_region_area_ha"],
        errors="raise",
    )

    observed_regions = set(
        area["region"]
    )

    if not set(
        FIG7_REGION_ORDER
    ).issubset(
        observed_regions
    ):
        raise ValueError(
            "Figure 7 is missing one or more SR1-SR5 regions."
        )

    reconciliation = (
        area
        .groupby(
            [
                "region",
                "period",
            ],
            as_index=False,
        )
        .agg(
            summed_class_area_ha=(
                "area_ha",
                "sum",
            ),
            valid_region_area_ha=(
                "valid_region_area_ha",
                "first",
            ),
        )
    )

    reconciliation[
        "difference_ha"
    ] = (
        reconciliation[
            "summed_class_area_ha"
        ]
        - reconciliation[
            "valid_region_area_ha"
        ]
    )

    maximum_error = float(
        reconciliation[
            "difference_ha"
        ]
        .abs()
        .max()
    )

    if maximum_error > 0.05:
        raise ValueError(
            "Figure 7 class areas do not reconcile with "
            f"valid regional area (max error {maximum_error:.3f} ha)."
        )

    return area


def _discover_gauge_file(
    data_root: Path,
    gauge_file: Path | None,
) -> Path:
    """Resolve the WaterNSW Darlington Point gauge CSV."""

    if gauge_file is not None:
        return require_file(
            gauge_file,
            "WaterNSW gauge CSV",
        )

    hydrology_directories = [
        data_root / "hydrology",
        data_root / "Hydrology",
    ]

    candidates = []

    for directory in hydrology_directories:
        candidates.extend([
            directory
            / "Darlington_Point_410021_daily_level_and_discharge.csv",
            directory
            / "Darlington_Point_410021_daily_discharge.csv",
        ])

    existing = [
        path
        for path in candidates
        if path.exists()
    ]

    if len(existing) == 1:
        return existing[0]

    discovered = []

    for directory in hydrology_directories:
        if directory.exists():
            discovered.extend(
                sorted(
                    directory.glob(
                        "*410021*.csv"
                    )
                )
            )

    discovered = list(
        dict.fromkeys(
            discovered
        )
    )

    if len(discovered) == 1:
        return discovered[0]

    if not discovered:
        raise FileNotFoundError(
            "Could not find a Darlington Point 410021 gauge CSV. "
            "Supply it explicitly with --gauge-file."
        )

    raise RuntimeError(
        "Multiple possible 410021 gauge CSV files were found. "
        "Supply the intended file with --gauge-file.\n"
        + "\n".join(
            str(path)
            for path in discovered
        )
    )


def _load_gauge_file(
    gauge_file: Path,
) -> pd.DataFrame:
    """Read and clean the WaterNSW daily level/discharge export."""

    # First try the original WaterNSW export layout used for the paper:
    # header information followed by a row beginning 'Date and time'.
    try:
        raw = pd.read_csv(
            gauge_file,
            header=None,
            dtype=str,
            encoding="utf-8-sig",
            engine="python",
        )
    except UnicodeDecodeError:
        raw = pd.read_csv(
            gauge_file,
            header=None,
            dtype=str,
            encoding="cp1252",
            engine="python",
        )

    header_matches = []

    if raw.shape[1] >= 5:
        first_column = (
            raw.iloc[:, 0]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        header_matches = raw.index[
            first_column.eq(
                "date and time"
            )
        ].tolist()

    if header_matches:

        start = int(
            header_matches[0]
        ) + 1

        gauge = raw.iloc[
            start:,
            [
                0,
                1,
                2,
                3,
                4,
            ],
        ].copy()

        gauge.columns = [
            "date",
            "water_level_m",
            "level_quality",
            "discharge_ml_day",
            "discharge_quality",
        ]

    else:
        # Also accept a clean CSV with named columns.
        clean = pd.read_csv(
            gauge_file
        )

        normalised = {
            re.sub(
                r"[^a-z0-9]+",
                "_",
                str(column).strip().lower(),
            ).strip("_"): column
            for column in clean.columns
        }

        def find_column(candidates):
            for candidate in candidates:
                if candidate in normalised:
                    return normalised[
                        candidate
                    ]
            return None

        date_column = find_column([
            "date",
            "date_and_time",
            "datetime",
        ])

        level_column = find_column([
            "water_level_m",
            "level_mean_metres",
            "level_mean_m",
            "level",
        ])

        discharge_column = find_column([
            "discharge_ml_day",
            "discharge_mean_ml_day",
            "discharge",
        ])

        if (
            date_column is None
            or level_column is None
            or discharge_column is None
        ):
            raise ValueError(
                "Could not identify date, water-level and discharge "
                "columns in the gauge CSV."
            )

        level_quality_column = find_column([
            "level_quality",
            "water_level_quality",
        ])

        discharge_quality_column = find_column([
            "discharge_quality",
        ])

        gauge = pd.DataFrame({
            "date": clean[
                date_column
            ],
            "water_level_m": clean[
                level_column
            ],
            "level_quality": (
                clean[
                    level_quality_column
                ]
                if level_quality_column is not None
                else np.nan
            ),
            "discharge_ml_day": clean[
                discharge_column
            ],
            "discharge_quality": (
                clean[
                    discharge_quality_column
                ]
                if discharge_quality_column is not None
                else np.nan
            ),
        })

    gauge["date"] = pd.to_datetime(
        gauge["date"],
        errors="coerce",
        dayfirst=True,
    )

    for column in [
        "water_level_m",
        "level_quality",
        "discharge_ml_day",
        "discharge_quality",
    ]:
        gauge[column] = pd.to_numeric(
            gauge[column],
            errors="coerce",
        )

    gauge = gauge.dropna(
        subset=[
            "date",
        ]
    )

    gauge.loc[
        gauge["level_quality"].eq(255),
        "water_level_m",
    ] = np.nan

    gauge.loc[
        gauge["discharge_quality"].eq(255),
        "discharge_ml_day",
    ] = np.nan

    gauge.loc[
        (~np.isfinite(gauge["water_level_m"]))
        | (gauge["water_level_m"] < 0),
        "water_level_m",
    ] = np.nan

    gauge.loc[
        (~np.isfinite(gauge["discharge_ml_day"]))
        | (gauge["discharge_ml_day"] < 0),
        "discharge_ml_day",
    ] = np.nan

    gauge = gauge.dropna(
        subset=[
            "water_level_m",
            "discharge_ml_day",
        ],
        how="all",
    )

    gauge = (
        gauge
        .sort_values(
            "date"
        )
        .drop_duplicates(
            subset=[
                "date",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    if gauge.empty:
        raise ValueError(
            "No valid hydrological observations remained after cleaning."
        )

    indexed = (
        gauge
        .set_index(
            "date"
        )
        .sort_index()
    )

    indexed[
        "water_level_30d"
    ] = (
        indexed[
            "water_level_m"
        ]
        .rolling(
            window="30D",
            min_periods=1,
        )
        .mean()
    )

    indexed[
        "discharge_30d"
    ] = (
        indexed[
            "discharge_ml_day"
        ]
        .rolling(
            window="30D",
            min_periods=1,
        )
        .mean()
    )

    return indexed.reset_index()


def _draw_regional_area_trajectory(
    axis,
    area: pd.DataFrame,
    class_label: str,
    title: str,
    period_x: list[pd.Timestamp],
    show_xlabel: bool,
) -> None:
    """Draw RRG or NFFP area trajectories for SR1-SR5."""

    period_order = list(
        FIG7_PERIODS.keys()
    )

    class_data = area[
        area["class_label"]
        == class_label
    ].copy()

    plotted_values = []

    for region in FIG7_REGION_ORDER:

        region_data = (
            class_data[
                class_data["region"]
                == region
            ]
            .set_index(
                "period"
            )
            .reindex(
                period_order
            )
            .reset_index()
        )

        if region_data[
            "area_ha"
        ].isna().any():
            raise ValueError(
                f"Figure 7: missing {class_label} area for {region}."
            )

        values = (
            region_data[
                "area_ha"
            ].to_numpy()
            / 1000.0
        )

        plotted_values.extend(
            values.tolist()
        )

        axis.plot(
            period_x,
            values,
            marker=FIG7_REGION_MARKERS[
                region
            ],
            markersize=3.4,
            linewidth=1.1,
            color=FIG7_REGION_COLOURS[
                region
            ],
            label=region,
        )

    axis.set_xticks(
        period_x
    )

    axis.set_xticklabels([
        FIG7_PERIODS[
            period
        ]["short_label"]
        for period in period_order
    ])

    axis.tick_params(
        axis="x",
        which="minor",
        bottom=False,
    )

    axis.set_xlim(
        pd.Timestamp(
            "2017-04-01"
        ),
        pd.Timestamp(
            "2026-09-01"
        ),
    )

    maximum_value = float(
        np.nanmax(
            plotted_values
        )
    )

    axis.set_ylim(
        0,
        maximum_value * 1.16,
    )

    if show_xlabel:
        axis.set_xlabel(
            "Observation period"
        )

    axis.set_ylabel(
        f"{class_label}-classified\n"
        "area (10³ ha)"
    )

    axis.set_title(
        title,
        pad=4,
    )

    clean_axis(
        axis,
        horizontal_grid=True,
        linewidth=0.7,
    )


def make_figure_7(
    area_file: Path,
    data_root: Path,
    gauge_file: Path | None,
    output_dir: Path,
    show: bool = False,
) -> None:
    """Generate final publication Figure 7."""

    print("\nGenerating Figure 7...")

    plt.rcParams.update({
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.7,
    })

    area = _load_figure7_area(
        area_file
    )

    resolved_gauge = _discover_gauge_file(
        data_root,
        gauge_file,
    )

    print(
        "Gauge file:",
        resolved_gauge,
    )

    gauge = _load_gauge_file(
        resolved_gauge
    )

    period_order = list(
        FIG7_PERIODS.keys()
    )

    period_midpoints = {
        period_key: (
            FIG7_PERIODS[
                period_key
            ]["start"]
            + (
                FIG7_PERIODS[
                    period_key
                ]["end"]
                - FIG7_PERIODS[
                    period_key
                ]["start"]
            ) / 2
        )
        for period_key in period_order
    }

    period_x = [
        period_midpoints[
            period_key
        ]
        for period_key in period_order
    ]

    figure = plt.figure(
        figsize=(
            3.5,
            6.4,
        )
    )

    outer_grid = figure.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[
            1.0,
            1.0,
            1.75,
        ],
        hspace=0.85,
    )

    axis_rrg = figure.add_subplot(
        outer_grid[0, 0]
    )

    axis_nffp = figure.add_subplot(
        outer_grid[1, 0]
    )

    hydrology_grid = (
        outer_grid[2, 0]
        .subgridspec(
            nrows=2,
            ncols=1,
            height_ratios=[
                1.0,
                1.0,
            ],
            hspace=0.10,
        )
    )

    axis_level = figure.add_subplot(
        hydrology_grid[0, 0]
    )

    axis_discharge = figure.add_subplot(
        hydrology_grid[1, 0],
        sharex=axis_level,
    )

    _draw_regional_area_trajectory(
        axis_rrg,
        area,
        class_label="RRG",
        title="RRG-classified area",
        period_x=period_x,
        show_xlabel=False,
    )

    _draw_regional_area_trajectory(
        axis_nffp,
        area,
        class_label="NFFP",
        title="NFFP-classified area",
        period_x=period_x,
        show_xlabel=True,
    )

    add_panel_label(
        axis_rrg,
        "(a)",
        x=-0.16,
        y=1.05,
        fontsize=8,
        bold=True,
    )

    add_panel_label(
        axis_nffp,
        "(b)",
        x=-0.16,
        y=1.05,
        fontsize=8,
        bold=True,
    )

    region_handles = [
        Line2D(
            [0],
            [0],
            color=FIG7_REGION_COLOURS[
                region
            ],
            marker=FIG7_REGION_MARKERS[
                region
            ],
            linewidth=1.1,
            markersize=3.4,
            label=region,
        )
        for region in FIG7_REGION_ORDER
    ]

    axis_nffp.legend(
        handles=region_handles,
        labels=FIG7_REGION_ORDER,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.42,
        ),
        columnspacing=0.9,
        handlelength=1.3,
        handletextpad=0.3,
    )

    for period_key in period_order:

        period = FIG7_PERIODS[
            period_key
        ]

        for axis in [
            axis_level,
            axis_discharge,
        ]:

            axis.axvspan(
                period["start"],
                period["end"],
                color=FIG7_WINDOW_COLOURS[
                    period_key
                ],
                alpha=0.40,
                linewidth=0,
                zorder=0,
            )

            axis.axvline(
                period["start"],
                color="#777777",
                linewidth=0.5,
                linestyle="--",
                alpha=0.55,
                zorder=1,
            )

            axis.axvline(
                period["end"],
                color="#777777",
                linewidth=0.5,
                linestyle="--",
                alpha=0.55,
                zorder=1,
            )

    axis_level.plot(
        gauge["date"],
        gauge["water_level_m"],
        color=FIG7_LEVEL_DAILY_COLOUR,
        linewidth=0.35,
        alpha=0.55,
        label="Daily water level",
        zorder=2,
    )

    axis_level.plot(
        gauge["date"],
        gauge["water_level_30d"],
        color=FIG7_LEVEL_MEAN_COLOUR,
        linewidth=1.0,
        label="30-day mean",
        zorder=3,
    )

    axis_level.set_ylabel(
        "Water level (m)"
    )

    axis_level.set_title(
        "Hydrological context",
        pad=4,
    )

    axis_level.set_ylim(
        bottom=0
    )

    clean_axis(
        axis_level,
        horizontal_grid=True,
        linewidth=0.7,
    )

    axis_level.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
        handlelength=1.3,
        handletextpad=0.3,
        columnspacing=0.7,
    )

    add_panel_label(
        axis_level,
        "(c)",
        x=-0.16,
        y=1.06,
        fontsize=8,
        bold=True,
    )

    axis_level.tick_params(
        axis="x",
        which="both",
        labelbottom=False,
    )

    axis_discharge.plot(
        gauge["date"],
        gauge["discharge_ml_day"],
        color=FIG7_DISCHARGE_DAILY_COLOUR,
        linewidth=0.35,
        alpha=0.55,
        label="Daily discharge",
        zorder=2,
    )

    axis_discharge.plot(
        gauge["date"],
        gauge["discharge_30d"],
        color=FIG7_DISCHARGE_MEAN_COLOUR,
        linewidth=1.0,
        label="30-day mean",
        zorder=3,
    )

    axis_discharge.set_ylabel(
        "Discharge\n"
        "(10³ ML d$^{-1}$)"
    )

    axis_discharge.set_xlabel(
        "Date"
    )

    axis_discharge.set_ylim(
        bottom=0
    )

    axis_discharge.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda value, _:
                f"{value / 1000:g}"
        )
    )

    clean_axis(
        axis_discharge,
        horizontal_grid=True,
        linewidth=0.7,
    )

    axis_discharge.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
        handlelength=1.3,
        handletextpad=0.3,
        columnspacing=0.7,
    )

    axis_discharge.set_xlim(
        pd.Timestamp(
            "2015-01-01"
        ),
        pd.Timestamp(
            "2026-10-01"
        ),
    )

    axis_discharge.xaxis.set_major_locator(
        mdates.YearLocator(
            base=2
        )
    )

    axis_discharge.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%Y"
        )
    )

    axis_discharge.xaxis.set_minor_locator(
        mdates.YearLocator(
            base=1
        )
    )

    for tick_label in axis_discharge.get_xticklabels():
        tick_label.set_rotation(
            0
        )
        tick_label.set_ha(
            "center"
        )

    figure.subplots_adjust(
        left=0.20,
        right=0.98,
        top=0.94,
        bottom=0.07,
    )

    output_base = (
        output_dir
        / "Figure_7_regional_classified_area_and_hydrological_context_single_column"
    )

    save_figure(
        figure,
        output_base,
    )

    if show:
        plt.show()

    plt.close(
        figure
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

    parser = argparse.ArgumentParser(
        description=(
            "Generate publication Figures 3, 4, 5 and 7 "
            "from ECHO analysis outputs."
        )
    )

    parser.add_argument(
        "--figure",
        choices=[
            "all",
            "3",
            "4",
            "5",
            "7",
        ],
        default="all",
        help=(
            "Generate all supported figures or one selected figure. "
            "Default: all"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repo_root
            / "figures"
            / "output"
        ),
        help=(
            "Output directory for publication figures."
        ),
    )

    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=(
            repo_root
            / "results"
            / "loro_ablation"
            / "loro_ablation_predictions.csv"
        ),
        help=(
            "Held-out predictions from script 06."
        ),
    )

    parser.add_argument(
        "--compact-predictions",
        type=Path,
        default=(
            repo_root
            / "results"
            / "compact_fusion"
            / "compact_fusion_predictions.csv"
        ),
        help=(
            "Held-out Compact Fusion predictions from script 08."
        ),
    )

    parser.add_argument(
        "--shap-dir",
        type=Path,
        default=(
            repo_root
            / "results"
            / "shap_analysis"
        ),
        help=(
            "SHAP output directory produced by script 07."
        ),
    )

    parser.add_argument(
        "--temporal-area",
        type=Path,
        default=(
            repo_root
            / "results"
            / "temporal_transfer"
            / "summaries"
            / "class_area_by_period_region.csv"
        ),
        help=(
            "Regional temporal class-area CSV produced by script 09."
        ),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            repo_root
            / "data"
        ),
        help=(
            "Root data directory, used to locate the hydrology CSV."
        ),
    )

    parser.add_argument(
        "--gauge-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit Darlington Point gauge CSV for Figure 7."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Display figures interactively after saving."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Generate the requested publication figure(s)."""

    configure_matplotlib()

    args = parse_args()

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    baseline_predictions = (
        args.baseline_predictions
        .expanduser()
        .resolve()
    )

    compact_predictions = (
        args.compact_predictions
        .expanduser()
        .resolve()
    )

    shap_dir = (
        args.shap_dir
        .expanduser()
        .resolve()
    )

    temporal_area = (
        args.temporal_area
        .expanduser()
        .resolve()
    )

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )

    gauge_file = (
        args.gauge_file
        .expanduser()
        .resolve()
        if args.gauge_file is not None
        else None
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    requested = (
        [
            "3",
            "4",
            "5",
            "7",
        ]
        if args.figure == "all"
        else [
            args.figure
        ]
    )

    print(
        "ECHO publication-figure generator"
    )

    print(
        "Figures:",
        ", ".join(
            requested
        ),
    )

    print(
        "Output directory:",
        output_dir,
    )

    if "3" in requested:
        make_figure_3(
            baseline_predictions=baseline_predictions,
            compact_predictions=compact_predictions,
            output_dir=output_dir,
            show=args.show,
        )

    if "4" in requested:
        make_figure_4(
            shap_dir=shap_dir,
            output_dir=output_dir,
            show=args.show,
        )

    if "5" in requested:
        make_figure_5(
            shap_dir=shap_dir,
            output_dir=output_dir,
            show=args.show,
        )

    if "7" in requested:
        make_figure_7(
            area_file=temporal_area,
            data_root=data_root,
            gauge_file=gauge_file,
            output_dir=output_dir,
            show=args.show,
        )

    print(
        "\nRequested publication figures generated successfully."
    )


if __name__ == "__main__":
    main()
