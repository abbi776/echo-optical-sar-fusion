# 🌿 ECHO: Interpretable Optical–SAR Fusion for Floodplain Vegetation Mapping

Reproducible workflow for evaluating and interpreting **multi-seasonal Sentinel-1 SAR and Sentinel-2 optical fusion** for floodplain vegetation mapping.

The framework, referred to as **ECHO**, combines Sentinel-1 backscatter, polarisation and texture predictors with Sentinel-2 spectral bands and vegetation/water indices. Random Forest (RF) and XGBoost models are evaluated under spatially independent **Leave-One-Region-Out (LORO)** validation, while SHAP is used to interpret sensor contributions and derive a compact fusion model.

This repository accompanies an **anonymous manuscript submission** and contains the Google Earth Engine preprocessing code, Python analysis pipeline, and publication-figure generation code needed to reproduce the main workflow.

---

## 🌱 Key Features

- **Sentinel-1 preprocessing in Google Earth Engine:** border-noise handling, Refined Lee speckle filtering, ellipsoidal incidence-angle normalisation, polarisation features and GLCM textures.
- **Sentinel-2 preprocessing in Google Earth Engine:** Cloud Score+ and Scene Classification Layer masking, seasonal quality mosaics, spectral bands and vegetation/water indices.
- **Multi-seasonal optical–SAR fusion:** four seasonal products combined into a **176-predictor annual feature space**.
- **Spatially independent validation:** five-fold Leave-One-Region-Out validation across five sub-regions.
- **Sensor and feature benchmarking:** eight baseline configurations covering Sentinel-1, Sentinel-2 and complete fusion representations.
- **Explainable AI:** class-specific and overall SHAP attribution for interpreting optical–SAR complementarity.
- **Compact fusion model:** fold-wise training-only SHAP ranking reduces the 176-predictor space to **25 predictors** using an 80% cumulative-attribution threshold.
- **Temporal transfer diagnostics:** the compact model is transferred across three hydrological periods to examine model consistency and condition sensitivity.
- **Reproducible publication figures:** code for generating manuscript Figures 3, 4, 5 and 7 from saved analysis outputs.

---

## 📂 Repository Structure

```text
echo-review-code/
│
├── gee/                                  # Google Earth Engine preprocessing
│   ├── 00_utils.js
│   ├── 01_s1_border_noise.js
│   ├── 02_s1_speckle_filter.js
│   ├── 03_s1_rtc.js
│   ├── 04_s1_feature_extraction.js
│   ├── 05_s2_preprocessing.js
│   ├── 06_s2_feature_extraction.js
│   ├── 07_download_s1_feature_stack.js
│   ├── 08_check_s2_clear_mosaic_window.js
│   └── 09_download_s2_feature_stack.js
│
├── scripts/                              # Python analysis pipeline
│   ├── 00_prepare_s1_seasonal_mosaics.py
│   ├── 01_prepare_s2_seasonal_mosaics.py
│   ├── 02_build_s1_annual_stacks.py
│   ├── 03_build_s2_annual_stacks.py
│   ├── 04_build_fusion_annual_stacks.py
│   ├── 05_prepare_reference_samples.py
│   ├── 06_train_loro_ablation_models.py
│   ├── 07_shap_feature_analysis.py
│   ├── 08_train_compact_fusion_model.py
│   └── 09_temporal_transfer_analysis.py
│
├── figures/
│   └── generate_publication_figures.py   # Figures 3, 4, 5 and 7
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the anonymous review repository and create a Python environment:

```bash
git clone <anonymous-repository-url>
cd echo-review-code

python -m venv venv
```

Activate the environment.

**Linux / macOS**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The Google Earth Engine scripts in `gee/` are run separately in the **Google Earth Engine Code Editor**.

---

# 🚀 Pipeline Steps

## 1️⃣ Sentinel-1 preprocessing in Google Earth Engine

Run the Sentinel-1 scripts in `gee/` in numerical order.

The workflow includes:

- Sentinel-1 GRD IW imagery
- VV and VH polarisations
- border-noise handling
- conversion between dB and linear units where required
- Refined Lee speckle filtering
- ellipsoidal incidence-angle normalisation
- VV and VH backscatter predictors
- polarisation/algebraic predictors
- GLCM texture predictors

Each season produces **26 Sentinel-1 predictors**:

- 2 backscatter predictors
- 10 polarisation/algebraic predictors
- 14 GLCM texture predictors

Seasonal feature tiles are exported from Earth Engine for subsequent local mosaicking.

Example:

```text
S1_WPE_FEATURES_SPRING_2025-*.tif
```

> Note: the Sentinel-1 normalisation used here is an ellipsoidal-incidence-angle approximation and is not full DEM-based radiometric terrain flattening.

---

## 2️⃣ Sentinel-2 preprocessing in Google Earth Engine

Run:

```text
05_s2_preprocessing.js
06_s2_feature_extraction.js
08_check_s2_clear_mosaic_window.js
09_download_s2_feature_stack.js
```

The Sentinel-2 workflow uses:

- Sentinel-2 Level-2A surface reflectance
- Cloud Score+
- Scene Classification Layer masking
- seasonal quality mosaicking
- 10 spectral bands
- 8 spectral indices

The eight indices are:

```text
NDVI
EVI
NDWI
LSWI
MSAVI2
DBSI
GNDVI
TVI
```

Each season therefore contains **18 Sentinel-2 predictors**.

Example export:

```text
S2_WPE_FEATURES_SPRING_2025-*.tif
```

---

## 3️⃣ Prepare seasonal Sentinel-1 mosaics

```bash
python scripts/00_prepare_s1_seasonal_mosaics.py
```

This script mosaics exported Sentinel-1 tiles and checks:

- 26 bands per seasonal product
- EPSG:3577
- 10 m spatial resolution
- consistent raster metadata

---

## 4️⃣ Prepare seasonal Sentinel-2 mosaics

```bash
python scripts/01_prepare_s2_seasonal_mosaics.py
```

This performs the equivalent mosaicking and metadata checks for the **18-band Sentinel-2 seasonal products**.

---

## 5️⃣ Build annual Sentinel-1 stacks

```bash
python scripts/02_build_s1_annual_stacks.py
```

Four seasonal Sentinel-1 products are combined in the following order:

```text
Winter
Spring
Summer
Autumn
```

This creates:

```text
26 predictors × 4 seasons = 104 Sentinel-1 predictors
```

---

## 6️⃣ Build annual Sentinel-2 stacks

```bash
python scripts/03_build_s2_annual_stacks.py
```

Four seasonal Sentinel-2 products are combined to create:

```text
18 predictors × 4 seasons = 72 Sentinel-2 predictors
```

---

## 7️⃣ Build annual optical–SAR fusion stacks

```bash
python scripts/04_build_fusion_annual_stacks.py
```

The Sentinel-1 and Sentinel-2 products are fused season by season.

Within each season:

```text
26 Sentinel-1 predictors
+
18 Sentinel-2 predictors
=
44 predictors
```

Across four seasons:

```text
44 × 4 = 176 predictors
```

The final ordering is:

```text
Winter S1
Winter S2
Spring S1
Spring S2
Summer S1
Summer S2
Autumn S1
Autumn S2
```

Spatial alignment, predictor identity and predictor order are checked before the final annual fusion stack is written.

---

## 8️⃣ Prepare reference samples

```bash
python scripts/05_prepare_reference_samples.py
```

Polygon-level predictor means are extracted from the 2025–2026 annual stacks.

The reference design contains:

```text
525 polygons
5 spatial regions
105 polygons per region
3 classes
35 polygons per class per region
```

Classes:

```text
RRG   = River Red Gum
NFFP  = Non-Forest Floodplain vegetation
Water = Water
```

The script produces model-ready tables for:

- Full Sentinel-1
- Full Sentinel-2
- Full Sentinel-1 + Sentinel-2 fusion

It also performs reference-design, geometry, raster and extraction-quality checks.

---

## 9️⃣ Run LORO baseline experiments

```bash
python scripts/06_train_loro_ablation_models.py
```

Eight baseline feature configurations are evaluated:

| Configuration | Predictors |
|---|---:|
| Sentinel-2 bands only | 40 |
| Sentinel-2 indices only | 32 |
| Sentinel-1 backscatter only | 8 |
| Sentinel-1 polarisation features | 40 |
| Sentinel-1 texture only | 56 |
| Full Sentinel-2 | 72 |
| Full Sentinel-1 | 104 |
| Full Sentinel-1 + Sentinel-2 fusion | 176 |

Each configuration is evaluated with:

- Random Forest
- XGBoost

Validation uses five-fold **Leave-One-Region-Out (LORO)** cross-validation:

```text
Fold 1 → SR1 held out
Fold 2 → SR2 held out
Fold 3 → SR3 held out
Fold 4 → SR4 held out
Fold 5 → SR5 held out
```

Each fold therefore contains:

```text
420 training polygons
105 test polygons
35 test polygons per class
```

Outputs include:

- pooled performance metrics
- fold-level metrics
- class-specific precision, recall and F1
- pooled and fold confusion matrices
- held-out predictions
- feature importance
- fitted baseline models

---

## 🔟 SHAP interpretation and compact feature selection

```bash
python scripts/07_shap_feature_analysis.py
```

Two SHAP analyses are kept deliberately separate.

### Descriptive SHAP

A Full Fusion XGBoost model fitted using all 525 reference polygons is interpreted to quantify:

- overall Sentinel-1 vs Sentinel-2 attribution
- class-specific sensor attribution
- feature-group attribution
- seasonal attribution
- top individual predictors
- signed class-specific SHAP effects

These results are used for **interpretation and visualisation only**.

### Fold-wise training-only SHAP

For compact feature selection, SHAP is calculated independently within each LORO training set.

For every fold:

```text
Held-out region excluded
        ↓
model fitted on four training regions
        ↓
SHAP calculated on training observations only
        ↓
mean |SHAP| by predictor and class
```

Importance is then averaged across classes and folds.

The smallest ranked subset reaching at least:

```text
80% cumulative SHAP attribution
```

is retained.

The resulting compact feature space contains:

```text
25 predictors
```

No predictor is manually forced into the compact subset.

---

## 1️⃣1️⃣ Train and evaluate Compact Fusion

```bash
python scripts/08_train_compact_fusion_model.py
```

The SHAP-selected **25-predictor Compact Fusion** configuration is evaluated using both:

- Random Forest
- XGBoost

under the same five-fold LORO design used for the baseline experiments.

The script also trains final all-sample Compact Fusion models.

The final operational model is:

```text
Compact Fusion XGBoost
```

The saved model package preserves:

- trained classifier
- exact predictor list
- predictor order
- label encoder
- class order
- SHAP feature-selection metadata

---

## 1️⃣2️⃣ Temporal transfer analysis

```bash
python scripts/09_temporal_transfer_analysis.py
```

The final Compact XGBoost model is transferred across three hydrological periods:

```text
2017–2018   lower-flow
2022–2023   high-flow
2025–2026   post-flood reference
```

Historical predictors are matched using:

```text
season + sensor + feature identity
```

rather than literal acquisition year.

Prediction is performed only where all 25 required predictors are valid.

The script produces:

- predicted class maps
- maximum class probability (`pmax`)
- class area by period and region
- combined class proportions
- complete 3 × 3 class-state transition matrices
- RRG/NFFP classified gain–loss summaries

---

# 📊 Publication Figures

Figures 3, 4, 5 and 7 can be regenerated using:

```bash
python figures/generate_publication_figures.py
```

Individual figures can also be generated:

```bash
python figures/generate_publication_figures.py --figure 3
python figures/generate_publication_figures.py --figure 4
python figures/generate_publication_figures.py --figure 5
python figures/generate_publication_figures.py --figure 7
```

The figure script reads analysis outputs rather than rerunning the models.

Supported figures are:

- **Figure 3:** pooled XGBoost confusion matrices for all nine feature configurations
- **Figure 4:** sensor, feature-group, seasonal and predictor-level SHAP attribution
- **Figure 5:** signed class-specific SHAP effects
- **Figure 7:** regional classified-area trajectories and hydrological context

Figures are exported as:

```text
PDF
SVG
600-dpi PNG
```

---

# 📁 Data & Results Organisation

Large raster datasets, reference data, trained models and generated outputs are **not stored directly in this repository**.

A compatible local project structure is:

```text
data/
├── images/
│   ├── 2017-2018/
│   ├── 2022-2023/
│   └── 2025-2026/
│
├── labels/
│
├── wetlands/
│
├── hydrology/
│
└── reference_samples/

models/
├── loro_ablation/
└── compact_fusion/

results/
├── loro_ablation/
├── shap_analysis/
├── compact_fusion/
└── temporal_transfer/

figures/
├── generate_publication_figures.py
└── output/
```

The scripts expose command-line path options where appropriate, so equivalent local directory structures can also be used.

---

# 📤 Main Outputs

The workflow produces:

- 104-band annual Sentinel-1 stacks
- 72-band annual Sentinel-2 stacks
- 176-band optical–SAR fusion stacks
- polygon-level modelling tables
- LORO validation metrics
- confusion matrices
- held-out predictions
- fitted Random Forest and XGBoost models
- SHAP attribution tables
- 25-feature compact model
- temporal-transfer class maps
- `pmax` reliability layers
- regional class-area summaries
- class-state transition matrices
- manuscript Figures 3, 4, 5 and 7

---

# 🔬 Reproducibility Notes

Several safeguards are included throughout the workflow:

- fixed random seed for RF and XGBoost
- identical reference samples across sensor configurations
- five spatially independent LORO folds
- predictor-count and predictor-order checks
- CRS and raster-alignment checks
- explicit NoData handling
- fold-wise training-only SHAP for compact feature selection
- exact feature-order storage with fitted models
- exact feature matching during temporal transfer

Model hyperparameters are fixed across feature configurations to keep comparisons consistent.

---

# ⚠️ Temporal-Transfer Interpretation

The 2017–2018 and 2022–2023 classifications do **not** have independent historical reference labels.

They should therefore be interpreted as:

> **model-consistent temporal transfers and reliability diagnostics**

rather than conventional validated ecological change-detection products.

Terms such as:

```text
NFFP → RRG = RRG-classified gain
RRG → NFFP = RRG-classified loss
```

describe **modelled class-state transitions only**.

They should not be interpreted directly as:

- recruitment
- mortality
- encroachment
- retreat

without independent historical validation.

Similarly, `pmax` is retained as a measure of **relative prediction decisiveness** and should not be interpreted as formally calibrated uncertainty.

---

# 🛰️ External Data Sources

The workflow uses publicly available satellite products accessed through Google Earth Engine, including:

- Sentinel-1 GRD
- Sentinel-2 Level-2A surface reflectance
- Google Cloud Score+

Reference polygons, wetland boundaries, ancillary imagery and hydrological observations required for the complete study are not redistributed in this anonymous review repository where redistribution or review-anonymity constraints apply.

---

# 🙏 Methodological Attribution

The Sentinel-1 preprocessing workflow uses established remote-sensing methods including:

- Refined Lee speckle filtering
- gamma-nought backscatter normalisation
- grey-level co-occurrence matrix texture analysis

Relevant methodological sources are cited in the accompanying manuscript.

The implementation in this repository was adapted specifically for the optical–SAR fusion and floodplain vegetation classification workflow described here.

---

# 🔒 Triple-Blind Review

This repository accompanies a manuscript submitted under triple-blind review.

To preserve reviewer anonymity requirements:

- author names and affiliations are omitted
- contact information and personal account identifiers are omitted
- identifying repository links and local filesystem paths have been removed

A de-anonymised version of the repository will be released following acceptance.
