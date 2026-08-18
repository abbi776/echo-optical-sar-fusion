/**
 * @name 05_s2_preprocessing
 * @description Sentinel-2 preprocessing utilities for the optical-SAR
 * fusion workflow.
 *
 * The workflow:
 * 1. Applies Sentinel-2 edge masking.
 * 2. Applies Cloud Score+ clear-pixel masking.
 * 3. Applies Scene Classification Layer (SCL) masking.
 * 4. Scales selected Sentinel-2 surface-reflectance bands.
 * 5. Retains the Cloud Score+ quality band for seasonal compositing.
 * 6. Builds a seasonal quality mosaic using cs_cdf.
 *
 * Required Sentinel-2 collection:
 *   COPERNICUS/S2_SR_HARMONIZED
 *
 * Required Cloud Score+ collection:
 *   GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED
 *
 * Cloud Score+ must be linked to the Sentinel-2 collection before
 * preprocessS2() is applied.
 */


// ======================================================
// Sentinel-2 band definitions
// ======================================================

/**
 * Sentinel-2 surface-reflectance bands used as predictors.
 *
 * Native spatial resolution:
 * - 10 m: B2, B3, B4, B8
 * - 20 m: B5, B6, B7, B8A, B11, B12
 *
 * The final export is performed at 10 m to provide a common working
 * resolution with the Sentinel-1 feature stack.
 */
var S2_REFLECTANCE_BANDS = [
  'B2',   // Blue
  'B3',   // Green
  'B4',   // Red
  'B5',   // Red Edge 1
  'B6',   // Red Edge 2
  'B7',   // Red Edge 3
  'B8',   // NIR
  'B8A',  // Narrow NIR / Red Edge 4
  'B11',  // SWIR 1
  'B12'   // SWIR 2
];


/**
 * Standardized output names used during optical feature extraction.
 */
var S2_REFLECTANCE_NAMES = [
  'BLUE',
  'GREEN',
  'RED',
  'RE1',
  'RE2',
  'RE3',
  'NIR',
  'NIR_NARROW',
  'SWIR1',
  'SWIR2'
];


// ======================================================
// Cloud Score+ masking
// ======================================================

/**
 * Masks Sentinel-2 pixels using Cloud Score+.
 *
 * Cloud Score+ values range from 0 to 1, with larger values representing
 * clearer observations.
 *
 * The workflow uses:
 *
 *   cs_cdf >= 0.50
 *
 * @param {ee.Image} image Sentinel-2 image with linked Cloud Score+ band.
 * @param {string} [qaBand='cs_cdf'] Cloud Score+ quality band.
 * @param {number} [clearThreshold=0.50] Minimum accepted clear-pixel score.
 * @returns {ee.Image} Cloud Score+ masked image.
 */
exports.maskCloudScorePlus = function(
  image,
  qaBand,
  clearThreshold
) {
  image = ee.Image(image);

  qaBand = (qaBand === undefined)
    ? 'cs_cdf'
    : qaBand;

  clearThreshold = (clearThreshold === undefined)
    ? 0.50
    : clearThreshold;

  var clearMask = image
    .select(qaBand)
    .gte(clearThreshold);

  return image
    .updateMask(clearMask)
    .copyProperties(
      image,
      image.propertyNames()
    );
};


// ======================================================
// Scene Classification Layer masking
// ======================================================

/**
 * Masks unsuitable Sentinel-2 Scene Classification Layer classes.
 *
 * SCL classes:
 *
 *  0 = No data
 *  1 = Saturated or defective
 *  2 = Dark area pixels
 *  3 = Cloud shadows
 *  4 = Vegetation
 *  5 = Bare soils
 *  6 = Water
 *  7 = Unclassified / low probability cloud
 *  8 = Medium probability cloud
 *  9 = High probability cloud
 * 10 = Thin cirrus
 * 11 = Snow or ice
 *
 * Water and dark-area pixels are retained because both may represent
 * legitimate floodplain surface conditions.
 *
 * Masked classes:
 * - No data
 * - Saturated/defective pixels
 * - Cloud shadows
 * - Medium/high probability cloud
 * - Thin cirrus
 * - Snow/ice
 *
 * @param {ee.Image} image Sentinel-2 image containing the SCL band.
 * @returns {ee.Image} SCL-masked image.
 */
exports.maskSCL = function(image) {
  image = ee.Image(image);

  var scl = image.select('SCL');

  var validMask = scl
    .neq(0)
    .and(scl.neq(1))
    .and(scl.neq(3))
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));

  return image
    .updateMask(validMask)
    .copyProperties(
      image,
      image.propertyNames()
    );
};


// ======================================================
// Edge masking
// ======================================================

/**
 * Applies additional Sentinel-2 image-edge masking.
 *
 * Masks from B8A and B9 are propagated to remove edge pixels that may
 * remain valid in the higher-resolution bands.
 *
 * @param {ee.Image} image Sentinel-2 image.
 * @returns {ee.Image} Edge-masked image.
 */
exports.maskEdges = function(image) {
  image = ee.Image(image);

  return image
    .updateMask(
      image.select('B8A').mask()
    )
    .updateMask(
      image.select('B9').mask()
    )
    .copyProperties(
      image,
      image.propertyNames()
    );
};


// ======================================================
// Reflectance scaling and band selection
// ======================================================

/**
 * Scales and selects Sentinel-2 surface-reflectance bands.
 *
 * Sentinel-2 L2A surface-reflectance values use a scale factor of
 * 0.0001.
 *
 * The Cloud Score+ quality band is retained without scaling because it
 * is required by qualityMosaic() during seasonal compositing.
 *
 * @param {ee.Image} image Sentinel-2 surface-reflectance image.
 * @param {string} [qaBand='cs_cdf'] Cloud Score+ quality band to retain.
 * @returns {ee.Image} Scaled reflectance bands plus Cloud Score+ band.
 */
exports.scaleAndSelectReflectance = function(
  image,
  qaBand
) {
  image = ee.Image(image);

  qaBand = (qaBand === undefined)
    ? 'cs_cdf'
    : qaBand;

  var reflectance = image
    .select(
      S2_REFLECTANCE_BANDS,
      S2_REFLECTANCE_NAMES
    )
    .multiply(0.0001)
    .toFloat();

  var quality = image
    .select(qaBand)
    .toFloat();

  return reflectance
    .addBands(quality)
    .copyProperties(
      image,
      image.propertyNames()
    );
};


// ======================================================
// Complete per-image Sentinel-2 preprocessing
// ======================================================

/**
 * Applies the complete per-image Sentinel-2 preprocessing workflow.
 *
 * Expected input:
 * - COPERNICUS/S2_SR_HARMONIZED image
 * - Linked Cloud Score+ cs_cdf band
 *
 * Processing order:
 * 1. Edge masking
 * 2. Cloud Score+ masking
 * 3. SCL masking
 * 4. Reflectance scaling
 * 5. Reflectance-band selection
 * 6. Retention of cs_cdf for quality mosaicking
 *
 * @param {ee.Image} image Sentinel-2 image with linked Cloud Score+ data.
 * @param {string} [qaBand='cs_cdf'] Cloud Score+ quality band.
 * @param {number} [clearThreshold=0.50] Clear-pixel threshold.
 * @returns {ee.Image} Preprocessed Sentinel-2 image.
 */
exports.preprocessS2 = function(
  image,
  qaBand,
  clearThreshold
) {
  image = ee.Image(image);

  qaBand = (qaBand === undefined)
    ? 'cs_cdf'
    : qaBand;

  clearThreshold = (clearThreshold === undefined)
    ? 0.50
    : clearThreshold;

  var edgeMasked = exports.maskEdges(
    image
  );

  var cloudMasked = exports.maskCloudScorePlus(
    edgeMasked,
    qaBand,
    clearThreshold
  );

  var sclMasked = exports.maskSCL(
    cloudMasked
  );

  var scaled = exports.scaleAndSelectReflectance(
    sclMasked,
    qaBand
  );

  return scaled.copyProperties(
    image,
    image.propertyNames()
  );
};


// ======================================================
// Seasonal quality mosaic
// ======================================================

/**
 * Builds a seasonal Sentinel-2 quality mosaic.
 *
 * The highest-quality available observation at each pixel is selected
 * using the linked Cloud Score+ cs_cdf score.
 *
 * Before calling this function, the collection should already be:
 * - filtered by study area
 * - filtered by seasonal date range
 * - linked with Cloud Score+
 * - processed using preprocessS2()
 *
 * The Cloud Score+ band is removed after compositing so that only optical
 * reflectance predictors are passed to feature extraction.
 *
 * @param {ee.ImageCollection} collection Preprocessed Sentinel-2 collection.
 * @param {ee.Geometry|ee.FeatureCollection} region Study-area region.
 * @param {string} [qaBand='cs_cdf'] Quality band used for mosaicking.
 * @returns {ee.Image} Seasonal Sentinel-2 reflectance composite.
 */
exports.makeQualityComposite = function(
  collection,
  region,
  qaBand
) {
  collection = ee.ImageCollection(collection);

  qaBand = (qaBand === undefined)
    ? 'cs_cdf'
    : qaBand;

  var geom = ee.FeatureCollection(region)
    .geometry();

  var composite = collection
    .qualityMosaic(qaBand)
    .select(S2_REFLECTANCE_NAMES)
    .clip(geom)
    .toFloat();

  return composite;
};