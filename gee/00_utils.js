/**
 * @name 00_utils
 * @description Shared utility functions for the Sentinel-1 and Sentinel-2
 * woody plant encroachment (WPE) workflow.
 *
 * Includes:
 * - Sentinel-1 dB/linear conversion
 * - Safe mathematical operations
 * - Sentinel-2 reflectance scaling
 * - Shared band lists
 * - Band-prefix helpers
 * - Seasonal metadata helpers
 */


// ======================================================
// Shared band lists
// ======================================================

/**
 * Sentinel-1 polarization bands used throughout the workflow.
 */
exports.S1_POL_BANDS = [
  'VV',
  'VH'
];


/**
 * Sentinel-1 geometry bands required during preprocessing.
 *
 * The incidence-angle band is used for border-noise masking and
 * ellipsoidal radiometric normalization. It is not used as a final
 * classification predictor.
 */
exports.S1_GEOMETRY_BANDS = [
  'angle'
];


/**
 * Sentinel-2 surface-reflectance bands used for optical feature generation.
 *
 * Native spatial resolutions:
 * - 10 m: B2, B3, B4, B8
 * - 20 m: B5, B6, B7, B8A, B11, B12
 *
 * Exports are generated at 10 m to provide a common working resolution
 * with the Sentinel-1 feature stack.
 */
exports.S2_REFLECTANCE_BANDS = [
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
 * Sentinel-2 quality bands that may be used during masking and
 * quality-control operations.
 *
 * The main Sentinel-2 preprocessing workflow uses scene classification
 * information together with Cloud Score+ for cloud screening.
 */
exports.S2_QUALITY_BANDS = [
  'SCL',
  'QA60',
  'MSK_CLDPRB',
  'MSK_SNWPRB'
];


// ======================================================
// Sentinel-1 conversion utilities
// ======================================================

/**
 * Converts Sentinel-1 backscatter from dB to linear power.
 *
 * Formula:
 *   linear = 10^(dB / 10)
 *
 * The incidence-angle band is retained because it is required by later
 * Sentinel-1 preprocessing operations.
 *
 * @param {ee.Image} image Sentinel-1 image containing VV, VH, and angle.
 * @returns {ee.Image} Image containing linear VV and VH plus angle.
 */
exports.dbToLin = function(image) {
  image = ee.Image(image);

  var vv = ee.Image(10)
    .pow(image.select('VV').divide(10))
    .rename('VV');

  var vh = ee.Image(10)
    .pow(image.select('VH').divide(10))
    .rename('VH');

  return ee.Image.cat([
    vv,
    vh,
    image.select('angle')
  ])
    .copyProperties(image, image.propertyNames());
};


/**
 * Converts Sentinel-1 backscatter from linear power to dB.
 *
 * Formula:
 *   dB = 10 * log10(linear)
 *
 * A small lower bound is applied before logarithmic conversion to avoid
 * undefined values caused by log10(0).
 *
 * @param {ee.Image} image Image containing VV and VH in linear power.
 * @param {number} [eps=1e-5] Minimum positive value before conversion.
 * @returns {ee.Image} Image containing VV and VH in dB.
 */
exports.linToDb = function(image, eps) {
  image = ee.Image(image);

  eps = (eps === undefined) ? 1e-5 : eps;

  var vv = image.select('VV')
    .max(eps)
    .log10()
    .multiply(10)
    .rename('VV');

  var vh = image.select('VH')
    .max(eps)
    .log10()
    .multiply(10)
    .rename('VH');

  return ee.Image.cat([
    vv,
    vh
  ])
    .copyProperties(image, image.propertyNames());
};


// ======================================================
// General mathematical utilities
// ======================================================

/**
 * Performs division while protecting against zero or near-zero
 * denominators.
 *
 * Unlike an absolute-value clamp, this implementation preserves the
 * original sign of the denominator.
 *
 * @param {ee.Image} numerator Numerator image.
 * @param {ee.Image} denominator Denominator image.
 * @param {number} [eps=1e-6] Minimum absolute denominator magnitude.
 * @returns {ee.Image} Safely divided image.
 */
exports.safeDivide = function(numerator, denominator, eps) {
  numerator = ee.Image(numerator);
  denominator = ee.Image(denominator);

  eps = (eps === undefined) ? 1e-6 : eps;

  var sign = denominator
    .gte(0)
    .multiply(2)
    .subtract(1);

  var safeDenominator = denominator.where(
    denominator.abs().lt(eps),
    sign.multiply(eps)
  );

  return numerator.divide(safeDenominator);
};


// ======================================================
// Sentinel-2 utilities
// ======================================================

/**
 * Scales Sentinel-2 surface-reflectance bands from integer values to
 * reflectance.
 *
 * Sentinel-2 L2A surface-reflectance bands use a scale factor of 0.0001.
 * Quality bands such as SCL and QA60 must not be passed to this function.
 *
 * @param {ee.Image} image Sentinel-2 surface-reflectance image.
 * @param {ee.List|Array} [bands] Reflectance bands to scale.
 * @returns {ee.Image} Selected bands scaled to reflectance.
 */
exports.scaleS2Reflectance = function(image, bands) {
  image = ee.Image(image);

  bands = (bands === undefined)
    ? exports.S2_REFLECTANCE_BANDS
    : bands;

  bands = ee.List(bands);

  return image
    .select(bands)
    .multiply(0.0001)
    .copyProperties(image, image.propertyNames());
};


// ======================================================
// Band naming helpers
// ======================================================

/**
 * Adds a prefix to every band name in an image.
 *
 * Examples:
 *   VV   -> S1_VV
 *   NDVI -> S2_NDVI
 *
 * This helps avoid ambiguous band names when Sentinel-1 and Sentinel-2
 * predictors are combined during downstream analysis.
 *
 * @param {ee.Image} image Input image.
 * @param {string} prefix Prefix such as 'S1_' or 'S2_'.
 * @returns {ee.Image} Image with prefixed band names.
 */
exports.addBandPrefix = function(image, prefix) {
  image = ee.Image(image);

  var oldNames = image.bandNames();

  var newNames = oldNames.map(function(name) {
    return ee.String(prefix)
      .cat(ee.String(name));
  });

  return image
    .rename(newNames)
    .copyProperties(image, image.propertyNames());
};


// ======================================================
// Metadata helpers
// ======================================================

/**
 * Adds seasonal-composite metadata to an image.
 *
 * @param {ee.Image} image Input image.
 * @param {string} cycle Analysis cycle, e.g. '2025_2026'.
 * @param {string} season Season name, e.g. 'winter'.
 * @param {string} startDate Composite start date.
 * @param {string} endDate Composite end date.
 * @returns {ee.Image} Image with seasonal metadata properties.
 */
exports.addSeasonMetadata = function(
  image,
  cycle,
  season,
  startDate,
  endDate
) {
  image = ee.Image(image);

  return image.set({
    'cycle': cycle,
    'season': season,
    'start_date': startDate,
    'end_date': endDate
  });
};