/**
 * @name 06_s2_feature_extraction
 * @description Generates Sentinel-2 spectral predictor stacks for
 * machine-learning analysis.
 *
 * The workflow:
 * 1. Retains 10 Sentinel-2 surface-reflectance bands.
 * 2. Calculates eight spectral indices:
 *    NDVI, EVI, NDWI, LSWI, MSAVI2, DBSI, GNDVI, and TVI.
 * 3. Optionally adds an S2_ prefix to all output predictor names.
 *
 * Requirements:
 * - Input image must be a preprocessed Sentinel-2 seasonal composite
 *   produced by 05_s2_preprocessing.
 * - Expected input bands:
 *   BLUE, GREEN, RED, RE1, RE2, RE3, NIR, NIR_NARROW, SWIR1, SWIR2.
 *
 * Notes:
 * - Input reflectance bands are expected to already be scaled.
 * - LSWI is calculated using SWIR2.
 * - DBSI is calculated using SWIR1 and NDVI.
 */


// ======================================================
// Internal helper: safe division
// ======================================================

/**
 * Performs division while protecting against zero or near-zero
 * denominators.
 *
 * The original denominator sign is preserved.
 *
 * @param {ee.Image} numerator Numerator image.
 * @param {ee.Image} denominator Denominator image.
 * @param {number} [eps=1e-6] Minimum absolute denominator magnitude.
 * @returns {ee.Image} Safely divided image.
 */
var safeDivide = function(numerator, denominator, eps) {
  numerator = ee.Image(numerator);
  denominator = ee.Image(denominator);

  eps = (eps === undefined)
    ? 1e-6
    : eps;

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
// Internal helper: band prefix
// ======================================================

/**
 * Adds a prefix to every band name in an image.
 *
 * @param {ee.Image} image Input image.
 * @param {string} prefix Prefix to add, e.g. 'S2_'.
 * @returns {ee.Image} Image with prefixed band names.
 */
var addBandPrefix = function(image, prefix) {
  image = ee.Image(image);

  var oldNames = image.bandNames();

  var newNames = oldNames.map(function(name) {
    return ee.String(prefix)
      .cat(ee.String(name));
  });

  return image
    .rename(newNames)
    .copyProperties(
      image,
      image.propertyNames()
    );
};


// ======================================================
// Sentinel-2 spectral indices
// ======================================================

/**
 * Calculates eight Sentinel-2 spectral indices.
 *
 * Indices:
 * 1. NDVI
 * 2. EVI
 * 3. NDWI
 * 4. LSWI
 * 5. MSAVI2
 * 6. DBSI
 * 7. GNDVI
 * 8. TVI
 *
 * @param {ee.Image} image Sentinel-2 image containing the required
 * reflectance bands.
 * @returns {ee.Image} Image containing eight spectral-index predictors.
 */
exports.computeIndices = function(image) {
  image = ee.Image(image);

  var BLUE = image.select('BLUE');     // B2
  var GREEN = image.select('GREEN');   // B3
  var RED = image.select('RED');       // B4
  var NIR = image.select('NIR');       // B8
  var SWIR1 = image.select('SWIR1');   // B11
  var SWIR2 = image.select('SWIR2');   // B12


  // ==================================================
  // 1. Normalized Difference Vegetation Index
  // ==================================================
  //
  // NDVI = (NIR - RED) / (NIR + RED)
  //

  var NDVI = safeDivide(
    NIR.subtract(RED),
    NIR.add(RED)
  ).rename('NDVI');


  // ==================================================
  // 2. Enhanced Vegetation Index
  // ==================================================
  //
  // EVI =
  // 2.5 * (NIR - RED) /
  // (NIR + 6 * RED - 7.5 * BLUE + 1)
  //

  var EVI = safeDivide(
    NIR
      .subtract(RED)
      .multiply(2.5),

    NIR
      .add(RED.multiply(6))
      .subtract(BLUE.multiply(7.5))
      .add(1)
  ).rename('EVI');


  // ==================================================
  // 3. Normalized Difference Water Index
  // ==================================================
  //
  // NDWI = (GREEN - NIR) / (GREEN + NIR)
  //

  var NDWI = safeDivide(
    GREEN.subtract(NIR),
    GREEN.add(NIR)
  ).rename('NDWI');


  // ==================================================
  // 4. Land Surface Water Index
  // ==================================================
  //
  // LSWI = (NIR - SWIR2) / (NIR + SWIR2)
  //

  var LSWI = safeDivide(
    NIR.subtract(SWIR2),
    NIR.add(SWIR2)
  ).rename('LSWI');


  // ==================================================
  // 5. Modified Soil Adjusted Vegetation Index 2
  // ==================================================
  //
  // MSAVI2 =
  // [2*NIR + 1 -
  // sqrt((2*NIR + 1)^2 - 8*(NIR - RED))] / 2
  //

  var msaviTerm = NIR
    .multiply(2)
    .add(1);

  var msaviDiscriminant = msaviTerm
    .pow(2)
    .subtract(
      NIR
        .subtract(RED)
        .multiply(8)
    )
    .max(0);

  var MSAVI2 = msaviTerm
    .subtract(
      msaviDiscriminant.sqrt()
    )
    .divide(2)
    .rename('MSAVI2');


  // ==================================================
  // 6. Dry Bare-Soil Index
  // ==================================================
  //
  // DBSI =
  // [(SWIR1 - GREEN) / (SWIR1 + GREEN)] - NDVI
  //

  var DBSI = safeDivide(
    SWIR1.subtract(GREEN),
    SWIR1.add(GREEN)
  )
    .subtract(NDVI)
    .rename('DBSI');


  // ==================================================
  // 7. Green Normalized Difference Vegetation Index
  // ==================================================
  //
  // GNDVI = (NIR - GREEN) / (NIR + GREEN)
  //

  var GNDVI = safeDivide(
    NIR.subtract(GREEN),
    NIR.add(GREEN)
  ).rename('GNDVI');


  // ==================================================
  // 8. Transformed Vegetation Index
  // ==================================================
  //
  // TVI = sqrt(NDVI + 0.5)
  //
  // Values below zero are clamped before the square-root operation.
  //

  var TVI = NDVI
    .add(0.5)
    .max(0)
    .sqrt()
    .rename('TVI');


  // ==================================================
  // Assemble spectral-index predictors
  // ==================================================

  return ee.Image.cat([
    NDVI,
    EVI,
    NDWI,
    LSWI,
    MSAVI2,
    DBSI,
    GNDVI,
    TVI
  ])
    .copyProperties(
      image,
      image.propertyNames()
    );
};


// ======================================================
// Complete Sentinel-2 feature stack
// ======================================================

/**
 * Builds the complete Sentinel-2 predictor stack.
 *
 * Final stack:
 * - 10 surface-reflectance predictors
 * - 8 spectral-index predictors
 *
 * Total:
 * - 18 Sentinel-2 predictors per season
 *
 * By default, all output bands receive an S2_ prefix to distinguish
 * optical predictors from Sentinel-1 predictors during downstream fusion.
 *
 * @param {ee.Image} composite Preprocessed Sentinel-2 seasonal composite.
 * @param {boolean} [addPrefix=true] Whether to prefix output bands with S2_.
 * @returns {ee.Image} Complete Sentinel-2 predictor stack.
 */
exports.buildS2FeatureStack = function(
  composite,
  addPrefix
) {
  composite = ee.Image(composite);

  addPrefix = (addPrefix === undefined)
    ? true
    : addPrefix;


  // ==================================================
  // Retain 10 spectral bands
  // ==================================================

  var spectralBands = composite.select([
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
  ]);


  // ==================================================
  // Calculate eight spectral indices
  // ==================================================

  var indices = exports.computeIndices(
    composite
  );


  // ==================================================
  // Assemble complete optical predictor stack
  // ==================================================

  var output = spectralBands
    .addBands(indices)
    .toFloat()
    .copyProperties(
      composite,
      composite.propertyNames()
    );


  // ==================================================
  // Prefix final predictor names
  // ==================================================

  if (addPrefix) {
    output = addBandPrefix(
      output,
      'S2_'
    );
  }


  return ee.Image(output);
};