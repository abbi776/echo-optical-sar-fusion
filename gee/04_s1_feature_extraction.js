/**
 * @name 04_s1_feature_extraction
 * @description Generates Sentinel-1 SAR predictor stacks for downstream
 * machine-learning analysis.
 *
 * The workflow:
 * 1. Retains VV and VH Gamma0 backscatter in dB.
 * 2. Calculates ten polarization/algebraic features from linear and dB
 *    Gamma0 backscatter.
 * 3. Calculates seven GLCM texture metrics independently for VV and VH.
 * 4. Optionally adds an S1_ prefix to all exported predictor names.
 *
 * Requirements:
 * - Input image must contain:
 *   - VV:    linear Gamma0 VV
 *   - VH:    linear Gamma0 VH
 *   - VV_dB: Gamma0 VV in dB
 *   - VH_dB: Gamma0 VH in dB
 *
 * Notes:
 * - Ratios and algebraic SAR features are calculated in linear power where
 *   appropriate.
 * - GLCM textures are calculated from quantized dB backscatter.
 * - Linear VV and VH are used internally but are not retained as final
 *   predictor bands.
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
// Internal helper: band prefix
// ======================================================

/**
 * Adds a prefix to every band name in an image.
 *
 * @param {ee.Image} image Input image.
 * @param {string} prefix Prefix to add, e.g. 'S1_'.
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
// Sentinel-1 backscatter and polarization features
// ======================================================

/**
 * Calculates Sentinel-1 backscatter and polarization/algebraic features.
 *
 * @param {ee.Image} image Image containing VV, VH, VV_dB, and VH_dB.
 * @returns {ee.Image} Sentinel-1 backscatter and derived SAR predictors.
 */
exports.computeIndices = function(image) {
  image = ee.Image(image);

  var VV = image.select('VV');
  var VH = image.select('VH');

  var VV_dB = image.select('VV_dB');
  var VH_dB = image.select('VH_dB');


  // ==================================================
  // 1. Normalized Difference Polarization Index
  // ==================================================

  var NDPI = safeDivide(
    VV.subtract(VH),
    VV.add(VH)
  ).rename('NDPI');


  // ==================================================
  // 2. Normalized Ratio Procedure Between Bands
  // ==================================================

  var NRPB = safeDivide(
    VH.subtract(VV),
    VH.add(VV)
  ).rename('NRPB');


  // ==================================================
  // 3. VV / VH polarization ratio
  // ==================================================

  var VV_VH_RATIO = safeDivide(
    VV,
    VH
  ).rename('VV_VH_RATIO');


  // ==================================================
  // 4. VH / VV polarization ratio
  // ==================================================

  var VH_VV_RATIO = safeDivide(
    VH,
    VV
  ).rename('VH_VV_RATIO');


  // ==================================================
  // 5. Dual-polarization Radar Vegetation Index
  // ==================================================

  var RVI = safeDivide(
    VH.multiply(4),
    VV.add(VH)
  ).rename('RVI');


  // ==================================================
  // 6. Polarization sum
  // ==================================================

  var SUM = VV
    .add(VH)
    .rename('SUM');


  // ==================================================
  // 7. Polarization difference
  // ==================================================

  var DIFF = VV
    .subtract(VH)
    .rename('DIFF');


  // ==================================================
  // 8. Polarization product
  // ==================================================

  var PROD = VV
    .multiply(VH)
    .rename('PROD');


  // ==================================================
  // 9. Vertical Dual Depolarization Index
  // ==================================================

  var VDDI = safeDivide(
    VV.add(VH),
    VV
  ).rename('VDDI');


  // ==================================================
  // 10. Logarithmic polarization ratio
  // ==================================================
  //
  // VV_dB - VH_dB is equivalent to:
  //
  // 10 * log10(VV / VH)
  //

  var LOG_RATIO = VV_dB
    .subtract(VH_dB)
    .rename('LOG_RATIO');


  // ==================================================
  // Assemble backscatter + polarization predictors
  // ==================================================

  return ee.Image.cat([
    VV_dB.rename('VV_dB'),
    VH_dB.rename('VH_dB'),
    NDPI,
    NRPB,
    VV_VH_RATIO,
    VH_VV_RATIO,
    RVI,
    SUM,
    DIFF,
    PROD,
    VDDI,
    LOG_RATIO
  ])
    .copyProperties(
      image,
      image.propertyNames()
    );
};


// ======================================================
// Sentinel-1 GLCM texture features
// ======================================================

/**
 * Calculates GLCM texture features from quantized VV and VH dB
 * backscatter.
 *
 * Settings:
 * - Quantization: 32 grey levels by default.
 * - VV quantization range: -25 to 5 dB.
 * - VH quantization range: -30 to 0 dB.
 * - GLCM size: 2, corresponding to a 5 x 5 neighbourhood.
 *
 * Retained metrics:
 * - Angular Second Moment (ASM)
 * - Correlation
 * - Variance
 * - Inverse Difference Moment (IDM)
 * - Sum Average
 * - Entropy
 * - Contrast
 *
 * Seven metrics are calculated independently for VV and VH,
 * producing 14 texture predictors.
 *
 * @param {ee.Image} image Image containing VV_dB and VH_dB.
 * @param {number} [levels=32] Number of grey-level quantization levels.
 * @returns {ee.Image} Input predictors with GLCM texture bands added.
 */
exports.computeGLCM = function(image, levels) {
  image = ee.Image(image);

  levels = (levels === undefined)
    ? 32
    : levels;

  var VV_dB = image.select('VV_dB');
  var VH_dB = image.select('VH_dB');


  // ==================================================
  // Quantize dB backscatter
  // ==================================================

  var VV_q = VV_dB
    .clamp(-25, 5)
    .unitScale(-25, 5)
    .multiply(levels - 1)
    .toInt()
    .rename('VV_q');

  var VH_q = VH_dB
    .clamp(-30, 0)
    .unitScale(-30, 0)
    .multiply(levels - 1)
    .toInt()
    .rename('VH_q');


  // ==================================================
  // Calculate GLCM textures
  // ==================================================
  //
  // size = 2 corresponds to a 5 x 5 moving neighbourhood.
  //

  var glcmVV = VV_q.glcmTexture({
    size: 2
  });

  var glcmVH = VH_q.glcmTexture({
    size: 2
  });


  // ==================================================
  // Retain selected texture metrics
  // ==================================================

  var metrics = [
    'asm',
    'corr',
    'var',
    'idm',
    'savg',
    'ent',
    'contrast'
  ];


  /**
   * Selects and standardizes GLCM predictor names.
   */
  var selectMetrics = function(glcmImage, polarization, quantizedName) {

    var oldNames = metrics.map(function(metric) {
      return quantizedName + '_' + metric;
    });

    var newNames = metrics.map(function(metric) {

      var cleanName = metric.toUpperCase();

      if (metric === 'corr') {
        cleanName = 'CORRELATION';
      }

      if (metric === 'var') {
        cleanName = 'VARIANCE';
      }

      if (metric === 'savg') {
        cleanName = 'SUMAVE';
      }

      if (metric === 'ent') {
        cleanName = 'ENTROPY';
      }

      return 'GLCM_' +
        cleanName +
        '_' +
        polarization;
    });

    return glcmImage.select(
      oldNames,
      newNames
    );
  };


  var vvTextures = selectMetrics(
    glcmVV,
    'VV',
    'VV_q'
  );

  var vhTextures = selectMetrics(
    glcmVH,
    'VH',
    'VH_q'
  );


  // ==================================================
  // Add texture predictors
  // ==================================================

  return image
    .addBands(vvTextures)
    .addBands(vhTextures)
    .copyProperties(
      image,
      image.propertyNames()
    );
};


// ======================================================
// Complete Sentinel-1 feature stack
// ======================================================

/**
 * Builds the complete Sentinel-1 predictor stack.
 *
 * Expected input:
 * - VV
 * - VH
 * - VV_dB
 * - VH_dB
 *
 * Final stack:
 * - 2 dB backscatter predictors
 * - 10 polarization/algebraic predictors
 * - 14 GLCM texture predictors
 *
 * Total:
 * - 26 Sentinel-1 predictors per season
 *
 * By default, all output bands receive an S1_ prefix to distinguish them
 * from Sentinel-2 predictors during downstream fusion.
 *
 * @param {ee.Image} baseImage Image containing VV, VH, VV_dB, and VH_dB.
 * @param {number} [levels=32] GLCM grey-level quantization levels.
 * @param {boolean} [addPrefix=true] Whether to prefix final bands with S1_.
 * @returns {ee.Image} Complete Sentinel-1 predictor stack.
 */
exports.buildS1FeatureStack = function(
  baseImage,
  levels,
  addPrefix
) {
  baseImage = ee.Image(baseImage);

  levels = (levels === undefined)
    ? 32
    : levels;

  addPrefix = (addPrefix === undefined)
    ? true
    : addPrefix;


  var withIndices = exports.computeIndices(
    baseImage
  );

  var withTextures = exports.computeGLCM(
    withIndices,
    levels
  );

  var output = ee.Image(
    withTextures
  )
    .toFloat()
    .copyProperties(
      baseImage,
      baseImage.propertyNames()
    );


  if (addPrefix) {
    output = addBandPrefix(
      output,
      'S1_'
    );
  }


  return ee.Image(output);
};