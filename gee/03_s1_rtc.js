/**
 * @name 03_s1_rtc
 * @description Applies ellipsoidal radiometric normalization to Sentinel-1
 * VV and VH backscatter.
 *
 * The workflow converts Sigma0 backscatter in linear power to approximate
 * Gamma0 using the ellipsoidal incidence angle provided by the Sentinel-1
 * GRD 'angle' band.
 *
 * Formula:
 *   Gamma0 = Sigma0 / cos(theta_i)
 *
 * This is an ellipsoidal Gamma0 normalization rather than DEM-based
 * radiometric terrain flattening.
 *
 * Reference:
 * Small, D. (2011). Flattening Gamma: Radiometric Terrain Correction
 * for SAR Imagery.
 */


/**
 * Applies ellipsoidal Gamma0 normalization to Sentinel-1 VV and VH.
 *
 * Requirements:
 * - Input backscatter must be in linear power.
 * - Input image must contain VV, VH, and angle bands.
 *
 * The incidence-angle band is retained unchanged for quality control
 * and downstream processing.
 *
 * @param {ee.Image} image Sentinel-1 image containing linear VV, VH,
 * and incidence angle.
 * @returns {ee.Image} Gamma0-normalized VV and VH with the original
 * incidence-angle band and image properties retained.
 */
exports.applyRTC = function(image) {
  image = ee.Image(image);

  // Protect against numerical instability in the cosine denominator.
  var eps = 1e-6;

  // Convert the ellipsoidal incidence angle from degrees to radians.
  var thetaI = image
    .select('angle')
    .multiply(Math.PI / 180);

  // Convert Sigma0 to approximate Gamma0:
  // Gamma0 = Sigma0 / cos(theta_i)
  var gamma0 = image
    .select([
      'VV',
      'VH'
    ])
    .divide(
      thetaI.cos().max(eps)
    )
    .rename([
      'VV',
      'VH'
    ]);

  return ee.Image.cat([
    gamma0,
    image.select('angle')
  ])
    .copyProperties(
      image,
      image.propertyNames()
    );
};