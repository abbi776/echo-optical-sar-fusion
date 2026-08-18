/**
 * @name 01_s1_border_noise
 * @description Masks Sentinel-1 pixels with incidence angles outside the
 * selected valid swath range.
 *
 * This provides a practical border-noise masking step for Sentinel-1 GRD
 * imagery before speckle filtering and radiometric normalization.
 *
 * Default incidence-angle range:
 * - Minimum: 30.63993 degrees
 * - Maximum: 45.23993 degrees
 */


/**
 * Masks Sentinel-1 pixels using the ellipsoidal incidence-angle band.
 *
 * Pixels outside the specified range are removed to reduce low-quality
 * swath-edge artefacts.
 *
 * @param {ee.Image} image Sentinel-1 image containing VV, VH, and angle.
 * @param {number} [minDeg=30.63993] Minimum valid incidence angle.
 * @param {number} [maxDeg=45.23993] Maximum valid incidence angle.
 * @returns {ee.Image} Masked Sentinel-1 image with properties preserved.
 */
exports.maskByIncidenceAngle = function(image, minDeg, maxDeg) {
  image = ee.Image(image);

  minDeg = (minDeg === undefined) ? 30.63993 : minDeg;
  maxDeg = (maxDeg === undefined) ? 45.23993 : maxDeg;

  var angle = image.select('angle');

  var validAngleMask = angle
    .gt(minDeg)
    .and(angle.lt(maxDeg));

  return image
    .updateMask(validAngleMask)
    .copyProperties(image, image.propertyNames());
};