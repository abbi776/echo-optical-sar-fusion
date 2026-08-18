/**
 * @name 02_s1_speckle_filter
 * @description Applies a Refined Lee speckle filter independently to the
 * Sentinel-1 VV and VH backscatter bands.
 *
 * The adaptive directional filter reduces speckle while preserving local
 * edges and spatial structure. Filtering is performed in linear power.
 * The Sentinel-1 incidence-angle band is retained unchanged for subsequent
 * radiometric normalization.
 *
 * Requirements:
 * - Input backscatter must be in linear power, not dB.
 * - Input image must contain VV, VH, and angle bands.
 */


/**
 * Applies the Refined Lee speckle filter to Sentinel-1 VV and VH.
 *
 * The implementation:
 * 1. Calculates local 3 x 3 mean and variance.
 * 2. Detects local gradients from a sampled 7 x 7 neighbourhood.
 * 3. Determines one of eight dominant edge directions.
 * 4. Estimates local noise variance.
 * 5. Calculates directional statistics.
 * 6. Applies the adaptive Refined Lee weighting.
 *
 * @param {ee.Image} image Sentinel-1 image containing linear VV, VH,
 * and incidence angle.
 * @returns {ee.Image} Filtered VV and VH bands with the original angle
 * band and image properties retained.
 */
exports.refinedLee = function(image) {
  image = ee.Image(image);

  // Prevent numerical instability in variance-based calculations.
  var eps = 1e-6;

  // Apply the filter independently to VV and VH.
  var bandNames = ee.List([
    'VV',
    'VH'
  ]);

  var filteredBands = ee.ImageCollection(
    bandNames.map(function(bandName) {
      bandName = ee.String(bandName);

      var img = image.select([bandName]);


      // ==================================================
      // 1. Local statistics: 3 x 3 neighbourhood
      // ==================================================

      var weights3 = ee.List.repeat(
        ee.List.repeat(1, 3),
        3
      );

      var kernel3 = ee.Kernel.fixed(
        3,
        3,
        weights3,
        1,
        1,
        false
      );

      var mean3 = img.reduceNeighborhood(
        ee.Reducer.mean(),
        kernel3
      );

      var variance3 = img.reduceNeighborhood(
        ee.Reducer.variance(),
        kernel3
      );


      // ==================================================
      // 2. Gradient estimation: sampled 7 x 7 neighbourhood
      // ==================================================

      var sampleWeights = ee.List([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0]
      ]);

      var sampleKernel = ee.Kernel.fixed(
        7,
        7,
        sampleWeights,
        3,
        3,
        false
      );

      var sampleMean = mean3.neighborhoodToBands(
        sampleKernel
      );

      var sampleVariance = variance3.neighborhoodToBands(
        sampleKernel
      );

      var gradients = sampleMean
        .select(1)
        .subtract(sampleMean.select(7))
        .abs();

      gradients = gradients.addBands(
        sampleMean
          .select(6)
          .subtract(sampleMean.select(2))
          .abs()
      );

      gradients = gradients.addBands(
        sampleMean
          .select(3)
          .subtract(sampleMean.select(5))
          .abs()
      );

      gradients = gradients.addBands(
        sampleMean
          .select(0)
          .subtract(sampleMean.select(8))
          .abs()
      );

      var maxGradient = gradients.reduce(
        ee.Reducer.max()
      );

      var gradientMask = gradients.eq(maxGradient);

      // Duplicate the four gradient masks for the eight possible
      // directional orientations.
      gradientMask = gradientMask.addBands(
        gradientMask
      );


      // ==================================================
      // 3. Determine dominant direction
      // ==================================================

      var directions = sampleMean
        .select(1)
        .subtract(sampleMean.select(4))
        .gt(
          sampleMean
            .select(4)
            .subtract(sampleMean.select(7))
        )
        .multiply(1);

      directions = directions.addBands(
        sampleMean
          .select(6)
          .subtract(sampleMean.select(4))
          .gt(
            sampleMean
              .select(4)
              .subtract(sampleMean.select(2))
          )
          .multiply(2)
      );

      directions = directions.addBands(
        sampleMean
          .select(3)
          .subtract(sampleMean.select(4))
          .gt(
            sampleMean
              .select(4)
              .subtract(sampleMean.select(5))
          )
          .multiply(3)
      );

      directions = directions.addBands(
        sampleMean
          .select(0)
          .subtract(sampleMean.select(4))
          .gt(
            sampleMean
              .select(4)
              .subtract(sampleMean.select(8))
          )
          .multiply(4)
      );

      // Add the four opposite directions.
      directions = directions.addBands(
        directions
          .select(0)
          .not()
          .multiply(5)
      );

      directions = directions.addBands(
        directions
          .select(1)
          .not()
          .multiply(6)
      );

      directions = directions.addBands(
        directions
          .select(2)
          .not()
          .multiply(7)
      );

      directions = directions.addBands(
        directions
          .select(3)
          .not()
          .multiply(8)
      );

      directions = directions
        .updateMask(gradientMask)
        .reduce(ee.Reducer.sum());


      // ==================================================
      // 4. Estimate local noise variance
      // ==================================================

      var sampleStats = sampleVariance.divide(
        sampleMean
          .multiply(sampleMean)
          .max(eps)
      );

      // Use the five smallest normalized local variances to estimate
      // the local speckle noise variance.
      var sigmaV = sampleStats
        .toArray()
        .arraySort()
        .arraySlice(0, 0, 5)
        .arrayReduce(
          ee.Reducer.mean(),
          [0]
        );


      // ==================================================
      // 5. Directional kernels
      // ==================================================

      var rectangularWeights = ee.List.repeat(
        ee.List.repeat(0, 7),
        3
      ).cat(
        ee.List.repeat(
          ee.List.repeat(1, 7),
          4
        )
      );

      var diagonalWeights = ee.List([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1]
      ]);

      var rectangularKernel = ee.Kernel.fixed(
        7,
        7,
        rectangularWeights,
        3,
        3,
        false
      );

      var diagonalKernel = ee.Kernel.fixed(
        7,
        7,
        diagonalWeights,
        3,
        3,
        false
      );


      // Direction 1.
      var directionalMean = img
        .reduceNeighborhood(
          ee.Reducer.mean(),
          rectangularKernel
        )
        .updateMask(
          directions.eq(1)
        );

      var directionalVariance = img
        .reduceNeighborhood(
          ee.Reducer.variance(),
          rectangularKernel
        )
        .updateMask(
          directions.eq(1)
        );


      // Direction 2.
      directionalMean = directionalMean.addBands(
        img
          .reduceNeighborhood(
            ee.Reducer.mean(),
            diagonalKernel
          )
          .updateMask(
            directions.eq(2)
          )
      );

      directionalVariance = directionalVariance.addBands(
        img
          .reduceNeighborhood(
            ee.Reducer.variance(),
            diagonalKernel
          )
          .updateMask(
            directions.eq(2)
          )
      );


      // Directions 3-8 generated by rotating the two base kernels.
      for (var i = 1; i < 4; i++) {

        directionalMean = directionalMean.addBands(
          img
            .reduceNeighborhood(
              ee.Reducer.mean(),
              rectangularKernel.rotate(i)
            )
            .updateMask(
              directions.eq(2 * i + 1)
            )
        );

        directionalVariance = directionalVariance.addBands(
          img
            .reduceNeighborhood(
              ee.Reducer.variance(),
              rectangularKernel.rotate(i)
            )
            .updateMask(
              directions.eq(2 * i + 1)
            )
        );

        directionalMean = directionalMean.addBands(
          img
            .reduceNeighborhood(
              ee.Reducer.mean(),
              diagonalKernel.rotate(i)
            )
            .updateMask(
              directions.eq(2 * i + 2)
            )
        );

        directionalVariance = directionalVariance.addBands(
          img
            .reduceNeighborhood(
              ee.Reducer.variance(),
              diagonalKernel.rotate(i)
            )
            .updateMask(
              directions.eq(2 * i + 2)
            )
        );
      }

      directionalMean = directionalMean.reduce(
        ee.Reducer.sum()
      );

      directionalVariance = directionalVariance.reduce(
        ee.Reducer.sum()
      );


      // ==================================================
      // 6. Refined Lee adaptive filtering
      // ==================================================

      var varX = directionalVariance
        .subtract(
          directionalMean
            .multiply(directionalMean)
            .multiply(sigmaV)
        )
        .divide(
          sigmaV.add(1.0)
        );

      var filterWeight = varX.divide(
        directionalVariance.max(eps)
      );

      var filtered = directionalMean
        .add(
          filterWeight.multiply(
            img.subtract(directionalMean)
          )
        )
        .arrayProject([0])
        .arrayFlatten([['filtered']])
        .rename([bandName]);

      return ee.Image(filtered);
    })
  )
    .toBands()
    .rename([
      'VV',
      'VH'
    ]);


  // ======================================================
  // Preserve incidence angle for subsequent preprocessing
  // ======================================================

  var output = ee.Image.cat([
    filteredBands,
    image.select('angle')
  ]);

  return output.copyProperties(
    image,
    image.propertyNames()
  );
};