/**
 * @name 08_check_s2_clear_mosaic_window
 * @description Evaluates Sentinel-2 clear-pixel availability around a
 * selected Sentinel-1 acquisition date.
 *
 * Purpose:
 * - Search Sentinel-2 imagery within a configurable temporal window around
 *   a Sentinel-1 reference date.
 * - Link Sentinel-2 SR imagery with Cloud Score+.
 * - Apply edge, Cloud Score+, and SCL masking.
 * - Evaluate valid clear-pixel coverage over the study region.
 * - Build a Cloud Score+ quality mosaic from multiple dates and MGRS tiles.
 * - Inspect the acquisition dates contributing to the final mosaic.
 *
 * This script is intended as a diagnostic step for selecting the temporal
 * window subsequently used by 09_download_s2_feature_stack.js.
 */


// ======================================================
// 1. Analysis settings
// ======================================================

/**
 * Sentinel-1 reference date.
 *
 * When a Sentinel-1 mosaic contains more than one acquisition date,
 * use a representative date for the acquisition window being evaluated.
 */
var S1_TARGET_DATE = '2025-07-11';


/**
 * Search window around the Sentinel-1 reference date.
 *
 * The values can be increased when insufficient cloud-free Sentinel-2
 * coverage is available within the initial window.
 */
var SEARCH_DAYS_BEFORE = 10;
var SEARCH_DAYS_AFTER = 10;


/**
 * Scene-level Sentinel-2 cloud filter.
 *
 * This is intentionally permissive because useful clear pixels may still
 * occur within scenes having high scene-level cloud percentages.
 */
var S2_MAX_CLOUDY_PIXEL_PERCENTAGE = 100;


/**
 * Cloud Score+ configuration.
 */
var CLOUD_SCORE_BAND = 'cs_cdf';
var CLOUD_SCORE_THRESHOLD = 0.50;


/**
 * Spatial scale used for coverage diagnostics.
 */
var STATS_SCALE = 10;


/**
 * Number of candidate records displayed in the console.
 */
var TOP_N = 20;


// ======================================================
// 2. Study area
// ======================================================

/**
 * Replace this placeholder with the Earth Engine asset containing the
 * study-area boundary.
 *
 * User-specific and institution-specific asset paths are intentionally
 * excluded from the review repository.
 */
var ROI_ASSET =
  'projects/your-project/assets/study_area_boundary';


var studyArea = ee.FeatureCollection(
  ROI_ASSET
);

var geom = studyArea.geometry();


Map.centerObject(
  geom,
  8
);

Map.addLayer(
  studyArea,
  {
    color: 'yellow'
  },
  'Study Area'
);


// ======================================================
// 3. Search dates
// ======================================================

var targetDate = ee.Date(
  S1_TARGET_DATE
);

var startDate = targetDate.advance(
  -SEARCH_DAYS_BEFORE,
  'day'
);

/**
 * Earth Engine filterDate() treats the end date as exclusive.
 * Adding one day therefore includes the complete requested final day.
 */
var endDate = targetDate.advance(
  SEARCH_DAYS_AFTER + 1,
  'day'
);


print(
  'Sentinel-1 reference date:',
  S1_TARGET_DATE
);

print(
  'Sentinel-2 search start:',
  startDate.format('YYYY-MM-dd')
);

print(
  'Sentinel-2 search end:',
  endDate.format('YYYY-MM-dd')
);

print(
  'Search window:',
  '-' + SEARCH_DAYS_BEFORE +
  ' / +' + SEARCH_DAYS_AFTER +
  ' days'
);


// ======================================================
// 4. Sentinel-2 band definitions
// ======================================================

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
// 5. Metadata helpers
// ======================================================

/**
 * Adds acquisition-date information relative to the Sentinel-1
 * reference date.
 *
 * @param {ee.Image} image Sentinel-2 image.
 * @returns {ee.Image} Image containing temporal diagnostic properties.
 */
function addDateInfo(image) {
  image = ee.Image(
    image
  );

  var imageDate = ee.Date(
    image.get('system:time_start')
  );

  var dateString = imageDate.format(
    'YYYY-MM-dd'
  );

  var daysFromS1 = imageDate.difference(
    targetDate,
    'day'
  );

  return image.set({
    'date_string': dateString,
    'days_from_s1': daysFromS1,
    'abs_days_from_s1': daysFromS1.abs()
  });
}


/**
 * Prints unique acquisition dates represented by an image collection.
 */
function printUniqueDates(
  collection,
  label
) {
  collection = ee.ImageCollection(
    collection
  );

  var dates = ee.List(
    collection.aggregate_array(
      'date_string'
    )
  ).distinct();

  print(
    label + ' unique dates:',
    dates
  );
}


/**
 * Prints unique Sentinel-2 MGRS tiles represented by a collection.
 */
function printUniqueTiles(
  collection,
  label
) {
  collection = ee.ImageCollection(
    collection
  );

  var tiles = ee.List(
    collection.aggregate_array(
      'MGRS_TILE'
    )
  ).distinct();

  print(
    label + ' unique MGRS tiles:',
    tiles
  );
}


// ======================================================
// 6. Sentinel-2 masking functions
// ======================================================

/**
 * Applies additional Sentinel-2 edge masking using the masks of B8A
 * and B9.
 *
 * @param {ee.Image} image Sentinel-2 image.
 * @returns {ee.Image} Edge-masked image.
 */
function maskEdges(image) {
  image = ee.Image(
    image
  );

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
}


/**
 * Applies Cloud Score+ clear-pixel masking.
 *
 * Pixels are retained when:
 *
 *   cs_cdf >= 0.50
 *
 * @param {ee.Image} image Sentinel-2 image with linked Cloud Score+ band.
 * @returns {ee.Image} Cloud Score+ masked image.
 */
function maskCloudScorePlus(image) {
  image = ee.Image(
    image
  );

  var clearMask = image
    .select(
      CLOUD_SCORE_BAND
    )
    .gte(
      CLOUD_SCORE_THRESHOLD
    );

  return image
    .updateMask(
      clearMask
    )
    .copyProperties(
      image,
      image.propertyNames()
    );
}


/**
 * Masks unsuitable Sentinel-2 Scene Classification Layer classes.
 *
 * Masked:
 *  0 = No data
 *  1 = Saturated or defective
 *  3 = Cloud shadow
 *  8 = Medium-probability cloud
 *  9 = High-probability cloud
 * 10 = Thin cirrus
 * 11 = Snow/ice
 *
 * Retained:
 *  2 = Dark area pixels
 *  4 = Vegetation
 *  5 = Bare soil
 *  6 = Water
 *  7 = Unclassified / low-probability cloud
 *
 * @param {ee.Image} image Sentinel-2 image containing SCL.
 * @returns {ee.Image} SCL-masked image.
 */
function maskSCL(image) {
  image = ee.Image(
    image
  );

  var scl = image.select(
    'SCL'
  );

  var validMask = scl
    .neq(0)
    .and(scl.neq(1))
    .and(scl.neq(3))
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));

  return image
    .updateMask(
      validMask
    )
    .copyProperties(
      image,
      image.propertyNames()
    );
}


// ======================================================
// 7. Preprocessing for quality mosaicking
// ======================================================

/**
 * Preprocesses a Sentinel-2 image while retaining the Cloud Score+
 * quality band required by qualityMosaic().
 *
 * A source-time band is also created for diagnostic inspection of the
 * acquisition dates contributing pixels to the final mosaic.
 *
 * @param {ee.Image} image Linked Sentinel-2 / Cloud Score+ image.
 * @returns {ee.Image} Preprocessed image containing reflectance,
 * Cloud Score+, and source-time bands.
 */
function preprocessS2ForQualityMosaic(image) {
  image = ee.Image(
    image
  );

  var original = image;


  // ==================================================
  // Apply masks
  // ==================================================

  var edgeMasked = maskEdges(
    image
  );

  var cloudMasked = maskCloudScorePlus(
    edgeMasked
  );

  var sclMasked = maskSCL(
    cloudMasked
  );


  // ==================================================
  // Scale and rename reflectance bands
  // ==================================================

  var reflectance = sclMasked
    .select(
      S2_REFLECTANCE_BANDS,
      S2_REFLECTANCE_NAMES
    )
    .multiply(0.0001)
    .toFloat();


  // ==================================================
  // Retain Cloud Score+ quality band
  // ==================================================

  var quality = sclMasked
    .select(
      CLOUD_SCORE_BAND
    )
    .rename(
      CLOUD_SCORE_BAND
    )
    .toFloat();


  // ==================================================
  // Add source acquisition time as diagnostic band
  // ==================================================

  var sourceTime = ee.Number(
    original.get(
      'system:time_start'
    )
  );

  var sourceTimeBand = ee.Image
    .constant(
      sourceTime
    )
    .rename(
      'source_time'
    )
    .toDouble()
    .updateMask(
      reflectance
        .select('RED')
        .mask()
    );


  // ==================================================
  // Assemble image
  // ==================================================

  var output = reflectance
    .addBands(
      quality
    )
    .addBands(
      sourceTimeBand
    )
    .copyProperties(
      original,
      original.propertyNames()
    );


  return output.set({
    'date_string':
      original.get('date_string'),

    'days_from_s1':
      original.get('days_from_s1'),

    'abs_days_from_s1':
      original.get('abs_days_from_s1'),

    'CLOUDY_PIXEL_PERCENTAGE':
      original.get('CLOUDY_PIXEL_PERCENTAGE'),

    'MGRS_TILE':
      original.get('MGRS_TILE'),

    'SPACECRAFT_NAME':
      original.get('SPACECRAFT_NAME'),

    'system_index':
      original.get('system:index')
  });
}


// ======================================================
// 8. Valid-pixel statistics
// ======================================================

/**
 * Calculates valid ROI coverage for an individual masked image.
 *
 * @param {ee.Image} image Preprocessed Sentinel-2 image.
 * @returns {ee.Image} Image with valid coverage properties.
 */
function addImageValidStats(image) {
  image = ee.Image(
    image
  );

  var validMask = image
    .select('RED')
    .mask();

  var validFraction = validMask.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: geom,
    scale: STATS_SCALE,
    maxPixels: 1e13,
    bestEffort: true
  }).get(
    'RED'
  );


  return image.set({
    'valid_fraction_roi':
      validFraction,

    'valid_percent_roi':
      ee.Number(
        validFraction
      ).multiply(100)
  });
}


/**
 * Calculates valid ROI coverage for a completed Sentinel-2 mosaic.
 *
 * @param {ee.Image} mosaic Sentinel-2 mosaic.
 * @returns {ee.Number} Fraction of valid pixels over the study region.
 */
function getMosaicValidFraction(mosaic) {
  mosaic = ee.Image(
    mosaic
  );

  var validMask = mosaic
    .select('RED')
    .mask();

  return validMask.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: geom,
    scale: STATS_SCALE,
    maxPixels: 1e13,
    bestEffort: true
  }).get(
    'RED'
  );
}


// ======================================================
// 9. Load Sentinel-2 and Cloud Score+
// ======================================================

var cloudScorePlus = ee.ImageCollection(
  'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'
);


var s2Raw = ee.ImageCollection(
  'COPERNICUS/S2_SR_HARMONIZED'
)
  .filterBounds(
    geom
  )
  .filterDate(
    startDate,
    endDate
  )
  .filter(
    ee.Filter.lt(
      'CLOUDY_PIXEL_PERCENTAGE',
      S2_MAX_CLOUDY_PIXEL_PERCENTAGE
    )
  )
  .map(
    addDateInfo
  );


print(
  'Raw Sentinel-2 candidate count:',
  s2Raw.size()
);

print(
  'Raw Sentinel-2 collection:',
  s2Raw
);

printUniqueDates(
  s2Raw,
  'Sentinel-2 raw'
);

printUniqueTiles(
  s2Raw,
  'Sentinel-2 raw'
);


// ======================================================
// 10. Link Cloud Score+ and preprocess
// ======================================================

var s2Linked = s2Raw.linkCollection(
  cloudScorePlus,
  [
    CLOUD_SCORE_BAND
  ]
);


var s2Processed = s2Linked
  .map(
    preprocessS2ForQualityMosaic
  )
  .map(
    addImageValidStats
  );


print(
  'Processed Sentinel-2 candidate count:',
  s2Processed.size()
);

print(
  'Processed Sentinel-2 collection:',
  s2Processed
);


// ======================================================
// 11. Candidate-image diagnostics
// ======================================================

/**
 * Rank individual images by valid ROI coverage.
 *
 * Temporal distance from Sentinel-1 is retained in the candidate table so
 * that coverage and acquisition proximity can be considered together when
 * selecting the final compositing window.
 */
var rankedImages = s2Processed.sort(
  'valid_fraction_roi',
  false
);


var candidateTable = rankedImages.map(
  function(image) {
    image = ee.Image(
      image
    );

    return ee.Feature(
      null,
      {
        'date':
          image.get('date_string'),

        'days_from_s1':
          image.get('days_from_s1'),

        'abs_days_from_s1':
          image.get('abs_days_from_s1'),

        'valid_percent_roi':
          image.get('valid_percent_roi'),

        'cloudy_pixel_percentage_tile':
          image.get(
            'CLOUDY_PIXEL_PERCENTAGE'
          ),

        'mgrs_tile':
          image.get('MGRS_TILE'),

        'spacecraft':
          image.get('SPACECRAFT_NAME'),

        'system_index':
          image.get('system_index')
      }
    );
  }
);


print(
  'Individual Sentinel-2 candidates ranked by valid ROI coverage:',
  candidateTable.limit(
    TOP_N
  )
);


// ======================================================
// 12. Build clear-pixel Sentinel-2 quality mosaic
// ======================================================

/**
 * qualityMosaic() selects, independently for each pixel, the observation
 * having the highest Cloud Score+ value among all valid observations within
 * the selected temporal window.
 *
 * The resulting mosaic may therefore combine pixels from multiple
 * Sentinel-2 dates and MGRS tiles.
 */
var s2ClearMosaicAllBands = s2Processed
  .qualityMosaic(
    CLOUD_SCORE_BAND
  )
  .clip(
    geom
  );

s2ClearMosaicAllBands = ee.Image(
  s2ClearMosaicAllBands
);


print(
  'Sentinel-2 quality mosaic:',
  s2ClearMosaicAllBands
);

print(
  'Sentinel-2 quality-mosaic bands:',
  s2ClearMosaicAllBands.bandNames()
);


// ======================================================
// 13. Reflectance-only diagnostic mosaic
// ======================================================

var s2ClearMosaic = s2ClearMosaicAllBands
  .select(
    S2_REFLECTANCE_NAMES
  )
  .toFloat()
  .set({
    's1_target_date':
      S1_TARGET_DATE,

    's2_search_start':
      startDate.format('YYYY-MM-dd'),

    's2_search_end':
      endDate.format('YYYY-MM-dd'),

    'cloud_score_band':
      CLOUD_SCORE_BAND,

    'cloud_score_threshold':
      CLOUD_SCORE_THRESHOLD,

    'search_days_before':
      SEARCH_DAYS_BEFORE,

    'search_days_after':
      SEARCH_DAYS_AFTER
  });


s2ClearMosaic = ee.Image(
  s2ClearMosaic
);


var mosaicValidFraction = getMosaicValidFraction(
  s2ClearMosaic
);

var mosaicValidPercent = ee.Number(
  mosaicValidFraction
).multiply(
  100
);


print(
  'Sentinel-2 quality-mosaic valid fraction:',
  mosaicValidFraction
);

print(
  'Sentinel-2 quality-mosaic valid percent:',
  mosaicValidPercent
);


// ======================================================
// 14. Source-date diagnostics
// ======================================================

/**
 * The source_time band records the acquisition timestamp of the image
 * selected by qualityMosaic() at each pixel.
 */
var sourceTimeStats = s2ClearMosaicAllBands
  .select(
    'source_time'
  )
  .reduceRegion({
    reducer: ee.Reducer.minMax(),
    geometry: geom,
    scale: STATS_SCALE,
    maxPixels: 1e13,
    bestEffort: true
  });


print(
  'Source acquisition-time min/max:',
  sourceTimeStats
);


// ======================================================
// 15. Visual quality-control layers
// ======================================================

Map.addLayer(
  s2ClearMosaic,
  {
    bands: [
      'RED',
      'GREEN',
      'BLUE'
    ],
    min: 0,
    max: 0.3
  },
  'Sentinel-2 clear-pixel quality mosaic'
);


Map.addLayer(
  s2ClearMosaic
    .select('RED')
    .mask(),
  {
    min: 0,
    max: 1,
    palette: [
      'red',
      'green'
    ]
  },
  'Sentinel-2 valid coverage',
  false
);


Map.addLayer(
  s2ClearMosaicAllBands.select(
    CLOUD_SCORE_BAND
  ),
  {
    min: 0,
    max: 1,
    palette: [
      'red',
      'yellow',
      'green'
    ]
  },
  'Selected Cloud Score+ values',
  false
);


// ======================================================
// 16. Display top individual candidates
// ======================================================

/**
 * Up to five high-coverage individual images are displayed for comparison
 * with the multi-date quality mosaic.
 */
var topCandidates = rankedImages
  .limit(5)
  .toList(5);


for (var i = 0; i < 5; i++) {

  var candidate = ee.Image(
    topCandidates.get(i)
  );

  Map.addLayer(
    candidate.select(
      S2_REFLECTANCE_NAMES
    ),
    {
      bands: [
        'RED',
        'GREEN',
        'BLUE'
      ],
      min: 0,
      max: 0.3
    },
    'Individual Sentinel-2 candidate ' +
      (i + 1),
    false
  );

  Map.addLayer(
    candidate
      .select('RED')
      .mask(),
    {
      min: 0,
      max: 1,
      palette: [
        'red',
        'green'
      ]
    },
    'Candidate valid mask ' +
      (i + 1),
    false
  );
}