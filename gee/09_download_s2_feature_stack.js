/**
 * @name 09_download_s2_feature_stack
 * @description Builds and exports a seasonal Sentinel-2 predictor stack.
 *
 * Workflow:
 * 1. Loads Sentinel-2 L2A surface-reflectance imagery.
 * 2. Links the collection with Cloud Score+.
 * 3. Applies edge, Cloud Score+, and SCL masking.
 * 4. Scales the selected surface-reflectance bands.
 * 5. Builds a clear-pixel seasonal quality mosaic using cs_cdf.
 * 6. Evaluates valid spatial coverage over the study region.
 * 7. Generates the complete Sentinel-2 predictor stack.
 * 8. Exports the stack to Google Drive.
 *
 * Before running:
 * - Upload the accompanying modules to a Google Earth Engine script repository.
 * - Replace MODULE_ROOT with the corresponding GEE module path.
 * - Replace ROI_ASSET with the study-area asset available to the user.
 */


// ======================================================
// 1. Module configuration
// ======================================================

/**
 * Root path containing the accompanying GEE modules.
 *
 * Example:
 * users/your_username/echo-review-code
 *
 * Do not include a trailing slash.
 */
var MODULE_ROOT =
  'users/your_username/echo-review-code';


var s2Preprocessing = require(
  MODULE_ROOT + '/05_s2_preprocessing'
);

var s2Features = require(
  MODULE_ROOT + '/06_s2_feature_extraction'
);


// ======================================================
// 2. Analysis settings
// ======================================================

/**
 * Update these values for each seasonal export.
 */
var CYCLE_NAME = '2025_2026';
var SEASON_NAME = 'AUTUMN_2026';


/**
 * Sentinel-1 reference date associated with this optical acquisition
 * window.
 *
 * This value is retained as metadata and does not affect Sentinel-2
 * processing.
 */
var S1_TARGET_DATE = '2026-04-08';


/**
 * Sentinel-2 acquisition window selected after clear-pixel assessment
 * using 08_check_s2_clear_mosaic_window.js.
 *
 * Earth Engine filterDate() uses an inclusive start date and exclusive
 * end date.
 */
var S2_START_DATE = '2026-04-10';
var S2_END_DATE = '2026-04-20';


/**
 * Scene-level cloud filter.
 *
 * This is intentionally permissive because usable clear pixels may occur
 * within scenes having substantial scene-level cloud cover.
 */
var S2_MAX_CLOUDY_PIXEL_PERCENTAGE = 100;


/**
 * Cloud Score+ configuration.
 */
var CLOUD_SCORE_BAND = 'cs_cdf';
var CLOUD_SCORE_THRESHOLD = 0.50;


/**
 * Export configuration.
 */
var EXPORT_FOLDER = 'GEE_S1_S2_WPE';
var EXPORT_SCALE = 10;
var EXPORT_CRS = 'EPSG:3577';


/**
 * Scale used for valid-pixel coverage diagnostics.
 */
var STATS_SCALE = 10;


// ======================================================
// 3. Study area
// ======================================================

/**
 * Replace this placeholder with the Earth Engine asset containing the
 * study-area boundary.
 *
 * User-specific and institution-specific Earth Engine asset paths are
 * intentionally excluded from the review repository.
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


print(
  'Study area:',
  studyArea
);

print(
  'Cycle:',
  CYCLE_NAME
);

print(
  'Season:',
  SEASON_NAME
);

print(
  'Sentinel-1 reference date:',
  S1_TARGET_DATE
);

print(
  'Sentinel-2 date window:',
  S2_START_DATE,
  'to',
  S2_END_DATE
);

print(
  'Cloud Score+ band:',
  CLOUD_SCORE_BAND
);

print(
  'Cloud Score+ threshold:',
  CLOUD_SCORE_THRESHOLD
);


// ======================================================
// 4. Metadata helpers
// ======================================================

/**
 * Adds a formatted acquisition-date property to a Sentinel-2 image.
 *
 * @param {ee.Image} image Sentinel-2 image.
 * @returns {ee.Image} Image with date_string property.
 */
function addDateInfo(image) {
  image = ee.Image(
    image
  );

  var dateString = ee.Date(
    image.get('system:time_start')
  ).format(
    'YYYY-MM-dd'
  );

  return image.set(
    'date_string',
    dateString
  );
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
// 5. Load Sentinel-2 and Cloud Score+
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
    S2_START_DATE,
    S2_END_DATE
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
  'Raw Sentinel-2 image count:',
  s2Raw.size()
);

print(
  'Raw Sentinel-2 collection:',
  s2Raw
);

printUniqueDates(
  s2Raw,
  'Sentinel-2'
);

printUniqueTiles(
  s2Raw,
  'Sentinel-2'
);


// ======================================================
// 6. Link Cloud Score+
// ======================================================

/**
 * Cloud Score+ is linked to Sentinel-2 using the harmonized collection.
 * The cs_cdf band is retained for masking and quality mosaicking.
 */
var s2Linked = s2Raw.linkCollection(
  cloudScorePlus,
  [
    CLOUD_SCORE_BAND
  ]
);


print(
  'Cloud Score+ linked image count:',
  s2Linked.size()
);


// ======================================================
// 7. Apply per-image Sentinel-2 preprocessing
// ======================================================

/**
 * 05_s2_preprocessing applies:
 *
 * 1. B8A/B9 edge masking
 * 2. Cloud Score+ masking
 * 3. SCL masking
 * 4. Reflectance scaling
 * 5. Reflectance-band selection
 * 6. Retention of cs_cdf for quality mosaicking
 */
var s2Processed = s2Linked.map(
  function(image) {
    return s2Preprocessing.preprocessS2(
      image,
      CLOUD_SCORE_BAND,
      CLOUD_SCORE_THRESHOLD
    );
  }
);


print(
  'Processed Sentinel-2 image count:',
  s2Processed.size()
);

print(
  'Processed Sentinel-2 collection:',
  s2Processed
);


// ======================================================
// 8. Build clear-pixel Sentinel-2 quality mosaic
// ======================================================

/**
 * qualityMosaic() selects, independently for each pixel, the valid
 * observation having the highest Cloud Score+ cs_cdf value within the
 * selected acquisition window.
 *
 * The resulting mosaic may therefore contain pixels contributed by
 * multiple Sentinel-2 acquisition dates and MGRS tiles.
 */
var s2ClearMosaic =
  s2Preprocessing.makeQualityComposite(
    s2Processed,
    studyArea,
    CLOUD_SCORE_BAND
  );


s2ClearMosaic = ee.Image(
  s2ClearMosaic
).set({
  'sensor': 'Sentinel-2',
  'cycle': CYCLE_NAME,
  'season': SEASON_NAME,
  's1_target_date': S1_TARGET_DATE,
  's2_start_date': S2_START_DATE,
  's2_end_date': S2_END_DATE,
  'cloud_score_band': CLOUD_SCORE_BAND,
  'cloud_score_threshold': CLOUD_SCORE_THRESHOLD,
  'max_cloudy_pixel_percentage':
    S2_MAX_CLOUDY_PIXEL_PERCENTAGE
});


print(
  'Sentinel-2 clear-pixel quality mosaic:',
  s2ClearMosaic
);

print(
  'Sentinel-2 mosaic bands:',
  s2ClearMosaic.bandNames()
);


// ======================================================
// 9. Evaluate valid Sentinel-2 spatial coverage
// ======================================================

/**
 * The RED-band mask is used to assess valid optical coverage over the
 * study region.
 *
 * Example:
 * 1.00 = 100% valid coverage
 * 0.95 = 95% valid coverage
 */
var s2ValidMask = s2ClearMosaic
  .select('RED')
  .mask();


var s2CoverageFraction = s2ValidMask.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: geom,
  scale: STATS_SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  bestEffort: true
}).get(
  'RED'
);


var s2CoveragePercent = ee.Number(
  s2CoverageFraction
).multiply(
  100
);


print(
  'Sentinel-2 valid coverage fraction:',
  s2CoverageFraction
);

print(
  'Sentinel-2 valid coverage percent:',
  s2CoveragePercent
);


Map.addLayer(
  s2ValidMask,
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


// ======================================================
// 10. Build Sentinel-2 predictor stack
// ======================================================

/**
 * The final seasonal Sentinel-2 stack contains:
 *
 * - 10 surface-reflectance predictors
 * - 8 spectral-index predictors
 *
 * Total = 18 predictors per season.
 */
var s2FeatureStack = s2Features
  .buildS2FeatureStack(
    s2ClearMosaic,
    true
  )
  .set({
    'sensor': 'Sentinel-2',
    'cycle': CYCLE_NAME,
    'season': SEASON_NAME,
    's1_target_date': S1_TARGET_DATE,
    's2_start_date': S2_START_DATE,
    's2_end_date': S2_END_DATE,
    'cloud_score_band': CLOUD_SCORE_BAND,
    'cloud_score_threshold': CLOUD_SCORE_THRESHOLD,
    'max_cloudy_pixel_percentage':
      S2_MAX_CLOUDY_PIXEL_PERCENTAGE,
    'valid_fraction_roi':
      s2CoverageFraction,
    'valid_percent_roi':
      s2CoveragePercent
  });


s2FeatureStack = ee.Image(
  s2FeatureStack
);


print(
  'Sentinel-2 feature stack:',
  s2FeatureStack
);

print(
  'Sentinel-2 feature bands:',
  s2FeatureStack.bandNames()
);

print(
  'Sentinel-2 predictor count:',
  s2FeatureStack.bandNames().size()
);


// ======================================================
// 11. Visual quality-control layers
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
  'Sentinel-2 clear-pixel RGB mosaic'
);


Map.addLayer(
  s2FeatureStack.select(
    'S2_NDVI'
  ),
  {
    min: -0.2,
    max: 0.8,
    palette: [
      'brown',
      'yellow',
      'green'
    ]
  },
  'Sentinel-2 NDVI',
  false
);


Map.addLayer(
  s2FeatureStack.select(
    'S2_NDWI'
  ),
  {
    min: -0.5,
    max: 0.5,
    palette: [
      'brown',
      'white',
      'blue'
    ]
  },
  'Sentinel-2 NDWI',
  false
);


// ======================================================
// 12. Export Sentinel-2 predictor stack
// ======================================================

var exportName =
  'S2_WPE_FEATURES_' +
  SEASON_NAME;


Export.image.toDrive({
  image: s2FeatureStack,
  description: exportName,
  folder: EXPORT_FOLDER,
  fileNamePrefix: exportName,
  region: geom,
  scale: EXPORT_SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13
});