/**
 * @name 07_download_s1_feature_stack
 * @description Builds and exports a seasonal Sentinel-1 predictor stack.
 *
 * Workflow:
 * 1. Loads Sentinel-1 GRD imagery for the specified acquisition window.
 * 2. Filters to IW mode, dual VV/VH polarization, and the selected orbit pass.
 * 3. Converts backscatter from dB to linear power.
 * 4. Applies incidence-angle masking.
 * 5. Applies Refined Lee speckle filtering.
 * 6. Applies ellipsoidal Gamma0 radiometric normalization.
 * 7. Mosaics available scenes over the study region.
 * 8. Evaluates valid Sentinel-1 spatial coverage.
 * 9. Generates the complete Sentinel-1 predictor stack.
 * 10. Exports the stack to Google Drive.
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
 * users/your_username/echo-review-code/
 *
 * Do not include a trailing slash.
 */
var MODULE_ROOT = 'users/your_username/echo-review-code';


var utils = require(
  MODULE_ROOT + '/00_utils'
);

var border = require(
  MODULE_ROOT + '/01_s1_border_noise'
);

var speckle = require(
  MODULE_ROOT + '/02_s1_speckle_filter'
);

var rtc = require(
  MODULE_ROOT + '/03_s1_rtc'
);

var s1Features = require(
  MODULE_ROOT + '/04_s1_feature_extraction'
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
 * Sentinel-1 acquisition window.
 *
 * Earth Engine filterDate() uses an inclusive start date and exclusive
 * end date.
 */
var S1_START_DATE = '2026-04-10';
var S1_END_DATE = '2026-04-15';


/**
 * Sentinel-1 configuration.
 */
var S1_ORBIT_PASS = 'DESCENDING';
var GLCM_LEVELS = 32;


/**
 * Export configuration.
 */
var EXPORT_FOLDER = 'GEE_S1_S2_WPE';
var EXPORT_SCALE = 10;
var EXPORT_CRS = 'EPSG:3577';


// ======================================================
// 3. Study area
// ======================================================

/**
 * Replace this placeholder with the Earth Engine asset containing the
 * study-area boundary.
 *
 * The review repository intentionally does not contain user-specific or
 * institution-specific Earth Engine asset paths.
 */
var ROI_ASSET = 'projects/your-project/assets/study_area_boundary';


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
  'Sentinel-1 date window:',
  S1_START_DATE,
  'to',
  S1_END_DATE
);

print(
  'Sentinel-1 orbit pass:',
  S1_ORBIT_PASS
);


// ======================================================
// 4. Helper functions
// ======================================================

/**
 * Adds a formatted acquisition-date property to an image.
 *
 * @param {ee.Image} image Sentinel-1 image.
 * @returns {ee.Image} Image with date_string property.
 */
function addDateString(image) {
  image = ee.Image(image);

  var dateString = ee.Date(
    image.get('system:time_start')
  ).format('YYYY-MM-dd');

  return image.set(
    'date_string',
    dateString
  );
}


/**
 * Prints unique acquisition dates represented by an image collection.
 *
 * @param {ee.ImageCollection} collection Image collection.
 * @param {string} label Label shown in the console.
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
      'system:time_start'
    )
  )
    .map(function(dateValue) {
      return ee.Date(
        dateValue
      ).format('YYYY-MM-dd');
    })
    .distinct();

  print(
    label + ' unique dates:',
    dates
  );
}


/**
 * Applies per-image Sentinel-1 preprocessing.
 *
 * Processing order:
 * 1. dB to linear power
 * 2. incidence-angle masking
 * 3. Refined Lee speckle filtering
 * 4. ellipsoidal Gamma0 normalization
 *
 * @param {ee.Image} image Sentinel-1 GRD image.
 * @returns {ee.Image} Preprocessed Sentinel-1 image.
 */
function preprocessS1Image(image) {
  image = ee.Image(image);

  var linear = utils.dbToLin(
    image
  );

  var angleMasked = border.maskByIncidenceAngle(
    linear
  );

  var filtered = speckle.refinedLee(
    angleMasked
  );

  var gamma0 = rtc.applyRTC(
    filtered
  );

  return gamma0.copyProperties(
    image,
    image.propertyNames()
  );
}


// ======================================================
// 5. Load and preprocess Sentinel-1
// ======================================================

var s1Collection = ee.ImageCollection(
  'COPERNICUS/S1_GRD'
)
  .filterBounds(
    geom
  )
  .filterDate(
    S1_START_DATE,
    S1_END_DATE
  )
  .filter(
    ee.Filter.eq(
      'instrumentMode',
      'IW'
    )
  )
  .filter(
    ee.Filter.eq(
      'orbitProperties_pass',
      S1_ORBIT_PASS
    )
  )
  .filter(
    ee.Filter.listContains(
      'transmitterReceiverPolarisation',
      'VV'
    )
  )
  .filter(
    ee.Filter.listContains(
      'transmitterReceiverPolarisation',
      'VH'
    )
  )
  .map(
    addDateString
  )
  .map(
    preprocessS1Image
  );


print(
  'Processed Sentinel-1 image count:',
  s1Collection.size()
);

print(
  'Processed Sentinel-1 collection:',
  s1Collection
);

printUniqueDates(
  s1Collection,
  'Sentinel-1'
);


// ======================================================
// 6. Build Sentinel-1 mosaic
// ======================================================

/**
 * Multiple Sentinel-1 scenes may be required to cover the complete study
 * area within a narrow acquisition window. mosaic() combines those scenes
 * into a single spatially continuous image where valid observations exist.
 */
var s1Mosaic = s1Collection
  .mosaic()
  .clip(
    geom
  );

s1Mosaic = ee.Image(
  s1Mosaic
);


print(
  'Sentinel-1 mosaic:',
  s1Mosaic
);


// ======================================================
// 7. Evaluate valid Sentinel-1 spatial coverage
// ======================================================

/**
 * The VV mask is used as the spatial-coverage indicator. The input
 * collection is restricted to scenes containing both VV and VH.
 */
var s1ValidMask = s1Mosaic
  .select('VV')
  .mask();


/**
 * The mean binary mask value gives the proportion of the study region
 * containing valid Sentinel-1 observations.
 *
 * Example:
 * 1.00 = 100% valid coverage
 * 0.95 = 95% valid coverage
 */
var s1CoverageFraction = s1ValidMask.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: geom,
  scale: EXPORT_SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  bestEffort: true
}).get('VV');


var s1CoveragePercent = ee.Number(
  s1CoverageFraction
).multiply(
  100
);


print(
  'Sentinel-1 valid coverage fraction:',
  s1CoverageFraction
);

print(
  'Sentinel-1 valid coverage percent:',
  s1CoveragePercent
);


Map.addLayer(
  s1ValidMask,
  {
    min: 0,
    max: 1,
    palette: [
      'red',
      'green'
    ]
  },
  'Sentinel-1 valid coverage',
  false
);


// ======================================================
// 8. Build Sentinel-1 predictor stack
// ======================================================

/**
 * Convert normalized linear Gamma0 VV and VH to dB.
 */
var s1Db = utils
  .linToDb(
    s1Mosaic
  )
  .select(
    [
      'VV',
      'VH'
    ],
    [
      'VV_dB',
      'VH_dB'
    ]
  );


/**
 * Construct the input expected by 04_s1_feature_extraction:
 *
 * - VV
 * - VH
 * - VV_dB
 * - VH_dB
 */
var s1Base = s1Mosaic
  .select([
    'VV',
    'VH'
  ])
  .addBands(
    s1Db
  );

s1Base = ee.Image(
  s1Base
);


/**
 * Build the complete Sentinel-1 feature stack.
 *
 * Per season:
 * - 2 backscatter predictors
 * - 10 polarization/algebraic predictors
 * - 14 GLCM texture predictors
 *
 * Total = 26 predictors.
 */
var s1FeatureStack = s1Features
  .buildS1FeatureStack(
    s1Base,
    GLCM_LEVELS,
    true
  )
  .set({
    'sensor': 'Sentinel-1',
    'cycle': CYCLE_NAME,
    'season': SEASON_NAME,
    'start_date': S1_START_DATE,
    'end_date': S1_END_DATE,
    'orbit_pass': S1_ORBIT_PASS,
    'valid_fraction_roi': s1CoverageFraction,
    'valid_percent_roi': s1CoveragePercent
  });


s1FeatureStack = ee.Image(
  s1FeatureStack
);


print(
  'Sentinel-1 feature stack:',
  s1FeatureStack
);

print(
  'Sentinel-1 feature bands:',
  s1FeatureStack.bandNames()
);

print(
  'Sentinel-1 predictor count:',
  s1FeatureStack.bandNames().size()
);


// ======================================================
// 9. Visual quality-control layers
// ======================================================

Map.addLayer(
  s1FeatureStack.select(
    'S1_VV_dB'
  ),
  {
    min: -25,
    max: 0
  },
  'Sentinel-1 VV Gamma0 dB'
);


Map.addLayer(
  s1FeatureStack.select(
    'S1_VH_dB'
  ),
  {
    min: -30,
    max: -5
  },
  'Sentinel-1 VH Gamma0 dB',
  false
);


Map.addLayer(
  s1FeatureStack.select(
    'S1_RVI'
  ),
  {
    min: 0,
    max: 1,
    palette: [
      'brown',
      'yellow',
      'green'
    ]
  },
  'Sentinel-1 RVI',
  false
);


// ======================================================
// 10. Export Sentinel-1 predictor stack
// ======================================================

var exportName =
  'S1_WPE_FEATURES_' +
  SEASON_NAME;


Export.image.toDrive({
  image: s1FeatureStack,
  description: exportName,
  folder: EXPORT_FOLDER,
  fileNamePrefix: exportName,
  region: geom,
  scale: EXPORT_SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e13
});