[![Python package](https://github.com/Molmed/conformist/actions/workflows/python-package.yml/badge.svg)](https://github.com/Molmed/conformist/actions/workflows/python-package.yml)

# Conformist v1.0.2

Conformist is an implementation of conformal prediction, specifically conformal risk control. It was written using Python 3.8.

*BaseCoP* contains utility functions common to all conformal predictors, such as splitting data into calibration and validation sets, and setting up runs. It is extended by *FNRCoP* that implements conformal risk control.

The *ValidationRun* class contains the data from a single run, which entails shuffling the data randomly, splitting it into calibration and validation datasets, calibrating the conformal predictor on the calibration data and creating prediction sets for the validation data.

The *ValidationTrial* class contains a list of runs and calculates statistics across these runs.

## Input file format

The input to Conformist is a CSV file with the following columns:
`id, known_class, predicted_class, [proba_columns]`

The proba_columns should contain class-specific probability scores and correspond to the names used in the `known_class` and `predicted_class` columns.

Example:
| id      | known_class | predicted_class | ClassA | ClassB | ClassC |
| --------| ----------- | --------------- | ------ | ------ | ------ |
| Sample1 | ClassB      | ClassA          | 0.70   | 0.25   | 0.05   |
| Sample2 | ClassA      | ClassA          | 0.98   | 0.02   | 0.0    |
| Sample3 | ClassC      | ClassC          | 0.01   | 0.01   | 0.98   |

## Example implementation
```
from conformist import FNRCoP, PredictionDataset, ValidationTrial

CALIB_DATA_CSV = 'path/to/calib.csv'
TEST_DATA_CSV = 'path/to/test.csv'
OUTPUT_DIR = 'path/to/output'
ALPHA = 0.05 # Select a reasonable False Negative Rate

# Read in formatted predictions
calpd = PredictionDataset(predictions_csv=CALIB_DATA_CSV,
                          dataset_name='my_cal_data')
testpd = PredictionDataset(predictions_csv=TEST_DATA_CSV,
                           dataset_name='my_test_data')

# Get class counts and prediction heatmap
calpd.run_reports(OUTPUT_DIR)

# Validation trial and reports
mcp = FNRCoP(calpd, alpha=ALPHA)
trial = mcp.do_validation_trial(n_runs=10000)
trial.run_reports(OUTPUT_DIR)

# Recalibrate conformal predictor
mcp = FNRCoP(calpd, alpha=ALPHA)
mcp.calibrate()

# Predict
formatted_predictions = mcp.predict(testpd,
                                    OUTPUT_DIR)
```
