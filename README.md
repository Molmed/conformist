# Conformist v1.0.0-alpha.1

Conformist is an implementation of conformal prediction. It was written using Python 3.8.

*BaseCoP* contains utility functions common to all conformal predictors, such as splitting data into calibration and validation sets, and setting up runs. It is extended by the single-class *SimpleConformalPredictor* and the *FNRCoP* that implements conformal risk control.

The *ValidationRun* class contains the data from a single run, which entails shuffling the data randomly, splitting it into calibration and validation datasets, calibrating the conformal predictor on the calibration data and creating prediction sets for the validation data.

The *ValidationTrial* class contains a list of runs and calculates statistics across these runs.

## Input file format

The input to Conformist is a CSV file of the following format:
id, known_class, predicted_class, [proba_columns]

The proba_columns should contain class-specific probability scores and correspond to the names used in the 'known_class' and 'predicted_class' columns.

TODO: example CSV
