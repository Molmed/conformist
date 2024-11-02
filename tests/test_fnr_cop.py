from src.conformist.prediction_dataset import PredictionDataset
from src.conformist.fnr_cop import FNRCoP
import os
import shutil

# Get current directory
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = f'{CURRENT_DIR}/output'
INPUT_CSV = f'{CURRENT_DIR}/cal.csv'


def _cleanup():
    print(f'Cleaning up {OUTPUT_DIR}')
    # Remove all files and dirs in the output directory
    for filename in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def test_prediction_dataset():
    # Create dummy output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        calpd = PredictionDataset(predictions_csv=INPUT_CSV,
                                  dataset_name='my_cal_data')

        calpd.run_reports(OUTPUT_DIR)
    finally:
        _cleanup()


def test_fnr_cop():
    # Create dummy output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        calpd = PredictionDataset(predictions_csv=INPUT_CSV,
                                  dataset_name='my_cal_data')
        # Validation trial and reports
        mcp = FNRCoP(calpd, alpha=0.2)
        trial = mcp.do_validation_trial(n_runs=10)
        trial.run_reports(OUTPUT_DIR)
    finally:
        print("hi")
        #_cleanup()
