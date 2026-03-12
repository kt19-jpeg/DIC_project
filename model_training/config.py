from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_drug_overdose_deaths.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "model_training"


TARGET_COLUMN = "death_count"
DATE_COLUMN = "Date"
GROUP_COLUMNS = ["State", "Indicator"]


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


RANDOM_STATE = 42
