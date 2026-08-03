import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "wisdm-dataset", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
PARAM_DIR = os.path.join(MODEL_DIR, "params")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAM_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Preprocessing parameters
WINDOW_SIZE = 4
TEST_SIZE = 0.2

# Model parameters (these are just defaults, won't be used if best_params.json exists)
DEFAULT_PARAMS = {
    "n_estimators": 150,
    "max_depth": 15,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "criterion": "entropy",
    "max_features": "sqrt",
    "max_samples": 0.8,
}

ACTIVITIES = {
    "A": "Walking",
    "B": "Jogging",
    "C": "Stairs",
    "D": "Sitting",
    "E": "Standing",
    "F": "Typing",
    "G": "Brushing Teeth",
    "H": "Eating Soup",
    "I": "Eating Chips",
    "J": "Eating Pasta",
    "K": "Drinking from Cup",
    "L": "Eating Sandwich",
    "M": "Kicking (Soccer Ball)",
    "O": "Playing Catch w/Tennis Ball",
    "P": "Dribbling (Basketball)",
    "Q": "Writing",
    "R": "Clapping",
    "S": "Folding Clothes",
}
