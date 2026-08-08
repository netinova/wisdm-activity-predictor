import os
import sys
from typing import Literal

import pandas as pd

try:
    from .config import PROCESSED_DIR
except ImportError:  # pragma: no cover - allows direct script execution
    from config import PROCESSED_DIR


def load_processed_data(
    mode: Literal["phone", "watch", "both"] = "phone", window=2, test_data=False
):
    split_name = "test" if test_data else "train"

    if mode == "both":
        filename = f"{mode}_aligned_feature_extracted_{split_name}({window}s).csv"
    else:
        filename = f"{mode}_feature_extracted_{split_name}({window}s).csv"

    filepath = os.path.join(PROCESSED_DIR, filename)

    if not os.path.exists(filepath):
        print(f"ERROR: Processed data not found for mode '{mode}'.")
        print(f"Expected file: {filepath}")
        print("Please run the preprocessing script first:")
        print(f"   python src/preprocessing.py --only {mode}")
        sys.exit(1)

    print(f"Loading processed data: {filename}")
    df = pd.read_csv(filepath)
    df["activity"] = df["activity"].astype("category")

    X = df.drop(columns="activity")
    y = df["activity"]

    print(f"Loaded {len(X)} samples, {X.shape[1]} features")
    return X, y
