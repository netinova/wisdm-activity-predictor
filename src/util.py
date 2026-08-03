import os
import sys
import pandas as pd
from typing import Literal

from config import PROCESSED_DIR


def load_processed_data(
    mode: Literal["phone", "watch", "both"] = "phone", window=2, test_data=False
):

    if mode == "both":
        filename = f"{mode}_aligned_feature_extracted_{"test" if test_data else "train"}({window}s).csv"
    else:
        filename = f"{mode}_feature_extracted_{"test" if test_data else "train"}({window}s).csv"

    filepath = os.path.join(PROCESSED_DIR, filename)

    if not os.path.exists(filepath):
        print(f"ERROR: Processed data not found for mode '{mode}'.")
        print(f"Expected file: {filepath}")
        print(f"**Please run the preprocessing script first:")
        print(f"   python src/preprocess.py --only {mode}")
        sys.exit(1)

    # Load the data
    print(f"Loading processed data: {filename}")
    df = pd.read_csv(filepath)
    df["activity"] = df["activity"].astype("category")

    X = df.drop(columns="activity")
    y = df["activity"]

    print(f"Loaded {len(X)} samples, {X.shape[1]} features")
    return X, y
