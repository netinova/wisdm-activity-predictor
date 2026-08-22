from util import load_processed_data, load_model
from config import WINDOW_SIZE, REPORTS_DIR
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt
import os
import pandas as pd


def window_type(value):
    try:
        int_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer") from exc
    if int_value < 1 or int_value > 60:
        raise argparse.ArgumentTypeError(
            f"Window must be between 1 and 60 (got {int_value})"
        )
    return int_value


def save_reports(mode, y_true, y_pred):
    report_dir = os.path.join(REPORTS_DIR, mode)
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, "classification_report_test.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(classification_report(y_true, y_pred))  # type: ignore

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=sorted(set(y_true.astype(str)))
    )
    fig, ax = plt.subplots(figsize=(12, 9), dpi=200)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    fig.tight_layout()
    confusion_path = os.path.join(report_dir, "confusion_matrix_test.png")
    fig.savefig(confusion_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved report to {report_path}")
    print(f"Saved confusion matrix to {confusion_path}")


def main():

    parser = argparse.ArgumentParser(
        description="an script to score the models on test data. ONLY use this just before deploying!"
    )

    parser.add_argument(
        "--mode",
        nargs="+",
        metavar="DataSet",
        type=str,
        choices=["phone", "watch", "both"],
        help="which model to test",
    )
    parser.add_argument(
        "-w",
        "--window",
        default=WINDOW_SIZE,
        type=window_type,
        help="Window size in seconds (1-60). Default: %(default)s.",
    )

    args = parser.parse_args()

    if args.mode is None:
        raise SystemExit("Please provide --mode phone, watch, or both")

    for mode in args.mode:
        X_test, y_test = load_processed_data(mode, WINDOW_SIZE, test_data=True)
        model: RandomForestClassifier = load_model(mode, WINDOW_SIZE)
        y_pred = model.predict(X_test)
        save_reports(mode, y_test, y_pred)


if __name__ == "__main__":
    main()
