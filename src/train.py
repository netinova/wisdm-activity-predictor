import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from config import (
    DEFAULT_PARAMS,
    MODELS_DIR,
    PARAM_DIR,
    REPORTS_DIR,
    WINDOW_SIZE,
)
from util import load_processed_data


def extract_windows(df, window_sec=2):
    df = df.copy()
    df = df.sort_values(["rel_time"])
    df = df.set_index("rel_time")

    sensor_cols = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro"]
    fs = 20.0

    grouped = df.groupby([pd.Grouper(freq=f"{window_sec}s")], observed=True)

    windows = []
    for _, group in grouped:
        features = {}

        for axis in sensor_cols:
            if axis not in group.columns:
                continue
            series = group[axis].values
            n = len(series)
            if n == 0:
                continue

            mean_val = np.mean(series)
            std_val = np.std(series, ddof=1) if n > 1 else np.nan
            p25 = np.percentile(series, 25)
            p75 = np.percentile(series, 75)
            kurt = pd.Series(series).kurtosis()

            with np.errstate(invalid="ignore", divide="ignore"):
                autocorr = pd.Series(series).autocorr(lag=1) if n > 2 else np.nan

            rms = np.sqrt(np.mean(np.square(series)))

            if n > 1:
                fft_vals = np.fft.rfft(series)
                freqs = np.fft.rfftfreq(n, d=1 / fs)
                magnitudes = np.abs(fft_vals)
                if len(magnitudes) > 1:
                    idx_max = np.argmax(magnitudes[1:]) + 1
                    dom_freq = freqs[idx_max]
                    spectral_energy = np.sum(magnitudes[1:] ** 2)
                else:
                    dom_freq = 0.0
                    spectral_energy = 0.0
            else:
                dom_freq = 0.0
                spectral_energy = 0.0

            prefix = axis + "_"
            features[prefix + "mean"] = mean_val
            features[prefix + "std"] = std_val
            features[prefix + "p25"] = p25
            features[prefix + "p75"] = p75
            features[prefix + "kurt"] = kurt
            features[prefix + "autocorr"] = autocorr
            features[prefix + "rms"] = rms
            features[prefix + "dom_freq"] = dom_freq
            features[prefix + "spectral_energy"] = spectral_energy

        if features:
            windows.append(features)

    windows_df = pd.DataFrame(windows)
    return windows_df.dropna()


def extract_windows_aligned(df, window_sec=2):
    df = df.copy()
    df = df.sort_values(["rel_time"])
    df = df.set_index("rel_time")

    sensor_cols = [
        "x_phone_accel",
        "y_phone_accel",
        "z_phone_accel",
        "x_phone_gyro",
        "y_phone_gyro",
        "z_phone_gyro",
        "x_watch_accel",
        "y_watch_accel",
        "z_watch_accel",
        "x_watch_gyro",
        "y_watch_gyro",
        "z_watch_gyro",
    ]
    fs = 20.0

    grouped = df.groupby([pd.Grouper(freq=f"{window_sec}s")], observed=True)

    windows = []
    for _, group in grouped:
        features = {}

        for axis in sensor_cols:
            if axis not in group.columns:
                continue
            series = group[axis].values
            n = len(series)
            if n == 0:
                continue
            mean_val = np.mean(series)
            std_val = np.std(series, ddof=1) if n > 1 else np.nan
            p25 = np.percentile(series, 25)
            p75 = np.percentile(series, 75)
            kurt = pd.Series(series).kurtosis()

            with np.errstate(invalid="ignore", divide="ignore"):
                autocorr = pd.Series(series).autocorr(lag=1) if n > 2 else np.nan

            rms = np.sqrt(np.mean(np.square(series)))

            if n > 1:
                fft_vals = np.fft.rfft(series)
                freqs = np.fft.rfftfreq(n, d=1 / fs)
                magnitudes = np.abs(fft_vals)
                if len(magnitudes) > 1:
                    idx_max = np.argmax(magnitudes[1:]) + 1
                    dom_freq = freqs[idx_max]
                    spectral_energy = np.sum(magnitudes[1:] ** 2)
                else:
                    dom_freq = 0.0
                    spectral_energy = 0.0
            else:
                dom_freq = 0.0
                spectral_energy = 0.0

            prefix = axis + "_"
            features[prefix + "mean"] = mean_val
            features[prefix + "std"] = std_val
            features[prefix + "p25"] = p25
            features[prefix + "p75"] = p75
            features[prefix + "kurt"] = kurt
            features[prefix + "autocorr"] = autocorr
            features[prefix + "rms"] = rms
            features[prefix + "dom_freq"] = dom_freq
            features[prefix + "spectral_energy"] = spectral_energy

        if features:
            windows.append(features)

    windows_df = pd.DataFrame(windows)
    return windows_df.dropna()


class RawFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mode="phone", window_sec=2):
        self.mode = mode
        self.window_sec = window_sec

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X is None:
            return X

        frame = X.copy()
        if "rel_time" not in frame.columns and "timeStamp" in frame.columns:
            frame["rel_time"] = frame.groupby("activity")["timeStamp"].transform(
                lambda value: value - value.min()
            )

        if self.mode == "both":
            return extract_windows_aligned(frame, window_sec=self.window_sec)
        return extract_windows(frame, window_sec=self.window_sec)


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


def resolve_requested_modes(requested):
    if not requested:
        return ["phone", "watch", "both"]

    modes = []
    if "phone" in requested:
        modes.append("phone")
    if "watch" in requested:
        modes.append("watch")
    if "both" in requested:
        modes.append("both")
    return modes


def load_best_params(mode):
    params_path = os.path.join(PARAM_DIR, f"best_params_{mode}.json")
    if not os.path.exists(params_path):
        return DEFAULT_PARAMS.copy()

    with open(params_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_classifier(params):
    model_params = {k: v for k, v in params.items() if v is not None}
    return RandomForestClassifier(
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
        **model_params,
    )


def build_pipeline(mode, classifier, window_sec):
    return Pipeline(
        [
            (
                "feature_engineering",
                RawFeatureTransformer(mode=mode, window_sec=window_sec),
            ),
            ("classifier", classifier),
        ]
    )


def save_model(model, mode, final_model=False):
    target_dir = MODELS_DIR
    os.makedirs(target_dir, exist_ok=True)
    suffix = "final" if final_model else "train"
    model_path = os.path.join(target_dir, f"{mode}_{suffix}.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    return model_path


def save_reports(mode, y_true, y_pred):
    report_dir = os.path.join(REPORTS_DIR, mode)
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(classification_report(y_true, y_pred))  # type: ignore

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=sorted(set(y_true.astype(str)))
    )
    fig, ax = plt.subplots(figsize=(12, 9), dpi=200)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    fig.tight_layout()
    confusion_path = os.path.join(report_dir, "confusion_matrix.png")
    fig.savefig(confusion_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved report to {report_path}")
    print(f"Saved confusion matrix to {confusion_path}")


def train_mode(mode, window, report=False, final_model=False):
    print(f"\nTraining {mode} model with window {window}s")

    if not os.path.isdir(PARAM_DIR):
        raise FileNotFoundError("Best params not found! Please run tune.py first.")

    params = load_best_params(mode)
    classifier = build_classifier(params)

    X_train, y_train = load_processed_data(mode=mode, window=window, test_data=False)

    if final_model:
        X_test, y_test = load_processed_data(mode=mode, window=window, test_data=True)
        X_fit = pd.concat([X_train, X_test], axis=0)
        y_fit = pd.concat([y_train, y_test], axis=0)
        print(f"Final model training on {len(X_fit)} samples (train + test)")
    else:
        X_fit, y_fit = X_train, y_train

    if report:
        y_pred = cross_val_predict(classifier, X_train, y_train, cv=5)
        print(
            f"Cross-validation accuracy (5-fold): {accuracy_score(y_train, y_pred):.4f}"
        )
        save_reports(mode, y_train, y_pred)

    classifier.fit(X_fit, y_fit)

    pipeline = build_pipeline(mode, classifier, window)
    save_model(pipeline, mode, final_model=final_model)


def main():
    parser = argparse.ArgumentParser(
        description="Train WISDM activity models from the preprocessed feature data"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="DataSet",
        choices=["watch", "phone", "both"],
        help="Which datasets to train. Default: all.",
    )
    parser.add_argument(
        "-w",
        "--window",
        default=WINDOW_SIZE,
        type=window_type,
        help="Window size in seconds (1-60). Default: %(default)s.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate a classification report and confusion matrix image from 5-fold cross-validation predictions on the training data.",
    )
    parser.add_argument(
        "--final_model",
        action="store_true",
        help="Fit the final model on the combined train+test preprocessed data.",
    )
    args = parser.parse_args()

    requested_modes = resolve_requested_modes(args.only)
    if not requested_modes:
        print("No modes selected. Nothing to train.")
        return

    for mode in requested_modes:
        train_mode(
            mode=mode,
            window=args.window,
            report=args.report,
            final_model=args.final_model,
        )


if __name__ == "__main__":
    main()
