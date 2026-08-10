import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
    def __init__(self, mode="phone", window_sec=4):
        self.mode = mode
        self.window_sec = window_sec

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X is None:
            return X

        frame = X.copy()
        if "rel_time" not in frame.columns and "timeStamp" in frame.columns:
            frame = frame.sort_values(by="timeStamp")
            frame["rel_time"] = frame["timeStamp"] - frame["timeStamp"].min()

        if self.mode == "both":
            return extract_windows_aligned(frame, window_sec=self.window_sec)

        return extract_windows(frame, window_sec=self.window_sec)
