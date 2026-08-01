import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# constants
DATA_DIR = f"{os.path.dirname(__file__).replace("\\","/")}/../data"
RAW_DIR = f"{DATA_DIR}/wisdm-dataset/raw"

SUBJECTS = range(1600, 1651)  # 51 subjects
SENSORS = [("phone", "accel"), ("phone", "gyro"), ("watch", "accel"), ("watch", "gyro")]
WINDOW = 2


def window_type(value):
    """Type function for argparse to validate int range."""
    try:
        intValue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer")
    if intValue < 1 or intValue > 60:
        raise argparse.ArgumentTypeError(
            f"Window must be between 1 and 60 (got {intValue})"
        )
    return intValue


def main():

    parser = argparse.ArgumentParser(
        description="a preprocessing script to prepare the data for model training."
    )

    parser.add_argument(
        "--only",
        nargs="+",
        metavar="DataSet",
        choices=["watch", "phone", "both"],
        help="what datasets you want to generate and prepare (don't use if you want all!).",
    )

    parser.add_argument(
        "-w",
        "--window",
        default=2,
        type=window_type,
        help="window size (in seconds) for aggregating/extracting data. "
        "Must be between 1 and 60. Default: %(default)s.",
    )

    args = parser.parse_args()

    if not os.path.isdir(RAW_DIR):
        sys.exit(
            "Error: Did not found the Data set! please run x first to download the data."
        )

    print("this may take a while.", end="\n\n")

    if not args.only or "watch" in args.only or "phone" in args.only:
        df_separate = load_all_data()
        df_separate = df_separate.dropna()
        df_separate = df_separate.drop(columns="subject_id")

        final_df = extract_windows(df_separate, args.window)
        final_df = final_df.dropna()
        final_df = final_df.drop(columns="window_start")

        final_df_phone = final_df[final_df["device"] == "phone"]
        final_df_phone = final_df_phone.drop(columns="device")
        final_df_watch = final_df[final_df["device"] == "watch"]
        final_df_watch = final_df_watch.drop(columns="device")

        final_df_phone.to_csv(
            f"{DATA_DIR}/processed/phone_feature_extracted({args.window}s).csv",
            index=False,
        )
        final_df_watch.to_csv(
            f"{DATA_DIR}/processed/watch_feature_extracted({args.window}s).csv",
            index=False,
        )

    if not args.only or "both" in args.only:
        df_aligned = load_all_data_aligned()
        df_aligned = df_aligned.dropna()
        df_aligned = df_aligned.drop(columns="subject_id")
        final_df = extract_windows_aligned(df_aligned, args.window)
        final_df = final_df.dropna()
        final_df = final_df.drop(columns="window_start")
        final_df.to_csv(
            f"{DATA_DIR}/processed/both_aligned_feature_extracted({args.window}s).csv",
            index=False,
        )


def get_sensor_data(filepath):
    df = pd.read_csv(
        filepath,
        header=None,
        names=["subject_id", "activity", "timeStamp", "x", "y", "z"],
    )

    df.iloc[:, -1] = df.iloc[:, -1].str.rstrip(";")

    df = df.astype({"z": float})

    df["timeStamp"] = pd.to_datetime(df["timeStamp"], unit="ns")
    df.drop(columns="subject_id", inplace=True)

    return df


def get_subject_data(subject_id, log):

    dfs: list[pd.DataFrame] = []

    for device, sensor in SENSORS:

        df = get_sensor_data(
            f"{RAW_DIR}/{device}/{sensor}/data_{subject_id}_{sensor}_{device}.txt"
        )

        df["device"] = device

        df = df.rename(
            columns={
                "x": f"x_{sensor}",
                "y": f"y_{sensor}",
                "z": f"z_{sensor}",
            }
        )

        dfs.append(df)

    for df in dfs:
        df.sort_values(by="timeStamp", inplace=True)

    merged_phone = pd.merge_asof(
        dfs[0] if dfs[0].shape[0] < dfs[1].shape[0] else dfs[1],
        dfs[0] if dfs[0].shape[0] >= dfs[1].shape[0] else dfs[1],
        on="timeStamp",
        by=["activity", "device"],
        direction="nearest",
        tolerance=pd.Timedelta("100ms"),
    )

    if log:
        print(f"{merged_phone.shape[0]} rows of phone data")

    # return merged_phone

    merged_watch = pd.merge_asof(
        dfs[2] if dfs[2].shape[0] < dfs[3].shape[0] else dfs[3],
        dfs[2] if dfs[2].shape[0] >= dfs[3].shape[0] else dfs[3],
        on="timeStamp",
        by=["activity", "device"],
        direction="nearest",
        tolerance=pd.Timedelta("100ms"),
    )

    if log:
        print(f"{merged_watch.shape[0]} rows of phone data")

    merged = pd.concat([merged_phone, merged_watch])

    merged = merged.iloc[:, [5, 0, 1, 2, 3, 4, 6, 7, 8]]

    merged["activity"] = merged["activity"].astype("category")
    merged["device"] = merged["device"].astype("category")

    return merged.sort_values(by=["device", "activity", "timeStamp"])


def load_all_data():

    merged = pd.DataFrame()

    for subject_id in SUBJECTS:

        print(f"loading {subject_id} data...")

        if subject_id == SUBJECTS.start:
            merged = get_subject_data(subject_id, False)
            merged["subject_id"] = np.ones(merged.shape[0], dtype=int) * subject_id
            continue

        other = get_subject_data(subject_id, False)
        other["subject_id"] = np.ones(other.shape[0], dtype=int) * subject_id
        merged = pd.concat([merged, other])

    merged["subject_id"] = merged["subject_id"].astype("category")
    merged["activity"] = merged["activity"].astype("category")

    merged = merged.reindex(
        columns=[
            "subject_id",
            "device",
            "activity",
            "timeStamp",
            "x_accel",
            "y_accel",
            "z_accel",
            "x_gyro",
            "y_gyro",
            "z_gyro",
        ]
    )

    return merged


def extract_windows(df, window_sec=2):

    df = df.copy()

    df = df.sort_values(["device", "activity", "timeStamp"])
    df = df.set_index("timeStamp")

    sensor_cols = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro"]

    fs = 20.0

    def _extract_features(group):

        device, activity, window_start = group.name
        features = {
            "device": device,
            "activity": activity,
            "window_start": window_start,
        }

        for axis in sensor_cols:
            series = group[axis].values
            n = len(series)
            if n == 0:
                continue

            # Time domain features
            mean_val = np.mean(series)
            std_val = std_val = np.std(series, ddof=1) if n > 1 else np.nan
            p25 = np.percentile(series, 25)
            p75 = np.percentile(series, 75)
            kurt = pd.Series(series).kurtosis()
            autocorr = np.corrcoef(series[:-1], series[1:])[0, 1] if n > 1 else np.nan
            rms = np.sqrt(np.mean(np.square(series)))

            # Frequency domain features
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

        return pd.Series(features)

    grouped = df.groupby(["device", "activity", pd.Grouper(freq=f"{window_sec}s")])
    windows = grouped.apply(_extract_features).reset_index(drop=True)
    windows["window_start"] = pd.to_datetime(windows["window_start"])
    return windows


def align_subject_sensors(subject_id):

    TOLERANCE = pd.Timedelta("100ms")

    sensor_dfs = {}
    for device, sensor in SENSORS:
        filepath = (
            f"{RAW_DIR}/{device}/{sensor}/data_{subject_id}_{sensor}_{device}.txt"
        )
        df = get_sensor_data(filepath)

        df["rel_time"] = df.groupby("activity")["timeStamp"].transform(
            lambda x: (x - x.min())
        )

        suffix = f"{device}_{sensor}"
        df.rename(
            columns={
                "x": f"x_{suffix}",
                "y": f"y_{suffix}",
                "z": f"z_{suffix}",
            },
            inplace=True,
        )

        cols = [
            "activity",
            "rel_time",
            f"x_{suffix}",
            f"y_{suffix}",
            f"z_{suffix}",
        ]
        sensor_dfs[(device, sensor)] = df[cols]

    phone_accel = sensor_dfs[("phone", "accel")]
    phone_gyro = sensor_dfs[("phone", "gyro")]

    phone_merged_list = []

    for act, accel_grp in phone_accel.groupby("activity"):
        if act not in phone_gyro["activity"].values:
            continue

        gyro_grp = phone_gyro[phone_gyro["activity"] == act]

        accel_grp = accel_grp.sort_values("rel_time")
        gyro_grp = gyro_grp.sort_values("rel_time")

        merged = pd.merge_asof(
            accel_grp if accel_grp.shape[0] < gyro_grp.shape[0] else gyro_grp,
            accel_grp if accel_grp.shape[0] >= gyro_grp.shape[0] else gyro_grp,
            on="rel_time",
            direction="nearest",
            tolerance=TOLERANCE,
        )

        merged.drop(columns=["activity_y"], inplace=True, errors="ignore")

        merged.rename(columns={"activity_x": "activity"}, inplace=True)

        phone_merged_list.append(merged)

    phone_merged = pd.concat(phone_merged_list, ignore_index=True)

    watch_accel = sensor_dfs[("watch", "accel")]
    watch_gyro = sensor_dfs[("watch", "gyro")]

    watch_merged_list = []
    for act, accel_grp in watch_accel.groupby("activity"):
        if act not in watch_gyro["activity"].values:
            continue
        gyro_grp = watch_gyro[watch_gyro["activity"] == act]

        accel_grp = accel_grp.sort_values("rel_time")
        gyro_grp = gyro_grp.sort_values("rel_time")

        merged = pd.merge_asof(
            accel_grp if accel_grp.shape[0] < gyro_grp.shape[0] else gyro_grp,
            accel_grp if accel_grp.shape[0] >= gyro_grp.shape[0] else gyro_grp,
            on="rel_time",
            direction="nearest",
            tolerance=TOLERANCE,
        )
        merged.drop(columns=["activity_y"], inplace=True, errors="ignore")
        merged.rename(columns={"activity_x": "activity"}, inplace=True)
        watch_merged_list.append(merged)

    watch_merged = pd.concat(watch_merged_list, ignore_index=True)

    final_list = []
    for act, phone_grp in phone_merged.groupby("activity"):
        if act not in watch_merged["activity"].values:
            continue
        watch_grp = watch_merged[watch_merged["activity"] == act]

        phone_grp = phone_grp.sort_values("rel_time")
        watch_grp = watch_grp.sort_values("rel_time")

        merged = pd.merge_asof(
            phone_grp if phone_grp.shape[0] < watch_grp.shape[0] else watch_grp,
            phone_grp if phone_grp.shape[0] >= watch_grp.shape[0] else watch_grp,
            on="rel_time",
            direction="nearest",
            tolerance=TOLERANCE,
        )
        merged.drop(columns=["activity_y"], inplace=True, errors="ignore")
        merged.rename(columns={"activity_x": "activity"}, inplace=True)
        final_list.append(merged)

    final = pd.concat(final_list, ignore_index=True)

    # Final clean‑up as in your version
    final.sort_values(["activity", "rel_time"], inplace=True)
    final.reset_index(drop=True, inplace=True)
    return final


def load_all_data_aligned():

    merged = pd.DataFrame()

    for subject_id in SUBJECTS:

        print(f"loading {subject_id} data...")

        if subject_id == SUBJECTS.start:
            merged = align_subject_sensors(subject_id)
            merged["subject_id"] = np.ones(merged.shape[0], dtype=int) * subject_id
            continue

        other = align_subject_sensors(subject_id)
        other["subject_id"] = np.ones(other.shape[0], dtype=int) * subject_id
        merged = pd.concat([merged, other])

    merged["subject_id"] = merged["subject_id"].astype("category")
    merged["activity"] = merged["activity"].astype("category")

    return merged


def extract_windows_aligned(df, window_sec=2):

    df = df.copy()

    df = df.sort_values(["activity", "rel_time"])
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

    def _extract_features(group):

        _, activity, window_start = group.name

        features = {
            "activity": activity,
            "window_start": window_start,
        }

        for axis in sensor_cols:
            series = group[axis].values
            n = len(series)
            if n == 0:
                continue

            # Time domain features
            mean_val = np.mean(series)
            std_val = std_val = np.std(series, ddof=1) if n > 1 else np.nan
            p25 = np.percentile(series, 25)
            p75 = np.percentile(series, 75)
            kurt = pd.Series(series).kurtosis()
            autocorr = np.corrcoef(series[:-1], series[1:])[0, 1] if n > 1 else np.nan
            rms = np.sqrt(np.mean(np.square(series)))

            # Frequency domain features
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

        return pd.Series(features)

    grouped = df.groupby(
        by=["subject_id", "activity", pd.Grouper(freq=f"{window_sec}s")]
    )
    windows = grouped.apply(_extract_features).reset_index(drop=True)
    return windows


if __name__ == "__main__":
    main()
