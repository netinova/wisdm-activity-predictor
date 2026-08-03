import argparse
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
import time

from config import PROCESSED_DIR, RAW_DIR

SUBJECTS = list(range(1600, 1651))  # 51 subjects
SENSORS = [("phone", "accel"), ("phone", "gyro"), ("watch", "accel"), ("watch", "gyro")]
WINDOW = 2


def window_type(value):
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
        description="Preprocessing script to prepare data for model training."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="DataSet",
        choices=["watch", "phone", "both"],
        help="Which datasets to generate. Default: all).",
    )
    parser.add_argument(
        "-w",
        "--window",
        default=2,
        type=window_type,
        help="Window size in seconds (1-60). Default: %(default)s.",
    )
    args = parser.parse_args()

    if not os.path.isdir(RAW_DIR):
        sys.exit("Error: Dataset not found! Please download it first (see README).")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    do_separate = not args.only or "watch" in args.only or "phone" in args.only
    do_aligned = not args.only or "both" in args.only

    print(f"Window size: {args.window}s")
    print("This may take a while.\n")

    start_time = time.time()

    # Accumulators for the two data formats
    separate_parts = []
    aligned_parts = []

    for subject_id in tqdm(SUBJECTS, desc="Processing subjects", unit="subject"):
        try:
            raw_dfs = read_subject_raw(subject_id)
        except FileNotFoundError as e:
            print(f"Warning: Skipping subject {subject_id} – {e}")
            continue

        if do_separate:
            df_sep = get_subject_data_from_raw(raw_dfs)
            df_sep["subject_id"] = subject_id
            separate_parts.append(df_sep)

        if do_aligned:
            df_al = align_subject_sensors_from_raw(raw_dfs)
            df_al["subject_id"] = subject_id
            aligned_parts.append(df_al)

    if do_separate and separate_parts:
        print("\nBuilding separate phone/watch dataset...")
        df_separate = pd.concat(separate_parts, ignore_index=True)
        df_separate["subject_id"] = df_separate["subject_id"].astype("category")
        df_separate["activity"] = df_separate["activity"].astype("category")
        df_separate = df_separate.dropna()

        final_df = extract_windows(df_separate, args.window)
        final_df = final_df.dropna()
        final_df = final_df.drop(columns="window_start")

        final_df_phone = final_df[final_df["device"] == "phone"].drop(columns="device")
        final_df_watch = final_df[final_df["device"] == "watch"].drop(columns="device")

        phone_path = os.path.join(
            PROCESSED_DIR, f"phone_feature_extracted({args.window}s).csv"
        )
        watch_path = os.path.join(
            PROCESSED_DIR, f"watch_feature_extracted({args.window}s).csv"
        )
        final_df_phone.to_csv(phone_path, index=False)
        final_df_watch.to_csv(watch_path, index=False)
        print(f"Saved {phone_path}")
        print(f"Saved {watch_path}")
    elif do_separate:
        print("No subjects processed for separate dataset.")

    if do_aligned and aligned_parts:
        print("\nBuilding aligned phone+watch dataset...")
        df_aligned = pd.concat(aligned_parts, ignore_index=True)
        df_aligned["subject_id"] = df_aligned["subject_id"].astype("category")
        df_aligned["activity"] = df_aligned["activity"].astype("category")
        df_aligned = df_aligned.dropna()

        final_df = extract_windows_aligned(df_aligned, args.window)
        final_df = final_df.dropna()
        final_df = final_df.drop(columns="window_start")

        aligned_path = os.path.join(
            PROCESSED_DIR, f"both_aligned_feature_extracted({args.window}s).csv"
        )
        final_df.to_csv(aligned_path, index=False)
        print(f"Saved {aligned_path}")
    elif do_aligned:
        print("No subjects processed for aligned dataset.")

    elapsed = time.time() - start_time
    print(f"\nAll done in {elapsed:.1f} seconds.")


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


def read_subject_raw(subject_id):

    raw = {}
    for device, sensor in SENSORS:
        filepath = os.path.join(
            RAW_DIR, device, sensor, f"data_{subject_id}_{sensor}_{device}.txt"
        )
        raw[(device, sensor)] = get_sensor_data(filepath)
    return raw


def get_subject_data_from_raw(raw_dfs):

    dfs = []
    for device, sensor in SENSORS:
        df = raw_dfs[(device, sensor)].copy()
        df["device"] = device
        df = df.rename(
            columns={
                "x": f"x_{sensor}",
                "y": f"y_{sensor}",
                "z": f"z_{sensor}",
            }
        )
        df.sort_values(by="timeStamp", inplace=True)
        dfs.append(df)

    # Merge phone accel + phone gyro
    merged_phone = pd.merge_asof(
        dfs[0] if dfs[0].shape[0] < dfs[1].shape[0] else dfs[1],
        dfs[0] if dfs[0].shape[0] >= dfs[1].shape[0] else dfs[1],
        on="timeStamp",
        by=["activity", "device"],
        direction="nearest",
        tolerance=pd.Timedelta("100ms"),
    )

    # Merge watch accel + watch gyro
    merged_watch = pd.merge_asof(
        dfs[2] if dfs[2].shape[0] < dfs[3].shape[0] else dfs[3],
        dfs[2] if dfs[2].shape[0] >= dfs[3].shape[0] else dfs[3],
        on="timeStamp",
        by=["activity", "device"],
        direction="nearest",
        tolerance=pd.Timedelta("100ms"),
    )

    merged = pd.concat([merged_phone, merged_watch])
    merged = merged.iloc[:, [5, 0, 1, 2, 3, 4, 6, 7, 8]]
    merged["activity"] = merged["activity"].astype("category")
    merged["device"] = merged["device"].astype("category")
    return merged.sort_values(by=["device", "activity", "timeStamp"])


def align_subject_sensors_from_raw(raw_dfs):

    TOLERANCE = pd.Timedelta("100ms")

    # Prepare sensor DataFrames with rel_time and renamed columns
    sensor_dfs = {}
    for device, sensor in SENSORS:
        df = raw_dfs[(device, sensor)].copy()
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
        cols = ["activity", "rel_time", f"x_{suffix}", f"y_{suffix}", f"z_{suffix}"]
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
    final.sort_values(["activity", "rel_time"], inplace=True)
    final.reset_index(drop=True, inplace=True)
    return final


def extract_windows(df, window_sec=2):
    df = df.copy()
    df = df.sort_values(["device", "activity", "timeStamp"])
    df = df.set_index("timeStamp")

    sensor_cols = ["x_accel", "y_accel", "z_accel", "x_gyro", "y_gyro", "z_gyro"]
    fs = 20.0

    grouped = df.groupby(
        ["device", "activity", pd.Grouper(freq=f"{window_sec}s")],
        observed=True,
    )

    windows = []
    for name, group in tqdm(
        grouped, total=grouped.ngroups, desc="Extracting windows (separate)"
    ):
        device, activity, window_start = name
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
            std_val = np.std(series, ddof=1) if n > 1 else np.nan
            p25 = np.percentile(series, 25)
            p75 = np.percentile(series, 75)
            kurt = pd.Series(series).kurtosis()

            with np.errstate(invalid="ignore", divide="ignore"):
                autocorr = pd.Series(series).autocorr(lag=1) if n > 2 else np.nan

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

        windows.append(features)

    windows_df = pd.DataFrame(windows)
    windows_df["window_start"] = pd.to_datetime(windows_df["window_start"])
    return windows_df


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

    grouped = df.groupby(
        ["subject_id", "activity", pd.Grouper(freq=f"{window_sec}s")],
        observed=True,
    )

    windows = []
    for name, group in tqdm(
        grouped, total=grouped.ngroups, desc="Extracting windows (aligned)"
    ):
        _, activity, window_start = name
        features = {"activity": activity, "window_start": window_start}

        for axis in sensor_cols:
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

        windows.append(features)

    windows_df = pd.DataFrame(windows)
    windows_df = windows_df.reset_index(drop=True)
    return windows_df


if __name__ == "__main__":
    main()
