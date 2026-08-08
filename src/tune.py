import argparse
import json
import os

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from config import PARAM_DIR, WINDOW_SIZE
from util import load_processed_data


def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 20, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
        ),
        "max_samples": trial.suggest_categorical("max_samples", [0.5, 0.7, 0.8, 0.9]),
    }
    model = RandomForestClassifier(**params, bootstrap=True, n_jobs=-1, random_state=42)
    score = cross_val_score(
        model, X, y, cv=3, scoring=make_scorer(f1_score, average="macro")
    ).mean()
    return score


def save_params(mode, params):
    os.makedirs(PARAM_DIR, exist_ok=True)
    params_filename = f"best_params_{mode}.json"
    params_path = os.path.join(PARAM_DIR, params_filename)
    with open(params_path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=4)
        print(f"Params saved to {params_path}")


def main():
    parser = argparse.ArgumentParser(description="Tune Random Forest for WISDM")
    parser.add_argument(
        "--mode",
        nargs="+",
        type=str,
        choices=["phone", "watch", "both"],
        help="Which sensor configuration to tune",
    )
    args = parser.parse_args()

    if args.mode is None:
        raise SystemExit("Please provide --mode phone, watch, or both")

    for mode in args.mode:
        X, y = load_processed_data(mode=mode, window=WINDOW_SIZE)

        print(f"Running Optuna (50 trials) for {mode}...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )

        study.optimize(lambda trial: objective(trial, X, y), n_trials=50)
        print(f"Best F1-Macro: {study.best_value:.4f}")
        print(f"Best Params: {study.best_params}")
        save_params(mode, study.best_params)


if __name__ == "__main__":
    main()
