import json
from ensurepip import bootstrap
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from common import optuna_hyperparameter_tuning
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split

SEED = 42


def main(args):
    assert args.features_folder.exists(), "Folder provided does not exist"

    # Load features from folder
    train_csv = args.features_folder / "train_features.csv"
    val_csv = args.features_folder / "val_features.csv"
    test_csv = args.features_folder / "test_features.csv"

    if val_csv.exists():
        train = pd.read_csv(train_csv)
        val = pd.read_csv(val_csv)
        test = pd.read_csv(test_csv)
    else:
        train = pd.read_csv(train_csv)
        val = None
        test = pd.read_csv(test_csv)

    assert args.label in train, "Label column not found in csv"

    train_X = train.filter(regex="pred|feature").dropna().values
    train_y = train[args.label].values

    # train_X = preprocessing.normalize(train_X, norm="l2")

    scaler = preprocessing.StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)

    if val is not None:
        val_X = val.filter(regex="pred|feature").dropna().values
        val_X = scaler.transform(val_X)
        val_y = val[args.label].values
    else:
        val_X = None
        val_y = None

    test_X = test.filter(regex="pred|feature").dropna().values
    test_X = scaler.transform(test_X)
    test_y = test[args.label].values

    start_time = time.time()
    best_params = optuna_hyperparameter_tuning(train_X, train_y, val_X, val_y, args.scoring, args.trials, args.n_jobs)
    classifier = sklearn.linear_model.LogisticRegression(**best_params, random_state=SEED, max_iter=1000)
    # classifier = sklearn.neighbors.KNeighborsClassifier(**best_params)
    classifier.fit(train_X, train_y)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    if args.scoring == "roc_auc":
        score = sklearn.metrics.roc_auc_score(test_y, classifier.predict_proba(test_X)[:, 1])
    else:
        score = classifier.score(test_X, test_y)

    print(f"Score using {args.scoring} = {score}")

    # Save best parameters
    with open(args.features_folder / "best_params.json", "w") as fp:
        json.dump(best_params, fp)

    results_df = test.copy()
    probs = classifier.predict_proba(test_X)
    _, n_classes = probs.shape

    for i in range(n_classes):
        results_df[f"conf_scores_class_{i}"] = probs[:, i]

    results_df["pred_class"] = classifier.predict(test_X)
    results_df["target"] = test_y

    results_df = results_df.drop(columns=[col for col in results_df.columns if col.startswith("feature")])
    results_df.to_csv(args.csv, index=False)

    results_df = val.copy()
    probs = classifier.predict_proba(val_X)
    _, n_classes = probs.shape

    for i in range(n_classes):
        results_df[f"conf_scores_class_{i}"] = probs[:, i]

    results_df["pred_class"] = classifier.predict(val_X)
    results_df["target"] = val_y

    results_df = results_df.drop(columns=[col for col in results_df.columns if col.startswith("feature")])
    results_df.to_csv(Path(args.csv).parent / "val_results.csv", index=False)

    # Save model
    joblib.dump(classifier, Path(args.csv).parent / "model.pkl")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("features_folder", help="Path to folder where extracted features are present", type=Path)
    parser.add_argument("label", help="Label column from the csv file")
    parser.add_argument("--scoring", help="Scoring metric to use", default=None)
    parser.add_argument("--trials", help="Number of trials for hyperparameter tuning", default=100, type=int)
    parser.add_argument("--csv", help="Output_csv", default="test_results.csv", type=str)
    parser.add_argument("--n_jobs", help="Number of jobs", default=3, type=int)

    args = parser.parse_args()

    main(args)
