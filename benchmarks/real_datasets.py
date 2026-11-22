"""Repeated out-of-fold comparisons on six small tabular datasets.

Run from the repository root: python benchmarks/real_datasets.py
Requires numpy, scikit-learn, openpyxl, torch, xgboost, smallgbm and smallmlp.
The protocol records the exact versions used. All preprocessing is fitted on
the outer training fold. The two regressions predict heating load (Energy)
and residuary resistance per unit displacement (Yacht).
"""
import argparse
import csv
import ctypes
import gzip
import hashlib
import io
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import torch
from openpyxl import load_workbook
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from smallgbm import SmallGBMClassifier
from smallgbm import tree as gbm_tree
from smallmlp import SmallMLPRegressor
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor


SEEDS = (11, 29, 47)
FOLDS = 5
FILES = {
    "hepatitis": ("46/hepatitis.zip", "8573c2898714fb32ef8162291dcabd483a1ba9701123cab262f5acace3112d87"),
    "ionosphere": ("52/ionosphere.zip", "4d218ece62756c99659011a13052d06464f13cb2b3d0410ce2aef16de1403860"),
    "parkinsons": ("174/parkinsons.zip", "9baccfbcdac15d6d9347b37c2a352fb31d65c52e367df5a47d7543a55c75b0ab"),
    "yacht": ("243/yacht+hydrodynamics.zip", "aa52b68f88c4bb552187a53ef4c5753fa178f6a36035a3771c5bc04e078487ac"),
    "energy": ("242/energy+efficiency.zip", "499441eee27929a4b00417f58fd8c63c9cc14b8a71520cd0dd27fcb626738351"),
}
UCI_BASE = "https://archive.ics.uci.edu/static/public/"


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    task: str
    unit: str


def archive(cache, name):
    suffix, digest = FILES[name]
    path = cache / f"{name}.zip"
    if path.exists():
        raw = path.read_bytes()
    else:
        with urlopen(UCI_BASE + suffix, timeout=60) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{name}: UCI archive differs from the pinned SHA-256")
    if not path.exists():
        path.write_bytes(raw)
    return ZipFile(io.BytesIO(raw))


def group_features(X):
    return np.unique(X, axis=0, return_inverse=True)[1].ravel()


def load_datasets(cache):
    with archive(cache, "hepatitis") as z:
        data = np.genfromtxt(io.BytesIO(z.read("hepatitis.data")), delimiter=",", missing_values="?")
    yield Dataset("Hepatitis", data[:, 1:], (data[:, 0] == 1).astype(int),
                  np.arange(len(data)), "classification", "row")

    X, y = load_breast_cancer(return_X_y=True)
    yield Dataset("Breast Cancer Wisconsin Diagnostic", X, (y == 0).astype(int),
                  np.arange(len(y)), "classification", "row")

    with archive(cache, "ionosphere") as z:
        rows = np.genfromtxt(io.BytesIO(z.read("ionosphere.data")), delimiter=",", dtype=str)
    X = rows[:, :-1].astype(float)
    yield Dataset("Ionosphere", X, (rows[:, -1] == "g").astype(int), group_features(X),
                  "classification", "unique predictor vector")

    with archive(cache, "parkinsons") as z:
        rows = list(csv.DictReader(io.StringIO(z.read("parkinsons.data").decode())))
    features = [col for col in rows[0] if col not in ("name", "status")]
    X = np.array([[float(row[key]) for key in features] for row in rows])
    # The released file has 32 distinct ID prefixes. The UCI description lists 31 people.
    patients = np.array([row["name"].rsplit("_", 1)[0] for row in rows])
    yield Dataset("Parkinsons", X, np.array([int(row["status"]) for row in rows]), patients,
                  "classification", "subject ID")

    with archive(cache, "yacht") as z:
        data = np.loadtxt(io.BytesIO(z.read("yacht_hydrodynamics.data")))
    yield Dataset("Yacht Hydrodynamics", data[:, :-1], data[:, -1], group_features(data[:, :5]),
                  "regression", "hull geometry")

    with archive(cache, "energy") as z:
        workbook = load_workbook(io.BytesIO(z.read("ENB2012_data.xlsx")), read_only=True, data_only=True)
        try:
            data = np.array([row[:10] for row in workbook.active.values
                             if isinstance(row[0], (int, float))], dtype=float)
        finally:
            workbook.close()
    yield Dataset("Energy Efficiency", data[:, :8], data[:, 8], group_features(data[:, :5]),
                  "regression", "building geometry")


def splits(dataset, seed):
    units, first, inverse = np.unique(dataset.groups, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    if dataset.task == "classification":
        labels = dataset.y[first]
        if not np.array_equal(labels[inverse], dataset.y):
            raise ValueError(f"{dataset.name}: inconsistent labels within a split unit")
        folds = StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(units, labels)
    else:
        folds = KFold(FOLDS, shuffle=True, random_state=seed).split(units)
    for train_units, test_units in folds:
        yield np.flatnonzero(np.isin(inverse, train_units)), np.flatnonzero(np.isin(inverse, test_units))


def preprocess(dataset, train, test):
    if dataset.name == "Energy Efficiency":
        # X6 is orientation and X8 is glazing-area distribution (nominal).
        transform = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), [0, 1, 2, 3, 4, 6]),
            ("nominal", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [5, 7]),
        ])
    else:
        transform = SimpleImputer(strategy="median")
    return transform.fit_transform(dataset.X[train]), transform.transform(dataset.X[test])


def models_for(task, seed):
    if task == "classification":
        return {
            "linear": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
            "xgboost": XGBClassifier(n_jobs=1, random_state=seed),
            "small": SmallGBMClassifier(random_state=seed),
        }
    return {
        "linear": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "xgboost": XGBRegressor(n_jobs=1, random_state=seed),
        "small": SmallMLPRegressor(random_state=seed),
    }


def out_of_fold(dataset, seed):
    n = len(dataset.y)
    fold_ids = np.full(n, -1, dtype=int)
    predictions = {name: np.full(n, np.nan) for name in ("linear", "xgboost", "small")}
    for fold, (train, test) in enumerate(splits(dataset, seed)):
        if np.intersect1d(dataset.groups[train], dataset.groups[test]).size or np.any(fold_ids[test] != -1):
            raise ValueError("Overlapping split units or repeated test rows")
        fold_ids[test] = fold
        Xt, Xv = preprocess(dataset, train, test)
        model_seed = seed * 100 + fold
        for key, model in models_for(dataset.task, model_seed).items():
            if isinstance(model, SmallGBMClassifier):
                # Version 1.5.0 also samples splits using its native C RNG.
                srand = gbm_tree._lib.srand
                srand.argtypes = [ctypes.c_uint]
                srand.restype = None
                srand(model_seed)
            model.fit(Xt, dataset.y[train])
            if dataset.task == "classification":
                pred = model.predict_proba(Xv)[:, 1]
            else:
                pred = model.predict(Xv)
            predictions[key][test] = pred
    if np.any(fold_ids < 0) or any(not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError(f"{dataset.name}: incomplete predictions")
    return fold_ids, predictions


def score(dataset, prediction):
    if dataset.unit == "subject ID":
        # One probability and one vote per participant, irrespective of recording count.
        units, first, inverse = np.unique(dataset.groups, return_index=True, return_inverse=True)
        sums = np.bincount(inverse.ravel(), weights=prediction, minlength=len(units))
        return roc_auc_score(dataset.y[first], sums / np.bincount(inverse.ravel()))
    if dataset.task == "classification":
        return roc_auc_score(dataset.y, prediction)
    return mean_absolute_error(dataset.y, prediction)


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/real_datasets"))
    parser.add_argument("--output", type=Path, default=Path("results/real_datasets"))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    repeats, summary = [], []
    with gzip.open(args.output / "out_of_fold.csv.gz", "wt", newline="") as out, threadpool_limits(limits=1):
        writer = csv.writer(out)
        writer.writerow(["dataset", "seed", "test_fold", "row", "group", "y", "linear", "xgboost", "small"])
        for ds in load_datasets(args.cache):
            trials = []
            for seed in SEEDS:
                folds, predictions = out_of_fold(ds, seed)
                trial = {k: score(ds, v) for k, v in predictions.items()}
                trials.append(trial)
                repeats.append(dict(dataset=ds.name, seed=seed, **trial))
                writer.writerows(zip([ds.name] * len(ds.y), [seed] * len(ds.y), folds, range(len(ds.y)),
                                      ds.groups, ds.y, predictions["linear"], predictions["xgboost"], predictions["small"]))
            entry = dict(dataset=ds.name, rows=len(ds.y), features=ds.X.shape[1],
                         split_units=len(np.unique(ds.groups)), split_unit=ds.unit,
                         metric="ROC-AUC" if ds.task == "classification" else "MAE")
            entry.update({k: float(np.mean([r[k] for r in trials])) for k in trials[0]})
            summary.append(entry)
            print(entry, flush=True)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "repeat_scores.csv", repeats)
    protocol = {
        "seeds": SEEDS, "outer_folds": FOLDS,
        "metric": "Pooled out-of-fold score per repeat, arithmetic mean across three repeats. Parkinsons AUC uses participant-mean probabilities. Regression MAE uses rows.",
        "targets": {"Hepatitis": "die (class 1 in UCI)", "Breast Cancer Wisconsin Diagnostic": "malignant",
                    "Ionosphere": "good radar return", "Parkinsons": "Parkinson's disease",
                    "Yacht Hydrodynamics": "residuary resistance per unit weight of displacement",
                    "Energy Efficiency": "Y1 heating load"},
        "linear": "StandardScaler + logistic C=1 or ridge alpha=1",
        "xgboost": "Library-default hyperparameters, CPU, one thread",
        "small": "Library defaults. SmallGBM 1.5.0 native RNG also seeded. SmallMLP retains its internal early-stopping validation split.",
        "data": {k: {"url": UCI_BASE + url, "sha256": digest} for k, (url, digest) in FILES.items()},
        "breast_cancer_source": "sklearn.datasets.load_breast_cancer (UCI 17)",
        "parkinsons_grouping": "32 distinct subject-name prefixes in the released file. The UCI catalog describes 31 people. The name and status columns are excluded from predictors.",
        "energy_encoding": "X6 orientation and X8 glazing distribution one-hot encoded inside each outer fold; X9 (Y1, heating) used as the target. Y2 excluded.",
        "data_exposure": "Retrospective public-data comparison. Earlier SmallMLP development benchmarks named these datasets, including Breast Cancer in an architectural ablation.",
        "versions": {k: version(k) for k in ("numpy", "scikit-learn", "openpyxl", "torch", "xgboost", "smallgbm", "smallmlp", "threadpoolctl")},
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
