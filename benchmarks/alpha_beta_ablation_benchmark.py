"""SmallMLP capacity sensitivity on three prespecified development datasets (Figure 7)."""
import argparse
import csv
import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import product
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from sklearn.datasets import load_iris, load_wine
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from smallmlp import SmallMLPClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
ALPHAS = tuple(range(2, 11))
BETAS = (0.5, 0.6, 0.7, 0.8, 0.9)
SELECTED = (6, 0.7)
SEED = 42
N_FOLDS = 5
HEART_URL = "https://archive.ics.uci.edu/static/public/145/data.csv"
HEART_SHA256 = "440ab32cedbc4c3e0587439e48cccf1792a99323df35611b17f33adb140f6baf"
METHOD = {"width_mode": "formula", "class_weight": None, "max_epochs": 500,
          "patience": 30, "random_state": SEED}


def load_development_data(cache):
    iris, wine = load_iris(return_X_y=True), load_wine(return_X_y=True)
    path = cache / "statlog_heart_145.csv"
    if not path.exists():
        with urlopen(HEART_URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != HEART_SHA256:
            raise ValueError("UCI Statlog Heart file changed. Review the source before rerunning.")
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest() != HEART_SHA256:
        raise ValueError(f"Unexpected SHA-256 for {path}.")
    matrix = np.genfromtxt(path, delimiter=",", skip_header=1)
    heart = (matrix[:, :-1], matrix[:, -1].astype(int))
    return {"Iris": iris, "Wine": wine, "Statlog (Heart)": heart}


def build_folds(datasets):
    folds, assignments = {}, []
    for name, (X, y) in datasets.items():
        folds[name] = []
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for fold, (train, test) in enumerate(cv.split(X, y), start=1):
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            x_train = imputer.fit_transform(X[train])
            x_test = imputer.transform(X[test])
            folds[name].append((x_train, y[train], x_test, y[test]))
            assignments.extend({"dataset": name, "row_index_zero_based": int(i), "test_fold": fold}
                               for i in test)
    return folds, sorted(assignments, key=lambda row: (row["dataset"], row["row_index_zero_based"]))


def evaluate(task):
    name, alpha, beta, folds = task
    torch.set_num_threads(1)
    rows = []
    with threadpool_limits(limits=1):
        for i, (x_train, y_train, x_test, y_test) in enumerate(folds, start=1):
            model = SmallMLPClassifier(alpha=alpha, beta=beta, **METHOD)
            model.fit(x_train, y_train)
            rows.append({"dataset": name, "alpha": alpha, "beta": beta, "fold": i,
                         "n_train": len(y_train), "n_test": len(y_test),
                         "accuracy": float(accuracy_score(y_test, model.predict(x_test)))})
    return rows


def write_csv(file, rows):
    with file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def summarize(runs, names):
    cells = []
    by_dataset = []
    for alpha, beta in product(ALPHAS, BETAS):
        scores = []
        for name in names:
            acc = [r["accuracy"] for r in runs
                   if (r["dataset"], r["alpha"], r["beta"]) == (name, alpha, beta)]
            assert len(acc) == N_FOLDS
            mean = float(np.mean(acc))
            by_dataset.append({"dataset": name, "alpha": alpha, "beta": beta, "mean_accuracy": mean})
            scores.append(mean)
        cells.append({"alpha": alpha, "beta": beta, "mean_accuracy": float(np.mean(scores))})
    return cells, by_dataset


def plot_grid(cells, path):
    values = np.array([r["mean_accuracy"] for r in cells]).reshape(len(ALPHAS), len(BETAS))
    norm = matplotlib.colors.Normalize(values.min() - 0.001, values.max() + 0.001)
    cmap = plt.get_cmap("Blues")
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(5.1, 4.15))
        image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(ALPHAS)):
            for j in range(len(BETAS)):
                text_color = "white" if norm(values[i, j]) > 0.66 else "#121722"
                ax.text(j, i, f"{values[i, j]:.4f}", ha="center", va="center",
                        color=text_color, fontsize=8)
        i = ALPHAS.index(SELECTED[0]); j = BETAS.index(SELECTED[1])
        ax.add_patch(Rectangle((j - 0.475, i - 0.475), 0.95, 0.95, fill=False,
                               ec="#c33d23", lw=2.0))
        ax.set_xticks(range(len(BETAS)), BETAS)
        ax.set_yticks(range(len(ALPHAS)), ALPHAS)
        ax.set_xlabel(r"Width decay $\beta$")
        ax.set_ylabel(r"Initial width multiplier $\alpha$")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        bar = fig.colorbar(image, ax=ax, shrink=0.82)
        bar.set_label("Mean accuracy")
        fig.tight_layout()
        fig.savefig(path)
        fig.savefig(path.with_suffix(".png"), dpi=240)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "meta_training")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    datasets = load_development_data(args.data_dir)
    folds, assignments = build_folds(datasets)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    out = ROOT / "results" / "meta_training" / stamp
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "folds.csv", assignments)
    tasks = [(name, alpha, beta, folds[name])
             for alpha, beta in product(ALPHAS, BETAS) for name in datasets]
    protocol = {
        "datasets": {k: {"n": len(y), "d": X.shape[1], "classes": len(np.unique(y)),
                         "missing_cells": int(np.isnan(X).sum())} for k, (X, y) in datasets.items()},
        "sources": {"Iris": "sklearn.datasets.load_iris", "Wine": "sklearn.datasets.load_wine",
                    "Statlog (Heart)": {"uci_id": 145, "url": HEART_URL, "sha256": HEART_SHA256}},
        "alphas": ALPHAS, "betas": BETAS, "selected_before_run": SELECTED, "model": METHOD,
        "split": "Five stratified shuffled folds (seed 42), fixed across all cells. Each fold fits its own median imputer.",
        "score": "Arithmetic mean of five fold accuracies per dataset, then arithmetic mean across three datasets.",
        "jobs": args.jobs, "versions": {p: version(p) for p in ("smallmlp", "scikit-learn", "numpy", "torch", "matplotlib")}}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    started = time.monotonic()
    runs = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, result in enumerate(pool.map(evaluate, tasks), start=1):
            runs.extend(result)
            if i % 9 == 0:
                print(f"{i}/{len(tasks)} model settings by dataset in {time.monotonic()-started:.1f}s", flush=True)
    write_csv(out / "runs.csv", runs)
    cells, by_dataset = summarize(runs, datasets)
    write_csv(out / "summary.csv", cells)
    write_csv(out / "by_dataset.csv", by_dataset)
    plot_grid(cells, out / "meta_grid_surface.pdf")
    shutil.copyfile(out / "meta_grid_surface.pdf", ROOT / "meta_grid_surface.pdf")
    shutil.copyfile(out / "meta_grid_surface.png", ROOT / "meta_grid_surface.png")
    shutil.copyfile(out / "meta_grid_surface.png", ROOT / "benchmarks" / "meta_grid_surface4.3.1.png")
    write_csv(ROOT / "meta_training_grid_results.csv",
              [{"alpha": r["alpha"], "beta": r["beta"], "mean_acc": r["mean_accuracy"]}
               for r in cells])
    print(f"Saved {len(runs)} fits and Figure 7 in {out}")


if __name__ == "__main__":
    main()
