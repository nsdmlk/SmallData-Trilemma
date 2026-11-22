#!/usr/bin/env python3
"""Coverage and width on heteroscedastic regression data (Figure 6)."""

import csv
import json
import math
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm, t
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from smallmlp import SmallMLPRegressor

# ---------------- Configuration ----------------
N_TOTAL = 150
CAL_FRAC = 0.25          # Single calibration split: n_fit=112, n_cal=38
N_TEST = 2000
ALPHA = 0.10             # Target marginal coverage = 0.90
DIMS = (10, 100)         # Low-d (d=10) and High-d (d=100, d ~ n)
SEEDS = tuple(range(101, 121))  # 20 independent seeds
MAX_EPOCHS = 500
PATIENCE = 30

ROOT_DIR = Path(__file__).resolve().parent.parent
GP = "GP (Type-II ML)"
SPLIT = "SmallMLP (Split Conformal)"
LOCAL = "SmallMLP (Locally Weighted)"
METHODS = (GP, SPLIT, LOCAL)


# ---------------- Data Generator ----------------
def make_heteroscedastic_data(d: int, seed: int):
    streams = np.random.SeedSequence([20261119, d, seed]).spawn(2)

    def draw(n, stream):
        rng = np.random.default_rng(stream)
        X = rng.normal(size=(n, d))
        f_x = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.8 * X[:, 2] * X[:, 3] + np.sin(X[:, 4])
        # Sharp 6x heteroscedastic noise regime based on X[:, 0]
        sigma_x = np.where(X[:, 0] <= 0.0, 0.25, 1.50)
        y = f_x + sigma_x * rng.normal(size=n)
        return np.ascontiguousarray(X, dtype=np.float64), y, sigma_x

    X, y, sig = draw(N_TOTAL, streams[0])
    Xt, yt, sig_t = draw(N_TEST, streams[1])
    return X, y, sig, Xt, yt, sig_t


# ---------------- Embedding & Conformal Helpers ----------------
def extract_penultimate_embedding(model: SmallMLPRegressor, X: np.ndarray) -> np.ndarray:
    """Apply the same input standardization and fitted backbone as predict()."""
    model._model.eval()
    with torch.no_grad():
        return model._model(model._prepare_query(X)).cpu().numpy()


def split_conformal_radius(residuals: np.ndarray, alpha: float) -> float:
    """Return the one-based rank ceil((m + 1)(1 - alpha)), including infinity."""
    if not 0 < alpha < 1 or len(residuals) == 0:
        raise ValueError("Expected calibration residuals and an alpha between 0 and 1.")
    k = math.ceil((len(residuals) + 1) * (1.0 - alpha))
    if k > len(residuals):
        return np.inf
    return float(np.partition(residuals, k - 1)[k - 1])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Computes the q-quantile of `values` with non-negative `weights`."""
    sorter = np.argsort(values)
    v_sorted = values[sorter]
    w_sorted = weights[sorter]
    cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
    idx = np.searchsorted(cdf, q, side="left")
    return float(v_sorted[min(idx, len(v_sorted) - 1)])


def locally_weighted_conformal_radii(
    h_cal: np.ndarray, r_cal: np.ndarray, h_test: np.ndarray, alpha: float
) -> tuple[np.ndarray, int]:
    """
    Empirical local interval rule (Eq. 9):
    h_bw = median(||h(x_i) - h(x_j)||) over calibration set.
    """
    # Pairwise Euclidean distances on calibration set for deterministic median heuristic
    diff_cal = h_cal[:, None, :] - h_cal[None, :, :]
    dist_cal = np.linalg.norm(diff_cal, axis=-1)
    triu = dist_cal[np.triu_indices(len(h_cal), k=1)]
    h_bw = float(np.median(triu))
    if h_bw < 1e-8:
        h_bw = 1.0

    # Distances from test points to calibration points
    diff_test = h_test[:, None, :] - h_cal[None, :, :]
    dist_test = np.linalg.norm(diff_test, axis=-1)
    W = np.exp(-0.5 * (dist_test / h_bw) ** 2)

    # A unit test weight at infinity participates in the weighted empirical quantile.
    radii = np.empty(len(h_test), dtype=np.float64)
    r_aug = np.append(r_cal, np.inf)
    for i in range(len(h_test)):
        w_aug = np.append(W[i], 1.0)
        radii[i] = weighted_quantile(r_aug, w_aug, 1.0 - alpha)
    # A test point can have too little nearby calibration weight even with finite weights.
    # Keep the original finite-cap convention and report how often it changes a radius.
    capped = np.isinf(radii)
    radii[capped] = np.max(r_cal)
    return radii, int(capped.sum())


# ---------------- Single Trial ----------------
def run_trial(d: int, seed: int):
    X, y, _, Xt, yt, sig_t = make_heteroscedastic_data(d, seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_TOTAL)
    n_cal = int(round(CAL_FRAC * N_TOTAL))
    cal_idx, fit_idx = perm[:n_cal], perm[n_cal:]
    X_fit, y_fit = X[fit_idx], y[fit_idx]
    X_cal, y_cal = X[cal_idx], y[cal_idx]

    low_mask = sig_t < 0.5
    high_mask = ~low_mask
    results = []

    def record(method, pred, radii, *, capped=None, convergence_warnings=None):
        radii = np.broadcast_to(np.asarray(radii, dtype=float), yt.shape)
        covered = np.abs(yt - np.asarray(pred, dtype=float)) <= radii
        widths = 2.0 * radii
        corr = float(np.corrcoef(widths, sig_t)[0, 1]) if np.std(widths) > 1e-12 else 0.0
        results.append({
            "d": d,
            "seed": seed,
            "method": method,
            "marginal_coverage": float(np.mean(covered)),
            "low_noise_coverage": float(np.mean(covered[low_mask])),
            "high_noise_coverage": float(np.mean(covered[high_mask])),
            "mean_width": float(np.mean(widths)),
            "low_noise_width": float(np.mean(widths[low_mask])),
            "high_noise_width": float(np.mean(widths[high_mask])),
            "width_sigma_corr": corr,
            "capped_test_points": capped,
            "gp_convergence_warnings": convergence_warnings,
        })

    # 1. Gaussian Process (RBF + WhiteKernel, Type-II Marginal Likelihood) on full N_TOTAL
    scaler = StandardScaler()
    X_gp = scaler.fit_transform(X)
    Xt_gp = scaler.transform(Xt)
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(d)) + WhiteKernel(noise_level=1.0)
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=2, random_state=seed, normalize_y=True
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        gp.fit(X_gp, y)
    n_warnings = sum(issubclass(w.category, ConvergenceWarning) for w in captured)
    mu_gp, std_gp = gp.predict(Xt_gp, return_std=True)
    z = norm.ppf(1.0 - ALPHA / 2.0)
    record(GP, mu_gp, z * std_gp, convergence_warnings=n_warnings)

    # 2. SmallMLPRegressor fitted on fit_idx, calibrated on cal_idx
    mlp = SmallMLPRegressor(random_state=seed, max_epochs=MAX_EPOCHS, patience=PATIENCE)
    mlp.fit(X_fit, y_fit)
    pred_cal = mlp.predict(X_cal)
    pred_test = mlp.predict(Xt)
    r_cal = np.abs(y_cal - pred_cal)

    # 2a. Unweighted Split Conformal (Eq. 8)
    record(SPLIT, pred_test, split_conformal_radius(r_cal, ALPHA))

    # 2b. Locally weighted intervals on the standardized penultimate embeddings
    h_cal = extract_penultimate_embedding(mlp, X_cal)
    h_test = extract_penultimate_embedding(mlp, Xt)
    radii_lw, capped = locally_weighted_conformal_radii(h_cal, r_cal, h_test, ALPHA)
    record(LOCAL, pred_test, radii_lw, capped=capped)

    return results


def mean_ci(values):
    """Pointwise 95% Student t interval for a mean over independent seeds."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    radius = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return mean, float(mean - radius), float(mean + radius)


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    metrics = ("marginal_coverage", "low_noise_coverage", "high_noise_coverage",
               "mean_width", "low_noise_width", "high_noise_width", "width_sigma_corr")
    summary, paired = [], []
    for d in DIMS:
        by_method = {}
        for method in METHODS:
            group = sorted((r for r in rows if r["d"] == d and r["method"] == method),
                           key=lambda r: r["seed"])
            assert tuple(r["seed"] for r in group) == SEEDS
            by_method[method] = group
            result = {"d": d, "method": method, "n_seeds": len(group)}
            for metric in metrics:
                mean, lo, hi = mean_ci([r[metric] for r in group])
                result.update({metric: mean, metric + "_lo": lo, metric + "_hi": hi})
            result["high_to_low_width_ratio"] = float(np.mean(
                [r["high_noise_width"] / r["low_noise_width"] for r in group]))
            result["cap_rate"] = (sum(r["capped_test_points"] for r in group) /
                                  (len(group) * N_TEST) if method == LOCAL else None)
            summary.append(result)
        for metric in ("marginal_coverage", "high_noise_coverage", "mean_width"):
            diffs = [a[metric] - b[metric] for a, b in zip(by_method[LOCAL], by_method[SPLIT])]
            mean, lo, hi = mean_ci(diffs)
            paired.append({"d": d, "comparison": "locally weighted minus split",
                           "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi,
                           "n_positive": sum(x > 0 for x in diffs), "n_seeds": len(diffs)})
    return summary, paired


def plot_results(summary, path):
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "axes.axisbelow": True}):
        fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))
        titles = ("Overall coverage", "High-noise coverage", "Mean interval width")
        metrics = ("marginal_coverage", "high_noise_coverage", "mean_width")
        colors = ("#93411d", "#405695", "#007467")
        labels = ("Gaussian process", "SmallMLP, unweighted", "SmallMLP, locally weighted")
        for axis, title, metric in zip(axes, titles, metrics):
            for i, method in enumerate(METHODS):
                rows = [r for r in summary if r["method"] == method]
                mean = np.array([r[metric] for r in rows])
                lo = np.array([r[metric + "_lo"] for r in rows])
                hi = np.array([r[metric + "_hi"] for r in rows])
                axis.errorbar(np.arange(len(rows)) + (i - 1) * 0.16, mean,
                              yerr=np.vstack((mean - lo, hi - mean)), fmt=("s", "o", "^")[i],
                              color=colors[i], capsize=3, ms=5, lw=1.4, label=labels[i])
            if metric != "mean_width":
                axis.axhline(1 - ALPHA, ls="--", color="#626262", lw=1, label="90% target")
                axis.set_ylim(0.68, 1.01)
                axis.set_yticks([0.70, 0.80, 0.90, 1.00], ["70%", "80%", "90%", "100%"])
            axis.set_xticks(range(len(DIMS)), [f"$d={d}$" for d in DIMS])
            axis.set_xlim(-0.38, len(DIMS)-0.62)
            axis.set_title(title, loc="left", fontweight="semibold")
            axis.grid(axis="y", lw=0.55, color="#d6d6d6")
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False,
                   fontsize=8, bbox_to_anchor=(0.5, 0.016))
        fig.subplots_adjust(left=0.057, right=0.99, bottom=0.22, top=0.88, wspace=0.29)
        fig.savefig(path)
        plt.close(fig)


def run_worker(pair):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        return run_trial(*pair)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1, help="Number of independent trials in parallel")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    out = ROOT_DIR / "results" / "conformal_uq" / stamp
    out.mkdir(parents=True, exist_ok=False)
    tasks = [(d, seed) for d in DIMS for seed in SEEDS]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            trials = list(pool.map(run_worker, tasks))
    else:
        trials = [run_worker(task) for task in tasks]
    rows = [r for trial in trials for r in trial]
    summary, paired = summarize(rows)
    save_csv(out / "runs.csv", rows)
    save_csv(out / "summary.csv", summary)
    save_csv(out / "paired.csv", paired)
    protocol = {
        "n_total": N_TOTAL, "n_test": N_TEST, "n_fit": 112, "n_cal": 38,
        "alpha": ALPHA, "dims": DIMS, "seeds": SEEDS,
        "split_calibration_rank_one_based": 36,
        "smallmlp_internal_early_stopping_fraction": 0.2,
        "gp_optimizer_restarts": 2, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "embedding": "SmallMLP's fitted backbone on its fitted input scaler, as in predict().",
        "local_quantile": "Gaussian weights with median pairwise bandwidth and unit mass at infinity. Infinite local radii are capped at the largest calibration residual.",
        "intervals": "Pointwise 95% Student t intervals across independent seeds. Paired differences use the same data and fitted SmallMLP.",
        "jobs": args.jobs,
        "versions": {p: version(p) for p in ("numpy", "torch", "scikit-learn", "scipy", "smallmlp", "matplotlib")},
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    plot_results(summary, out / "conformal_uq.pdf")
    shutil.copyfile(out / "conformal_uq.pdf", ROOT_DIR / "conformal_uq.pdf")
    print(f"Saved 40 trials, paired comparisons, and Figure 6 at {out}")


if __name__ == "__main__":
    main()
