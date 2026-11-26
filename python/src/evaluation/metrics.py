"""Metrics of the experiments (after the GP literature).
Predictive, test targets y against a Gaussian predictive N(mu, var) (lower is better unless noted):
  rmse, mae
  smse            : MSE / variance of the test targets                                (Rasmussen & Williams 2006)
  nlpd            : mean negative log predictive density                              (Quinonero-Candela et al. 2006)
  msll            : nlpd minus that of N(mean(y_train), var(y_train))                 (Rasmussen & Williams 2006)
  crps            : continuous ranked probability score, Gaussian closed form         (Gneiting & Raftery 2007)
  interval_score  : width + 2 / alpha * miss distance of the central 1 - alpha interval (Gneiting & Raftery 2007)
  coverage        : fraction of y inside the central 95% interval (target 0.95); coverage_error = |coverage - 0.95|
  zscore_ms       : mean squared standardized residual (target 1)
Samplers, implied covariance K^ = R R^T against the exact K = L L^T, W = L^{-1} K^ L^{-T} - I:
  cov_error       : max |K^ - K|;  whitened_error: ||W||_2;  kl_gaussian: KL(N(0, K^) || N(0, K))
  wasserstein2    : 2-Wasserstein distance of two Gaussians (Wilson et al. 2021); of nearly equal covariances it
                    cancels to its sqrt(eps tr K) floor, so exact samplers are scored by cov_error
Cost: timed (seconds after a call that compiles), the models' nbytes (storage). The experiment's single CSV table
(write_csv, read_csv: figures and tables are made again from it without evaluating the methods again)."""
import csv
import time

import numpy as np
from scipy.special import erf

from kernel_packets.backend import bk

Z95 = 1.959963984540054


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def max_error(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def nlpd(y, mean, var):
    """Mean negative log predictive density of y under N(mean, var)."""
    y, mean, var = np.asarray(y), np.asarray(mean), np.asarray(var)
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (y - mean) ** 2 / var))


def coverage(y, mean, var, z=Z95):
    """Fraction of y inside mean +- z sd."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(mean)) <= z * np.sqrt(np.clip(np.asarray(var), 0, None))))


def predictive(y, mean, var, y_train):
    """Every predictive metric of the test targets y under N(mean, var) (variances floored at 1e-12)."""
    y, mean, var = np.asarray(y, float), np.asarray(mean, float), np.maximum(np.asarray(var, float), 1e-12)
    sd, r = np.sqrt(var), y - mean
    z = r / sd
    crps = sd * (z * erf(z / np.sqrt(2)) + np.sqrt(2 / np.pi) * np.exp(-z ** 2 / 2) - 1 / np.sqrt(np.pi))
    lo, hi = mean - Z95 * sd, mean + Z95 * sd
    score = (hi - lo) + 2 / 0.05 * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    mt, vt = float(np.mean(y_train)), float(np.var(y_train))
    cov = coverage(y, mean, var)
    out = dict(rmse=rmse(mean, y), mae=float(np.mean(np.abs(r))), smse=float(np.mean(r ** 2) / np.var(y)),
               nlpd=nlpd(y, mean, var), msll=nlpd(y, mean, var) - nlpd(y, np.full_like(y, mt), np.full_like(y, vt)),
               crps=float(np.mean(crps)), interval_score=float(np.mean(score)), coverage=cov,
               coverage_error=abs(cov - 0.95), zscore_ms=float(np.mean(z ** 2)))
    return out


def predictive_w2(y, mean, var):
    """Per-point 2-Wasserstein distance W2(N(mean, diag(var)), delta_y) / sqrt(m) = sqrt(mean((mean - y)^2 + var))."""
    y, mean, var = np.asarray(y, float), np.asarray(mean, float), np.maximum(np.asarray(var, float), 0.0)
    return float(np.sqrt(np.mean((mean - y) ** 2 + var)))


def covariance_error(L, K):
    """max |L L^T - K| of a sampler's implied factor L (its draws from the unit vectors)."""
    L = np.asarray(L)
    return max_error(L @ L.T, K)


def sampler(L, K):
    """cov_error, whitened_error, kl_gaussian and wasserstein2 of the implied factor L against the exact K."""
    L, K = np.asarray(L), np.asarray(K)
    C = np.linalg.cholesky(K)
    R = np.linalg.solve(C, L)
    lam = np.linalg.eigvalsh(R @ R.T - np.eye(len(K)))
    return dict(cov_error=covariance_error(L, K), whitened_error=float(np.max(np.abs(lam))),
                kl_gaussian=float(0.5 * np.sum(lam - np.log1p(np.maximum(lam, -1 + 1e-300)))),
                wasserstein2=factor_wasserstein2(L, K))


def _psd_sqrt(M):
    w, V = np.linalg.eigh((M + M.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def factor_wasserstein2(L, K):
    """W2(N(0, L L^T), N(0, K)) = ||K^{1/2} U - L V||_F where K^{1/2} L = U S V^T (no trace cancellation)."""
    L, S = np.asarray(L, float), _psd_sqrt(np.asarray(K, float))
    U, _, Vt = np.linalg.svd(S @ L, full_matrices=False)
    return float(np.linalg.norm(S @ U - L @ Vt.T, "fro"))


def wasserstein2(Kh, K, dm=0.0):
    """W2(N(m + dm, Kh), N(m, K)) = sqrt(|dm|^2 + tr K + tr Kh - 2 tr (K^{1/2} Kh K^{1/2})^{1/2})."""
    Kh, K, S = np.asarray(Kh), np.asarray(K), _psd_sqrt(np.asarray(K))
    w2 = np.sum(np.asarray(dm) ** 2) + np.trace(K) + np.trace(Kh) - 2 * np.trace(_psd_sqrt(S @ Kh @ S))
    return float(np.sqrt(max(w2, 0.0)))


def draws_wasserstein2(F, mean, cov):
    """W2 between the Gaussian of the draws F (m, p) (sample mean and covariance) and N(mean, cov)."""
    F = np.asarray(F)
    return wasserstein2(np.cov(F), cov, F.mean(1) - np.asarray(mean))


def timed(fn, *args, repeat=1, **kwargs):
    """(result, seconds): the mean of `repeat` calls after a first one that compiles (repeat = 0: the first call)."""
    t = time.perf_counter()
    out = bk.sync(fn(*args, **kwargs))
    if not repeat:
        return out, time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(repeat):
        out = bk.sync(fn(*args, **kwargs))
    return out, (time.perf_counter() - t) / repeat


def write_csv(rows, path, part):
    """Rows (dicts) of one experiment part into the experiment's single table at path: a `part` column, the union of
    the columns; earlier rows of the same part are replaced. Printed too."""
    old = []
    if path.exists():
        with open(path, newline="") as f:
            old = [r for r in csv.DictReader(f) if r["part"] != part]
    rows = [{"part": part, **r} for r in rows]
    cols = list(dict.fromkeys(k for r in old + rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(old)
        w.writerows({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    for r in rows:
        print(", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


def read_csv(path, part):
    """The rows of one part of the experiment's table, numbers as floats (empty cells as NaN, other text kept)."""
    def value(v):
        try:
            return float(v) if v != "" else float("nan")
        except ValueError:
            return v
    with open(path, newline="") as f:
        return [{k: value(v) for k, v in r.items()} for r in csv.DictReader(f) if r["part"] == part]
