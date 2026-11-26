"""Regression on toy data: exact kernel packets in 1-D, sparse-grid inducing points in d dimensions, against the exact
GP by Cholesky; random training and test sets from different distributions, metrics on the test set over SEEDS
replicates (mean and sd). Hyperparameters are trained by default (maximum likelihood; the sparse-grid model on its
variational bound, Cholesky on the exact likelihood), or fixed for both methods by name=value arguments.
    scripts/run_regression.sh python toy [parts] [framework=jax|torch|numpy] [rho=.. sigma2=.. tau2=..]
    scripts/run_regression.sh python toy replot [parts]   (the figures again from the saved results, no evaluation)
    parts: posterior_1d accuracy_1d scaling_1d sparse_grid scaling_nd
Results in results/: metrics.csv (every part and replicate, a `part` column), posterior_1d.npz (the plotted curves);
one figure per kind (.png): posterior_1d (the posterior), accuracy_1d (RMSE, NLPD, W2), scaling_1d (training time and
storage against the training size n), and per dimension d: accuracy_2d (RMSE, NLPD, W2 against the inducing points),
scaling_sparse_grid_2d, scaling_2d (training time and storage against n), and so on for d = 4. Arrays in
KP_FRAMEWORK (jax, torch, or numpy; overridden by framework=...) on the GPU if there is one."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import kernel_packets as kp
from benchmarks.dense_gp import DenseGP
from data import (BIG, INIT, K, KS, M, M_W2, N_1D, N_ACC, N_DENSE, ND_SIZES, NOISE, PARTS, SCALING_RHO, SEEDS,
                  SG_DIMS, SG_LEVELS, SG_N, SIZES, data_1d, data_nd, f, scaling_design)
from evaluation.arrays import array, host, set_framework
from evaluation.metrics import max_error, predictive, predictive_w2, read_csv, rmse, timed, wasserstein2, write_csv
from evaluation.plotting import STYLE, bottom_legend, metric_plot, posterior_plot, save_figure

OUT = Path(__file__).resolve().parent / "results"
METHOD = "stable"                           # method=original: the original KPs (results in results/original)
FIXED = {}                                  # name=value arguments: fixed hyperparameters (else trained)
ACCURACY_1D = [("rmse", "test RMSE", True), ("nlpd", "test NLPD", False), ("w2", r"test $W_2$", True)]
ACCURACY_SG = [("rmse", "test RMSE", True), ("nlpd", "test NLPD", False), ("w2", r"$W_2$ to the exact posterior", True)]
COST = [("time", "time of fit and prediction (s)"), ("storage_mb", "storage of the fitted model (MB)")]


def mean_sd(rows, key, value, **match):
    """Keys in order, and the mean and sd over the replicates of rows[value] at each key, for the rows matching."""
    sel = [r for r in rows if all(r[a] == b for a, b in match.items())]
    keys = list(dict.fromkeys(r[key] for r in sel))
    vals = [[r[value] for r in sel if r[key] == x] for x in keys]
    return keys, (np.array([np.mean(v) for v in vals]), np.array([np.std(v) for v in vals]))


def padded(rows, key, value, methods=tuple(STYLE), **match):
    """All keys in order and each method's (mean, sd) at them, NaN where a method has no rows (Cholesky beyond
    N_DENSE, W2 without the exact reference)."""
    keys = list(dict.fromkeys(r[key] for r in rows if all(r[a] == b for a, b in match.items())))
    out = {}
    for method in methods:
        ks, (mu, sd) = mean_sd([r for r in rows if not np.isnan(r.get(value, np.nan))], key, value, method=method, **match)
        at = {k: i for i, k in enumerate(ks)}
        out[method] = tuple(np.array([a[at[k]] if k in at else np.nan for k in keys]) for a in (mu, sd))
    return keys, out


def fit_1d(X, y, k=K, noisy=False):
    """The KernelPacket at rho (and, noisy, the noise ratio tau2 of the stable method) by maximum likelihood (or
    fixed), the mean beta and sigma2 at their MLEs (or fixed)."""
    noisy = noisy and METHOD == "stable"
    m = kp.KernelPacket(X, k, FIXED.get("rho", 0.1), FIXED.get("tau2", 0.0) if noisy else 0.0, method=METHOD)
    train = noisy and "tau2" not in FIXED
    _, t_train = timed(kp.mle, m, y, bounds=(1e-3, 1.0), tau2_bounds=(1e-8, 10.0) if train else None, repeat=0) \
        if "rho" not in FIXED else (None, 0.0)
    _, beta, s2 = kp.loglike(m, y)
    return m, beta[0], FIXED.get("sigma2", s2), t_train


def posterior_1d():
    """The posterior of noise-free data at N_1D clustered random points for Matern-1/2, 3/2, 5/2 (k = 3, 5, 7), rho by
    maximum likelihood (or fixed): KP and Cholesky means, the KP 95% band, the observations and the truth."""
    X, y, _, _ = data_1d(N_1D, np.random.default_rng(0))
    grid = np.linspace(0, 1, 1000)
    curves = dict(X=X, y=y, grid=grid, truth=f(grid))
    for k in KS:
        m, b, s2, _ = fit_1d(X, y, k)
        mean, var = kp.posterior(m, grid, y - b, var=True)
        ref, ref_var = DenseGP(array(X), array(y - b), k, m.rho, s2).predict(array(grid))
        curves |= {f"rho_{k}": m.rho, f"KP_mean_{k}": np.asarray(mean) + b, f"KP_var_{k}": s2 * np.asarray(var),
                   f"Cholesky_mean_{k}": host(ref) + b, f"Cholesky_var_{k}": host(ref_var)}
    np.savez(OUT / "posterior_1d.npz", **curves)
    plot_posterior_1d()


def plot_posterior_1d():
    c = np.load(OUT / "posterior_1d.npz")
    fig, axes = plt.subplots(1, len(KS), figsize=(5 * len(KS), 3.8), sharey=True)
    for ax, k in zip(axes, KS):
        fits = {mth: (c[f"{mth}_mean_{k}"], c[f"{mth}_var_{k}"]) for mth in STYLE}
        handles = posterior_plot(ax, c["grid"], c["truth"], c["X"], c["y"], fits,
                                 rf"Matérn-${k - 2}/2$, $n = {len(c['X'])}$, $\rho = {float(c[f'rho_{k}']):.3f}$")
    bottom_legend(fig, *handles)
    save_figure(fig, OUT / "posterior_1d.png")


def accuracy_1d():
    """Test metrics against the number of training points N_ACC (clustered data with noise sd NOISE; rho and the noise
    ratio by maximum likelihood): RMSE, NLPD and predictive W2 for KP (n = 10^2 .. 10^8) and Cholesky (n <= N_DENSE) at
    M test points, and the W2 of the KP posterior to the exact one at M_W2 test points."""
    rows = []
    for n in N_ACC:
        for seed in range(SEEDS if n <= BIG else 1):
            X, y, xs, ys = data_1d(n, np.random.default_rng(seed), NOISE)
            m, b, s2, t_train = fit_1d(X, y, noisy=True)
            (mean, var), t = timed(kp.posterior, m, xs, y - b, var=True)
            mean, var = np.asarray(mean) + b, s2 * np.asarray(var)
            row = dict(seed=seed, n=n, m=M, rho=m.rho, sigma2=s2, tau2=m.tau2, train_time=t_train)
            w2_kp = predictive_w2(ys, mean, var)
            if n > N_DENSE:                             # no exact reference: the KP test metrics only
                rows.append(dict(row, method="KP", test_time=t, w2=w2_kp, **predictive(ys, mean, var, y)))
                del m, X, y, xs, ys, mean, var
                continue
            exact, t_fit = timed(DenseGP, array(X), array(y - b), K, m.rho, s2, s2 * m.tau2)
            (ref, ref_var), t_ref = timed(exact.predict, array(xs))
            ref, ref_var = host(ref) + b, host(ref_var)
            xw = xs[:M_W2]
            mean_w, cov_w = kp.posterior(m, xw, y - b, cov=True)
            ref_w, ref_cov = exact.predict(array(xw), cov=True)
            w2_exact = wasserstein2(s2 * np.asarray(cov_w), host(ref_cov), np.asarray(mean_w) - host(ref_w))
            for method, mu, v, tm, w, we in [("KP", mean, var, t, w2_kp, w2_exact),
                                             ("Cholesky", ref, ref_var, t_fit + t_ref, predictive_w2(ys, ref, ref_var), 0.0)]:
                rows.append(dict(row, method=method, test_time=tm, w2=w, w2_exact=we,
                                 mean_vs_cholesky=max_error(mu, ref), var_vs_cholesky=max_error(v, ref_var),
                                 **predictive(ys, mu, v, y)))
    write_csv(rows, OUT / "metrics.csv", "accuracy_1d")
    plot_accuracy_1d()


def plot_accuracy_1d():
    rows = read_csv(OUT / "metrics.csv", "accuracy_1d")
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for a, (col, title, logy) in zip(ax, ACCURACY_1D):
        ns, series = padded(rows, "n", col)
        metric_plot(a, ns, series, r"training size $n$", title, logy=logy)
    save_figure(fig, OUT / "accuracy_1d.png")


def scaling_1d():
    """Training time (rho by maximum likelihood, or fixed, and the fit) and storage against the training size n = 10^2
    .. 10^8 on the jittered unit grid of constant conditioning (Cholesky up to n = N_DENSE)."""
    rows = []
    for n in SIZES:
        for seed in range(min(SEEDS, 3) if n <= BIG else 1):
            X, y, xs, ys = scaling_design(n, np.random.default_rng(seed))
            def train_kp():
                m = kp.KernelPacket(X, K, FIXED.get("rho", SCALING_RHO), method=METHOD)
                if "rho" not in FIXED:
                    kp.mle(m, y, bounds=(1.0, 1e3))
                return m
            m, t = timed(train_kp, repeat=0)   # C++ only: no compilation to exclude
            rows.append(dict(seed=seed, method="KP", n=n, m=M, rho=m.rho, train_time=t,
                             storage_mb=m.nbytes / 2**20, rmse=rmse(kp.posterior(m, xs, y), ys)))
            if n <= N_DENSE:
                train = (lambda: (DenseGP(array(X), array(y), K, FIXED["rho"]), 0.0)) if "rho" in FIXED else \
                    (lambda: DenseGP.mle(array(X), array(y), K, bounds=(1.0, 1e3)))
                (exact, beta), t = timed(train, repeat=0 if n >= 5000 else 1)
                rows.append(dict(seed=seed, method="Cholesky", n=n, m=M, rho=exact.rho, train_time=t,
                                 storage_mb=exact.nbytes / 2**20, rmse=rmse(host(exact.predict(array(xs))[0]) + beta, ys)))
            del m, X, y, xs, ys
    write_csv(rows, OUT / "metrics.csv", "scaling_1d")
    plot_scaling_1d()


def plot_scaling_1d():
    rows = read_csv(OUT / "metrics.csv", "scaling_1d")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ns, s_train = padded(rows, "n", "train_time")
    metric_plot(ax[0], ns, s_train, r"training size $n$", r"training time: maximum likelihood of $\rho$ and fit (s)", (1, 3))
    metric_plot(ax[1], ns, padded(rows, "n", "storage_mb")[1], r"training size $n$", "storage of the fitted model (MB)", (1, 2))
    save_figure(fig, OUT / "scaling_1d.png")


def sparse_grid():
    """Scattered noisy data (n = SG_N) in d dimensions: KP with sparse-grid inducing points of levels eta = d +
    SG_LEVELS and the automatic level (SparseGrid.for_data), trained on the variational bound (or fixed), against the
    exact GP by Cholesky (exact maximum likelihood, or fixed): every predictive metric, the W2 to the exact posterior
    at M_W2 test points, time and storage."""
    rows, p0 = [], {**INIT, **FIXED}
    for d in SG_DIMS:
        for seed in range(SEEDS):
            X, y, xs, ys = data_nd(d, SG_N, np.random.default_rng(seed))
            ys = ys + NOISE * np.random.default_rng(100 + seed).standard_normal(M)   # noisy test targets
            exact = DenseGP(array(X), array(y), K, p0["rho"], p0["sigma2"], p0["tau2"])
            _, t_train = timed(exact.train, repeat=0) if not FIXED else (None, 0.0)
            _, t_fit = timed(exact.fit, exact.rho, exact.sigma2, exact.tau2)
            (ref, ref_var), t = timed(exact.predict, array(xs))
            ref_w, ref_cov = (host(a) for a in exact.predict(array(xs[:M_W2]), cov=True))
            rows.append(dict(seed=seed, method="Cholesky", d=d, level="exact", inducing=np.nan, n=SG_N, rho=exact.rho,
                             sigma2=exact.sigma2, tau2=exact.tau2, train_time=t_train, fit_time=t_fit, test_time=t,
                             time=t_fit + t, storage_mb=exact.nbytes / 2**20, w2=0.0,
                             **predictive(ys, host(ref), host(ref_var) + exact.tau2, y)))
            auto = kp.SparseGrid.for_data(X, K, p0["rho"], p0["tau2"])
            for lev in [*SG_LEVELS[d], "auto"]:
                g = auto if lev == "auto" else kp.SparseGrid(d, d + lev, K, p0["rho"])
                m = kp.SparseGridRegression(array(X), array(y), g, p0["sigma2"], p0["tau2"])
                _, t_train = timed(m.train, repeat=0) if not FIXED else (None, 0.0)
                _, t_fit = timed(lambda: (setattr(m, "_key", None), m.fit()))
                (mean, var), t = timed(m.predict, array(xs), var=True)
                mean_w, cov_w = (host(a) for a in m.predict(array(xs[:M_W2]), cov=True))
                rows.append(dict(seed=seed, method="KP", d=d, level=lev if lev == "auto" else str(g.eta),
                                 inducing=len(g.X), n=SG_N, rho=float(g.rho[0]), sigma2=m.sigma2, tau2=m.tau2,
                                 train_time=t_train, fit_time=t_fit, test_time=t, time=t_fit + t,
                                 storage_mb=m.nbytes / 2**20, w2=wasserstein2(cov_w, ref_cov, mean_w - ref_w),
                                 **predictive(ys, host(mean), host(var) + m.tau2, y)))
    write_csv(rows, OUT / "metrics.csv", "sparse_grid")
    plot_sparse_grid()


def plot_sparse_grid():
    rows = read_csv(OUT / "metrics.csv", "sparse_grid")
    swept = [r for r in rows if r["level"] not in ("auto", "exact")]
    for name, panels in [("accuracy", ACCURACY_SG), ("scaling_sparse_grid", [(c, t, True) for c, t in COST])]:
        for d in SG_DIMS:                                      # accuracy_2d.png, scaling_sparse_grid_2d.png, ...
            fig, ax = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.8))
            for a, (col, title, logy) in zip(ax, panels):
                nz, kps = mean_sd(swept, "inducing", col, method="KP", d=d)
                series = {"KP": kps}
                if col != "w2":                                   # Cholesky is the exact posterior: W2 = 0
                    _, (cm, cs) = mean_sd(rows, "d", col, method="Cholesky", d=d)
                    series["Cholesky"] = (np.full(len(nz), cm[0]), np.full(len(nz), cs[0]))
                metric_plot(a, nz, series, r"inducing points $n_Z$ (sparse-grid levels)",
                            rf"{title}, $d = {d}$, $n = {SG_N}$", logy=logy)
            save_figure(fig, OUT / f"{name}_{d}d.png")


def scaling_nd():
    """Training time (the fit at the given hyperparameters INIT, the automatic sparse grid's selection included) and
    storage against the training size n = ND_SIZES in d = SG_DIMS dimensions: KP with the automatic sparse grid
    (SparseGrid.for_data) against the exact GP by Cholesky (up to n = N_DENSE)."""
    rows, p0 = [], {**INIT, **FIXED}
    for d in SG_DIMS:
        for n in ND_SIZES:
            for seed in range(min(SEEDS, 3)):
                X, y, xs, ys = data_nd(d, n, np.random.default_rng(seed))
                def train_kp():
                    g = kp.SparseGrid.for_data(X, K, p0["rho"], p0["tau2"])
                    return kp.SparseGridRegression(array(X), array(y), g, p0["sigma2"], p0["tau2"])
                m, t = timed(train_kp, repeat=0)   # C++ only: no compilation to exclude
                rows.append(dict(seed=seed, method="KP", d=d, n=n, inducing=len(m.sg.X), train_time=t,
                                 storage_mb=m.nbytes / 2**20, rmse=rmse(host(m.predict(array(xs))), ys)))
                if n <= N_DENSE:
                    exact, t = timed(DenseGP, array(X), array(y), K, p0["rho"], p0["sigma2"], p0["tau2"],
                                     repeat=0 if n >= 5000 else 1)
                    rows.append(dict(seed=seed, method="Cholesky", d=d, n=n, inducing=np.nan, train_time=t,
                                     storage_mb=exact.nbytes / 2**20, rmse=rmse(host(exact.predict(array(xs))[0]), ys)))
    write_csv(rows, OUT / "metrics.csv", "scaling_nd")
    plot_scaling_nd()


def plot_scaling_nd():
    rows = read_csv(OUT / "metrics.csv", "scaling_nd")
    for d in SG_DIMS:                                          # scaling_2d.png, scaling_4d.png, ...
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ns = list(dict.fromkeys(r["n"] for r in rows if r["d"] == d))
        for a, col, title, slopes in [(ax[0], "train_time", "training time: fit at given hyperparameters (s)", (1, 3)),
                                      (ax[1], "storage_mb", "storage of the fitted model (MB)", (1, 2))]:
            series = {}
            for method in STYLE:
                ks, (mu, sd) = mean_sd(rows, "n", col, method=method, d=d)
                pad = len(ns) - len(ks)
                series[method] = (np.r_[mu, [np.nan] * pad], np.r_[sd, [np.nan] * pad])
            metric_plot(a, ns, series, r"training size $n$", rf"{title}, $d = {d}$", slopes)
        save_figure(fig, OUT / f"scaling_{d}d.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    named = dict(a.split("=") for a in args if "=" in a)
    fw_flags = [a[2:] for a in args if a in ("--jax", "--torch", "--numpy")]
    if "framework" in named:
        set_framework(named.pop("framework"))
    elif fw_flags:
        set_framework(fw_flags[-1])
    METHOD = named.pop("method", METHOD)
    FIXED.update({k: float(v) for k, v in named.items()})
    OUT = OUT if METHOD == "stable" else OUT / METHOD
    OUT.mkdir(parents=True, exist_ok=True)
    replot = "replot" in args
    skip = {"replot", "--jax", "--torch", "--numpy"}
    for part in [a for a in args if "=" not in a and a not in skip] or PARTS:
        print(f"== {part}")
        globals()[f"plot_{part}" if replot else part]()
