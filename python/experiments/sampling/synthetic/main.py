"""Prior and posterior sampling on synthetic designs: exact kernel packets in 1-D and sparse-grid inducing points in d
dimensions (KP), against the exact GP by Cholesky; random inputs, SEEDS replicates (mean and sd). Hyperparameters are
trained by default (maximum likelihood in 1-D; the variational bound in d dimensions, on the automatic sparse grid
SparseGrid.for_data), or fixed by name=value arguments.
    scripts/run_sampling.sh python synthetic [parts] [framework=jax|torch|numpy] [rho=.. sigma2=.. tau2=..]
    scripts/run_sampling.sh python synthetic replot [parts]   (the figures again from the saved results, no evaluation)
    parts: prior_draws prior_accuracy prior_sensitivity prior_scaling posterior_draws posterior_accuracy posterior_sensitivity scaling_1d posterior_sparse_grid
Results in results/: metrics.csv (every part and replicate, a `part` column), prior_draws_1d.npz and
posterior_draws_1d.npz (the plotted draws and curves); the figures (.png). Arrays in KP_FRAMEWORK (jax, torch, or numpy;
overridden by framework=...) on the GPU if there is one."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import kernel_packets as kp
from benchmarks.dense_gp import DenseGP
from data import (BIG, K, KS, M_POST, N_DENSE, N_POST, NOISE, P_ACC, P_POST, P_TIME, PARTS, POST_ACC_N, POST_SENS_RHOS,
                  POST_SENS_TAU2S, PRIOR_ACC_N, PRIOR_RHO, SCALING_RHO, SEEDS, SENS_KS, SENS_N, SENS_RHOS, SG_DIMS,
                  SG_M, SG_N, SIZES, f, griewank, nodes, rho, scaling_data, scaling_design, test_inputs, train_inputs)
from evaluation.arrays import array, host, set_framework
from evaluation.metrics import draws_wasserstein2, max_error, predictive_w2, read_csv, rmse, sampler, timed, write_csv
from evaluation.plotting import STYLE, bottom_legend, draws_plot, metric_plot, save_figure

OUT = Path(__file__).resolve().parent / "results"
METHOD = "stable"                           # method=original: the original KPs (results in results/original)
FIXED = {}                                  # name=value arguments: fixed hyperparameters (else trained)


def mean_sd(rows, key, value, **match):
    """Keys in order, and the mean and sd over the replicates of rows[value] at each key, for the rows matching."""
    sel = [r for r in rows if all(r[a] == b for a, b in match.items())]
    keys = list(dict.fromkeys(r[key] for r in sel))
    vals = [[r[value] for r in sel if r[key] == x] for x in keys]
    return keys, (np.array([np.mean(v) for v in vals]), np.array([np.std(v) for v in vals]))


def padded(rows, key, value, methods=tuple(STYLE), **match):
    """All keys in order and each method's (mean, sd) at them, NaN where a method has no rows."""
    keys = list(dict.fromkeys(r[key] for r in rows if all(r[a] == b for a, b in match.items())))
    out = {}
    for method in methods:
        ks, (mu, sd) = mean_sd([r for r in rows if not np.isnan(r.get(value, np.nan))], key, value, method=method, **match)
        at = {k: i for i, k in enumerate(ks)}
        out[method] = tuple(np.array([a[at[k]] if k in at else np.nan for k in keys]) for a in (mu, sd))
    return keys, out


def prior_draws():
    """Three prior draws of KP and Cholesky at 500 random nodes for Matern-1/2, 3/2, 5/2, with the prior 95% band."""
    X = nodes(500, np.random.default_rng(0))
    curves = dict(X=X)
    for k in KS:
        curves |= {f"KP_{k}": np.asarray(kp.prior_sample(kp.KernelPacket(X, k, PRIOR_RHO, method=METHOD), 3, rng=0)),
                   f"Cholesky_{k}": host(DenseGP.prior_sample(array(X), k, PRIOR_RHO, 3, rng=1))}
    np.savez(OUT / "prior_draws_1d.npz", **curves)
    plot_prior_draws()


def plot_prior_draws():
    c = np.load(OUT / "prior_draws_1d.npz")
    X = c["X"]
    fig, axes = plt.subplots(1, len(KS), figsize=(5 * len(KS), 3.8), sharey=True)
    for ax, k in zip(axes, KS):
        handles = draws_plot(ax, X, {mth: c[f"{mth}_{k}"] for mth in STYLE},
                             rf"Matérn-${k - 2}/2$, $\rho = {PRIOR_RHO}$, $n = {len(X)}$",
                             band=(np.zeros(len(X)), np.ones(len(X)), "prior mean and 95% interval"))
    bottom_legend(fig, *handles)
    save_figure(fig, OUT / "prior_draws_1d.png")


def prior_accuracy():
    """Each sampler's Gaussian N(0, L L^T) (L its draws from the unit vectors) against N(0, K) over PRIOR_ACC_N
    points: wasserstein2, cov_error, whitened_error, kl_gaussian."""
    rows = []
    for k in SENS_KS:
        for n in PRIOR_ACC_N:
            for seed in range(SEEDS):
                X = scaling_design(n, np.random.default_rng(seed))
                Kd = np.asarray(kp.matern(X, X, k, SCALING_RHO))
                cond = float(np.linalg.cond(Kd))
                for method, L in [("KP", kp.KernelPacket(X, k, SCALING_RHO, method=METHOD).sample_(np.eye(n))),
                                  ("Cholesky", np.linalg.cholesky(Kd))]:
                    rows.append(dict(seed=seed, k=k, n=n, rho=SCALING_RHO, cond=cond, method=method,
                                     **sampler(np.asarray(L), Kd)))
    write_csv(rows, OUT / "metrics.csv", "prior_accuracy")
    plot_prior_accuracy()


def plot_prior_accuracy():
    rows = read_csv(OUT / "metrics.csv", "prior_accuracy")
    ns = list(dict.fromkeys(r["n"] for r in rows))
    fig, axes = plt.subplots(1, len(SENS_KS), figsize=(15, 3.8), sharey=True)
    for ax, k in zip(axes, SENS_KS):
        series = {m: mean_sd(rows, "n", "wasserstein2", method=m, k=k)[1] for m in STYLE}
        series = {m: (np.maximum(v[0], 1e-17), v[1]) for m, v in series.items()}
        metric_plot(ax, ns, series, r"number of points $n$", rf"$W_2$ to the exact prior, Matérn-${k - 2}/2$")
    save_figure(fig, OUT / "prior_accuracy_1d.png")


def prior_sensitivity():
    """Prior sampler implied covariance N(0, L L^T) against N(0, K) at SENS_N random nodes across lengthscales
    SENS_RHOS and kernels SENS_KS."""
    rows = []
    for k in SENS_KS:
        for r in SENS_RHOS:
            for seed in range(SEEDS):
                X = nodes(SENS_N, np.random.default_rng(seed))
                Kd = np.asarray(kp.matern(X, X, k, r))
                cond = float(np.linalg.cond(Kd))
                for method, L in [("KP", kp.KernelPacket(X, k, r, method=METHOD).sample_(np.eye(SENS_N))),
                                  ("Cholesky", np.linalg.cholesky(Kd))]:
                    rows.append(dict(seed=seed, k=k, rho=r, n=SENS_N, cond=cond, method=method,
                                     **sampler(np.asarray(L), Kd)))
    write_csv(rows, OUT / "metrics.csv", "prior_sensitivity")
    plot_prior_sensitivity()


def plot_prior_sensitivity():
    rows = read_csv(OUT / "metrics.csv", "prior_sensitivity")
    rhos = list(dict.fromkeys(r["rho"] for r in rows))
    fig, axes = plt.subplots(1, len(SENS_KS), figsize=(15, 3.8), sharey=True)
    for ax, k in zip(axes, SENS_KS):
        series = {m: mean_sd(rows, "rho", "wasserstein2", method=m, k=k)[1] for m in STYLE}
        series = {m: (np.maximum(v[0], 1e-17), v[1]) for m, v in series.items()}
        metric_plot(ax, rhos, series, r"lengthscale $\rho$",
                    rf"$W_2$ to the exact prior, Matérn-${k - 2}/2$, $n = {SENS_N}$")
    save_figure(fig, OUT / "prior_sensitivity_1d.png")


def prior_scaling():
    """Setup and P_TIME prior draws against n = 10^2 .. 10^8 (Cholesky up to N_DENSE): time and storage."""
    rows = []
    for n in SIZES:
        for seed in range(min(SEEDS, 3) if n <= BIG else 1):
            X = scaling_design(n, np.random.default_rng(seed))
            def run_kp():
                m = kp.KernelPacket(X, K, SCALING_RHO, method=METHOD)
                kp.prior_sample(m, P_TIME, seed)
                return m
            m, t = timed(run_kp, repeat=0)
            rows.append(dict(seed=seed, method="KP", n=n, draws=P_TIME, time=t, storage_mb=m.nbytes / 2**20))
            if n <= N_DENSE:
                F, t = timed(DenseGP.prior_sample, array(X), K, SCALING_RHO, P_TIME, seed,
                             repeat=0 if n >= 5000 else 1)
                rows.append(dict(seed=seed, method="Cholesky", n=n, draws=P_TIME, time=t,
                                 storage_mb=8 * n * n / 2**20))
            del m, X
    write_csv(rows, OUT / "metrics.csv", "prior_scaling")
    plot_prior_scaling()


def plot_prior_scaling():
    rows = read_csv(OUT / "metrics.csv", "prior_scaling")
    sizes = list(dict.fromkeys(r["n"] for r in rows))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, col, title, slopes in [(ax[0], "time", f"time of setup and {P_TIME} prior draws (s)", (1, 3)),
                                  (ax[1], "storage_mb", "storage of the sampler (MB)", (1, 2))]:
        series = {}
        for method in STYLE:
            ns, (mu, sd) = mean_sd(rows, "n", col, method=method)
            pad = len(sizes) - len(ns)
            series[method] = (np.r_[mu, [np.nan] * pad], np.r_[sd, [np.nan] * pad])
        metric_plot(a, sizes, series, r"number of points $n$", title, slopes)
    save_figure(fig, OUT / "prior_scaling_1d.png")


def posterior_draws():
    """P_POST posterior draws at M_POST random test points given noise-free data at N_POST clustered random points (rho
    by maximum likelihood, or fixed): W2 of the draws' Gaussian to the exact posterior, KP and Cholesky samplers, for
    Matern-1/2, 3/2, 5/2; the figure shows three draws of each with the exact posterior band."""
    rows, curves = [], {}
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        X = np.sort(train_inputs(N_POST, 1, rng)[:, 0])
        xs, y = np.sort(test_inputs(M_POST, 1, rng)[:, 0]), f(X)
        for k in KS:
            m = kp.KernelPacket(X, k, FIXED.get("rho", 0.1), method=METHOD)
            _, t_train = timed(kp.mle, m, y, bounds=(1e-2, 1.0), repeat=0) if "rho" not in FIXED else (None, 0.0)
            _, beta, s2 = kp.loglike(m, y)
            b, s2 = beta[0], FIXED.get("sigma2", s2)
            exact = DenseGP(array(X), array(y - b), k, m.rho, s2)
            mean, cov = (host(a) for a in exact.predict(array(xs), cov=True))
            draws = {}
            for method, fn in [("KP", lambda: b + np.sqrt(s2) * np.asarray(kp.posterior_sample(m, xs, (y - b) / np.sqrt(s2), P_POST, seed))),
                               ("Cholesky", lambda: b + host(exact.sample(array(xs), P_POST, seed)))]:
                F, t = timed(fn)
                F = host(F)
                draws[method] = F
                rows.append(dict(seed=seed, k=k, method=method, n=N_POST, m=M_POST, draws=P_POST, rho=m.rho, sigma2=s2,
                                 train_time=t_train, test_time=t, w2=draws_wasserstein2(F, mean + b, cov),
                                 mean_error=max_error(F.mean(1), mean + b), var_error=max_error(F.var(1), np.diag(cov)),
                                 rmse=rmse(F.mean(1), f(xs))))
            if seed == 0:
                curves |= {f"rho_{k}": m.rho, f"mean_{k}": mean + b, f"var_{k}": np.diag(cov),
                           **{f"{mth}_{k}": v[:, :3] for mth, v in draws.items()}}
                curves |= dict(X=X, y=y, xs=xs, truth=f(xs))
    write_csv(rows, OUT / "metrics.csv", "posterior_draws")
    np.savez(OUT / "posterior_draws_1d.npz", **curves)
    plot_posterior_draws()


def plot_posterior_draws():
    c = np.load(OUT / "posterior_draws_1d.npz")
    fig, axes = plt.subplots(1, len(KS), figsize=(5 * len(KS), 3.8), sharey=True)
    for ax, k in zip(axes, KS):
        handles = draws_plot(ax, c["xs"], {mth: c[f"{mth}_{k}"] for mth in STYLE},
                             rf"Matérn-${k - 2}/2$, $n = {len(c['X'])}$, $\rho = {float(c[f'rho_{k}']):.3f}$",
                             c["X"], c["y"], c["truth"],
                             band=(c[f"mean_{k}"], c[f"var_{k}"], "exact posterior mean and 95% interval"))
    bottom_legend(fig, *handles)
    save_figure(fig, OUT / "posterior_draws_1d.png")


def posterior_accuracy():
    """Posterior draws (P_ACC draws at M_POST test points) against training size POST_ACC_N: 2-Wasserstein distance
    to the ground truth (KP up to 10^6, Cholesky up to N_DENSE) and to the exact posterior (up to N_DENSE)."""
    rows = []
    for n in POST_ACC_N:
        for seed in range(min(SEEDS, 3) if n <= 10**5 else 1):
            rng = np.random.default_rng(seed)
            X = np.sort(train_inputs(n, 1, rng)[:, 0])
            xs = np.sort(test_inputs(M_POST, 1, rng)[:, 0])
            y, truth = f(X) + NOISE * rng.standard_normal(n), f(xs)
            m = kp.KernelPacket(X, K, PRIOR_RHO, NOISE**2, method=METHOD)
            Fk = np.asarray(kp.posterior_sample(m, xs, y, P_ACC, rng=seed))
            w2_kp = predictive_w2(truth, Fk.mean(1), Fk.var(1))
            if n > N_DENSE:
                rows.append(dict(seed=seed, method="KP", n=n, m=M_POST, draws=P_ACC, rho=PRIOR_RHO, tau2=NOISE**2,
                                 w2=w2_kp, rmse=rmse(Fk.mean(1), truth)))
                del m, X, y, Fk
                continue
            exact = DenseGP(array(X), array(y), K, PRIOR_RHO, 1.0, NOISE**2)
            mean, cov = (host(a) for a in exact.predict(array(xs), cov=True))
            Fc = host(exact.sample(array(xs), P_ACC, seed))
            for method, F, w2_t in [("KP", Fk, w2_kp), ("Cholesky", Fc, predictive_w2(truth, Fc.mean(1), Fc.var(1)))]:
                rows.append(dict(seed=seed, method=method, n=n, m=M_POST, draws=P_ACC, rho=PRIOR_RHO, tau2=NOISE**2,
                                 w2=w2_t, w2_exact=draws_wasserstein2(F, mean, cov), rmse=rmse(F.mean(1), truth)))
    write_csv(rows, OUT / "metrics.csv", "posterior_accuracy")
    plot_posterior_accuracy()


def plot_posterior_accuracy():
    rows = read_csv(OUT / "metrics.csv", "posterior_accuracy")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, col, title in [(ax[0], "w2", r"$W_2$ to the ground truth"),
                          (ax[1], "w2_exact", r"$W_2$ to the exact posterior")]:
        ns, series = padded(rows, "n", col)
        metric_plot(a, ns, series, r"training size $n$", title)
    save_figure(fig, OUT / "posterior_accuracy_1d.png")


def posterior_sensitivity():
    """P_POST posterior draws at M_POST test points (SENS_N clustered nodes): W2 to the exact posterior across
    lengthscales POST_SENS_RHOS (tau2 = NOISE^2) and noise ratios POST_SENS_TAU2S (rho = PRIOR_RHO)."""
    rows = []
    for sweep, key, vals in [("rho", "rho", POST_SENS_RHOS), ("tau2", "tau2", POST_SENS_TAU2S)]:
        for val in vals:
            r, t2 = (val, NOISE**2) if sweep == "rho" else (PRIOR_RHO, val)
            for seed in range(SEEDS):
                rng = np.random.default_rng(seed)
                X = np.sort(train_inputs(SENS_N, 1, rng)[:, 0])
                xs = np.sort(test_inputs(M_POST, 1, rng)[:, 0])
                y = f(X) + np.sqrt(t2) * rng.standard_normal(SENS_N)
                m = kp.KernelPacket(X, K, r, t2, method=METHOD)
                exact = DenseGP(array(X), array(y), K, r, 1.0, t2)
                mean, cov = (host(a) for a in exact.predict(array(xs), cov=True))
                for method, fn in [("KP", lambda: np.asarray(kp.posterior_sample(m, xs, y, P_POST, rng=seed))),
                                   ("Cholesky", lambda: host(exact.sample(array(xs), P_POST, seed)))]:
                    F = fn()
                    rows.append(dict(seed=seed, sweep=sweep, method=method, n=SENS_N, m=M_POST, draws=P_POST,
                                     rho=r, tau2=t2, w2=draws_wasserstein2(F, mean, cov),
                                     rmse=rmse(F.mean(1), f(xs))))
    write_csv(rows, OUT / "metrics.csv", "posterior_sensitivity")
    plot_posterior_sensitivity()


def plot_posterior_sensitivity():
    rows = read_csv(OUT / "metrics.csv", "posterior_sensitivity")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, sweep, key, xlabel, title in [
        (ax[0], "rho", "rho", r"lengthscale $\rho$", rf"$W_2$ to the exact posterior ($\tau^2 = {NOISE**2:g}$)"),
        (ax[1], "tau2", "tau2", r"noise ratio $\tau^2$", rf"$W_2$ to the exact posterior ($\rho = {PRIOR_RHO:g}$)"),
    ]:
        ks, series = padded(rows, key, "w2", sweep=sweep)
        metric_plot(a, ks, series, xlabel, title)
    save_figure(fig, OUT / "posterior_sensitivity_1d.png")


def scaling_1d():
    """Posterior sampling: training time (rho by maximum likelihood, or fixed) and storage against the training size
    n = 10^2 .. 10^8 (Cholesky up to N_DENSE), on the jittered unit grid of constant conditioning."""
    rows = []
    for n in SIZES:
        for seed in range(min(SEEDS, 3) if n <= BIG else 1):
            X, y, _ = scaling_data(n, np.random.default_rng(seed))
            def train_kp():
                m = kp.KernelPacket(X, K, FIXED.get("rho", SCALING_RHO), method=METHOD)
                if "rho" not in FIXED:
                    kp.mle(m, y, bounds=(1.0, 1e3))
                return m
            m, t = timed(train_kp, repeat=0)   # C++ only: no compilation to exclude
            rows.append(dict(seed=seed, method="KP", n=n, rho=m.rho, train_time=t, storage_mb=m.nbytes / 2**20))
            if n <= N_DENSE:
                train = (lambda: DenseGP(array(X), array(y), K, FIXED["rho"])) if "rho" in FIXED else \
                    (lambda: DenseGP.mle(array(X), array(y), K, bounds=(1.0, 1e3))[0])
                exact, t = timed(train, repeat=0 if n >= 5000 else 1)
                rows.append(dict(seed=seed, method="Cholesky", n=n, rho=exact.rho, train_time=t,
                                 storage_mb=exact.nbytes / 2**20))
            del m, X, y
    write_csv(rows, OUT / "metrics.csv", "scaling_1d")
    plot_scaling_1d()


def plot_scaling_1d():
    rows = read_csv(OUT / "metrics.csv", "scaling_1d")
    sizes = list(dict.fromkeys(r["n"] for r in rows))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, col, title, slopes in [(ax[0], "train_time", r"training time: maximum likelihood of $\rho$ (s)", (1, 3)),
                                  (ax[1], "storage_mb", "storage of the fitted model (MB)", (1, 2))]:
        series = {}
        for method in STYLE:
            ns, (mu, sd) = mean_sd(rows, "n", col, method=method)
            pad = len(sizes) - len(ns)
            series[method] = (np.r_[mu, [np.nan] * pad], np.r_[sd, [np.nan] * pad])
        metric_plot(a, sizes, series, r"training size $n$", title, slopes)
    save_figure(fig, OUT / "posterior_scaling_1d.png")


def posterior_sparse_grid():
    """P_POST posterior draws at SG_M random test points of noisy data at SG_N clustered random points in d
    dimensions: KP with the automatic sparse grid (SparseGrid.for_data; hyperparameters on the variational bound, or
    fixed) against the exact GP by Cholesky at the same hyperparameters; W2 to the exact posterior, time, storage."""
    rows = []
    for d in SG_DIMS:
        for seed in range(min(SEEDS, 3)):
            rng = np.random.default_rng(seed)
            X, xs = train_inputs(SG_N, d, rng), test_inputs(SG_M, d, rng)
            y = griewank(X) + NOISE * rng.standard_normal(SG_N)
            p = dict(rho=rho(d), sigma2=1.0, tau2=NOISE**2) | FIXED
            g = kp.SparseGrid.for_data(X, K, p["rho"], p["tau2"])
            m = kp.SparseGridRegression(array(X), array(y), g, p["sigma2"], p["tau2"])
            _, t_train = timed(m.train, repeat=0) if not FIXED else (None, 0.0)
            exact = DenseGP(array(X), array(y), K, float(g.rho[0]), m.sigma2, m.tau2)
            mean, cov = (host(a) for a in exact.predict(array(xs), cov=True))
            for method, fn, nbytes in [("KP", lambda: kp.posterior_sample(m, array(xs), None, P_POST, seed), m.nbytes),
                                       ("Cholesky", lambda: exact.sample(array(xs), P_POST, seed), exact.nbytes)]:
                F, t = timed(fn)
                F = host(F)
                rows.append(dict(seed=seed, d=d, method=method, level=g.eta, inducing=len(g.X), n=SG_N, m=SG_M,
                                 draws=P_POST, rho=float(g.rho[0]), sigma2=m.sigma2, tau2=m.tau2, train_time=t_train,
                                 test_time=t, storage_mb=nbytes / 2**20, w2=draws_wasserstein2(F, mean, cov),
                                 rmse=rmse(F.mean(1), griewank(xs))))
    write_csv(rows, OUT / "metrics.csv", "posterior_sparse_grid")


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
        fn = globals().get(f"plot_{part}" if replot else part)
        if fn is None:
            continue
        print(f"== {part}")
        fn()
