"""Exact Gaussian process by dense Cholesky factorization: the reference of every experiment, O(n^3) time, O(n^2)
memory, in the framework and on the device of its data (NumPy arrays, JAX arrays via XLA, or PyTorch tensors via MKL /
cuSOLVER)."""
from contextlib import nullcontext
from functools import partial

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from kernel_packets import matern
from kernel_packets.backend import bk


_JAX_FNS = None


def _jax_fns():
    global _JAX_FNS
    jax, jnp = bk.jax()
    if _JAX_FNS is None:
        from jax.scipy.linalg import solve_triangular

        @partial(jax.jit, static_argnames="k")
        def _fit_jax(X, y, k, rho, sigma2, tau2):
            K = sigma2 * matern(X, X, k, rho)
            L = jnp.linalg.cholesky(K.at[jnp.diag_indices(X.shape[0])].add(tau2))
            alpha = solve_triangular(L, solve_triangular(L, y, lower=True), lower=True, trans=1)
            ll = -0.5 * (jnp.dot(y, alpha) + 2.0 * jnp.log(jnp.diagonal(L)).sum() + X.shape[0] * np.log(2.0 * np.pi))
            return L, alpha, ll

        @partial(jax.jit, static_argnames="k")
        def _profile_jax(X, y, ones, k, rho):
            L = jnp.linalg.cholesky(matern(X, X, k, rho))
            a = solve_triangular(L, ones, lower=True)
            c = solve_triangular(L, y, lower=True)
            beta = jnp.dot(a, c) / jnp.dot(a, a)
            r = c - beta * a
            s2 = jnp.dot(r, r) / X.shape[0]
            nll = 0.5 * (X.shape[0] * jnp.log(s2) + 2.0 * jnp.log(jnp.diagonal(L)).sum() + X.shape[0])
            return L, beta, s2, nll

        @partial(jax.jit, static_argnames="k")
        def _predict_batch_jax(Xb, X, L, alpha, k, rho, sigma2):
            Ks = sigma2 * matern(Xb, X, k, rho)
            V = solve_triangular(L, Ks.T, lower=True)
            return Ks @ alpha, sigma2 - (V ** 2).sum(0)

        _JAX_FNS = (_fit_jax, _profile_jax, _predict_batch_jax)
    return _JAX_FNS


class DenseGP:
    """Kernel sigma2 * matern (k, rho), noise tau2, data (X, y)."""

    def __init__(self, X, y, k=5, rho=1.0, sigma2=1.0, tau2=0.0):
        self.X, self.y, self.k = X, y, k
        self.fit(rho, sigma2, tau2)

    def fit(self, rho, sigma2, tau2):
        """Factor K + tau2 I and the log marginal likelihood `loglike` at the given parameters."""
        self.rho, self.sigma2, self.tau2 = rho, sigma2, tau2
        if bk.is_jax(self.X):
            _fit_jax, _, _ = _jax_fns()
            self.L, self.alpha, ll = _fit_jax(self.X, self.y, self.k, float(rho), float(sigma2), float(tau2))
            self.loglike = float(ll)
            return self
        with (bk.torch().no_grad() if bk.is_torch(self.X) else nullcontext()):
            K = sigma2 * matern(self.X, self.X, self.k, rho)
            if tau2:
                bk.add_diag(K, tau2)
            self.L = bk.cholesky(K)
            self.alpha = bk.tri_solve(self.L, bk.tri_solve(self.L, self.y), trans=True)
            ns = bk.xp(K)
            self.loglike = -0.5 * (float(self.y @ self.alpha) + 2 * float(ns.log(ns.diagonal(self.L)).sum())
                                   + len(self.y) * np.log(2 * np.pi))
        return self

    def train(self, bounds=((1e-2, 1e2), (1e-4, 1e4), (1e-8, 1e2))):
        """Exact maximum likelihood over (rho, sigma2, tau2) in bounds (Nelder-Mead on their logarithms, as the
        sparse-grid model's training)."""
        def nll(z):  # +inf where K + tau2 I is numerically singular (NaN in JAX, exception in torch/NumPy)
            try:
                out = -self.fit(*np.exp(z)).loglike
            except (RuntimeError, np.linalg.LinAlgError):
                return np.inf
            return out if np.isfinite(out) else np.inf

        z0 = np.log([self.rho, self.sigma2, max(self.tau2, 1e-6)])
        z = minimize(nll, z0, method="Nelder-Mead", bounds=np.log(bounds), options=dict(xatol=1e-3, fatol=1e-3)).x
        return self.fit(*np.exp(z))

    @property
    def nbytes(self):
        """Bytes held: the n x n Cholesky factor, the data and the weights."""
        return 8 * (int(np.prod(self.L.shape)) + int(np.prod(self.X.shape)) + len(self.alpha))

    @staticmethod
    def mle(X, y, k, bounds=(1e-3, 1.0)):
        """Maximum likelihood of rho (Brent), the constant mean and sigma2 profiled out, no noise (as kernel_packets.mle):
        the DenseGP at the optimum, its mean beta."""
        n = len(y)
        ones = bk.like(np.ones(n), y)
        ns = bk.xp(y)

        def profile(rho):
            if bk.is_jax(X):
                _, _profile_jax, _ = _jax_fns()
                _, beta, s2, out = _profile_jax(X, y, ones, k, float(rho))
                return float(beta), float(s2), float(out)
            with (bk.torch().no_grad() if bk.is_torch(X) else nullcontext()):
                L = bk.cholesky(matern(X, X, k, rho))
                a, c = bk.tri_solve(L, ones), bk.tri_solve(L, y)
                beta = float(a @ c) / float(a @ a)
                r = c - beta * a
                s2 = float(r @ r) / n
                return beta, s2, 0.5 * (n * np.log(s2) + 2 * float(ns.log(ns.diagonal(L)).sum()) + n)

        def nll(log_rho):
            try:
                _, _, out = profile(np.exp(log_rho))
            except (RuntimeError, np.linalg.LinAlgError):
                return np.inf
            return out if np.isfinite(out) else np.inf

        rho = float(np.exp(minimize_scalar(nll, bounds=np.log(bounds), method="bounded", options=dict(xatol=1e-3)).x))
        beta, s2, _ = profile(rho)
        return DenseGP(X, y - beta, k, rho, s2), beta

    def predict(self, xnew, cov=False):
        """Posterior mean and variance (or covariance) of f at xnew; the variance in batches of test points (memory of
        2^24 numbers, not m n)."""
        use_torch = bk.is_torch(self.L)
        with (bk.torch().no_grad() if use_torch else nullcontext()):
            if cov:
                Ks = self.sigma2 * matern(xnew, self.X, self.k, self.rho)
                V = bk.tri_solve(self.L, Ks.T)
                return Ks @ self.alpha, self.sigma2 * matern(xnew, xnew, self.k, self.rho) - V.T @ V
            bs = bk.batch(len(self.X))
            use_jax = bk.is_jax(self.L)
            if use_jax:
                _, _, _predict_batch_jax = _jax_fns()
            means, variances = [], []
            for i in range(0, len(xnew), bs):
                Xb = xnew[i:i + bs]
                if use_jax:
                    mu, var = _predict_batch_jax(Xb, self.X, self.L, self.alpha, self.k, float(self.rho), float(self.sigma2))
                else:
                    Ks = self.sigma2 * matern(Xb, self.X, self.k, self.rho)
                    mu = Ks @ self.alpha
                    var = self.sigma2 - (bk.tri_solve(self.L, Ks.T) ** 2).sum(0)
                if len(xnew) <= bs:
                    return mu, var
                means.append(mu)
                variances.append(var)
            cat = bk.torch().cat if use_torch else bk.xp(self.L).concatenate
            return cat(means), cat(variances)

    def sample(self, xnew, p=1, rng=None):
        """p posterior draws at xnew, shape (m, p)."""
        with (bk.torch().no_grad() if bk.is_torch(self.L) else nullcontext()):
            mean, C = self.predict(xnew, cov=True)
            Z = bk.like(np.random.default_rng(rng).standard_normal((len(mean), p)), C)
            return mean[:, None] + bk.cholesky(bk.add_diag(C, 1e-12)) @ Z

    @staticmethod
    def prior_sample(X, k, rho, p=1, rng=None):
        """p prior draws at X by the Cholesky factor of K."""
        with (bk.torch().no_grad() if bk.is_torch(X) else nullcontext()):
            K = matern(X, X, k, rho)
            Z = bk.like(np.random.default_rng(rng).standard_normal((len(X), p)), K)
            try:
                L = bk.cholesky(K)
            except (RuntimeError, np.linalg.LinAlgError):
                return bk.like(np.full((len(X), p), np.nan), K)
            return L @ Z
