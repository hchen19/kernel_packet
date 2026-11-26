"""Regression: exact in 1-D through the KP basis (C++), scattered data in d dimensions with a sparse grid as inducing
points (dense n_Z x n_Z algebra in NumPy, JAX, or PyTorch, kernel-packet solves and kernel sums in C++)."""
from contextlib import nullcontext

import numpy as np
from scipy.optimize import minimize

from .backend import bk
from .kernel_packets import matern


_SG_FACTOR_JAX = None


def _sg_factor_jax(Kzz, G, b, y, s, t, ld_zz, trq):
    global _SG_FACTOR_JAX
    jax, jnp = bk.jax()
    if _SG_FACTOR_JAX is None:
        from jax.scipy.linalg import solve_triangular

        @jax.jit
        def _fn(Kzz, G, b, y, s, t, ld_zz, trq):
            A = Kzz + (s / t) * G
            n_z = A.shape[0]
            A = A.at[jnp.diag_indices(n_z)].add(n_z * np.finfo(np.float64).eps * jnp.diagonal(A).mean())
            L = jnp.linalg.cholesky(A)
            alpha = solve_triangular(L, solve_triangular(L, b, lower=True), lower=True, trans=1) / t
            quad = (jnp.dot(y, y) - s * jnp.dot(b, alpha)) / t
            logdet = 2.0 * jnp.log(jnp.diagonal(L)).sum() - ld_zz + y.shape[0] * jnp.log(t)
            elbo = -0.5 * (quad + logdet + y.shape[0] * np.log(2.0 * np.pi)) - 0.5 * s * (y.shape[0] - trq) / t
            return L, alpha, elbo

        _SG_FACTOR_JAX = _fn
    return _SG_FACTOR_JAX(Kzz, G, b, y, s, t, ld_zz, trq)


def posterior(kp, xnew, y, var=False, cov=False):
    """Posterior mean K(xnew, X) (K + tau2 I)^{-1} y of the KernelPacket kp (zero prior mean, unit variance, noise
    ratio kp.tau2) and, if var, the variance, or, if cov, the covariance (C++)."""
    xs = np.ascontiguousarray(bk.to_numpy(xnew).ravel())
    m = xs.size
    mean, v = np.empty(m), np.empty(m) if var else None
    bk.lib.kp_posterior(kp._h, xs, m, np.ascontiguousarray(bk.to_numpy(y).ravel()), mean,
                        None if v is None else v.ctypes.data)
    out = [bk.like(mean, xnew)] + ([bk.like(v, xnew)] if var else [])
    if cov:
        C = np.empty((m, m))
        bk.lib.kp_posterior_cov(kp._h, xs, m, C.reshape(-1))
        out.append(bk.like(C.T, xnew))
    return out[0] if len(out) == 1 else tuple(out)


def predict(xnew, X, w, k, rho):
    """Posterior mean K(xnew, X) w, w = K^{-1} (y - mean), summed on the fly (no m x N matrix); on the GPU for device
    arrays."""
    if bk.on_gpu(xnew):
        torch = bk.torch()
        xs = bk.device(xnew).reshape(len(xnew), -1).contiguous()
        Xd = torch.as_tensor(bk.to_numpy(X).reshape(len(X), -1), device=xs.device) if not bk.on_gpu(X) else bk.device(X)
        W = (torch.as_tensor(bk.to_numpy(w), device=xs.device) if not bk.on_gpu(w) else bk.device(w)).reshape(len(Xd), -1).T.contiguous()
        out = xs.new_empty((W.shape[0], len(xs)))
        rho = np.ascontiguousarray(np.broadcast_to(np.asarray(rho, np.float64), xs.shape[1]))
        bk.lib.kernel_predict_cuda(xs.data_ptr(), len(xs), Xd.contiguous().data_ptr(), len(Xd), xs.shape[1], W.data_ptr(),
                                   W.shape[0], k, rho, out.data_ptr(), bk.stream())
        return bk.back(out.T.reshape((len(xs),) + tuple(np.shape(w)[1:])), xnew)
    xs = np.ascontiguousarray(bk.to_numpy(xnew).reshape(len(xnew), -1))
    Xh = np.ascontiguousarray(bk.to_numpy(X).reshape(len(X), -1))
    W = bk.cols(w)
    out = np.empty((len(W), len(xs)))
    rho = np.ascontiguousarray(np.broadcast_to(np.asarray(rho, np.float64), xs.shape[1]))
    bk.lib.kernel_predict(xs, len(xs), Xh, len(Xh), xs.shape[1], W.reshape(-1), len(W), k, rho, out.reshape(-1))
    return bk.like(out.T.reshape((len(xs),) + np.shape(w)[1:]), xnew)


class SparseGridRegression:
    """Scattered data (X, y) in d dimensions with the points Z of a SparseGrid as inducing points (Titsias VFE), kernel
    sigma2 * matern and noise tau2.  With K = matern and Sigma = K_ZZ + sigma2 K_ZX K_XZ / tau2 = L L^T:
        mean(x) = sigma2 K(x, Z) alpha,  alpha = Sigma^{-1} K_ZX y / tau2,
        var(x) = sigma2 (1 - K(x, Z) K_ZZ^{-1} K(Z, x) + |L^{-1} K(Z, x)|^2).
    K_ZZ^{-1} and log|K_ZZ| come from kernel packets (C++); K_ZX K_XZ, K_ZX y and the Cholesky factor of Sigma are dense
    BLAS work in NumPy, JAX, or PyTorch on the data's device, accumulated in chunks (memory of n_Z^2, not n n_Z)."""

    def __init__(self, X, y, sg, sigma2=1.0, tau2=1e-4):
        self.X, self.y, self.sg, self.sigma2, self.tau2 = X.reshape(X.shape[0], -1), y.reshape(-1), sg, sigma2, tau2
        self.Z = bk.like(sg.X, X)
        self.fit()

    def fit(self):
        """Factor Sigma and the variational lower bound `elbo` of the log marginal likelihood at the current parameters."""
        X, y, sg, s, t = self.X, self.y, self.sg, self.sigma2, self.tau2
        n = len(y)
        key = tuple(sg.rho)
        use_jax = bk.is_jax(X)
        with (bk.torch().no_grad() if bk.is_torch(X) else nullcontext()):
            if getattr(self, "_key", None) != key:   # G, b, K_ZZ and the trace depend on rho only: reused across sigma2, tau2
                G = b = None
                c = bk.batch(len(sg.X))
                for i in range(0, n, c):
                    Kc = matern(self.Z, X[i:i + c], sg.k, sg.rho)
                    Gi, bi = Kc @ Kc.T, Kc @ y[i:i + c]
                    if G is None:
                        G, b = Gi, bi
                    elif use_jax:
                        G, b = G + Gi, b + bi
                    else:
                        G += Gi
                        b += bi
                # tr(K_XZ K_ZZ^{-1} K_ZX) = tr(K_ZZ^{-1} G): n_Z columns, not n (C++)
                self._key, self._G, self._b = key, G, b
                self._Kzz, self._trq, self._ld_zz = matern(self.Z, self.Z, sg.k, sg.rho), sg.trace_solve(G), sg.logdet
            G, b, trq = self._G, self._b, self._trq
            if use_jax:
                self.L, self.alpha, elbo = _sg_factor_jax(self._Kzz, G, b, y, s, t, self._ld_zz, trq)
                self.elbo = float(elbo)
                return self
            A = self._Kzz + (s / t) * G
            # A is positive definite but cond(A) can exceed 1 / eps: a diagonal of n_Z eps times its mean keeps the
            # factor defined in every BLAS (CPU or GPU) at a relative change of ~1e-13
            ns = bk.xp(A)
            bk.add_diag(A, A.shape[0] * np.finfo(np.float64).eps * float(ns.diagonal(A).mean()))
            self.L = bk.cholesky(A)
            self.alpha = bk.tri_solve(self.L, bk.tri_solve(self.L, b), trans=True) / t
            quad = (float(y @ y) - s * float(b @ self.alpha)) / t
            logdet = 2 * float(ns.log(ns.diagonal(self.L)).sum()) - self._ld_zz + n * np.log(t)
            self.elbo = -0.5 * (quad + logdet + n * np.log(2 * np.pi)) - 0.5 * s * (n - trq) / t
        return self

    @property
    def nbytes(self):
        """Bytes held: the sparse grid, the n_Z x n_Z factor and the cached terms of training."""
        arrays = [self.L, self.alpha, getattr(self, "_G", None), getattr(self, "_Kzz", None), self.Z]
        return self.sg.nbytes + sum(int(np.prod(a.shape)) * 8 for a in arrays if a is not None)

    def predict(self, xnew, var=False, cov=False):
        """Posterior mean (C++ kernel sum) and variance, or covariance if cov, at the rows of xnew."""
        xnew = xnew.reshape(xnew.shape[0], -1)
        mean = self.sigma2 * predict(xnew, self.Z, self.alpha, self.sg.k, self.sg.rho)
        if not (var or cov):
            return mean
        M = matern(xnew, self.Z, self.sg.k, self.sg.rho)
        S, V = self.sg.solve(M.T), bk.tri_solve(self.L, M.T)    # K_ZZ^{-1} K(Z, x), L^{-1} K(Z, x)
        if cov:
            return mean, self.sigma2 * (matern(xnew, xnew, self.sg.k, self.sg.rho) - M @ S + V.T @ V)
        return mean, self.sigma2 * (1 - (M * S.T).sum(1) + (V ** 2).sum(0))

    def train(self, bounds=((1e-2, 1e2), (1e-4, 1e4), (1e-8, 1e2))):
        """Maximize the bound over (rho, sigma2, tau2) in bounds (Nelder-Mead on their logarithms: the bound is noisy at
        small tau2, finite-difference gradients are not reliable); rho is shared by the axes."""
        def nelbo(z):
            self.sg.rho, self.sigma2, self.tau2 = np.exp(z)
            return -self.fit().elbo

        z0 = np.log([self.sg.rho[0], self.sigma2, self.tau2])
        z = minimize(nelbo, z0, method="Nelder-Mead", bounds=np.log(bounds), options=dict(xatol=1e-3, fatol=1e-3)).x
        nelbo(z)
        return self
