"""Matern-(k-2)/2 kernel matrices (NumPy, JAX, or PyTorch, for the dense algebra) and kernel packets on sorted 1-D nodes (C++
core).  method "stable" (default, k <= 7, noise ratio tau2): the stable KPs: the factored one-sided
whitening K + tau2 I = C~^{-1} G C~^{-T} on dense nodes (c h <= 1), else the two-sided KPs of Phi + tau2 A, the Green-
form mean and selected-inverse variance, the likelihood by Givens rotations.  method "original" (noise-free): K^{-1} =
A Phi^{-1}, log|K| = log|Phi| - log|A|, K = Phi B^{-1} Phi^T with B = A^T Phi banded."""
import numpy as np

from .backend import bk


def _poly(k):
    """Coefficients of P in K(r) = P(z) e^{-z}, z = sqrt(k-2) |r| / rho, lowest degree first."""
    c, p = [1.0], (k - 3) // 2
    for j in range(p):
        c.append(c[-1] * 2 * (p - j) / ((2 * p - j) * (j + 1)))
    return c


def _matern(X1, X2, k, c, xp):
    """Kernel matrix accumulated one dimension at a time (memory of one (m, n) matrix); c = sqrt(k-2) / rho."""
    X1, X2 = X1.reshape(X1.shape[0], -1), X2.reshape(X2.shape[0], -1)
    P, S = 1.0, 0.0
    for i in range(X1.shape[1]):
        Z = xp.abs(X1[:, i, None] - X2[None, :, i]) * c[i]
        Pi = 0.0
        for a in _poly(k)[::-1]:
            Pi = Pi * Z + a
        P, S = P * Pi, S + Z
    return P * xp.exp(-S)


_MATERN_JAX = None


def _matern_jax(X1, X2, k, c):
    global _MATERN_JAX
    jax, jnp = bk.jax()
    if _MATERN_JAX is None:
        _MATERN_JAX = jax.jit(lambda a, b, kk, cc: _matern(a, b, kk, cc, jnp), static_argnums=2)
    return _MATERN_JAX(X1, X2, k, c)


def scale(k, rho, d):
    """c = sqrt(k-2) / rho for each of d dimensions."""
    return np.broadcast_to(np.sqrt(k - 2.0) / np.asarray(rho, np.float64), d)


def matern(X1, X2, k, rho):
    """Kernel matrix between the rows of X1 (m, d) and X2 (n, d) (or 1-D points) with lengthscales rho."""
    if bk.is_torch(X1):
        torch = bk.torch()
        d = X1.reshape(X1.shape[0], -1).shape[1]
        return _matern(X1, X2, k, torch.as_tensor(scale(k, rho, d).copy(), dtype=X1.dtype, device=X1.device), torch)
    if bk.is_jax(X1):
        _, jnp = bk.jax()
        d = X1.reshape(X1.shape[0], -1).shape[1]
        return _matern_jax(X1, X2, k, jnp.broadcast_to(jnp.sqrt(k - 2.0) / jnp.asarray(rho, X1.dtype), (d,)))
    X1, X2 = np.asarray(X1, np.float64), np.asarray(X2, np.float64)
    d = X1.reshape(X1.shape[0], -1).shape[1]
    return _matern(X1, X2, k, scale(k, rho, d), np)


METHODS = {"stable": 0, "original": 1}


class KernelPacket:
    """Kernel packets of the Matern-(k-2)/2 kernel with lengthscale rho on sorted nodes x, noise ratio tau2 (stable);
    method "stable" (k <= 7 and n >= k, else the original is taken) or "original"."""

    def __init__(self, x, k=5, rho=None, tau2=0.0, method="stable"):
        self.x = np.ascontiguousarray(bk.to_numpy(x).ravel())
        self.n, self.k, self._tau2, self.method, self._h = self.x.size, k, float(tau2), method, None
        self.rho = np.sqrt(k - 2.0) if rho is None else rho

    @property
    def tau2(self):
        return self._tau2

    @tau2.setter
    def tau2(self, tau2):
        self._tau2 = float(tau2)
        if self._h is not None:
            bk.lib.kp_set_params(self._h, self._rho, self._tau2)

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, rho):
        self._rho = float(rho)
        if self._h is None:
            self._h = bk.lib.kp_new(self.x, self.n, self.k, self._rho, self._tau2, METHODS[self.method])
            self.method = list(METHODS)[bk.lib.kp_method(self._h)]
        else:
            bk.lib.kp_set_params(self._h, self._rho, self._tau2)

    def set_params(self, rho, tau2):
        self._rho, self._tau2 = float(rho), float(tau2)
        bk.lib.kp_set_params(self._h, self._rho, self._tau2)

    def __del__(self):
        if self._h:
            bk.lib.kp_free(self._h)

    @property
    def logdet(self):
        return bk.lib.kp_logdet(self._h)

    @property
    def nbytes(self):
        """Bytes held by the factorization (C++)."""
        return bk.lib.kp_bytes(self._h)

    def solve(self, Y, a=True):
        """(K + tau2 I)^{-1} Y (original, not a: Phi^{-1} Y) for Y of shape (n,) or (n, p)."""
        Z = bk.cols(Y)
        bk.lib.kp_solve(self._h, Z.reshape(-1), len(Z), int(a))
        return bk.like(Z.T.reshape(np.shape(Y)), Y)

    def sample_(self, Z):
        """N(0, K) from N(0, I) for Z of shape (n,) or (n, p): C~^{-1} L_B Z (stable), Phi L^{-T} Z (original)."""
        W = bk.cols(Z)
        bk.lib.kp_sample(self._h, W.reshape(-1), len(W))
        return bk.like(W.T.reshape(np.shape(Z)), Z)

    def basis(self, xnew):
        """Original KPs at xnew: phi_{start[j] + t}(xnew[j]) = V[j, t]."""
        xnew = np.ascontiguousarray(bk.to_numpy(xnew).ravel())
        start, V = np.empty(xnew.size, np.int64), np.empty((xnew.size, bk.lib.kp_width(self._h)))
        bk.lib.kp_basis(self._h, xnew, xnew.size, start, V)
        return start, V
