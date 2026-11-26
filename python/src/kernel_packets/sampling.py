"""Sampling: 1-D prior draws C~^{-1} L_B z (stable) or Phi L^{-T} z (original) and posterior draws by Matheron's rule
with noise ratio tau2 (C++); posterior draws from the sparse-grid inducing posterior in d dimensions."""
import numpy as np

from .backend import bk
from .kernel_packets import matern
from .regression import SparseGridRegression


def prior_sample(kp, p=1, rng=None):
    """p prior draws at the nodes of the KernelPacket kp, shape (n, p)."""
    return kp.sample_(np.random.default_rng(rng).standard_normal((kp.n, p)))


def posterior_sample(model, xnew, y=None, p=1, rng=None):
    """p posterior draws at xnew, shape (m, p).  KernelPacket model with data y at its nodes: prior draws f on the nodes
    and xnew, then f(xnew) + K(xnew, X) (K + tau2 I)^{-1} (y - f(X) - tau e) (C++, Matheron's rule).  SparseGridRegression model: mean +
    sqrt(sigma2) K(xnew, Z) L^{-T} z."""
    rng = np.random.default_rng(rng)
    if isinstance(model, SparseGridRegression):
        M = matern(xnew.reshape(xnew.shape[0], -1), model.Z, model.sg.k, model.sg.rho)
        Z = bk.like(rng.standard_normal((len(model.sg.X), p)), M)
        return model.predict(xnew)[:, None] + np.sqrt(model.sigma2) * (M @ bk.tri_solve(model.L, Z, trans=True))
    xs = np.ascontiguousarray(bk.to_numpy(xnew).ravel())
    rows = model.n + xs.size + (model.n if model.tau2 > 0 else 0)   # prior on the union, then the noise
    Z, out = bk.cols(rng.standard_normal((rows, p))), np.empty((p, xs.size))
    bk.lib.kp_posterior_sample(model._h, xs, xs.size, np.ascontiguousarray(bk.to_numpy(y).ravel()), Z.reshape(-1), p,
                               out.reshape(-1))
    return bk.like(out.T, xnew)
