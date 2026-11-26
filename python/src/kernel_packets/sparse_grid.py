"""Sparse grids from nested 1-D designs, the inducing points of d-dimensional regression (C++ core): design, K^{-1} by
the combination technique over tensor grids of 1-D kernel packets, log|K|, tr(K^{-1} G)."""
import numpy as np

from .backend import bk


def hc(j, lo=0.0, hi=1.0):
    """Hyperbolic cross (bisection) design of level j: 2^j - 1 points in (lo, hi), nested in j."""
    return lo + (hi - lo) * np.arange(1, 2**j) / 2**j


_PL = np.array([[.5, .5], [.125, .875], [.25, .75], [0, 1], [.375, .625], [.25 - 1 / 16, .75 + 1 / 16],
                [1 - 1 / 16, 1 / 16], [.5 - 1 / 16, .5 + 1 / 16], [.3125, .6875], [.25 + 1 / 32, .75 - 1 / 32],
                [1 / 32, 1 - 1 / 32], [.875 + 1 / 32, .125 - 1 / 32], [.875 - 1 / 32, .125 + 1 / 32],
                [.375 + 1 / 32, .625 - 1 / 32], [.5 - 1 / 32, .5 + 1 / 32], [.25 - 1 / 32, .75 + 1 / 32]])


def pl(j, lo=0.0, hi=1.0):
    """Plumlee's design of level j <= 16, nested in j."""
    return lo + np.unique(_PL[:j]) * (hi - lo)


class SparseGrid:
    """Sparse grid of level eta >= d in [lo, hi]^d: K^{-1} is the sum over |j| = max(d, eta-d+1) .. eta of
    (-1)^(eta-|j|) C(d-1, eta-|j|) times the tensor-grid K^{-1} of level j; points X with sorted rows."""

    def __init__(self, d, eta, k=5, rho=None, design=hc, lo=0.0, hi=1.0):
        self.d, self.eta, self.k, self._hd = d, eta, k, None
        nodes = [np.asarray(design(j, lo, hi), np.float64) for j in range(eta - d + 2)]
        self._rho = np.broadcast_to(np.sqrt(k - 2.0) if rho is None else rho, d).astype(np.float64)
        self._h = bk.lib.sg_new(d, eta, np.concatenate(nodes), np.cumsum([0] + [len(x) for x in nodes]), k, self._rho)
        self.X = np.empty((bk.lib.sg_size(self._h), d))
        bk.lib.sg_points(self._h, self.X.reshape(-1))

    @classmethod
    def for_data(cls, X, k=5, rho=None, tau2=1e-2, n_max=4000, kl=1.0, design=hc, lo=0.0, hi=1.0):
        """The sparse grid of the smallest level eta with tr(K_XX - Q_XX) / tau2 <= kl for the data X (n, d) (the
        expected KL of the variational posterior), within n_max points.
        tr(K_XX - Q_XX) = sum_i (1 - sum over the grid's levels k of prod_j (v_{j,k_j-1} - v_{j,k_j})(x_ij)), v_{j,l}
        the 1-D KP posterior variance of axis j with the level-l nodes (C++), summed over the levels by a recursion
        over the axes."""
        from .kernel_packets import KernelPacket
        from .regression import posterior
        X = bk.to_numpy(X).reshape(len(X), -1)
        n, d = X.shape
        n_max = min(n_max, n)
        rho = np.broadcast_to(np.sqrt(k - 2.0) if rho is None else rho, d).astype(np.float64)
        L = int(np.log2(n_max + 1))
        sizes = np.array([len(design(l, lo, hi)) for l in range(L + 1)])
        D = np.empty((d, L, n))                                  # D[j, l - 1] = v_{j,l-1} - v_{j,l} at X[:, j]
        for j in range(d):
            v = [np.ones(n)] + [np.asarray(posterior(KernelPacket(design(l, lo, hi), k, rho[j]), X[:, j],
                                                     np.zeros(sizes[l]), var=True)[1]) for l in range(1, L + 1)]
            D[j] = -np.diff(v, axis=0)
        dn = np.diff(sizes).astype(float)                        # new points of level l
        S, C = np.ones((1, n)), np.ones(1)                       # S[s]: sum over k_1..k_j with sum s of prod D
        for j in range(d):
            S2, C2 = np.zeros((len(S) + L, n)), np.zeros(len(C) + L)
            for l in range(1, L + 1):
                S2[l:l + len(S)] += D[j, l - 1] * S
                C2[l:l + len(C)] += dn[l - 1] * C
            S, C = S2, C2
        q, N = np.cumsum(S.sum(1)), np.cumsum(C)                 # at eta = index: sum over |k| <= eta, and n_Z
        etas = [e for e in range(d, len(N)) if N[e] <= n_max]
        eta = next((e for e in etas if (n - q[e]) / tau2 <= kl), etas[-1])
        return cls(d, eta, k, rho, design, lo, hi)

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, rho):
        self._rho = np.broadcast_to(rho, self.d).astype(np.float64)
        bk.lib.sg_set_rho(self._h, self._rho)
        self._free_device()

    def _free_device(self):
        if getattr(self, "_hd", None):
            bk.lib.sg_cuda_free(self._hd)
        self._hd = None

    def _device(self):
        """The device copy of the 1-D factors (made on the first GPU call after a change of rho)."""
        if self._hd is None:
            self._hd = bk.lib.sg_cuda_new(self._h)
        return self._hd

    def __del__(self):
        self._free_device()
        if getattr(self, "_h", None):
            bk.lib.sg_free(self._h)

    def solve(self, Y):
        """K^{-1} Y for Y of shape (N,) or (N, p); on the GPU for device arrays."""
        if bk.on_gpu(Y):
            T = bk.device(Y).reshape(len(self.X), -1).T.contiguous()
            W = T.new_empty(T.shape)
            bk.lib.sg_cuda_solve(self._device(), T.data_ptr(), W.data_ptr(), T.shape[0], bk.stream())
            return bk.back(W.T.reshape(Y.shape), Y)
        Z = bk.cols(Y)
        W = np.empty_like(Z)
        bk.lib.sg_solve(self._h, Z.reshape(-1), W.reshape(-1), len(Z))
        return bk.like(W.T.reshape(np.shape(Y)), Y)

    @property
    def logdet(self):
        return bk.lib.sg_logdet(self._h)

    @property
    def nbytes(self):
        """Bytes held by the design and the 1-D factors (C++)."""
        return bk.lib.sg_bytes(self._h)

    def trace_solve(self, G):
        """tr(K^{-1} G) for an N x N matrix G (C++, O(N) memory per thread; on the GPU for a device array)."""
        if bk.on_gpu(G):
            T = bk.device(G).T.contiguous()
            return bk.lib.sg_cuda_trace_solve(self._device(), T.data_ptr(), bk.stream())
        return bk.lib.sg_trace_solve(self._h, bk.cols(G).reshape(-1))
