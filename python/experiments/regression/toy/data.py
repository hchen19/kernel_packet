"""Toy regression data: standard test functions of GP regression (Surjanovic & Bingham, Virtual Library
of Simulation Experiments, https://www.sfu.ca/~ssurjano/).
  1-D: the Gramacy & Lee (2012) function sin(10 pi t) / (2 t) + (t - 1)^4, t = 0.5 + 2x on x in [0, 1]
       (https://www.sfu.ca/~ssurjano/grlee12.html).
  d-D: the Genz (1984) oscillatory function cos(2 pi w + sum_i a_i x_i) on [0, 1]^d, a_i = 9 / i^2, w = 0.5
       (https://www.sfu.ca/~ssurjano/oscil.html).
Training and test inputs are random and drawn from different distributions: training from two Beta clusters (a sparse
middle region where the posterior widens), test from Beta(0.7, 0.7) (heavier near the ends); the only structured
points are the sparse grids, as inducing points."""
import numpy as np

PARTS = ["posterior_1d", "accuracy_1d", "scaling_1d", "sparse_grid", "scaling_nd"]
K = 5                                       # Matern-3/2
KS = (3, 5, 7)                              # Matern-1/2, 3/2, 5/2 of the posterior figure
SEEDS = 5                                   # random replicates of every metric (mean and sd)
M = 1000                                    # test points
N_1D = 60                                   # training size of the posterior figure
N_ACC = [100, 500, 10**3, 5 * 10**3, 10**4, 5 * 10**4, 10**5, 5 * 10**5, 10**6, 5 * 10**6, 10**7, 5 * 10**7, 10**8]
M_W2 = 300                                  # test points of the W2 to the exact posterior (a dense m x m covariance)
SIZES = N_ACC  # training sizes of the time and storage figure
BIG = 10**6                                 # above: one replicate (minutes per size)
SCALING_RHO, N_DENSE = 20.0, 10**4          # lengthscale of the scaling design; largest n of Cholesky
NOISE = 0.05                                # noise sd of the 1-D accuracy data and the d-dimensional data
SG_N, SG_DIMS = 2000, (2, 4)                # training points; dimensions
SG_LEVELS = {2: range(2, 7), 4: range(1, 6)}   # inducing levels eta - d per dimension
INIT = dict(rho=0.5, sigma2=1.0, tau2=NOISE**2)       # starting values of the training
ND_SIZES = [100, 500, 10**3, 5 * 10**3, 10**4, 5 * 10**4, 10**5, 5 * 10**5, 10**6]   # training sizes in d dimensions


def f(x):
    t = 0.5 + 2 * x
    return np.sin(10 * np.pi * t) / (2 * t) + (t - 1) ** 4


def train_inputs(n, d, rng):
    """Two clusters on each axis: Beta(5, 15) and Beta(15, 5) (means 0.25, 0.75, sd 0.09), inside (0, 1) (no clipping,
    so no repeated inputs)."""
    a = rng.choice([5.0, 15.0], (n, d))
    return rng.beta(a, 20.0 - a)


def test_inputs(m, d, rng):
    return rng.beta(0.7, 0.7, (m, d))


def data_1d(n, rng, noise=0.0):
    """Training data (sorted, noise sd `noise`) and the noise-free truth at the test inputs, in [0, 1]."""
    X, xs = np.sort(train_inputs(n, 1, rng)[:, 0]), test_inputs(M, 1, rng)[:, 0]
    return X, f(X) + noise * rng.standard_normal(n), xs, f(xs)


def scaling_design(n, rng, m=M):
    """Unit grid with jitter, y = sin(x / 50) (lengthscale 20: the conditioning does not change with n); m test points
    random in [0, n - 1]."""
    X = np.arange(n) + rng.uniform(-0.3, 0.3, n)
    xs = rng.uniform(0, n - 1, m)
    return X, np.sin(X / 50.0), xs, np.sin(xs / 50.0)


def f_nd(X):
    """Genz oscillatory function on [0, 1]^d."""
    return np.cos(np.pi + X @ (9.0 / np.arange(1, X.shape[1] + 1) ** 2))


def data_nd(d, n, rng):
    """Noisy training data (noise sd NOISE) and noise-free test values."""
    X, xs = train_inputs(n, d, rng), test_inputs(M, d, rng)
    return X, f_nd(X) + NOISE * rng.standard_normal(n), xs, f_nd(xs)
