"""Synthetic designs for prior and posterior sampling: data from the Griewank function (Griewank 1981,
https://www.sfu.ca/~ssurjano/griewank.html) on [-5, 5]^d, rescaled to [0, 1]^d, with noise s.d. NOISE in d
dimensions. Inputs are random: prior nodes from Beta(0.7, 0.7), training inputs from two clusters (a sparse middle
region), test inputs from Beta(0.7, 0.7); the only structured points are the sparse grids, as inducing points."""
import numpy as np

PARTS = ["prior_draws", "prior_accuracy", "prior_sensitivity", "prior_scaling", "posterior_draws",
         "posterior_accuracy", "posterior_sensitivity", "scaling_1d", "posterior_sparse_grid"]
K = 5                                       # Matern-3/2
SEEDS = 5                                   # random replicates (mean and sd)
NOISE = 0.1
KS = (3, 5, 7)                              # Matern-1/2, 3/2, 5/2 of the draws figures
PRIOR_RHO = 0.2                             # lengthscale of the prior draws
PRIOR_ACC_N = [50, 100, 200, 500, 1000]     # sizes for prior W2 accuracy against the exact K
POST_ACC_N = [100, 500, 10**3, 5 * 10**3, 10**4, 5 * 10**4, 10**5, 5 * 10**5, 10**6]
P_ACC = 512                                 # posterior draws per size for posterior_accuracy
SENS_N, SENS_KS = 100, (3, 5, 7)            # nodes and kernels for sensitivity sweeps
SENS_RHOS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
POST_SENS_RHOS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
POST_SENS_TAU2S = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
SIZES = [100, 500, 10**3, 5 * 10**3, 10**4, 5 * 10**4, 10**5, 5 * 10**5, 10**6, 5 * 10**6, 10**7, 5 * 10**7, 10**8]
BIG = 10**6                                 # above: one replicate of the timing (minutes per size)
SCALING_RHO, P_TIME, N_DENSE = 20.0, 16, 10**4   # draws per timing; largest n of Cholesky
N_POST, M_POST, P_POST = 20, 200, 4096      # posterior draws: training points, test points, draws
SG_DIMS, SG_N, SG_M = (2, 4, 8), 2**10, 200 # posterior with sparse-grid inducing points


def griewank(X):
    """sum z_i^2 / 4000 - prod cos(z_i / sqrt(i)) + 1, z = 10 x - 5, X (n, d) in [0, 1]^d."""
    Z = 10 * np.asarray(X) - 5
    return (Z ** 2).sum(1) / 4000 - np.cos(Z / np.sqrt(np.arange(1, Z.shape[1] + 1))).prod(1) + 1


def f(x):
    return griewank(np.asarray(x)[:, None])


def nodes(n, rng):
    """Sorted random nodes on [0, 1] (Beta(0.7, 0.7))."""
    return np.sort(rng.beta(0.7, 0.7, n))


def scaling_design(n, rng):
    """Unit grid with jitter (lengthscale 20: the conditioning does not change with n)."""
    return np.arange(n) + rng.uniform(-0.3, 0.3, n)


def scaling_data(n, rng, m=200):
    """The jittered unit grid with y = sin(x / 50), and m test points random in [0, n - 1]."""
    X = scaling_design(n, rng)
    return X, np.sin(X / 50.0), np.sort(rng.uniform(0, n - 1, m))


def train_inputs(n, d, rng):
    """Two clusters on each axis: Beta(5, 15) and Beta(15, 5) (means 0.25, 0.75, sd 0.09), inside (0, 1) (no clipping,
    so no repeated inputs)."""
    a = rng.choice([5.0, 15.0], (n, d))
    return rng.beta(a, 20.0 - a)


def test_inputs(m, d, rng):
    return rng.beta(0.7, 0.7, (m, d))


def rho(d):
    """Lengthscale in d dimensions: the product kernel decorrelates like rho^-d, so rho grows with d."""
    return 0.25 * np.sqrt(d)
