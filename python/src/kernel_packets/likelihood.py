"""Profile likelihood of the lengthscale (C++) and its maximization, for KernelPacket and SparseGrid models."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .backend import bk
from .sparse_grid import SparseGrid


def loglike(model, y, F=None):
    """Log-likelihood at model.rho with mean F beta and variance sigma2 at their MLEs: (loglike, beta, sigma2)."""
    y = np.ascontiguousarray(bk.to_numpy(y).ravel())
    F = bk.cols(np.ones((y.size, 1)) if F is None else F)
    beta, sigma2 = np.empty(len(F)), np.empty(1)
    L = (bk.lib.sg_loglike if isinstance(model, SparseGrid) else bk.lib.kp_loglike)(model._h, y, F.reshape(-1), len(F),
                                                                                    beta, sigma2)
    return L, beta, float(sigma2[0])


def mle(model, y, F=None, bounds=(5e-2, 5.0), tau2_bounds=None):
    """Maximize the log-likelihood over a scalar rho in bounds (Brent on log rho), or, with tau2_bounds (a KernelPacket
    of the stable method), over rho and the noise ratio tau2 (L-BFGS-B on their logarithms, from the bounds' geometric
    means); returns (theta, L_hat, L_init), model.rho (and model.tau2) set."""
    L_init = loglike(model, y, F)[0]

    def nll(z):
        if tau2_bounds is None:
            model.rho = float(np.exp(z))
        else:
            model.set_params(float(np.exp(z[0])), float(np.exp(z[1])))
        L = loglike(model, y, F)[0]
        return float(-L) if np.isfinite(L) else 1e30

    if tau2_bounds is None:
        model.rho = float(np.exp(minimize_scalar(nll, bounds=np.log(bounds), method="bounded",
                                                 options=dict(xatol=1e-3)).x))
    else:
        box = np.log([bounds, tau2_bounds])
        z = minimize(nll, box.mean(1), method="L-BFGS-B", bounds=box, options=dict(eps=1e-3, ftol=1e-6)).x
        model.set_params(float(np.exp(z[0])), float(np.exp(z[1])))
    L_hat, beta, sigma2 = loglike(model, y, F)
    return dict(rho=model.rho, tau2=getattr(model, "tau2", 0.0), beta=beta, sigma2=sigma2), L_hat, L_init
