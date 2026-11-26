"""Kernel packets against dense Gaussian process algebra: backward error of the solves at any conditioning, values
against dense solves where K is well conditioned, and symmetric dispatch across NumPy, JAX, and PyTorch."""
import os
import subprocess
import sys

import numpy as np
import pytest

import kernel_packets as kp

FRAMEWORKS = ["numpy", "jax", "torch"]


def _to_fw(a, fw):
    arr = np.asarray(a, dtype=np.float64)
    if fw == "numpy":
        return arr
    if fw == "jax":
        pytest.importorskip("jax")
        return kp.backend.bk.jax()[1].asarray(arr, dtype=np.float64)
    if fw == "torch":
        torch = pytest.importorskip("torch")
        return torch.as_tensor(arr, dtype=torch.float64)
    raise ValueError(fw)


def _assert_fw(a, fw):
    if fw == "numpy":
        assert isinstance(a, np.ndarray) and not kp.backend.bk.is_torch(a) and not kp.backend.bk.is_jax(a)
    elif fw == "jax":
        assert kp.backend.bk.is_jax(a)
    elif fw == "torch":
        assert kp.backend.bk.is_torch(a)


def dense(X, k, rho):
    return np.asarray(kp.matern(X, X, k, rho))


def backward_error(K, W, Y):
    return np.abs(K @ W - Y).max() / (np.abs(K).sum(1).max() * np.abs(W).max())


@pytest.mark.parametrize("k", [3, 5, 7, 9])
@pytest.mark.parametrize("n", [4, 30, 300])
@pytest.mark.parametrize("method", ["stable", "original"])
@pytest.mark.parametrize("rho", [0.03, 1.0, 20.0])
def test_solve(k, n, rho, method):
    rng = np.random.default_rng(k * n)
    x = np.sort(rng.uniform(0, 1, n))
    K, Y, m = dense(x, k, rho), rng.standard_normal((n, 3)), kp.KernelPacket(x, k, rho, method=method)
    assert backward_error(K, m.solve(Y), Y) < 1e-13
    if np.linalg.cond(K) < 1e8:
        assert abs(m.logdet - np.linalg.slogdet(K)[1]) < 1e-14 * n * np.linalg.cond(K)


@pytest.mark.parametrize("k", [3, 5, 7])
@pytest.mark.parametrize("method", ["stable", "original"])
@pytest.mark.parametrize("n", [4, 30])
def test_posterior(k, n, method):
    rng = np.random.default_rng(k * n)
    x, xnew, y = np.sort(rng.uniform(0, 1, n)), rng.uniform(-0.2, 1.2, 50), rng.standard_normal(n)
    rho = 0.01 * (k - 2)
    K, Ks = dense(x, k, rho), np.asarray(kp.matern(xnew, x, k, rho))
    mean, cov = kp.posterior(kp.KernelPacket(x, k, rho, method=method), xnew, y, cov=True)
    np.testing.assert_allclose(mean, Ks @ np.linalg.solve(K, y), atol=1e-10)
    np.testing.assert_allclose(cov, dense(xnew, k, rho) - Ks @ np.linalg.solve(K, Ks.T), atol=1e-10)


@pytest.mark.parametrize("k", [3, 5, 7])
@pytest.mark.parametrize("rho", [0.05, 1.0])
def test_posterior_noise(k, rho):
    """Stable KPs with noise ratio tau2, on both routes (two-sided at rho = 0.05, factored at rho = 1)."""
    rng = np.random.default_rng(k)
    x, xnew, y, tau2 = np.sort(rng.uniform(0, 1, 40)), rng.uniform(-0.2, 1.2, 30), rng.standard_normal(40), 1e-6
    K, Ks = dense(x, k, rho) + tau2 * np.eye(40), np.asarray(kp.matern(xnew, x, k, rho))
    mean, var, cov = kp.posterior(kp.KernelPacket(x, k, rho, tau2), xnew, y, var=True, cov=True)
    ref = dense(xnew, k, rho) - Ks @ np.linalg.solve(K, Ks.T)
    np.testing.assert_allclose(mean, Ks @ np.linalg.solve(K, y), atol=1e-9 * np.abs(mean).max())
    np.testing.assert_allclose(cov, ref, atol=1e-9)
    np.testing.assert_allclose(var, np.diag(ref), atol=1e-9)


@pytest.mark.parametrize("k", [3, 5, 7])
def test_smoother_route(k):
    """Stable KPs on the smoother route (tau2 > 0.03 (c h)^{2q-1}) and set_params."""
    rng = np.random.default_rng(10 + k)
    n, m_pts, rho, tau2 = 200, 25, 0.2, 1e-3
    x = np.sort(rng.beta(5.0, 15.0, n))
    xnew, y = rng.uniform(-0.1, 1.1, m_pts), rng.standard_normal(n)
    kp_m = kp.KernelPacket(x, k, 0.5, 1e-2)
    kp_m.set_params(rho, tau2)
    K, Ks = dense(x, k, rho) + tau2 * np.eye(n), np.asarray(kp.matern(xnew, x, k, rho))
    mean, var, cov = kp.posterior(kp_m, xnew, y, var=True, cov=True)
    ref = dense(xnew, k, rho) - Ks @ np.linalg.solve(K, Ks.T)
    np.testing.assert_allclose(mean, Ks @ np.linalg.solve(K, y), atol=1e-10)
    np.testing.assert_allclose(cov, ref, atol=1e-10)
    np.testing.assert_allclose(var, np.diag(ref), atol=1e-10)
    assert backward_error(K, kp_m.solve(y), y) < 1e-12
    F_ones = np.ones((n, 1))
    beta = np.linalg.solve(F_ones.T @ np.linalg.solve(K, F_ones), F_ones.T @ np.linalg.solve(K, y))
    r = y - F_ones @ beta
    s2 = r @ np.linalg.solve(K, r) / n
    L, b, s = kp.loglike(kp_m, y)
    np.testing.assert_allclose([L, b[0], s], [-0.5 * (n * np.log(s2) + np.linalg.slogdet(K)[1] + n), beta[0], s2],
                               rtol=1e-9)
    draws = np.asarray(kp.posterior_sample(kp_m, xnew[:5], y, 10000, rng=0))
    np.testing.assert_allclose(draws.mean(1), mean[:5], atol=5e-2 * np.sqrt(var[:5]).max())
    assert kp.KernelPacket(x, k, rho, tau2).sample_(np.eye(n)).shape == (n, n)
    m_prior = kp.KernelPacket(x, k, rho, tau2)
    kp.prior_sample(m_prior, 2, rng=0)
    assert m_prior.nbytes == 8 * n * (1 + (k - 1))


@pytest.mark.parametrize("d,eta", [(1, 4), (2, 5), (3, 6), (4, 7)])
def test_sparse_grid(d, eta):
    g = kp.SparseGrid(d, eta, 5, [0.5, 0.9, 1.3, 0.7][:d])
    K, Y = dense(g.X, 5, g.rho), np.random.default_rng(d).standard_normal((len(g.X), 2))
    assert backward_error(K, g.solve(Y), Y) < 1e-13
    np.testing.assert_allclose(g.logdet, np.linalg.slogdet(K)[1], rtol=1e-9)


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_sparse_grid_regression(d, fw):
    rng = np.random.default_rng(d)
    X, s2, t2 = rng.uniform(0, 1, (300, d)), 1.3, 1e-2
    y, xs = np.sin(4 * X).sum(1) + 0.1 * rng.standard_normal(300), rng.uniform(0, 1, (40, d))
    g = kp.SparseGrid(d, d + 3, 5, 0.6)
    m = kp.SparseGridRegression(_to_fw(X, fw), _to_fw(y, fw), g, s2, t2)
    M = lambda A, B: np.asarray(kp.matern(A, B, 5, g.rho))
    Kuu, Kux, Ms = M(g.X, g.X), M(g.X, X), M(xs, g.X)
    A = Kuu + s2 / t2 * Kux @ Kux.T
    A += len(A) * np.finfo(float).eps * np.diag(A).mean() * np.eye(len(A))   # the method's jitter
    Q = Kux.T @ np.linalg.solve(Kuu, Kux)
    C = s2 * Q + t2 * np.eye(len(y))
    elbo = -0.5 * (y @ np.linalg.solve(C, y) + np.linalg.slogdet(C)[1] + len(y) * np.log(2 * np.pi)) \
        - 0.5 * s2 * (len(y) - np.trace(Q)) / t2
    var = s2 * (1 - np.sum(Ms * np.linalg.solve(Kuu, Ms.T).T, 1) + np.sum(Ms * np.linalg.solve(A, Ms.T).T, 1))
    cov_ref = s2 * (M(xs, xs) - Ms @ np.linalg.solve(Kuu, Ms.T) + Ms @ np.linalg.solve(A, Ms.T))
    xs_fw = _to_fw(xs, fw)
    mean, v = m.predict(xs_fw, var=True)
    mean2, cov = m.predict(xs_fw, cov=True)
    for arr in (mean, v, mean2, cov):
        _assert_fw(arr, fw)
    np.testing.assert_allclose(kp.backend.bk.to_numpy(mean), s2 * Ms @ np.linalg.solve(A, Kux @ y) / t2, atol=1e-9)
    np.testing.assert_allclose(kp.backend.bk.to_numpy(mean2), kp.backend.bk.to_numpy(mean), atol=1e-12)
    np.testing.assert_allclose(kp.backend.bk.to_numpy(v), var, atol=1e-9)
    np.testing.assert_allclose(kp.backend.bk.to_numpy(cov), cov_ref, atol=1e-9)
    np.testing.assert_allclose(m.elbo, elbo, rtol=1e-7)
    draws = kp.posterior_sample(m, xs_fw[:5], p=16, rng=0)
    _assert_fw(draws, fw)
    assert draws.shape == (5, 16)


@pytest.mark.parametrize("k", [3, 5, 7])
@pytest.mark.parametrize("method", ["stable", "original"])
@pytest.mark.parametrize("n", [4, 40])
def test_prior_sample(k, n, method):
    """The draws are linear in the normals: their implied factor L (L L^T = K) from unit vectors, K well conditioned."""
    x, rho = np.sort(np.random.default_rng(n).uniform(0, 1, n)), 0.01 * (k - 2)
    L = np.asarray(kp.KernelPacket(x, k, rho, method=method).sample_(np.eye(n)))
    np.testing.assert_allclose(L @ L.T, dense(x, k, rho), atol=1e-12)


@pytest.mark.parametrize("method,tau2", [("stable", 0.0), ("stable", 1e-2), ("original", 0.0)])
def test_posterior_sample(method, tau2):
    rng = np.random.default_rng(3)
    x, xnew, y = np.sort(rng.uniform(0, 1, 30)), rng.uniform(0, 1, 5), rng.standard_normal(30)
    m = kp.KernelPacket(x, 5, 0.2, tau2, method)
    F = np.asarray(kp.posterior_sample(m, xnew, y, 20000, rng=0))
    mean, cov = kp.posterior(m, xnew, y, cov=True)
    np.testing.assert_allclose(F.mean(1), mean, atol=5e-2 * np.sqrt(np.diag(cov)).max())
    np.testing.assert_allclose(np.cov(F), cov, atol=5e-2 * np.abs(cov).max())


def test_lazy_imports():
    """Importing kernel_packets and running on NumPy loads neither JAX nor torch; torch never loads JAX and vice versa."""
    import importlib.util

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}
    common = (
        "import sys, numpy as np, kernel_packets as kp; "
        "from benchmarks.dense_gp import DenseGP; "
        "from evaluation.arrays import array, host, set_framework; "
        "from evaluation.metrics import timed; "
        "rng = np.random.default_rng(0); "
        "x = np.sort(rng.uniform(0, 1, 20)); y = rng.standard_normal(20); xs = rng.uniform(0, 1, 10); "
        "X2 = rng.uniform(0, 1, (30, 2)); y2 = rng.standard_normal(30); xs2 = rng.uniform(0, 1, (8, 2)); "
        "set_framework({fw!r}); "
        "m = kp.KernelPacket(x, 5, 0.3); kp.posterior(m, array(xs), array(y), var=True); "
        "kp.matern(array(X2), array(xs2), 5, 0.5); "
        "sg = kp.SparseGrid(2, 4, 5, 0.6); sg.solve(array(np.ones(len(sg.X)))); "
        "kp.SparseGridRegression(array(X2), array(y2), sg, 1.0, 1e-2).predict(array(xs2), var=True); "
        "DenseGP(array(X2), array(y2), 5, 0.6, 1.0, 1e-2).predict(array(xs2)); "
        "host(timed(lambda: kp.matern(array(xs2), array(xs2), 5, 0.5))[0]); "
    )
    subprocess.run([sys.executable, "-c", common.format(fw="numpy") +
                    "assert 'jax' not in sys.modules and 'torch' not in sys.modules"], env=env, check=True)
    if importlib.util.find_spec("torch") is not None:
        subprocess.run([sys.executable, "-c", common.format(fw="torch") +
                        "assert 'torch' in sys.modules and 'jax' not in sys.modules"], env=env, check=True)
    if importlib.util.find_spec("jax") is not None:
        subprocess.run([sys.executable, "-c", common.format(fw="jax") +
                        "assert 'jax' in sys.modules and 'torch' not in sys.modules"], env=env, check=True)


@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_matern_frameworks(fw):
    rng = np.random.default_rng(11)
    X1, X2, rho = rng.uniform(0, 1, (15, 3)), rng.uniform(0, 1, (10, 3)), [0.3, 0.6, 1.1]
    ref = kp.matern(X1, X2, 5, rho)
    out = kp.matern(_to_fw(X1, fw), _to_fw(X2, fw), 5, rho)
    _assert_fw(out, fw)
    np.testing.assert_allclose(kp.backend.bk.to_numpy(out), ref, atol=1e-13)


@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_framework_roundtrip(fw):
    rng = np.random.default_rng(1)
    x, xnew, y = np.sort(rng.uniform(0, 1, 40)), rng.uniform(0, 1, 20), rng.standard_normal(40)
    m, g = kp.KernelPacket(x, 5, 0.3), kp.SparseGrid(2, 5, 5, 0.7)
    _assert_fw(m.solve(_to_fw(y, fw)), fw)
    np.testing.assert_allclose(kp.backend.bk.to_numpy(m.solve(_to_fw(y, fw))), m.solve(y), atol=1e-13)
    for a, b in zip(kp.posterior(m, _to_fw(xnew, fw), _to_fw(y, fw), cov=True), kp.posterior(m, xnew, y, cov=True)):
        _assert_fw(a, fw)
        np.testing.assert_allclose(kp.backend.bk.to_numpy(a), b, atol=1e-13)
    draws = kp.posterior_sample(m, _to_fw(xnew[:5], fw), _to_fw(y, fw), 4, rng=0)
    _assert_fw(draws, fw)
    np.testing.assert_allclose(kp.backend.bk.to_numpy(draws), kp.posterior_sample(m, xnew[:5], y, 4, rng=0), atol=1e-13)
    Y = rng.standard_normal(len(g.X))
    _assert_fw(g.solve(_to_fw(Y, fw)), fw)
    np.testing.assert_allclose(kp.backend.bk.to_numpy(g.solve(_to_fw(Y, fw))), g.solve(Y), atol=0)
    G = rng.standard_normal((len(g.X), len(g.X)))
    np.testing.assert_allclose(g.trace_solve(_to_fw(G, fw)), g.trace_solve(G), atol=1e-13)
    X3, U3, w3 = rng.uniform(0, 1, (20, 3)), rng.uniform(0, 1, (15, 3)), rng.standard_normal((15, 2))
    pred = kp.predict(_to_fw(X3, fw), _to_fw(U3, fw), _to_fw(w3, fw), 5, [0.3, 0.5, 0.7])
    _assert_fw(pred, fw)
    np.testing.assert_allclose(kp.backend.bk.to_numpy(pred), kp.predict(X3, U3, w3, 5, [0.3, 0.5, 0.7]), atol=1e-13)


@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_dense_gp_frameworks(fw):
    from benchmarks.dense_gp import DenseGP

    rng = np.random.default_rng(19)
    X, xs = rng.uniform(0, 1, (35, 2)), rng.uniform(0, 1, (12, 2))
    y = np.sin(3 * X[:, 0]) + 0.1 * rng.standard_normal(35)
    ref = DenseGP(X, y, k=5, rho=0.4, sigma2=1.2, tau2=1e-2)
    cur = DenseGP(_to_fw(X, fw), _to_fw(y, fw), k=5, rho=0.4, sigma2=1.2, tau2=1e-2)
    _assert_fw(cur.L, fw)
    _assert_fw(cur.alpha, fw)
    np.testing.assert_allclose(cur.loglike, ref.loglike, rtol=1e-12)
    for a, b in zip(cur.predict(_to_fw(xs, fw)), ref.predict(xs)):
        _assert_fw(a, fw)
        np.testing.assert_allclose(kp.backend.bk.to_numpy(a), b, atol=1e-11)
    for a, b in zip(cur.predict(_to_fw(xs, fw), cov=True), ref.predict(xs, cov=True)):
        _assert_fw(a, fw)
        np.testing.assert_allclose(kp.backend.bk.to_numpy(a), b, atol=1e-11)
    ref_mle, ref_beta = DenseGP.mle(X, y, 5)
    cur_mle, cur_beta = DenseGP.mle(_to_fw(X, fw), _to_fw(y, fw), 5)
    np.testing.assert_allclose([cur_mle.rho, cur_beta, cur_mle.sigma2], [ref_mle.rho, ref_beta, ref_mle.sigma2], rtol=1e-6)
    ref_tr = DenseGP(X, y, k=5, rho=0.5, sigma2=1.0, tau2=1e-2).train()
    cur_tr = DenseGP(_to_fw(X, fw), _to_fw(y, fw), k=5, rho=0.5, sigma2=1.0, tau2=1e-2).train()
    np.testing.assert_allclose(cur_tr.loglike, ref_tr.loglike, rtol=1e-6)
    pdraw = DenseGP.prior_sample(_to_fw(xs, fw), 5, 0.4, p=4, rng=0)
    _assert_fw(pdraw, fw)
    np.testing.assert_allclose(kp.backend.bk.to_numpy(pdraw), DenseGP.prior_sample(xs, 5, 0.4, p=4, rng=0), atol=1e-11)
    sdraw = cur.sample(_to_fw(xs, fw), p=4, rng=0)
    _assert_fw(sdraw, fw)
    np.testing.assert_allclose(kp.backend.bk.to_numpy(sdraw), ref.sample(xs, p=4, rng=0), atol=1e-10)


def test_posterior_var_predict():
    rng = np.random.default_rng(5)
    x, xs, y = np.sort(rng.uniform(0, 1, 60)), rng.uniform(-0.1, 1.1, 30), rng.standard_normal(60)
    m = kp.KernelPacket(x, 5, 0.05)
    mean, var, cov = kp.posterior(m, xs, y, var=True, cov=True)
    np.testing.assert_allclose(var, np.diag(cov), atol=1e-13)
    X, U, w = rng.uniform(0, 1, (40, 3)), rng.uniform(0, 1, (50, 3)), rng.standard_normal((50, 2))
    np.testing.assert_allclose(kp.predict(X, U, w, 5, [0.3, 0.5, 0.7]),
                               np.asarray(kp.matern(X, U, 5, [0.3, 0.5, 0.7])) @ w, atol=1e-12)


@pytest.mark.parametrize("method,tau2", [("stable", 0.0), ("stable", 1e-3), ("original", 0.0)])
def test_loglike(method, tau2):
    x, y = np.linspace(0, 1, 50), np.sin(6 * np.linspace(0, 1, 50))
    m = kp.KernelPacket(x, 5, 0.2, tau2, method)
    K, F = dense(x, 5, 0.2) + tau2 * np.eye(50), np.ones((50, 1))
    beta = np.linalg.solve(F.T @ np.linalg.solve(K, F), F.T @ np.linalg.solve(K, y))
    r = y - F @ beta
    s2 = r @ np.linalg.solve(K, r) / 50
    L, b, s = kp.loglike(m, y)
    np.testing.assert_allclose([L, b[0], s], [-0.5 * (50 * np.log(s2) + np.linalg.slogdet(K)[1] + 50), beta[0], s2],
                               rtol=1e-8)


def test_cuda():
    """The CUDA path against the CPU path: the sparse-grid solve and trace equal to the bit (the same operations in the
    same order, no FMA contraction), kernel sums and the sparse-grid regression to rounding."""
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available() and kp.backend.bk.lib.cuda):
        pytest.skip("no GPU or no CUDA build")
    rng = np.random.default_rng(7)
    for d, eta in [(2, 7), (4, 8), (8, 11)]:
        g = kp.SparseGrid(d, eta, 5, 0.6)
        Y = rng.standard_normal((len(g.X), 3))
        Yd = torch.as_tensor(Y, device="cuda")
        np.testing.assert_array_equal(g.solve(Yd).cpu().numpy(), g.solve(Y))
        G = rng.standard_normal((len(g.X), len(g.X)))
        assert g.trace_solve(torch.as_tensor(G, device="cuda")) == g.trace_solve(G)
    X, U, w = rng.uniform(0, 1, (300, 3)), rng.uniform(0, 1, (200, 3)), rng.standard_normal((200, 2))
    dev = lambda a: torch.as_tensor(a, device="cuda")
    np.testing.assert_allclose(kp.predict(dev(X), dev(U), dev(w), 5, [0.3, 0.5, 0.7]).cpu().numpy(),
                               kp.predict(X, U, w, 5, [0.3, 0.5, 0.7]), rtol=1e-13, atol=1e-13)
    X, y, xs = rng.uniform(0, 1, (2000, 3)), rng.standard_normal(2000), rng.uniform(0, 1, (100, 3))
    g = kp.SparseGrid(3, 7, 5, 0.6)
    a = kp.SparseGridRegression(dev(X), dev(y), g, 1.0, 1e-2)
    b = kp.SparseGridRegression(X, y, g, 1.0, 1e-2)
    np.testing.assert_allclose(a.elbo, b.elbo, rtol=1e-7)   # cuBLAS / cuSOLVER against the CPU BLAS: rounding of the dense steps
    for u, v in zip(a.predict(dev(xs), var=True), b.predict(xs, var=True)):  # through A^{-1}: cond(A) amplifies them
        np.testing.assert_allclose(u.cpu().numpy(), np.asarray(v), atol=1e-5 * np.abs(np.asarray(v)).max())
