// Kernel packets against dense Gaussian process algebra, in C++ alone (as python/tests/test_kernel_packets.py):
// backward error of the solves at any conditioning, log-determinants, posterior mean / variance, the sparse-grid
// solve, the samplers' implied covariance, kernel sums, profile likelihoods.  Exit status 1 on a failure.
//   scripts/setup_env.sh cpp      (builds and runs build/test_kernel_packets)
#include <cstdio>
#include <random>

#include "c_api.hpp"
#include "sparse_grid.hpp"

using Matrix = std::vector<double>;  // n x n, column-major

static int failures = 0;

static void check(bool ok, const char* what, double value) {
    std::printf("%s %-48s %.3e\n", ok ? "ok  " : "FAIL", what, value);
    failures += !ok;
}

static Matrix kernel(const std::vector<double>& X, int64_t n, int d, int k, const double* rho) {
    kp::Kernel K((k - 1) / 2, rho, d);
    Matrix M(n * n);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < n; ++i) M[i + n * j] = K(&X[i * d], &X[j * d]);
    return M;
}

// dense Cholesky K = L L^T in place (lower), log|K|, and K^{-1} b (long double: references of ill-conditioned K)
template <typename T>
static double cholesky(std::vector<T>& L, int64_t n) {
    T ld = 0.0;
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t k = 0; k < j; ++k)
            for (int64_t i = j; i < n; ++i) L[i + n * j] -= L[i + n * k] * L[j + n * k];
        T p = std::sqrt(L[j + n * j]);
        ld += 2 * std::log(p);
        for (int64_t i = j; i < n; ++i) L[i + n * j] /= p;
    }
    return ld;
}

template <typename T>
static std::vector<T> chol_solve(const std::vector<T>& L, int64_t n, std::vector<T> b) {
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t k = 0; k < i; ++k) b[i] -= L[i + n * k] * b[k];
        b[i] /= L[i + n * i];
    }
    for (int64_t i = n - 1; i >= 0; --i) {
        for (int64_t k = i + 1; k < n; ++k) b[i] -= L[k + n * i] * b[k];
        b[i] /= L[i + n * i];
    }
    return b;
}

static double backward_error(const Matrix& K, int64_t n, const std::vector<double>& w, const std::vector<double>& y) {
    double r = 0, kn = 0, wn = 0;
    for (int64_t i = 0; i < n; ++i) {
        double s = -y[i], row = 0;
        for (int64_t j = 0; j < n; ++j) s += K[i + n * j] * w[j], row += std::abs(K[i + n * j]);
        r = std::max(r, std::abs(s)), kn = std::max(kn, row), wn = std::max(wn, std::abs(w[i]));
    }
    return r / (kn * wn);
}

int main() {
    std::mt19937_64 rng(0);
    std::uniform_real_distribution<double> U(0, 1);
    std::normal_distribution<double> N;
    auto sorted = [&](int64_t n) {
        std::vector<double> x(n);
        for (auto& v : x) v = U(rng);
        std::sort(x.begin(), x.end());
        return x;
    };

    for (int k : {3, 5, 7, 9})                                       // solve and log|K|
        for (int64_t n : {4, 30, 300})
            for (double rho : {0.03, 1.0, 20.0}) {
                auto x = sorted(n);
                std::vector<double> y(n), w;
                for (auto& v : y) v = N(rng);
                kp::KernelPacket P(x.data(), n, k, rho);
                w = y;
                P.solve(w.data(), 1, true);
                char what[128];
                std::snprintf(what, sizeof what, "solve k=%d n=%lld rho=%g", k, static_cast<long long>(n), rho);
                check(backward_error(kernel(x, n, 1, k, &rho), n, w, y) < 1e-13, what,
                      backward_error(kernel(x, n, 1, k, &rho), n, w, y));
            }

    for (auto [method, rho] : {std::pair{0, 0.03}, {0, 0.3}, {1, 0.03}, {1, 0.3}}) {   // stable two-sided,
        int64_t n = 30, m = 20;          // stable factored, original: posterior, logdet against dense
        auto x = sorted(n);
        std::vector<double> xs(m), y(n), mean(m), var(m), all(x);
        for (auto& v : xs) v = 1.4 * U(rng) - 0.2;
        for (auto& v : y) v = N(rng);
        void* h = kp_new(x.data(), n, 5, rho, 0.0, method);
        kp_posterior(h, xs.data(), m, y.data(), mean.data(), var.data());
        Matrix Kd = kernel(x, n, 1, 5, &rho);
        std::vector<long double> L(Kd.begin(), Kd.end());
        double ld = cholesky(L, n);
        auto alpha = chol_solve(L, n, std::vector<long double>(y.begin(), y.end()));
        kp::Matern K(2, rho);
        double em = 0, ev = 0, scale = 0;
        for (int64_t j = 0; j < m; ++j) {
            std::vector<long double> ks(n);
            long double mu = 0;
            for (int64_t i = 0; i < n; ++i) ks[i] = K(xs[j] - x[i]), mu += ks[i] * alpha[i];
            auto g = chol_solve(L, n, ks);
            long double v = 1;
            for (int64_t i = 0; i < n; ++i) v -= ks[i] * g[i];
            scale = std::max(scale, double(std::abs(mu)));
            em = std::max(em, double(std::abs(mean[j] - mu))), ev = std::max(ev, double(std::abs(var[j] - v)));
        }
        char what[128];
        const char* name = method ? "original" : "stable";
        std::snprintf(what, sizeof what, "%s rho=%g posterior mean against dense (relative)", name, rho);
        check(em < 1e-8 * scale, what, em / scale);
        std::snprintf(what, sizeof what, "%s rho=%g posterior variance against dense", name, rho);
        check(ev < 1e-10, what, ev);
        std::snprintf(what, sizeof what, "%s rho=%g log|K| against dense", name, rho);
        check(std::abs(kp_logdet(h) - ld) < 1e-8, what, std::abs(kp_logdet(h) - ld));
        kp_free(h);
    }

    {                                                                  // stable smoother route and kp_set_params
        int64_t n = 200, m = 25;
        double rho = 0.2, tau2 = 1e-3;
        auto x = sorted(n);
        std::vector<double> xs(m), y(n), F(n, 1.0), mean(m), var(m), cov(m * m), w;
        for (auto& v : xs) v = 1.4 * U(rng) - 0.2;
        for (auto& v : y) v = N(rng);
        void* h = kp_new(x.data(), n, 5, 0.5, 1e-2, 0);
        kp_set_params(h, rho, tau2);
        kp_posterior(h, xs.data(), m, y.data(), mean.data(), var.data());
        kp_posterior_cov(h, xs.data(), m, cov.data());
        w = y;
        kp_solve(h, w.data(), 1, 1);
        double beta, sigma2, ll = kp_loglike(h, y.data(), F.data(), 1, &beta, &sigma2);
        Matrix Kd = kernel(x, n, 1, 5, &rho);
        for (int64_t i = 0; i < n; ++i) Kd[i + n * i] += tau2;
        double be = backward_error(Kd, n, w, y);
        std::vector<long double> L(Kd.begin(), Kd.end());
        double ld = cholesky(L, n);
        auto alpha = chol_solve(L, n, std::vector<long double>(y.begin(), y.end()));
        auto fsol = chol_solve(L, n, std::vector<long double>(F.begin(), F.end()));
        long double fy = 0, ff = 0;
        for (int64_t i = 0; i < n; ++i) fy += alpha[i], ff += fsol[i];
        long double be_ref = fy / ff, s2_ref = 0;
        std::vector<long double> r(n);
        for (int64_t i = 0; i < n; ++i) r[i] = y[i] - be_ref;
        auto gr = chol_solve(L, n, r);
        for (int64_t i = 0; i < n; ++i) s2_ref += r[i] * gr[i] / n;
        double ll_ref = -0.5 * (n * std::log(double(s2_ref)) + ld + n);
        kp::Matern K(2, rho);
        double em = 0, ev = 0, ec = 0;
        for (int64_t b = 0; b < m; ++b) {
            std::vector<long double> ks(n);
            long double mu = 0;
            for (int64_t i = 0; i < n; ++i) ks[i] = K(xs[b] - x[i]), mu += ks[i] * alpha[i];
            auto g = chol_solve(L, n, ks);
            long double v = 1;
            for (int64_t i = 0; i < n; ++i) v -= ks[i] * g[i];
            em = std::max(em, double(std::abs(mean[b] - mu))), ev = std::max(ev, double(std::abs(var[b] - v)));
            ec = std::max(ec, std::abs(cov[b + m * b] - var[b]));
        }
        check(em < 1e-10, "smoother posterior mean against dense", em);
        check(ev < 1e-10, "smoother posterior variance against dense", ev);
        check(ec < 1e-10, "smoother posterior cov diag against var", ec);
        check(be < 1e-12, "smoother solve backward error", be);
        check(std::abs(ll / ll_ref - 1) < 1e-10, "smoother loglike against dense", std::abs(ll / ll_ref - 1));
        kp_free(h);
    }

    for (int d : {1, 2, 3, 4}) {                                       // sparse grid solve and logdet
        std::vector<double> nodes, rho(d, 0.7);
        std::vector<int64_t> noff{0};
        for (int j = 0; j <= 4; ++j) {
            for (int64_t i = 1; i < (int64_t(1) << j); ++i) nodes.push_back(double(i) / (int64_t(1) << j));
            noff.push_back(nodes.size());
        }
        kp::SparseGrid G(d, d + 3, nodes.data(), noff.data(), 5, rho.data());
        std::vector<double> y(G.N), w(G.N);
        for (auto& v : y) v = N(rng);
        G.solve(y.data(), w.data(), 1);
        Matrix K = kernel(G.X, G.N, d, 5, rho.data()), L = K;
        double ld = cholesky(L, G.N);
        char what[128];
        std::snprintf(what, sizeof what, "sparse grid solve d=%d N=%lld", d, static_cast<long long>(G.N));
        check(backward_error(K, G.N, w, y) < 1e-13, what, backward_error(K, G.N, w, y));
        std::snprintf(what, sizeof what, "sparse grid log|K| d=%d", d);
        check(std::abs(G.logdet() / ld - 1) < 1e-9, what, std::abs(G.logdet() / ld - 1));
    }

    for (int method : {0, 1}) {                                        // prior sampler: implied covariance
        int64_t n = 40;
        double rho = 0.03;
        auto x = sorted(n);
        void* h = kp_new(x.data(), n, 5, rho, 0.0, method);
        Matrix L(n * n, 0.0), K = kernel(x, n, 1, 5, &rho);
        for (int64_t i = 0; i < n; ++i) L[i + n * i] = 1;
        kp_sample(h, L.data(), n);
        kp_free(h);
        double e = 0;
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j) {
                double s = 0;
                for (int64_t c = 0; c < n; ++c) s += L[i + n * c] * L[j + n * c];
                e = std::max(e, std::abs(s - K[i + n * j]));
            }
        check(e < (method ? 1e-7 : 1e-13), method ? "original prior sampler L L^T against K (~ eps cond(B))"
                                                  : "stable prior sampler L L^T against K", e);
    }

    for (auto [method, tau2] : {std::pair{0, 0.0}, {0, 1e-2}, {1, 0.0}}) {   // profile likelihood (and with noise)
        int64_t n = 50;
        double rho = 0.2, beta, sigma2;
        std::vector<double> x(n), y(n), F(n, 1.0);
        for (int64_t i = 0; i < n; ++i) x[i] = double(i) / (n - 1), y[i] = std::sin(6 * x[i]);
        void* h = kp_new(x.data(), n, 5, rho, tau2, method);
        double L = kp_loglike(h, y.data(), F.data(), 1, &beta, &sigma2);
        Matrix C = kernel(x, n, 1, 5, &rho);
        for (int64_t i = 0; i < n; ++i) C[i + n * i] += tau2;
        double ld = cholesky(C, n);
        auto a = chol_solve(C, n, y), b = chol_solve(C, n, F);
        double fy = 0, ff = 0;
        for (int64_t i = 0; i < n; ++i) fy += a[i], ff += b[i];
        double be = fy / ff, s2 = 0;
        std::vector<double> r(n);
        for (int64_t i = 0; i < n; ++i) r[i] = y[i] - be;
        auto g = chol_solve(C, n, r);
        for (int64_t i = 0; i < n; ++i) s2 += r[i] * g[i] / n;
        double ref = -0.5 * (n * std::log(s2) + ld + n);
        char what[128];
        std::snprintf(what, sizeof what, "%s tau2=%g profile log-likelihood against dense",
                      method ? "original" : "stable", tau2);
        check(std::abs(L / ref - 1) < 1e-8, what, std::abs(L / ref - 1));
        kp_free(h);
    }

    {                                                                  // kernel sum
        std::vector<double> X(40 * 3), Xu(50 * 3), w(50), out(40), r3{0.3, 0.5, 0.7};
        for (auto& v : X) v = U(rng);
        for (auto& v : Xu) v = U(rng);
        for (auto& v : w) v = N(rng);
        kernel_predict(X.data(), 40, Xu.data(), 50, 3, w.data(), 1, 5, r3.data(), out.data());
        kp::Kernel K3(2, r3.data(), 3);
        double e = 0;
        for (int i = 0; i < 40; ++i) {
            double s = 0;
            for (int j = 0; j < 50; ++j) s += K3(&X[i * 3], &Xu[j * 3]) * w[j];
            e = std::max(e, std::abs(s - out[i]));
        }
        check(e < 1e-12, "kernel sum K(xs, X) w against direct", e);
    }
    std::printf("%d failure(s)\n", failures);
    return failures != 0;
}
