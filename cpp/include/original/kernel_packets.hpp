// Kernel packets (KP) of the Matern-(q-1/2) kernel on sorted 1-D nodes, k = 2q+1 nodes per packet.
// Columns of A are the KP coefficients, Phi = K A is banded (half-bandwidth q-1), so K^{-1} = A Phi^{-1} and
// log|K| = log|Phi| - log|A|; B = A^T K A = A^T Phi is banded positive definite and K = Phi B^{-1} Phi^T, so
// Phi L^{-T} z ~ N(0, K) with B = L L^T.  With fewer than k nodes, A = I and Phi = B = K (dense).
#pragma once
#include "banded_linalg.hpp"

namespace kp {

constexpr double TAYLOR_SPAN = 3.0;  // Taylor-normalized moment basis if c * (node span) <= TAYLOR_SPAN, else exp basis

// Taylor-normalized basis u_m (u_m^{(j)}(0) = delta_jm) of span{z^l e^z}_{l<np} + {z^l e^{-z}}_{l<nm}: the first row
// of exp(z C), C the companion matrix of (s-1)^np (s+1)^nm, as sum_k coef[k][m] z^k for |z| <= TAYLOR_SPAN / 2
struct Taylor {
    int d = 0, K = 0;
    std::vector<double> coef;
    Taylor() = default;
    Taylor(int np, int nm) : d(np + nm) {
        std::vector<double> p(d + 1, 0.0), R(d, 0.0);
        p[0] = R[0] = 1.0;
        int deg = 0;
        auto root = [&](double r) {
            for (int i = deg; i >= 0; --i) p[i + 1] -= r * p[i];
            ++deg;
        };
        for (int i = 0; i < np; ++i) root(1.0);
        for (int i = 0; i < nm; ++i) root(-1.0);
        double zmax = 0.5 * TAYLOR_SPAN, f = 1.0;  // f = 1 / k!
        double tol = 1e-18 * std::pow(zmax, d - 1) / std::tgamma(d);
        for (int k = 0; k <= 80; ++k, f /= k) {
            double rmax = 0.0;
            for (int m = 0; m < d; ++m) {
                coef.push_back(R[m] * f);
                rmax = std::max(rmax, std::abs(R[m]));
            }
            if (k >= d && std::pow(zmax, k) * f * rmax < tol) break;
            double last = R[d - 1];  // R <- R C
            for (int j = d - 1; j > 0; --j) R[j] = R[j - 1] - last * p[d - j];
            R[0] = -last * p[d];
        }
        K = static_cast<int>(coef.size()) / d;
    }
    double operator()(int m, double z) const {
        double s = 0.0;
        for (int k = K - 1; k >= 0; --k) s = s * z + coef[k * d + m];
        return s;
    }
};

// Null vector a (max |a_j| = 1) of the d x (d+1) row-major matrix M (overwritten): Householder QR of M^T with rows,
// then columns equilibrated (scale[j], so each a_j keeps its relative accuracy)
inline void null_vec(int d, double* M, double* tau, double* scale, double* a) {
    int s = d + 1;
    for (int k = 0; k < d; ++k) {
        double* v = M + k * s;
        double mx = 0.0;
        for (int j = 0; j < s; ++j) mx = std::max(mx, std::abs(v[j]));
        for (int j = 0; j < s; ++j) v[j] /= mx;
    }
    for (int j = 0; j < s; ++j) {
        scale[j] = 0.0;
        for (int k = 0; k < d; ++k) scale[j] = std::max(scale[j], std::abs(M[k * s + j]));
        for (int k = 0; k < d; ++k) M[k * s + j] /= scale[j];
    }
    for (int k = 0; k < d; ++k) {  // column k of M^T is row k of M; reflector k kept in M[k][k:]
        double* v = M + k * s;
        double nx = 0.0;
        for (int j = k; j < s; ++j) nx += v[j] * v[j];
        nx = std::sqrt(nx);
        v[k] += v[k] >= 0 ? nx : -nx;
        double vv = 0.0;
        for (int j = k; j < s; ++j) vv += v[j] * v[j];
        tau[k] = vv > 0 ? 2.0 / vv : 0.0;
        for (int i = k + 1; i < d; ++i) {
            double* r = M + i * s;
            double t = 0.0;
            for (int j = k; j < s; ++j) t += v[j] * r[j];
            t *= tau[k];
            for (int j = k; j < s; ++j) r[j] -= t * v[j];
        }
    }
    std::fill(a, a + s, 0.0);
    a[d] = 1.0;
    for (int k = d - 1; k >= 0; --k) {  // a = H_0 ... H_{d-1} e_d
        const double* v = M + k * s;
        double t = 0.0;
        for (int j = k; j < s; ++j) t += v[j] * a[j];
        t *= tau[k];
        for (int j = k; j < s; ++j) a[j] -= t * v[j];
    }
    double mx = 0.0;
    for (int j = 0; j < s; ++j) mx = std::max(mx, std::abs(a[j] /= scale[j]));
    for (int j = 0; j < s; ++j) a[j] /= mx;
}

struct KernelPacket {
    int64_t n;
    int k, q, w;  // w: coefficients per column (k, or n if dense)
    double rho;
    bool dense;
    Matern ker;
    std::vector<double> x, a, slo, shi;  // a: n x w coefficients; column i is supported on (slo[i], shi[i])
    std::vector<int64_t> lo;             // column i has coefficients on nodes lo[i] .. lo[i] + len[i] - 1
    std::vector<int> len;
    Band phi, A, B, Phi;  // LU of Phi and A; Cholesky of B and Phi itself, formed on the first sample()
    double logdet;

    KernelPacket(const double* xp, int64_t n_, int k, double rho)
        : n(n_), k(k), q((k - 1) / 2), w(n_ < k ? static_cast<int>(n_) : k), rho(rho), dense(n_ < k), ker(q, rho),
          x(xp, xp + n_),
          a(n_ * w, 0.0), slo(n_, -std::numeric_limits<double>::infinity()),
          shi(n_, std::numeric_limits<double>::infinity()), lo(n_, 0), len(n_, w),
          phi(n_, dense ? static_cast<int>(n_) - 1 : q - 1), A(n_, dense ? 0 : q) {
        if (dense)
            for (int64_t i = 0; i < n; ++i) a[i * w + i] = A.at(i, i) = 1.0;
        else
            coefficients();
        parallel_for(n, 4096 / (w * w) + 1, [&](int64_t i0, int64_t i1) {  // Phi = K A
            for (int64_t i = i0; i < i1; ++i)
                for (int t = 0; t < len[i]; ++t)
                    if (slo[i] < x[lo[i] + t] && x[lo[i] + t] < shi[i]) phi.at(lo[i] + t, i) = value(i, x[lo[i] + t]);
        });
        logdet = phi.factor() - A.factor();
    }

    // bytes held by the factorization (the sampler's B and Phi when formed)
    int64_t bytes() const {
        auto b = [](const Band& m) { return int64_t(m.v.size() * sizeof(double) + m.piv.size() * sizeof(int64_t)); };
        return int64_t((x.size() + a.size() + slo.size() + shi.size()) * sizeof(double) + lo.size() * sizeof(int64_t) +
                       len.size() * sizeof(int)) + b(phi) + b(A) + b(B) + b(Phi);
    }

    // phi_i(xv), exactly 0 off the support
    double value(int64_t i, double xv) const {
        if (xv <= slo[i] || xv >= shi[i]) return 0.0;
        double s = 0.0;
        for (int u = 0; u < len[i]; ++u) s += a[i * w + u] * ker(xv - x[lo[i] + u]);
        return s;
    }

    // Column i < q: nodes 0 .. q+i, q moments of e^z and i of e^{-z}, support (-inf, x[q+i]); mirrored on the right;
    // interior: nodes i-q .. i+q, q moments of each, support (x[i-q], x[i+q])
    void coefficients() {
        std::vector<Taylor> left(q), right(q);
        for (int i = 0; i < q; ++i) left[i] = Taylor(q, i), right[i] = Taylor(i, q);
        Taylor mid(q, q);
        parallel_for(n, 1024, [&](int64_t i0, int64_t i1) {
            std::vector<double> M(2 * q * (2 * q + 1)), tau(2 * q), scale(2 * q + 1), z(2 * q + 1);
            for (int64_t i = i0; i < i1; ++i) {
                int np = q, nm = q;
                const Taylor* T = &mid;
                if (i < q) {
                    nm = static_cast<int>(i), T = &left[i];
                } else if (i >= n - q) {
                    np = static_cast<int>(n - 1 - i), T = &right[np];
                    lo[i] = i - q;
                } else {
                    lo[i] = i - q;
                }
                int s = np + nm + 1;
                len[i] = s;
                const double* xi = x.data() + lo[i];
                if (nm == q) slo[i] = xi[0];  // q moments of e^{-z} vanish left of the nodes, of e^z right of them
                if (np == q) shi[i] = xi[s - 1];
                double m = 0.5 * (xi[0] + xi[s - 1]);
                for (int j = 0; j < s; ++j) z[j] = ker.c * (xi[j] - m);
                if (z[s - 1] - z[0] <= TAYLOR_SPAN) {
                    for (int r = 0; r < s - 1; ++r)
                        for (int j = 0; j < s; ++j) M[r * s + j] = (*T)(r, z[j]);
                } else {
                    for (int j = 0; j < s; ++j) {
                        double ep = std::exp(z[j] - z[s - 1]), em = std::exp(z[0] - z[j]), zl = 1.0;
                        for (int l = 0; l < std::max(np, nm); ++l, zl *= z[j]) {
                            if (l < np) M[l * s + j] = zl * ep;
                            if (l < nm) M[(np + l) * s + j] = zl * em;
                        }
                    }
                }
                double* ai = a.data() + i * w;
                null_vec(s - 1, M.data(), tau.data(), scale.data(), ai);
                for (int t = 0; t < s; ++t) A.at(lo[i] + t, i) = ai[t];
            }
        });
    }

    // y <- K^{-1} y (or Phi^{-1} y if !with_a) for one column, u: workspace of length n
    void solve_col(double* y, double* u, bool with_a = true) const {
        phi.solve(y);
        if (!with_a || dense) return;
        std::copy_n(y, n, u);
        std::fill_n(y, n, 0.0);
        for (int64_t i = 0; i < n; ++i)
            for (int t = 0; t < len[i]; ++t) y[lo[i] + t] += a[i * w + t] * u[i];
    }

    // B <- K^{-1} B (or Phi^{-1} B if !with_a) for p columns of length n stored contiguously
    void solve(double* B, int64_t p, bool with_a) const {
        parallel_for(p, 65536 / (n * (phi.b + 1)) + 1, [&](int64_t c0, int64_t c1) {
            std::vector<double> u(n);
            for (int64_t c = c0; c < c1; ++c) solve_col(B + c * n, u.data(), with_a);
        });
    }

    // KPs at xs: phi_{start[j] + t}(xs[j]) = val[j * w + t]
    void basis(const double* xs, int64_t m, int64_t* start, double* val) const {
        parallel_for(m, 1024, [&](int64_t j0, int64_t j1) {
            for (int64_t j = j0; j < j1; ++j) {
                double xv = xs[j];
                int64_t r = std::upper_bound(x.begin(), x.end(), xv) - x.begin();
                int64_t i0 = dense ? 0 : std::max<int64_t>(0, r - q - 1), i1 = dense ? n : std::min(n, r + q + 1);
                while (i0 < i1 - 1 && !(slo[i0] < xv && xv < shi[i0])) ++i0;
                int64_t s = std::min(i0, n - w);
                double* v = val + j * w;
                std::fill(v, v + w, 0.0);
                for (int64_t i = i0; i < i1; ++i)
                    if (slo[i] < xv && xv < shi[i]) v[i - s] = value(i, xv);
                start[j] = s;
            }
        });
    }

    // Z <- Phi L^{-T} Z for p columns of length n: draws of N(0, K) from standard normals
    void sample(double* Z, int64_t p) {
        if (B.n != n) {
            int b = dense ? static_cast<int>(n) - 1 : 2 * q - 1;
            B = Band(n, b), Phi = Band(n, phi.b);
            parallel_for(n, 4096 / (w * w) + 1, [&](int64_t j0, int64_t j1) {  // B_ij = sum_u a_i[u] phi_j(x_u)
                for (int64_t j = j0; j < j1; ++j) {
                    for (int64_t r = std::max<int64_t>(0, j - Phi.b); r < std::min(n, j + Phi.b + 1); ++r)
                        Phi.at(r, j) = value(j, x[r]);
                    for (int64_t i = j; i < std::min(n, j + b + 1); ++i) {
                        double s = 0.0;
                        for (int u = 0; u < len[i]; ++u) s += a[i * w + u] * value(j, x[lo[i] + u]);
                        B.at(i, j) = B.at(j, i) = s;
                    }
                }
            });
            B.cholesky();
        }
        parallel_for(p, 65536 / (n * (B.b + 1)) + 1, [&](int64_t c0, int64_t c1) {
            std::vector<double> y(n);
            for (int64_t c = c0; c < c1; ++c) {
                B.ltsolve(Z + c * n);
                Phi.mul(Z + c * n, y.data());
                std::copy(y.begin(), y.end(), Z + c * n);
            }
        });
    }
};

}  // namespace kp
