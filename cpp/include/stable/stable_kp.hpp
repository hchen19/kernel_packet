// Stable kernel packets: the Matern-(q-1/2) GP, q <= 3, on sorted 1-D nodes
// with noise ratio tau2, h the mean spacing, three routes:
//   smoother   tau2 > 0.03 (c h)^{2q-1}: square-root filter / RTS smoother on the factored one-sided KP state;
//   factored   c h <= 1, tau2 <= 0.1 (c h)^{2q-1}: K + tau2 I = C~^{-1} G C~^{-T}, alpha = C~^T G^{-1} C~ y, G banded;
//   two-sided  otherwise: the two-sided KPs of Phi_tau = Phi + tau2 A in block LU.
// Posterior means in the Green form of the local KPs, O(1) variances from a selected inverse or smoothed band, the
// likelihood by Givens rotations of the one-sided whitening (or prediction-error decomposition on the smoother route),
// prior draws C~^{-1} L_B z.  kp_model.hpp holds this or the original KernelPacket (original/kernel_packets.hpp) behind
// one handle of the C interface.
#pragma once
#include "../common.hpp"
#include "likelihood.hpp"
#include "regression.hpp"
#include "sampling.hpp"
#include "smoother.hpp"

namespace kp {

struct StableKP {
    enum Route { Factored, TwoSided, Smoother };
    int64_t n;
    int q;
    double rho, c, tau2;
    Route route;
    Matern ker;
    std::vector<double> x;
    // formed on the first solve or prediction (the likelihood needs none of them):
    mutable std::vector<double> F, L, Z;       // factored: F (n, q), chol(G) (q+1, n), band (n, 3q) of G^{-1}
    mutable std::vector<double> A, lu, Eh, Fh; // two-sided: A (n, 2q+1), block LU of Phi_tau; Z (n, 4q-1)
    mutable std::vector<uint8_t> piv;
    mutable std::vector<double> sfh, sal, sband;         // smoother: fhat (n), alpha (n), band (n, 2q)
    mutable std::vector<double> sF, sG, sKg, sGs;        // smoother gains: F (n, q), G (n, q), Kg (n, q), Gs (n, q^2)
    std::vector<double> pF, plow;              // prior sampler on the nodes, formed on the first sample()

    StableKP(const double* xp, int64_t n_, int k, double rho_, double tau2_)
        : n(n_), q((k - 1) / 2), rho(rho_), c(std::sqrt(2.0 * q - 1.0) / rho_), tau2(tau2_), ker(q, rho_),
          x(xp, xp + n_) {
        set_params(rho_, tau2_);
    }

    void set_params(double rho_, double tau2_) {
        rho = rho_;
        c = std::sqrt(2.0 * q - 1.0) / rho;
        tau2 = tau2_;
        ker = Matern(q, rho);
        double h = c * (x[n - 1] - x[0]) / std::max<int64_t>(n - 1, 1);
        if (tau2 > 0.0 && tau2 > 0.03 * std::pow(h, 2 * q - 1))
            route = Smoother;
        else if (h <= 1.0 && tau2 <= 0.1 * std::pow(h, 2 * q - 1))
            route = Factored;
        else
            route = TwoSided;
        F.clear(); L.clear(); Z.clear();
        A.clear(); lu.clear(); Eh.clear(); Fh.clear(); piv.clear();
        sfh.clear(); sal.clear(); sband.clear();
        sF.clear(); sG.clear(); sKg.clear(); sGs.clear();
        pF.clear(); plow.clear();
    }

    void fit() const {
        if (!Z.empty()) return;
        if (route == Factored) {
            F.resize(n * q), L.resize((q + 1) * n), Z.resize(n * 3 * q);
            DISPATCH(q, skp::fit_factored, n, x.data(), c, tau2, nullptr, F.data(), nullptr, nullptr, Z.data(),
                     L.data());
        } else {
            int64_t M = 2 * q, nb = (n + M - 1) / M;
            A.resize(n * (M + 1)), lu.resize(nb * M * M), piv.resize(nb * M), Eh.resize((nb - 1) * q * q),
                Fh.resize((nb - 1) * q * q), Z.resize(n * (2 * M - 1));
            DISPATCH(q, skp::fit, n, x.data(), c, tau2, nullptr, A.data(), nullptr, Z.data(), lu.data(), piv.data(),
                     Eh.data(), Fh.data());
        }
    }

    void fit_gains() const {
        if (!sF.empty()) return;
        sF.resize(n * q), sG.resize(n * q), sKg.resize(n * q), sGs.resize(n * q * q);
        DISPATCH(q, skp::smooth, n, x.data(), c, tau2, nullptr, nullptr, nullptr, sF.data(), sG.data(), sKg.data(),
                 sGs.data());
    }

    int64_t bytes() const {                    // of the fitted model
        if (Z.empty() && sband.empty() && sF.empty() && pF.empty()) {
            if (route == Smoother)
                fit_gains();
            else
                fit();
        }
        return int64_t((x.size() + F.size() + L.size() + Z.size() + A.size() + lu.size() + Eh.size() + Fh.size() +
                        sfh.size() + sal.size() + sband.size() + sF.size() + sG.size() + sKg.size() + sGs.size() +
                        pF.size() + plow.size()) * sizeof(double) + piv.size());
    }

    // The weights of the K columns of Y (n x K, column-major), in place: (K + tau2 I)^{-1} Y on the factored and
    // smoother routes, Phi_tau^{-1} Y otherwise
    void weights(double* Y, int64_t K) const {
        if (route == Smoother) {
            fit_gains();
            columns(K, [&](double* fh, int64_t b) {
                double* y = Y + n * b;
                DISPATCH(q, skp::smooth_cols, n, 1, sF.data(), sG.data(), sKg.data(), sGs.data(), y, fh);
                for (int64_t r = 0; r < n; ++r) y[r] = (y[r] - fh[r]) / tau2;
            }, n);
            return;
        }
        fit();
        columns(K, [&](double* out, int64_t b) {
            const double* y = Y + n * b;
            if (route == Factored)
                DISPATCH(q, skp::solve_factored, n, 1, F.data(), L.data(), y, out);
            else if (q == 1)
                skp::block_solve<2>(n, 1, lu.data(), piv.data(), Eh.data(), Fh.data(), y, out);
            else if (q == 2)
                skp::block_solve<4>(n, 1, lu.data(), piv.data(), Eh.data(), Fh.data(), y, out);
            else
                skp::block_solve<6>(n, 1, lu.data(), piv.data(), Eh.data(), Fh.data(), y, out);
            std::copy(out, out + n, Y + n * b);
        }, n);
    }

    // fn(buffer of len, b) for the K columns b, on threads (one buffer each)
    template <typename Fn>
    static void columns(int64_t K, Fn&& fn, int64_t len) {
        parallel_for(K, 1, [&](int64_t b0, int64_t b1) {
            std::vector<double> buf(len);
            for (int64_t b = b0; b < b1; ++b) fn(buf.data(), b);
        });
    }

    // K^{-1} Y (with tau2: (K + tau2 I)^{-1} Y) in place
    void solve(double* Y, int64_t K) const {
        weights(Y, K);
        if (route != TwoSided) return;
        int S = 2 * q + 1;                      // (A w)_r = sum_j A[j, r - j + q] w_j
        std::vector<double> w(n);
        for (int64_t b = 0; b < K; ++b) {
            double* y = Y + n * b;
            std::copy(y, y + n, w.begin());
            for (int64_t r = 0; r < n; ++r) {
                double s = 0.0;
                for (int64_t j = std::max<int64_t>(0, r - q); j <= std::min<int64_t>(n - 1, r + q); ++j)
                    s += A[j * S + r - j + q] * w[j];
                y[r] = s;
            }
        }
    }

    // Posterior mean at xs for the weights w of one data column y (from weights()), and the variance if var
    void predict(const double* xs, int64_t m, const double* y, const double* w, double* mean, double* var) const {
        fit();
        if (route == Factored) {
            std::vector<double> fh(n);
            for (int64_t r = 0; r < n; ++r) fh[r] = y[r] - tau2 * w[r];
            DISPATCH(q, skp::predict_factored, n, m, x.data(), F.data(), w, fh.data(), Z.data(), xs, c, tau2, mean,
                     var);
        } else {
            DISPATCH(q, skp::predict, n, m, x.data(), A.data(), w, Z.data(), xs, c, tau2, mean, var);
        }
    }

    // Posterior means (m x K) at xs of the K data columns V (n x K), column-major
    void means(const double* xs, int64_t m, const double* V, int64_t K, double* out) const {
        if (route == Smoother) {
            fit_gains();
            int64_t M = 2 * q;
            std::vector<int64_t> J(m * M);
            std::vector<double> pW(m * M), cW(m * M);
            DISPATCH(q, skp::psi_window, n, m, x.data(), xs, c, J.data(), pW.data(), cW.data());
            columns(K, [&](double* fh, int64_t b) {
                const double* v = V + n * b;
                DISPATCH(q, skp::smooth_cols, n, 1, sF.data(), sG.data(), sKg.data(), sGs.data(), v, fh);
                for (int64_t j = 0; j < m; ++j) {
                    double s = 0.0;
                    for (int t = 0; t < M; ++t) {
                        int64_t col = J[j * M + t];
                        s += pW[j * M + t] * ((v[col] - fh[col]) / tau2) - cW[j * M + t] * fh[col];
                    }
                    out[j + m * b] = s;
                }
            }, n);
            return;
        }
        std::vector<double> W(V, V + n * K);
        weights(W.data(), K);
        columns(K, [&](double*, int64_t b) { predict(xs, m, V + n * b, W.data() + n * b, out + m * b, nullptr); }, 0);
    }

    void posterior(const double* xs, int64_t m, const double* y, double* mean, double* var) const {
        if (route == Smoother) {
            sfh.resize(n);
            sal.resize(n);
            sband.resize(n * 2 * q);
            DISPATCH(q, skp::smooth, n, x.data(), c, tau2, y, sfh.data(), sband.data());
            for (int64_t k = 0; k < n; ++k) sal[k] = (y[k] - sfh[k]) / tau2;
            DISPATCH(q, skp::smoother_predict, n, m, x.data(), sfh.data(), sal.data(), sband.data(), xs, c, tau2,
                     mean, var);
            return;
        }
        std::vector<double> w(y, y + n);
        weights(w.data(), 1);
        predict(xs, m, y, w.data(), mean, var);
    }

    // K(xs, xs) - K(xs, X) (K + tau2 I)^{-1} K(X, xs) (m x m), column b from the posterior mean of K(X, xs_b)
    void posterior_cov(const double* xs, int64_t m, double* cov) const {
        std::vector<double> V(n * m);
        parallel_for(m, 64, [&](int64_t b0, int64_t b1) {
            for (int64_t b = b0; b < b1; ++b)
                for (int64_t r = 0; r < n; ++r) V[r + n * b] = ker(x[r] - xs[b]);
        });
        means(xs, m, V.data(), m, cov);
        for (int64_t b = 0; b < m; ++b)
            for (int64_t a = 0; a < m; ++a) cov[a + m * b] = ker(xs[a] - xs[b]) - cov[a + m * b];
    }

    // log det(K + tau2 I) and S = U^T U for R^T U = C Y (Y: n x K)
    double gram(const double* Y, int K, double* S) const {
        if (route == Smoother) return DISPATCH(q, skp::smoother_nll, n, x.data(), c, tau2, Y, K, S);
        return DISPATCH(q, skp::nll, n, x.data(), c, tau2, Y, K, S);
    }

    // Prior draws in place: Z (n x p) normals -> C~^{-1} L_B Z
    void sample(double* Zc, int64_t p) {
        if (pF.empty()) {
            pF.resize(n * q), plow.resize(q * n);
            DISPATCH(q, skp::prior_setup, n, x.data(), c, pF.data(), plow.data());
        }
        draw(pF, plow, n, Zc, p);
    }

    void draw(const std::vector<double>& Fp, const std::vector<double>& low, int64_t nu, double* Zc, int64_t p) const {
        columns(p, [&](double* out, int64_t b) {
            DISPATCH(q, skp::prior_draw, nu, 1, Fp.data(), low.data(), Zc + nu * b, out, false);
            std::copy(out, out + nu, Zc + nu * b);
        }, nu);
    }

    // Posterior draws at xs by pathwise conditioning: a prior draw f on the union of the nodes and xs, then
    // f(xs) + K(xs, X) (K + tau2 I)^{-1} (y - f(X) - tau e).  Z: (n + m) x p normals for f (the first |union| rows),
    // with n more rows for e when tau2 > 0; out: m x p
    void posterior_sample(const double* xs, int64_t m, const double* y, const double* Zn, int64_t p,
                          double* out) const {
        int64_t rows = n + m + (tau2 > 0.0 ? n : 0);
        std::vector<double> u(x);
        u.insert(u.end(), xs, xs + m);
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end()), u.end());
        int64_t nu = u.size();
        auto at = [&](double v) { return std::lower_bound(u.begin(), u.end(), v) - u.begin(); };
        std::vector<double> Fp(nu * q), low(q * nu), f(nu * p), V(n * p);
        DISPATCH(q, skp::prior_setup, nu, u.data(), c, Fp.data(), low.data());
        for (int64_t b = 0; b < p; ++b) std::copy(Zn + b * rows, Zn + b * rows + nu, f.begin() + b * nu);
        draw(Fp, low, nu, f.data(), p);
        double tau = std::sqrt(tau2);
        for (int64_t b = 0; b < p; ++b)
            for (int64_t r = 0; r < n; ++r)
                V[r + n * b] = y[r] - f[at(x[r]) + nu * b] - (tau2 > 0.0 ? tau * Zn[b * rows + n + m + r] : 0.0);
        means(xs, m, V.data(), p, out);
        for (int64_t b = 0; b < p; ++b)
            for (int64_t j = 0; j < m; ++j) out[j + m * b] += f[at(xs[j]) + nu * b];
    }
};

}  // namespace kp
