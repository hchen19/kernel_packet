#pragma once
// Stable KP regression (as regression.py): KP coefficients A (n, 2q+1), Phi_tau in blocks, block LU,
// w = Phi_tau^{-1} Y, selected inverse; posterior mean and O(1) variance; phi_J(x*); Phi_tau^{-1} b and Phi_tau^{-T} b.
#include "banded_linalg.hpp"
#include "kernel_packets.hpp"

namespace skp {

template <int Q>
void fit(int64_t n, const double* X, double c, double tau2, const double* Y, double* A, double* w, double* Z,
         double* lu, uint8_t* piv, double* Eh, double* Fh) {
    constexpr int M = 2 * Q, S = 2 * Q + 1, BW = 2 * M - 1;
    int64_t nb = (n + M - 1) / M;
    std::vector<double> Dv(nb * M * M, 0.0), Ev, Fv, luv;
    std::vector<uint8_t> pivv;
    if (!Eh) {
        Ev.resize((nb - 1) * Q * Q);
        Fv.resize((nb - 1) * Q * Q);
        luv.resize(nb * M * M);
        pivv.resize(nb * M);
        Eh = Ev.data(), Fh = Fv.data(), lu = luv.data(), piv = pivv.data();
    }
    std::fill(Eh, Eh + (nb - 1) * Q * Q, 0.0);
    std::fill(Fh, Fh + (nb - 1) * Q * Q, 0.0);
    double* D = Dv.data();
    int workers = num_workers(nb, 256);

    // Phi_tau[r, j] = phi_j(x_r) + tau2 A[j, r - j + q], columns scaled to max |Phi_j| = 1, into D, E, F
    parallel_for(nb, workers, [&](int64_t klo, int64_t khi, int) {
        for (int64_t k = klo; k < khi; ++k) {
            for (int64_t j = k * M; j < std::min(n, (k + 1) * M); ++j) {
                double a[S], pts[S], vals[S]{};
                int kind, s = column_kp<Q>(n, X, c, j, a, pts, kind);
                int off = static_cast<int>(std::max<int64_t>(0, j - Q) - j + Q);
                double smax = 0.0;
                for (int l = 1; l < S - 1; ++l) {                // rows strictly inside the nodes
                    int64_t r = j - Q + l;
                    if (r >= 0 && r < n) {
                        vals[l] = auto_value<Q>(X[r], pts, a + off, s, kind, c);
                        smax = std::max(smax, std::abs(vals[l]));
                    }
                }
                int cb = static_cast<int>(j - k * M);
                for (int l = 0; l < S; ++l) {
                    double asc = a[l] / smax;
                    A[j * S + l] = asc;
                    int64_t r = j - Q + l;
                    if (r < 0 || r >= n) continue;
                    double v = vals[l] / smax + tau2 * asc;
                    int64_t kr = r / M;
                    int rr = static_cast<int>(r - kr * M);
                    if (kr == k) D[(k * M + rr) * M + cb] = v;
                    else if (kr == k + 1) Eh[(k * Q + rr) * Q + cb - Q] = v;
                    else Fh[((k - 1) * Q + rr - Q) * Q + cb] = v;
                }
            }
            for (int64_t j = n; j < (k + 1) * M; ++j) D[(k * M + j - k * M) * M + j - k * M] = 1.0;
        }
    });

    // forward Schur sweep and block LU
    std::vector<double> Sv(Z ? nb * M * M : 0);
    double Sk[M][M];
    std::memcpy(Sk, D, sizeof(Sk));
    for (int64_t k = 0; k < nb; ++k) {
        if (Z) std::memcpy(&Sv[k * M * M], Sk, sizeof(Sk));
        auto L = reinterpret_cast<double(*)[M]>(lu + k * M * M);
        std::memcpy(L, Sk, sizeof(Sk));
        lu_small<M>(L, piv + k * M);
        if (k + 1 == nb) break;
        double Fp[M][Q]{};
        for (int i = 0; i < Q; ++i)
            for (int j = 0; j < Q; ++j) Fp[M - Q + i][j] = Fh[(k * Q + i) * Q + j];
        lu_solve_small<M, Q>(L, piv + k * M, Fp);
        std::memcpy(Sk, D + (k + 1) * M * M, sizeof(Sk));
        for (int i = 0; i < Q; ++i)
            for (int j = 0; j < Q; ++j)
                for (int r = 0; r < Q; ++r) Sk[i][j] -= Eh[(k * Q + i) * Q + r] * Fp[M - Q + r][j];
    }
    if (Y) block_solve<M>(n, 1, lu, piv, Eh, Fh, Y, w);
    if (!Z) return;

    // selected inverse: central 2M - 1 diagonals of Phi_tau^{-1}, Z[row, M - 1 + col - row]
    if (nb == 1) {
        double Zd[M][M];
        inverse_small<M>(reinterpret_cast<double(*)[M]>(D), Zd);
        for (int64_t r = 0; r < n; ++r)
            for (int d = -(M - 1); d < M; ++d) Z[r * BW + M - 1 + d] = (r + d >= 0 && r + d < M) ? Zd[r][r + d] : 0.0;
        return;
    }
    // backward sweep R_k = D_k - F_k R_{k+1}^{-1} E_k, overwriting D (R_0 is not needed)
    for (int64_t k = nb - 2; k >= 1; --k) {
        double Rl[M][M], Ep[M][Q]{};
        uint8_t pv[M]{};
        std::memcpy(Rl, D + (k + 1) * M * M, sizeof(Rl));
        lu_small<M>(Rl, pv);
        for (int i = 0; i < Q; ++i)
            for (int j = 0; j < Q; ++j) Ep[i][j] = Eh[(k * Q + i) * Q + j];
        lu_solve_small<M, Q>(Rl, pv, Ep);
        auto Rk = reinterpret_cast<double(*)[M]>(D + k * M * M);
        for (int i = 0; i < Q; ++i)
            for (int j = 0; j < Q; ++j)
                for (int r = 0; r < Q; ++r) Rk[M - Q + i][M - Q + j] -= Fh[(k * Q + i) * Q + r] * Ep[r][j];
    }
    // Z_kk = (S_k - F_k R_{k+1}^{-1} E_k)^{-1};  Z_{k+1,k} = -R_{k+1}^{-1} E_k Z_kk;  Z_{k,k+1} = -Z_kk F_k R_{k+1}^{-1}
    auto diag_block = [&](int64_t k, double Zd[M][M], double luR[M][M], uint8_t pvR[M], double RE[M][Q]) {
        double Md[M][M];
        std::memcpy(Md, &Sv[k * M * M], sizeof(Md));
        if (k + 1 < nb) {
            std::memcpy(luR, D + (k + 1) * M * M, sizeof(Md));
            lu_small<M>(luR, pvR);
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < Q; ++j) RE[i][j] = i < Q ? Eh[(k * Q + i) * Q + j] : 0.0;
            lu_solve_small<M, Q>(luR, pvR, RE);
            for (int i = 0; i < Q; ++i)
                for (int j = 0; j < Q; ++j)
                    for (int r = 0; r < Q; ++r) Md[M - Q + i][M - Q + j] -= Fh[(k * Q + i) * Q + r] * RE[r][j];
        }
        inverse_small<M>(Md, Zd);
    };
    parallel_for(nb, workers, [&](int64_t klo, int64_t khi, int) {
        double Zp[M][M]{}, REp[M][Q]{}, luR[M][M], dummy[M][M];
        uint8_t pvR[M];
        if (klo > 0) diag_block(klo - 1, Zp, dummy, pvR, REp);
        for (int64_t k = klo; k < khi; ++k) {
            double Zd[M][M], RE[M][Q]{}, Zl[M][M]{}, Zu[M][M]{};
            diag_block(k, Zd, luR, pvR, RE);
            if (k > 0)
                for (int i = 0; i < M; ++i)
                    for (int j = 0; j < M; ++j)
                        for (int r = 0; r < Q; ++r) Zl[i][j] -= REp[i][r] * Zp[M - Q + r][j];
            if (k + 1 < nb) {
                double FZ[M][M]{};                                   // (Z_kk F_k)^T padded, then R^{-T}
                for (int j = 0; j < Q; ++j)
                    for (int i = 0; i < M; ++i)
                        for (int r = 0; r < Q; ++r) FZ[j][i] += Zd[i][M - Q + r] * Fh[(k * Q + r) * Q + j];
                lu_solve_T_small<M, M>(luR, pvR, FZ);
                for (int i = 0; i < M; ++i)
                    for (int j = 0; j < M; ++j) Zu[i][j] = -FZ[j][i];
            }
            for (int r = 0; r < M && k * M + r < n; ++r) {
                double* z = Z + (k * M + r) * BW;
                for (int d = -(M - 1); d < M; ++d) {
                    int cl = r + d;
                    z[M - 1 + d] = cl < 0 ? Zl[r][cl + M] : (cl >= M ? Zu[r][cl - M] : Zd[r][cl]);
                }
            }
            std::memcpy(Zp, Zd, sizeof(Zd));
            std::memcpy(REp, RE, sizeof(RE));
        }
    });
}

// phi_J(x) on the window J = i-q..i+q-1 (i = #nodes < x), zero for J outside 0..n-1; J clipped
template <int Q>
inline void window_pt(int64_t n, const double* X, const double* A, double c, double x, int64_t i, int64_t* J,
                      double* ph) {
    constexpr int S = 2 * Q + 1;
    for (int t = 0; t < 2 * Q; ++t) {
        int64_t col = i - Q + t;
        J[t] = std::clamp<int64_t>(col, 0, n - 1);
        ph[t] = 0.0;
        if (col < 0 || col >= n) continue;
        int64_t lo = std::max<int64_t>(0, col - Q), hi = std::min<int64_t>(n - 1, col + Q);
        int kind = col < Q ? LEFT : (col >= n - Q ? RIGHT : TWO);
        ph[t] = auto_value<Q>(x, X + lo, A + col * S + (lo - col + Q), static_cast<int>(hi - lo + 1), kind, c);
    }
}

template <int Q>
void predict(int64_t n, int64_t m, const double* X, const double* A, const double* w, const double* Z,
             const double* xs, double c, double tau2, double* mean, double* var) {
    constexpr int M = 2 * Q, BW = 2 * M - 1;
    parallel_for(m, num_workers(m, 256), [&](int64_t lo, int64_t hi, int) {
        for (int64_t b = lo; b < hi; ++b) {
            double x = xs[b], ph[M];
            int64_t i = std::lower_bound(X, X + n, x) - X, J[M];
            window_pt<Q>(n, X, A, c, x, i, J, ph);
            if (mean) {
                double mu = 0.0;
                for (int t = 0; t < M; ++t) mu += ph[t] * w[J[t]];
                mean[b] = mu;
            }
            if (!var) continue;
            // v_J = psi(x_J) + tau2 cbar_J,  var = psi(x*) - phi_J^T Z_JJ v_J
            double pW[M], cW[M], v[M];
            double ps = local_psi<Q>(n, X, c, x, i, pW, cW);
            for (int t = 0; t < M; ++t) v[t] = pW[t] + tau2 * cW[t];
            double quad = 0.0;
            for (int r = 0; r < M; ++r) {
                const double* z = Z + J[r] * BW + M - 1;
                double acc = 0.0;
                for (int t = 0; t < M; ++t) acc += z[J[t] - J[r]] * v[t];
                quad += ph[r] * acc;
            }
            var[b] = ps - quad;
        }
    });
}

template <int Q>
void window(int64_t n, int64_t m, const double* X, const double* A, const double* xs, double c, int64_t* J,
            double* ph) {
    parallel_for(m, num_workers(m, 256), [&](int64_t lo, int64_t hi, int) {
        for (int64_t b = lo; b < hi; ++b) {
            int64_t i = std::lower_bound(X, X + n, xs[b]) - X;
            window_pt<Q>(n, X, A, c, xs[b], i, J + b * 2 * Q, ph + b * 2 * Q);
        }
    });
}

// ----------------------------------------------------------------------------- factored route (as regression.py)
// G = B~ + tau2 C~ C~^T (half-bandwidth Q), alpha = C~^T G^{-1} C~ Y, fhat = Y - tau2 alpha, Z: band (n, 3Q) of G^{-1}
// (if given), L: lower band (Q+1, n) of chol(G) (if given, for solve_factored)

// C~ v in place by the Q bidiagonal stages (positions k0 + i, v before k0 taken as 0)
template <int Q>
inline void stages(const double* F, int64_t k0, int64_t len, double* v) {
    for (int j = 1; j <= Q; ++j)
        for (int64_t i = len - 1; i >= 0; --i) {
            int64_t k = k0 + i;
            if (k < j) continue;
            double prev = i ? v[i - 1] : 0.0, s = j < Q ? F[k * Q + j] : 1.0;
            v[i] = (v[i] - F[k * Q] * prev) / s;
        }
}

// C~^T v in place over all n: stages L_q^T, ..., L_1^T, (L_j^T v)_m = v_m / s_m - e_{m+1} v_{m+1} / s_{m+1}
template <int Q>
inline void stages_T(const double* F, int64_t n, double* v) {
    for (int j = Q; j >= 1; --j)
        for (int64_t m = 0; m < n; ++m) {
            double out = m >= j ? v[m] / (j < Q ? F[m * Q + j] : 1.0) : v[m];
            if (m + 1 < n && m + 1 >= j) out -= F[(m + 1) * Q] * v[m + 1] / (j < Q ? F[(m + 1) * Q + j] : 1.0);
            v[m] = out;
        }
}

template <int Q>
void fit_factored(int64_t n, const double* X, double c, double tau2, const double* Y, double* F, double* al,
                  double* fh, double* Z, double* L) {
    constexpr int M = Q + 1;
    std::vector<double> B = factored_B<Q>(n, X, c, F), G(M * n, 0.0), R(n * M, 0.0), low(M * n, 0.0);
    for (int d = 0; d < Q; ++d) std::memcpy(&G[d * n], &B[d * n], n * sizeof(double));
    if (tau2 > 0.0) {                                       // rows of C~: R[k, l] = coefficient of x_{k-Q+l}
        for (int64_t k = 0; k < n; ++k) R[k * M + Q] = 1.0;
        for (int j = 1; j <= Q; ++j)
            for (int64_t k = n - 1; k >= j; --k) {
                double s = j < Q ? F[k * Q + j] : 1.0, e = F[k * Q];
                for (int l = 0; l < M; ++l)
                    R[k * M + l] = (R[k * M + l] - e * (l + 1 < M ? R[(k - 1) * M + l + 1] : 0.0)) / s;
            }
        for (int d = 0; d < M; ++d)
            for (int64_t j = 0; j + d < n; ++j) {
                double v = 0.0;
                for (int l = 0; l + d < M; ++l) v += R[(j + d) * M + l] * R[j * M + l + d];
                G[d * n + j] += tau2 * v;
            }
    }
    band_chol<M>(n, G, low.data());
    if (Y) {
        std::memcpy(al, Y, n * sizeof(double));
        stages<Q>(F, 0, n, al);
        band_solve(n, M, low.data(), al);
        stages_T<Q>(F, n, al);
        for (int64_t k = 0; k < n; ++k) fh[k] = Y[k] - tau2 * al[k];
    }
    if (Z) band_selinv(n, M, low.data(), 3 * Q - 1, Z);
    if (L) std::memcpy(L, low.data(), M * n * sizeof(double));
}

// alpha = C~^T G^{-1} C~ v for the K columns of V (n, K) from L of fit_factored; threads over columns
template <int Q>
void solve_factored(int64_t n, int64_t K, const double* F, const double* L, const double* V, double* out) {
    parallel_for(K, num_workers(K * n, 1 << 14), [&](int64_t lo, int64_t hi, int) {
        std::vector<double> v(n);
        for (int64_t b = lo; b < hi; ++b) {
            for (int64_t k = 0; k < n; ++k) v[k] = V[k * K + b];
            stages<Q>(F, 0, n, v.data());
            band_solve(n, Q + 1, L, v.data());
            stages_T<Q>(F, n, v.data());
            for (int64_t k = 0; k < n; ++k) out[k * K + b] = v[k];
        }
    });
}

template <int Q>
void predict_factored(int64_t n, int64_t m, const double* X, const double* F, const double* al, const double* fh,
                      const double* Z, const double* xs, double c, double tau2, double* mean, double* var) {
    constexpr int M = 2 * Q, W = 3 * Q;
    parallel_for(m, num_workers(m, 256), [&](int64_t lo, int64_t hi, int) {
        for (int64_t b = lo; b < hi; ++b) {
            double x = xs[b], pW[M], cW[M], w[W]{};
            int64_t i = std::lower_bound(X, X + n, x) - X, k0 = i - Q;
            double ps = local_psi<Q>(n, X, c, x, i, pW, cW), mu = 0.0, v = ps;
            for (int t = 0; t < M; ++t) {
                int64_t col = k0 + t;
                if (col < 0 || col >= n) continue;
                mu += pW[t] * al[col] - cW[t] * fh[col];
                if (!var) continue;
                w[t] = pW[t] + tau2 * cW[t];
                v += cW[t] * w[t];
            }
            if (!var) {
                mean[b] = mu;
                continue;
            }
            int64_t s0 = std::max<int64_t>(k0, 0);                    // C~ w on indices s0 .. k0 + W - 1 (< n)
            int len = static_cast<int>(std::min<int64_t>(k0 + W, n) - s0), off = static_cast<int>(s0 - k0);
            if (len > 0) stages<Q>(F, s0, len, w + off);
            for (int a = 0; a < len; ++a)
                for (int bb = 0; bb < len; ++bb) {
                    int64_t ka = s0 + a, kb = s0 + bb;
                    v -= w[off + a] * w[off + bb] * Z[std::min(ka, kb) * W + std::abs(ka - kb)];
                }
            mean[b] = mu;
            if (var) var[b] = v;
        }
    });
}

}  // namespace skp
