#pragma once
// KP state-space smoother: square-root filter / RTS smoother on the q-dimensional state of the
// factored one-sided KP whitening (F and L_B~ of kernel_packets.hpp); likelihood by
// prediction-error decomposition, fhat and the band Cov(f_k, f_{k+m} | Y) (m < 2q), prediction with the local KP psi;
// the gains (independent of Y) for the smoothed means of many data columns (posterior sampling).
#include "kernel_packets.hpp"

namespace skp {

// Householder triangularization of the row-major r x c matrix A (row stride ld): R in the first min(r, c) rows
inline void house_r(double* A, int r, int c, int ld) {
    for (int j = 0; j < std::min(r, c); ++j) {
        double nrm = 0.0;
        for (int i = j; i < r; ++i) nrm += A[i * ld + j] * A[i * ld + j];
        nrm = std::sqrt(nrm);
        if (nrm == 0.0) continue;
        double a0 = A[j * ld + j], alpha = a0 >= 0.0 ? -nrm : nrm, v0 = a0 - alpha;
        double vn2 = v0 * v0;
        for (int i = j + 1; i < r; ++i) vn2 += A[i * ld + j] * A[i * ld + j];
        for (int k = j + 1; k < c; ++k) {
            double sk = v0 * A[j * ld + k];
            for (int i = j + 1; i < r; ++i) sk += A[i * ld + j] * A[i * ld + k];
            sk *= 2.0 / vn2;
            A[j * ld + k] -= sk * v0;
            for (int i = j + 1; i < r; ++i) A[i * ld + k] -= sk * A[i * ld + j];
        }
        A[j * ld + j] = alpha;
        for (int i = j + 1; i < r; ++i) A[i * ld + j] = 0.0;
    }
}

// q-dimensional state z_k[j] = E[u_{j,k+j} | eta_<=k] (z_k[0] = f_k): z_k = A_k z_{k-1} + g_k eta_k, A_k upper
// bidiagonal (diagonal e_{k+j}, superdiagonal s_{k+j,j+1}); g_k[j] = d u_{j,k+j} / d eta_k from column k of L_B
// through u_{j-1,t} = s_tj u_{j,t} + e_t u_{j-1,t-1} (t >= j, else u_{j,t}); beyond the data (k + j >= n) e = 1, s = 0,
// g = 0 (A_k invertible, so the RTS residual has one row; these components never reach z[0] at k < n)
template <int Q>
inline void transition(const double* F, const double* G, int64_t n, int64_t k, double (&A)[Q][Q], double (&g)[Q]) {
    for (int j = 0; j < Q; ++j) {
        for (int t = 0; t < Q; ++t) A[j][t] = 0.0;
        bool in = k + j < n;
        A[j][j] = in ? F[(k + j) * Q] : 1.0;
        if (j + 1 < Q) A[j][j + 1] = in ? F[(k + j) * Q + j + 1] : 0.0;
        g[j] = G[k * Q + j];
    }
}

template <int Q>
struct Model {
    int64_t n;
    std::vector<double> F, G;                              // G[k * Q + j] = g_k[j]
    Model(int64_t n_, const double* X, double c) : n(n_), F(n_ * Q), G(n_ * Q, 0.0) {
        std::vector<double> low(Q * n_, 0.0);
        band_chol<Q>(n, factored_B<Q>(n, X, c, F.data()), low.data());
        for (int64_t k = 0; k < n; ++k) {
            double r[Q];
            for (int i = 0; i < Q; ++i) r[i] = k + i < n ? low[i * n + k] : 0.0;    // u_{q,k+i} = L_B[k+i, k]
            for (int j = Q; j >= 1; --j) {
                double prev = 0.0;
                for (int i = 0; i < Q; ++i) {
                    int64_t t = k + i;
                    if (t >= n) break;
                    if (t >= j) r[i] = (j < Q ? F[t * Q + j] : 1.0) * r[i] + F[t * Q] * prev;
                    prev = r[i];
                }
                if (k + j - 1 < n) G[k * Q + j - 1] = r[j - 1];
            }
        }
    }
};

// one filter step: (m, S) at k-1 -> filtered at k (S lower, P = S S^T); returns the standardized innovation, r = r_11;
// K (if given): the gain, m = A m + K (y - (A m)_0)
template <int Q>
inline double filter_step(const Model<Q>& md, int64_t k, double tau, double y, double* m,
                          double (&S)[Q][Q], double& r, double* K = nullptr) {
    constexpr int D = Q;
    double A[D][D], g[D], mp[D]{};
    transition<Q>(md.F.data(), md.G.data(), md.n, k, A, g);
    for (int i = 0; i < D; ++i)
        for (int t = 0; t < D; ++t) mp[i] += A[i][t] * m[t];
    double T[D + 1][D];                                    // [(A S)^T; g^T]
    for (int i = 0; i < D; ++i)
        for (int t = 0; t < D; ++t) {
            double v = 0.0;
            for (int l = 0; l < D; ++l) v += A[t][l] * S[l][i];
            T[i][t] = v;
        }
    for (int t = 0; t < D; ++t) T[D][t] = g[t];
    house_r(&T[0][0], D + 1, D, D);
    double pre[D + 1][D + 1]{};                            // [[tau, 0], [S^T e_1, S^T]], S = R^T
    pre[0][0] = tau;
    for (int i = 0; i < D; ++i) {
        pre[i + 1][0] = T[i][0];                           // (S^T e_1)_i = S[0][i] = R[i][0]
        for (int t = i; t < D; ++t) pre[i + 1][t + 1] = T[i][t];
    }
    house_r(&pre[0][0], D + 1, D + 1, D + 1);
    r = pre[0][0];
    double v = (y - mp[0]) / r;
    for (int i = 0; i < D; ++i) m[i] = mp[i] + pre[0][i + 1] * v;
    if (K)
        for (int i = 0; i < D; ++i) K[i] = pre[0][i + 1] / r;
    for (int i = 0; i < D; ++i)
        for (int t = 0; t < D; ++t) S[i][t] = t <= i ? pre[t + 1][i + 1] : 0.0;
    return v;
}

// log det(K + tau2 I) and Gram matrix S = U^T U (K x K) of the standardized innovations for K column-major columns Y
template <int Q>
double smoother_nll(int64_t n, const double* X, double c, double tau2, const double* Y, int K, double* S) {
    constexpr int D = Q;
    Model<Q> md(n, X, c);
    std::vector<double> m(K * D, 0.0), v(K);
    double Sf[D][D]{}, ld = 0.0, tau = std::sqrt(tau2);
    if (S) std::fill(S, S + K * K, 0.0);
    for (int64_t k = 0; k < n; ++k) {
        double A[D][D], g[D];
        transition<Q>(md.F.data(), md.G.data(), md.n, k, A, g);
        double T[D + 1][D];
        for (int i = 0; i < D; ++i)
            for (int t = 0; t < D; ++t) {
                double val = 0.0;
                for (int l = 0; l < D; ++l) val += A[t][l] * Sf[l][i];
                T[i][t] = val;
            }
        for (int t = 0; t < D; ++t) T[D][t] = g[t];
        house_r(&T[0][0], D + 1, D, D);
        double pre[D + 1][D + 1]{};
        pre[0][0] = tau;
        for (int i = 0; i < D; ++i) {
            pre[i + 1][0] = T[i][0];
            for (int t = i; t < D; ++t) pre[i + 1][t + 1] = T[i][t];
        }
        house_r(&pre[0][0], D + 1, D + 1, D + 1);
        double r = pre[0][0];
        for (int j = 0; j < K; ++j) {
            double* mj = &m[j * D];
            double mp[D]{};
            for (int i = 0; i < D; ++i)
                for (int t = 0; t < D; ++t) mp[i] += A[i][t] * mj[t];
            v[j] = (Y[j * n + k] - mp[0]) / r;
            for (int i = 0; i < D; ++i) mj[i] = mp[i] + pre[0][i + 1] * v[j];
        }
        if (S)
            for (int a = 0; a < K; ++a)
                for (int b = 0; b < K; ++b) S[a * K + b] += v[a] * v[b];
        ld += 2.0 * std::log(std::abs(r));
        for (int i = 0; i < D; ++i)
            for (int t = 0; t < D; ++t) Sf[i][t] = t <= i ? pre[t + 1][i + 1] : 0.0;
    }
    return ld;
}

// fhat (n) and band (n, 2q): band[k * 2q + m] = Cov(f_k, f_{k+m} | Y); gains (if given): F (n, Q), G (n, Q),
// the filter gains Kg (n, D) and the smoother gains Gs (n, D, D), Gs[k] = G_k^T (Gs[n-1] unused), for smooth_cols;
// fh == nullptr: the gains only (no backward recursion)
template <int Q>
void smooth(int64_t n, const double* X, double c, double tau2, const double* Y, double* fh, double* band,
            double* Fo = nullptr, double* Go = nullptr, double* Kg = nullptr, double* Gs = nullptr) {
    constexpr int D = Q, M = 2 * Q;
    Model<Q> md(n, X, c);
    if (Fo) std::memcpy(Fo, md.F.data(), n * Q * sizeof(double)), std::memcpy(Go, md.G.data(), n * Q * sizeof(double));
    std::vector<double> mf(n * D), Sf(n * D * D);
    double m[D]{}, S[D][D]{}, r;
    for (int64_t k = 0; k < n; ++k) {
        filter_step<Q>(md, k, std::sqrt(tau2), Y ? Y[k] : 0.0, m, S, r, Kg ? Kg + k * D : nullptr);
        std::memcpy(&mf[k * D], m, sizeof(m));
        std::memcpy(&Sf[k * D * D], S, sizeof(S));
    }
    // per step, independent of the recursion (threads): QR of [[(A S_k)^T, S_k^T], [g^T, 0]] = [[R11, R12], [0, R22]],
    // G_k^T = R11^{-1} R12 (rows with a zero pivot, deterministic states, set to 0) and the row R22
    std::vector<double> own(Gs ? 0 : n * D * D), R2(fh ? n * D : 0);
    double* Gt_all = Gs ? Gs : own.data();
    parallel_for(n - 1, num_workers(n, 1 << 12), [&](int64_t lo, int64_t hi, int) {
        for (int64_t k = lo; k < hi; ++k) {
            double A[D][D], g[D], Sk[D][D];
            transition<Q>(md.F.data(), md.G.data(), md.n, k + 1, A, g);
            std::memcpy(Sk, &Sf[k * D * D], sizeof(Sk));
            double W[D + 1][2 * D]{};
            for (int i = 0; i < D; ++i)
                for (int t = 0; t < D; ++t) {
                    double v = 0.0;
                    for (int l = 0; l < D; ++l) v += A[t][l] * Sk[l][i];
                    W[i][t] = v;
                    W[i][D + t] = Sk[t][i];
                }
            for (int t = 0; t < D; ++t) W[D][t] = g[t];
            house_r(&W[0][0], D + 1, 2 * D, 2 * D);
            double dmax = 0.0, Gt[D][D];
            for (int i = 0; i < D; ++i) dmax = std::max(dmax, std::abs(W[i][i]));
            for (int i = D - 1; i >= 0; --i) {
                bool zero = std::abs(W[i][i]) <= 1e-14 * dmax;
                for (int t = 0; t < D; ++t) {
                    double v = W[i][D + t];
                    for (int l = i + 1; l < D; ++l) v -= W[i][l] * Gt[l][t];
                    Gt[i][t] = zero ? 0.0 : v / W[i][i];
                }
            }
            std::memcpy(Gt_all + k * D * D, Gt, sizeof(Gt));
            if (fh)
                for (int t = 0; t < D; ++t) R2[k * D + t] = W[D][D + t];
        }
    });
    if (!fh) return;
    // backward recursion: m^s_k = m_k + G_k (m^s_{k+1} - A_{k+1} m_k), P^s_k = R22^T R22 + G_k P^s_{k+1} G_k^T
    // (PSD terms, no subtraction); cs[m] = Cov(s_{k+1}, f_{k+1+m})
    double ms[D], Ps[D][D], cs[M - 1][D]{};
    std::memcpy(ms, &mf[(n - 1) * D], sizeof(ms));
    {
        const double* Sl = &Sf[(n - 1) * D * D];
        for (int i = 0; i < D; ++i)
            for (int t = 0; t < D; ++t) {
                double v = 0.0;
                for (int l = 0; l < D; ++l) v += Sl[i * D + l] * Sl[t * D + l];
                Ps[i][t] = v;
            }
    }
    for (int i = 0; i < D; ++i) cs[0][i] = Ps[i][0];
    fh[n - 1] = ms[0];
    for (int t = 0; t < M; ++t) band[(n - 1) * M + t] = t ? 0.0 : cs[0][0];
    for (int64_t k = n - 2; k >= 0; --k) {
        double A[D][D], g[D];
        transition<Q>(md.F.data(), md.G.data(), md.n, k + 1, A, g);
        const double* mk = &mf[k * D];
        const double* Gt = Gt_all + k * D * D;              // Gt[i * D + t] = G[t][i]
        const double* r = &R2[k * D];
        double d[D], nm[D];
        for (int i = 0; i < D; ++i) {
            double v = 0.0;
            for (int t = 0; t < D; ++t) v += A[i][t] * mk[t];
            d[i] = ms[i] - v;
        }
        for (int i = 0; i < D; ++i) {
            double v = mk[i];
            for (int t = 0; t < D; ++t) v += Gt[t * D + i] * d[t];
            nm[i] = v;
        }
        double GP[D][D];                                   // G P^s_{k+1}
        for (int i = 0; i < D; ++i)
            for (int t = 0; t < D; ++t) {
                double v = 0.0;
                for (int l = 0; l < D; ++l) v += Gt[l * D + i] * Ps[l][t];
                GP[i][t] = v;
            }
        for (int i = 0; i < D; ++i)
            for (int t = 0; t < D; ++t) {
                double v = r[i] * r[t];
                for (int l = 0; l < D; ++l) v += GP[i][l] * Gt[l * D + t];
                Ps[i][t] = v;
            }
        std::memcpy(ms, nm, sizeof(ms));
        double cn[M][D];
        for (int i = 0; i < D; ++i) cn[0][i] = Ps[i][0];
        for (int mm = 1; mm < M; ++mm)
            for (int i = 0; i < D; ++i) {
                double v = 0.0;
                for (int t = 0; t < D; ++t) v += Gt[t * D + i] * cs[mm - 1][t];
                cn[mm][i] = v;
            }
        for (int mm = 0; mm < M; ++mm) band[k * M + mm] = cn[mm][0];
        for (int mm = 0; mm + 1 < M; ++mm) std::memcpy(cs[mm], cn[mm], sizeof(cs[mm]));
        fh[k] = ms[0];
    }
}

// smoothed means fhat (n, B) of the data columns V (n, B) from the gains of smooth: filter means
// m_k = A_k m_{k-1} + K_k (v_k - (A_k m_{k-1})_0), then m^s_k = m_k + G_k (m^s_{k+1} - A_{k+1} m_k); W columns per thread
template <int Q, int W>
void smooth_cols_w(int64_t n, int64_t B, int64_t c0, int w, const double* F, const double* G, const double* Kg,
                   const double* Gs, const double* V, double* out) {
    constexpr int D = Q;
    std::vector<double> mf(n * D * W);
    double m[W][D]{}, A[D][D], g[D];
    for (int64_t k = 0; k < n; ++k) {
        transition<Q>(F, G, n, k, A, g);
        for (int b = 0; b < w; ++b) {
            double mp[D]{};
            for (int i = 0; i < D; ++i)
                for (int t = 0; t < D; ++t) mp[i] += A[i][t] * m[b][t];
            double e = V[k * B + c0 + b] - mp[0];
            for (int i = 0; i < D; ++i) m[b][i] = mp[i] + Kg[k * D + i] * e, mf[(k * W + b) * D + i] = m[b][i];
        }
    }
    for (int b = 0; b < w; ++b) out[(n - 1) * B + c0 + b] = m[b][0];          // m now holds m^s_{n-1}
    for (int64_t k = n - 2; k >= 0; --k) {
        transition<Q>(F, G, n, k + 1, A, g);
        const double* Gt = Gs + k * D * D;
        for (int b = 0; b < w; ++b) {
            const double* mk = &mf[(k * W + b) * D];
            double d[D];
            for (int i = 0; i < D; ++i) {
                double v = m[b][i];
                for (int t = 0; t < D; ++t) v -= A[i][t] * mk[t];
                d[i] = v;
            }
            for (int i = 0; i < D; ++i) {
                double v = mk[i];
                for (int t = 0; t < D; ++t) v += Gt[t * D + i] * d[t];
                m[b][i] = v;
            }
            out[k * B + c0 + b] = m[b][0];
        }
    }
}

template <int Q>
void smooth_cols(int64_t n, int64_t B, const double* F, const double* G, const double* Kg, const double* Gs,
                 const double* V, double* out) {
    if (B == 1) return smooth_cols_w<Q, 1>(n, 1, 0, 1, F, G, Kg, Gs, V, out);
    constexpr int W = 4;
    int64_t nc = (B + W - 1) / W;
    parallel_for(nc, num_workers(nc * n, 1 << 14), [&](int64_t lo, int64_t hi, int) {
        for (int64_t ch = lo; ch < hi; ++ch)
            smooth_cols_w<Q, W>(n, B, ch * W, static_cast<int>(std::min<int64_t>(W, B - ch * W)), F, G, Kg, Gs, V, out);
    });
}

// J (m, 2q), psi_J and cbar_J (zero outside 0..n-1) of the local KPs through xs, for KPSmoother.mean_cols
template <int Q>
void psi_window(int64_t n, int64_t m, const double* X, const double* xs, double c, int64_t* J, double* pW,
                     double* cW) {
    constexpr int M = 2 * Q;
    parallel_for(m, num_workers(m, 256), [&](int64_t lo, int64_t hi, int) {
        for (int64_t b = lo; b < hi; ++b) {
            int64_t i = std::lower_bound(X, X + n, xs[b]) - X;
            local_psi<Q>(n, X, c, xs[b], i, pW + b * M, cW + b * M);
            for (int t = 0; t < M; ++t) {
                int64_t col = i - Q + t;
                bool ok = col >= 0 && col < n;
                J[b * M + t] = std::clamp<int64_t>(col, 0, n - 1);
                if (!ok) pW[b * M + t] = cW[b * M + t] = 0.0;
            }
        }
    });
}

template <int Q>
void smoother_predict(int64_t n, int64_t m, const double* X, const double* fh, const double* al, const double* band,
                      const double* xs, double c, double tau2, double* mean, double* var) {
    constexpr int M = 2 * Q;
    parallel_for(m, num_workers(m, 256), [&](int64_t lo, int64_t hi, int) {
        for (int64_t b = lo; b < hi; ++b) {
            double x = xs[b], pW[M], cW[M], w[M];
            int64_t i = std::lower_bound(X, X + n, x) - X;
            double ps = local_psi<Q>(n, X, c, x, i, pW, cW), mu = 0.0, v = ps;
            for (int t = 0; t < M; ++t) {
                int64_t col = i - Q + t;
                if (col < 0 || col >= n) {
                    w[t] = 0.0;
                    continue;
                }
                mu += pW[t] * al[col] - cW[t] * fh[col];
                if (!var) continue;
                v -= cW[t] * pW[t] + pW[t] * pW[t] / tau2;
                w[t] = cW[t] + pW[t] / tau2;
            }
            if (!var) {
                mean[b] = mu;
                continue;
            }
            for (int s = 0; s < M; ++s)
                for (int t = 0; t < M; ++t) {
                    int64_t a = i - Q + s, bb = i - Q + t;
                    if (a < 0 || bb < 0 || a >= n || bb >= n) continue;
                    v += w[s] * w[t] * band[std::min(a, bb) * M + std::abs(a - bb)];
                }
            mean[b] = mu;
            var[b] = v;
        }
    });
}

}  // namespace skp
