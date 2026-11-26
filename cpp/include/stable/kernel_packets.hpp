// Matern-(q-1/2) kernel and S(t), KP coefficients and smallest-|terms| evaluation, and the one-sided KP whitening
// C K C^T = B (as kernel_packets.py).
#pragma once
#include "banded_linalg.hpp"
#include "gauss_rules.hpp"

namespace skp {

constexpr int TWO = 0, LEFT = 1, RIGHT = 2;
constexpr double GREEN_MAX = 3.0;
constexpr double AUTO_SPAN = 3.0;

// ----------------------------------------------------------------------------- kernel and S(t)
template <int Q>
struct SSeries {                                  // Taylor coefficients of S(z) / z in z^2, truncated for |z| <= 2
    static constexpr int N = 15;
    double c[N]{};
    int len = N;

    SSeries() {
        for (int k = 0; k < N; ++k) {
            double d = FACT[2 * k + 1];
            if constexpr (Q == 1) c[k] = 1.0 / d;
            else if constexpr (Q == 2) c[k] = -2.0 * k / d;
            else c[k] = 4.0 * k * (k - 1) / (3.0 * d);
        }
        double ref = std::abs(c[Q - 1]);
        for (int k = Q; k < N; ++k)
            if (std::abs(c[k]) * std::pow(4.0, k - Q + 1) < 1e-18 * ref) {
                len = k;
                break;
            }
    }
};

template <int Q>
inline void kernel_s(double t, double c, double& k, double& sv) {
    // K(t) = P(|z|) e^{-|z|} and S(t), z = c t, from one exp: for |z| > 2, sinh and cosh use e^{|z|} = 1 / e^{-|z|}
    static const SSeries<Q> ser;
    double z = c * t, az = std::abs(z), z2 = z * z;
    double e = std::exp(-az);
    if constexpr (Q == 1) k = e;
    else if constexpr (Q == 2) k = (1.0 + az) * e;
    else k = (1.0 + az + z2 / 3.0) * e;
    if (az <= 2.0) {
        double acc = 0.0;
        for (int j = ser.len - 1; j >= 0; --j) acc = acc * z2 + ser.c[j];
        sv = acc * z;
        return;
    }
    double sh = std::copysign(0.5 * (1.0 / e - e), z), ch = 0.5 * (1.0 / e + e);
    if constexpr (Q == 1) sv = sh;
    else if constexpr (Q == 2) sv = sh - z * ch;
    else sv = (1.0 + z2 / 3.0) * sh - z * ch;
}

// ----------------------------------------------------------------------------- Taylor-normalized basis
struct TaylorTable {                              // coef[k][m] = e_1^T C^k e_m / k!, C companion of (s-1)^n+ (s+1)^n-
    static constexpr int KCAP = 60;
    double coef[KCAP + 1][6]{};
    int K = 0;
    int d = 0;

    void init(int n_plus, int n_minus) {
        d = n_plus + n_minus;
        double p[8]{1.0};                         // characteristic polynomial [1, p1, ..., pd]
        int deg = 0;
        auto mul_root = [&](double r) {
            for (int i = deg; i >= 0; --i) p[i + 1] -= r * p[i];
            ++deg;
        };
        for (int i = 0; i < n_plus; ++i) mul_root(1.0);
        for (int i = 0; i < n_minus; ++i) mul_root(-1.0);

        double R[KCAP + 1][6]{};
        double Rmax[KCAP + 1]{};
        R[0][0] = 1.0;
        Rmax[0] = 1.0;
        for (int k = 0; k < KCAP; ++k) {          // R[k+1] = R[k] C
            double mx = 0.0;
            for (int j = 0; j < d; ++j) {
                double v = (j > 0 ? R[k][j - 1] : 0.0) - R[k][d - 1] * p[d - j];
                R[k + 1][j] = v;
                mx = std::max(mx, std::abs(v));
            }
            Rmax[k + 1] = mx;
        }
        // truncate once the remaining terms are < 1e-18 * (smallest basis function ~ z^{d-1}/(d-1)!), |z| <= 4
        double lz = std::log(4.0);
        double rhs = std::log(1e-18) + (d - 1) * lz - std::lgamma(static_cast<double>(d));
        K = KCAP;
        for (int k = d; k <= KCAP; ++k)
            if (k * lz - std::lgamma(k + 1.0) + std::log(4.0 * Rmax[k]) < rhs) {
                K = k;
                break;
            }
        for (int k = 0; k <= K; ++k)
            for (int m = 0; m < d; ++m) coef[k][m] = R[k][m] / FACT[k];
    }
};

inline const TaylorTable& taylor_table(int n_plus, int n_minus) {
    static const struct Cache {
        TaylorTable tbl[4][4];
        Cache() {
            for (int np = 0; np <= 3; ++np)
                for (int nm = 0; nm <= 3; ++nm)
                    if (np + nm > 0) tbl[np][nm].init(np, nm);
        }
    } cache;
    return cache.tbl[n_plus][n_minus];
}

// ----------------------------------------------------------------------------- KP coefficients
// Unit null vector of the D x (D+1) matrix M (rows equilibrated): column-pivoted Householder QR of M^T.
template <int D>
inline void null_vec_qr(const double M[D][D + 1], double out[D + 1]) {
    double R[D + 1][D];
    for (int i = 0; i < D; ++i) {
        double rmax = 0.0;
        for (int j = 0; j <= D; ++j) rmax = std::max(rmax, std::abs(M[i][j]));
        for (int j = 0; j <= D; ++j) R[j][i] = M[i][j] / rmax;
    }
    double V[D][D + 1]{};
    double tau[D]{};
    for (int k = 0; k < D; ++k) {
        if (k < D - 1) {
            int piv = k;
            double best = -1.0;
            for (int c = k; c < D; ++c) {
                double n2 = 0.0;
                for (int i = k; i <= D; ++i) n2 += R[i][c] * R[i][c];
                if (n2 > best) {
                    best = n2;
                    piv = c;
                }
            }
            if (piv != k)
                for (int i = k; i <= D; ++i) std::swap(R[i][k], R[i][piv]);
        }
        double n2 = 0.0;
        for (int i = k; i <= D; ++i) {
            V[k][i] = R[i][k];
            n2 += R[i][k] * R[i][k];
        }
        double nx = std::sqrt(n2);
        V[k][k] += (R[k][k] >= 0.0 ? nx : -nx);
        double vn2 = 0.0;
        for (int i = k; i <= D; ++i) vn2 += V[k][i] * V[k][i];
        tau[k] = vn2 > 0.0 ? 2.0 / std::max(vn2, 1e-300) : 0.0;
        for (int c = k + 1; c < D; ++c) {
            double dot = 0.0;
            for (int i = k; i <= D; ++i) dot += V[k][i] * R[i][c];
            double sc = tau[k] * dot;
            for (int i = k + 1; i <= D; ++i) R[i][c] -= sc * V[k][i];
        }
    }
    for (int i = 0; i < D; ++i) out[i] = 0.0;
    out[D] = 1.0;
    for (int k = D - 1; k >= 0; --k) {
        double dot = 0.0;
        for (int i = k; i <= D; ++i) dot += V[k][i] * out[i];
        double sc = tau[k] * dot;
        for (int i = k; i <= D; ++i) out[i] -= sc * V[k][i];
    }
}

// KP with D+1 sorted nodes, NP conditions t^l e^{ct} and NM conditions t^l e^{-ct}, t = node - shift:
// Taylor basis if c * (node span) <= AUTO_SPAN, else exp basis.
template <int NP, int NM>
inline void kp_coef(const double* nodes, double shift, double c, double* out) {
    constexpr int D = NP + NM;
    double M[D][D + 1];
    if (c * (nodes[D] - nodes[0]) <= AUTO_SPAN) {
        const TaylorTable& tbl = taylor_table(NP, NM);
        for (int j = 0; j <= D; ++j) {
            double z = c * (nodes[j] - shift);
            if constexpr (NP == NM) {             // even / odd basis functions: Horner in z^2
                double z2 = z * z;
                for (int m = 0; m < D; ++m) {
                    int par = m & 1;
                    double acc = 0.0;
                    for (int k = tbl.K - ((tbl.K - par) & 1); k >= par; k -= 2) acc = acc * z2 + tbl.coef[k][m];
                    M[m][j] = par ? acc * z : acc;
                }
            } else {
                for (int m = 0; m < D; ++m) {
                    double acc = 0.0;
                    for (int k = tbl.K; k >= 0; --k) acc = acc * z + tbl.coef[k][m];
                    M[m][j] = acc;
                }
            }
        }
    } else {
        for (int j = 0; j <= D; ++j) {
            double t = nodes[j] - shift;
            double ec = NP > 0 ? std::exp(c * t) : 0.0;
            double em = NP > 0 ? 1.0 / ec : std::exp(-c * t);
            double tp = 1.0;
            for (int l = 0; l < NP; ++l, tp *= t) M[l][j] = tp * ec;
            tp = 1.0;
            for (int l = 0; l < NM; ++l, tp *= t) M[NP + l][j] = tp * em;
        }
    }
    null_vec_qr<D>(M, out);
}

// boundary KPs: n_plus + n_minus <= 2q - 1
inline void kp_coef_dyn(const double* nodes, int n_plus, int n_minus, double shift, double c, double* out) {
    switch (n_plus * 4 + n_minus) {
        case 1: return kp_coef<0, 1>(nodes, shift, c, out);
        case 2: return kp_coef<0, 2>(nodes, shift, c, out);
        case 3: return kp_coef<0, 3>(nodes, shift, c, out);
        case 4: return kp_coef<1, 0>(nodes, shift, c, out);
        case 5: return kp_coef<1, 1>(nodes, shift, c, out);
        case 6: return kp_coef<1, 2>(nodes, shift, c, out);
        case 7: return kp_coef<1, 3>(nodes, shift, c, out);
        case 8: return kp_coef<2, 0>(nodes, shift, c, out);
        case 9: return kp_coef<2, 1>(nodes, shift, c, out);
        case 10: return kp_coef<2, 2>(nodes, shift, c, out);
        case 11: return kp_coef<2, 3>(nodes, shift, c, out);
        case 12: return kp_coef<3, 0>(nodes, shift, c, out);
        case 13: return kp_coef<3, 1>(nodes, shift, c, out);
        case 14: return kp_coef<3, 2>(nodes, shift, c, out);
        case 15: return kp_coef<3, 3>(nodes, shift, c, out);
    }
}

// KP value at x from the exact representation with the smallest sum of |terms| (kernel, causal or
// anti-causal Green sum); s nodes pts with coefficients a.
template <int Q>
inline double auto_value(double x, const double* pts, const double* a, int s, int kind, double c) {
    double first = pts[0], last = pts[s - 1];
    if ((kind != LEFT && x <= first) || (kind != RIGHT && x >= last)) return 0.0;
    double sk = 0.0, mk = 0.0, sc = 0.0, mc = 0.0, sa = 0.0, ma = 0.0;
    for (int l = 0; l < s; ++l) {
        double d = x - pts[l];
        double kv, sv;
        kernel_s<Q>(d, c, kv, sv);
        double tk = a[l] * kv;
        sk += tk;
        mk += std::abs(tk);
        double aS = 2.0 * a[l] * sv;
        if (d > 0.0) {
            sc -= aS;
            mc += std::abs(aS);
        } else if (d < 0.0) {
            sa += aS;
            ma += std::abs(aS);
        }
    }
    double val = sk, bm = mk;
    if (kind != LEFT && c * (x - first) <= GREEN_MAX && mc < bm) {
        bm = mc;
        val = sc;
    }
    if (kind != RIGHT && c * (last - x) <= GREEN_MAX && ma < bm) val = sa;
    return val;
}

// Coefficients of the KP of column j (nodes x_{j-q}..x_{j+q} clipped to 0..n-1) into a[0..2q] (zero outside).
template <int Q>
inline int column_kp(int64_t n, const double* X, double c, int64_t j, double* a, double* pts, int& kind) {
    constexpr int S = 2 * Q + 1;
    int64_t lo = std::max<int64_t>(0, j - Q), hi = std::min<int64_t>(n - 1, j + Q);
    int s = static_cast<int>(hi - lo + 1);
    for (int l = 0; l < s; ++l) pts[l] = X[lo + l];
    kind = j < Q ? LEFT : (j >= n - Q ? RIGHT : TWO);
    for (int l = 0; l < S; ++l) a[l] = 0.0;
    int off = static_cast<int>(lo - j + Q);
    if (kind == TWO) kp_coef<Q, Q>(pts, 0.5 * (pts[0] + pts[S - 1]), c, a);
    else if (kind == LEFT) kp_coef_dyn(pts, Q, static_cast<int>(j), pts[s - 1], c, a + off);   // eq. (11)
    else kp_coef_dyn(pts, static_cast<int>(n - j - 1), Q, pts[0], c, a + off);                 // eq. (10)
    return s;
}

// Local KP psi through x (coefficient of K(., x) = 1; two-sided, one-sided at the ends; i = #nodes < x): returns
// psi(x), and psi(x_J), cbar_J (coefficient of K(., x_J)) on the window J = i-q..i+q-1, zero for J outside 0..n-1.
template <int Q>
inline double local_psi(int64_t n, const double* X, double c, double x, int64_t i, double* pW, double* cW) {
    constexpr int M = 2 * Q, S = 2 * Q + 1;
    int kind = i < Q ? LEFT : (i > n - Q ? RIGHT : TWO);
    int64_t base = kind == LEFT ? 0 : (kind == RIGHT ? n - Q : i - Q);
    double pts[S], a[S];
    int s = S, pos = Q;
    if (kind == TWO) {
        for (int l = 0; l < Q; ++l) pts[l] = X[i - Q + l], pts[Q + 1 + l] = X[i + l];
        pts[Q] = x;
        kp_coef<Q, Q>(pts, 0.5 * (pts[0] + pts[S - 1]), c, a);
    } else {
        s = Q + 1;
        const double* src = kind == LEFT ? X : X + n - Q;
        pos = 0;
        while (pos < Q && src[pos] < x) ++pos;
        for (int l = 0; l < pos; ++l) pts[l] = src[l];
        pts[pos] = x;
        for (int l = pos; l < Q; ++l) pts[l + 1] = src[l];
        if (kind == LEFT) kp_coef<Q, 0>(pts, pts[Q], c, a);
        else kp_coef<0, Q>(pts, pts[0], c, a);
    }
    double inv = 1.0 / a[pos];
    for (int l = 0; l < s; ++l) a[l] *= inv;
    for (int t = 0; t < M; ++t) {
        int64_t col = i - Q + t;
        pW[t] = cW[t] = 0.0;
        if (col < 0 || col >= n) continue;
        int64_t k = col - base;
        int sl = static_cast<int>(k + (k >= pos));
        cW[t] = (k >= 0 && k < M && sl < s) ? a[sl] : 0.0;
        pW[t] = auto_value<Q>(X[col], pts, a, s, kind, c);
    }
    return auto_value<Q>(x, pts, a, s, kind, c);
}

// ----------------------------------------------------------------------------- one-sided KP whitening
// Gauss-Legendre rules of gauss_rules.hpp: the unfactored B uses on
// [x_i, x_{i+1}] the first order >= 16 + 4 c (x_{i+1} - x_i), the factored B~ the first order
// >= 6 + 2 c (x_{i+1} - x_i);
// Gauss-Laguerre rules with q + 1 points for (-inf, x_0].

inline int gl_order(double need) {
    int o = 0;
    while (o < GL_N_ORDERS - 1 && GL_ORDERS[o] < need) ++o;
    return o;
}

// Lower band B[d * n + j] = B[j+d, j] (d < Q) from per-interval contributions: fn(iv, P) adds to P[r][r'] (r' <= r)
// the integral over (x_iv, x_{iv+1}) of g_{iv+1+r} g_{iv+1+r'}; intervals on threads.
template <int Q, typename Fn>
std::vector<double> interval_band(int64_t n, Fn&& fn) {
    std::vector<double> B(Q * n, 0.0);
    int64_t niv = n - 1;
    if (niv <= 0) return B;
    int workers = num_workers(niv, 4096);
    struct Edge {
        double v[Q][Q]{};                          // contributions to columns iv_hi + 1 .. iv_hi + q - 1
    };
    std::vector<Edge> edges(workers);
    parallel_for(niv, workers, [&](int64_t lo, int64_t hi, int t) {
        for (int64_t iv = lo; iv < hi; ++iv) {
            double P[Q][Q]{};
            fn(iv, P);
            for (int d = 0; d < Q; ++d)
                for (int rp = 0; rp + d < Q; ++rp) {
                    int64_t col = iv + 1 + rp;
                    if (col + d >= n) continue;
                    if (col <= hi) B[d * n + col] += P[rp + d][rp];      // columns > hi belong to the next range
                    else edges[t].v[d][col - hi - 1] += P[rp + d][rp];
                }
        }
    });
    for (int t = 0; t < workers; ++t) {
        int64_t hi = niv * (t + 1) / workers;
        for (int d = 0; d < Q; ++d)
            for (int o = 0; o + 1 < Q; ++o)
                if (hi + 1 + o + d < n) B[d * n + hi + 1 + o] += edges[t].v[d][o];
    }
    return B;
}

// L_B = chol(B) as the lower band low[d * n + j] = L_B[j+d, j]; block Cholesky (blocks of size q):
// L_kk = chol(D_k - W W^T), W = L_{k,k-1} = E_{k-1} L_{k-1}^{-T}
template <int Q>
void band_chol(int64_t n, const std::vector<double>& B, double* low) {
    int64_t nb = (n + Q - 1) / Q;
    double Lp[Q][Q]{};
    for (int64_t k = 0; k < nb; ++k) {
        double Dk[Q][Q]{}, Lk[Q][Q], Wk[Q][Q]{};
        for (int i = 0; i < Q; ++i)
            for (int j = 0; j < Q; ++j) {
                int64_t gi = k * Q + i, gj = k * Q + j;
                Dk[i][j] = (gi < n && gj < n) ? B[std::abs(i - j) * n + std::min(gi, gj)] : (i == j);
            }
        if (k > 0) {
            for (int i = 0; i < Q; ++i) {                            // E[i][j] = B[kQ + i, (k-1)Q + j], j > i
                if (k * Q + i >= n) continue;
                for (int j = 0; j < Q; ++j) {
                    double v = j > i ? B[(Q + i - j) * n + (k - 1) * Q + j] : 0.0;
                    for (int t = 0; t < j; ++t) v -= Wk[i][t] * Lp[j][t];
                    Wk[i][j] = v / Lp[j][j];
                }
            }
            for (int i = 0; i < Q; ++i)
                for (int j = 0; j < Q; ++j)
                    for (int t = 0; t < Q; ++t) Dk[i][j] -= Wk[i][t] * Wk[j][t];
        }
        chol_small<Q>(Dk, Lk);
        for (int j = 0; j < Q; ++j)                                  // low[d, j] = L[j + d, j]
            for (int d = 0; d < Q && k * Q + j + d < n; ++d) low[d * n + k * Q + j] = j + d < Q ? Lk[j + d][j] : 0.0;
        for (int j = 1; j < Q && k > 0; ++j)                         // L[kQ + i, (k-1)Q + j], d = q + i - j
            for (int i = 0; i < j && k * Q + i < n; ++i) low[(Q + i - j) * n + (k - 1) * Q + j] = Wk[i][j];
        std::memcpy(Lp, Lk, sizeof(Lk));
    }
}

// (L L^T)^{-1} b in place, L as the lower band low[d * n + j] = L[j+d, j] (d < m)
inline void band_solve(int64_t n, int m, const double* low, double* b) {
    for (int64_t k = 0; k < n; ++k) {
        double v = b[k];
        for (int d = 1; d < m && d <= k; ++d) v -= low[d * n + k - d] * b[k - d];
        b[k] = v / low[k];
    }
    for (int64_t k = n - 1; k >= 0; --k) {
        double v = b[k];
        for (int d = 1; d < m && k + d < n; ++d) v -= low[d * n + k] * b[k + d];
        b[k] = v / low[k];
    }
}

// band Z[j * (W+1) + a] = (L L^T)^{-1}_{j+a, j}, a <= W (Takahashi recursion with a W x W window, W <= 12)
inline void band_selinv(int64_t n, int m, const double* low, int W, double* Z) {
    double Zw[12][12]{}, Zn[12][12];                        // Zw[a-1][b-1] = Z_{j+a, j+b}
    for (int64_t j = n - 1; j >= 0; --j) {
        double dj = low[j], lj[4]{}, col[12];
        for (int d = 1; d < m; ++d) lj[d - 1] = j + d < n ? low[d * n + j] : 0.0;
        for (int a = 0; a < W; ++a) {
            double v = 0.0;
            for (int d = 0; d + 1 < m; ++d) v += Zw[a][d] * lj[d];
            col[a] = -v / dj;
        }
        double zjj = 1.0 / dj;
        for (int d = 0; d + 1 < m; ++d) zjj -= lj[d] * col[d];
        zjj /= dj;
        Zn[0][0] = zjj;
        for (int a = 1; a < W; ++a) Zn[0][a] = Zn[a][0] = col[a - 1];
        for (int a = 1; a < W; ++a)
            for (int b = 1; b < W; ++b) Zn[a][b] = Zw[a - 1][b - 1];
        std::memcpy(Zw, Zn, sizeof(Zw));
        Z[j * (W + 1)] = zjj;
        for (int a = 1; a <= W; ++a) Z[j * (W + 1) + a] = j + a < n ? col[a - 1] : 0.0;
    }
}

// one-sided rows: C[k, l] = coefficient of x_{k-q+l} (l < q, zero for k-q+l < 0), coefficient of x_k = 1;
// rows k >= q: q conditions t^m e^{-ct}, m < q; rows k < q: k conditions.
template <int Q>
inline void onesided_row(const double* X, double c, int64_t k, double* Ck) {
    int kk = static_cast<int>(std::min<int64_t>(k, Q));
    for (int l = 0; l < Q; ++l) Ck[l] = 0.0;
    if (kk == 0) return;
    double M[Q][Q], rhs[Q][1];
    for (int m = 0; m < kk; ++m) {
        double row[Q + 1], rmax = 0.0;
        for (int l = 0; l <= kk; ++l) {
            double t = X[k - kk + l] - X[k];
            row[l] = std::pow(t, m) * std::exp(-c * t);
            rmax = std::max(rmax, std::abs(row[l]));
        }
        for (int l = 0; l < kk; ++l) M[m][l] = row[l] / rmax;
        rhs[m][0] = -row[kk] / rmax;
    }
    uint8_t pv[Q]{};
    if (kk == Q) {
        lu_small<Q>(M, pv);
        lu_solve_small<Q, 1>(M, pv, rhs);
    } else if (kk == 1) {
        rhs[0][0] /= M[0][0];
    } else if constexpr (Q == 3) {                 // kk == 2
        double M2[2][2] = {{M[0][0], M[0][1]}, {M[1][0], M[1][1]}}, r2[2][1] = {{rhs[0][0]}, {rhs[1][0]}};
        lu_small<2>(M2, pv);
        lu_solve_small<2, 1>(M2, pv, r2);
        rhs[0][0] = r2[0][0], rhs[1][0] = r2[1][0];
    }
    for (int l = 0; l < kk; ++l) Ck[Q - kk + l] = rhs[l][0];
}

// unfactored one-sided C (n, Q) and the lower band B[d * n + j] = B[j+d, j], d < Q (the likelihood)
template <int Q>
std::vector<double> onesided_B(int64_t n, const double* X, double c, double* C) {
    parallel_for(n, num_workers(n, 4096), [&](int64_t lo, int64_t hi, int) {
        for (int64_t k = lo; k < hi; ++k) onesided_row<Q>(X, c, k, C + k * Q);
    });
    // B[j+d, j] = int g_{j+d} g_j ds, g_k(s) = sum_l C_kl h(x_l - s) 1{s < x_l}
    double sig2 = std::pow(2.0 * c, 2 * Q - 1) / FACT[2 * Q - 2];
    double sig = std::sqrt(sig2);
    std::vector<double> B = interval_band<Q>(n, [&](int64_t iv, double (&P)[Q][Q]) {
        double a = X[iv], bb = X[iv + 1], half = 0.5 * (bb - a), mid = 0.5 * (a + bb);
        int o = gl_order(16 + std::ceil(4.0 * c * (bb - a)));
        for (int p = 0; p < GL_ORDERS[o]; ++p) {
            double s = half * g_gl_x[o][p] + mid, g[Q]{};
            for (int r = 0; r < Q; ++r) {
                int64_t k = iv + 1 + r;
                if (k >= n) continue;
                // side with fewer nodes (Section 7.2): left form needs all q moment conditions (k >= q)
                bool left = k >= Q && 2 * (Q - r) < Q + 1;
                const double* Ck = C + k * Q;
                for (int l = left ? 0 : Q - r; l < (left ? Q - r : Q + 1); ++l) {
                    int64_t idx = k - Q + l;
                    if (idx < 0) continue;
                    double tt = X[idx] - s;
                    double h = sig * (Q == 1 ? 1.0 : (Q == 2 ? tt : tt * tt)) * std::exp(-c * tt);
                    g[r] += (left ? -h : h) * (l < Q ? Ck[l] : 1.0);
                }
            }
            double wp = half * g_gl_w[o][p];
            for (int r = 0; r < Q; ++r)
                for (int rp = 0; rp <= r; ++rp) P[r][rp] += wp * g[r] * g[rp];
        }
    });
    // (-inf, x_1] for rows k, j < q:  g_k = sig e^{-cu} sum_r binom(q-1, r) Mom_kr u^{q-1-r}
    int mb = static_cast<int>(std::min<int64_t>(Q, n));
    double Mom[Q][Q]{}, Wm[Q][Q];
    auto binom = [](int a, int b) { return FACT[a] / (FACT[b] * FACT[a - b]); };
    for (int k = 0; k < mb; ++k)
        for (int r = k; r < Q; ++r)
            for (int idx = 0; idx <= k; ++idx) {
                double d = X[idx] - X[0];
                Mom[k][r] += (idx < k ? C[k * Q + Q - k + idx] : 1.0) * std::pow(d, r) * std::exp(-c * d);
            }
    for (int r = 0; r < Q; ++r)
        for (int rr = 0; rr < Q; ++rr)
            Wm[r][rr] = binom(Q - 1, r) * binom(Q - 1, rr) * FACT[2 * Q - 2 - r - rr] /
                        std::pow(2.0 * c, 2 * Q - 1 - r - rr);
    for (int k = 0; k < mb; ++k)
        for (int j = 0; j <= k; ++j) {
            double v = 0.0;
            for (int r = 0; r < Q; ++r)
                for (int rr = 0; rr < Q; ++rr) v += Mom[k][r] * Wm[r][rr] * Mom[j][rr];
            B[(k - j) * n + j] += sig2 * v;
        }
    return B;
}

// unfactored C (n, Q) and L_B = chol(B) as the lower band low[d * n + j] = L_B[j+d, j] (Q, n): the likelihood's factors
template <int Q>
void onesided_setup(int64_t n, const double* X, double c, double* C, double* low) {
    band_chol<Q>(n, onesided_B<Q>(n, X, c, C), low);
}

// ----------------------------------------------------------------------------- factored one-sided KP whitening
// (as kernel_packets.py) C~ = L_q ... L_1, F[k, 0] = e^{-c(x_k - x_{k-1})}, F[k, j] = c (x_k - x_{k-j}) (j < q);
// g~_k(s) = sig e^{-c(x_k - s)} N_{k-q,q}(s) / c^{q-1} (k >= q),
//          sig e^{-c(x_k - s)} [x_0..x_k](. - s)_+^{q-1} / c^k (k < q).

// N_{0,m}(s) on knots t[0..m] (Cox-de Boor, convex combinations)
inline double bspline(const double* t, double s, int m) {
    double N[4];
    for (int l = 0; l < m; ++l) N[l] = (t[l] <= s && s < t[l + 1]) ? 1.0 : 0.0;
    for (int p = 2; p <= m; ++p)
        for (int l = 0; l + p <= m; ++l)
            N[l] = (s - t[l]) / (t[l + p - 1] - t[l]) * N[l] + (t[l + p] - s) / (t[l + p] - t[l + 1]) * N[l + 1];
    return N[0];
}

// [t_0..t_k](. - s)_+^p (p >= k - 1): D(j, p) = (t_j - s) D(j, p-1) + D(j-1, p-1) for s < t_j (else 0),
// D(0, p) = (t_0 - s)_+^p, D(j, j-1) = N_{0,j} / (t_j - t_0); non-negative terms only
inline double trunc_dd(const double* t, double s, int k, int p) {
    double D[4][4]{};
    for (int pp = 0; pp <= p; ++pp)
        for (int j = 0; j <= std::min(k, pp + 1); ++j) {
            if (j == 0) D[0][pp] = t[0] > s ? std::pow(t[0] - s, pp) : 0.0;
            else if (pp == j - 1) D[j][pp] = bspline(t, s, j) / (t[j] - t[0]);
            else D[j][pp] = s < t[j] ? (t[j] - s) * D[j][pp - 1] + D[j - 1][pp - 1] : 0.0;
        }
    return D[k][p];
}

template <int Q>
inline double g_factored(int64_t k, const double* X, double c, double sig, double s) {
    double ex = std::exp(-c * (X[k] - s));
    if (k >= Q) return sig / std::pow(c, Q - 1) * ex * bspline(X + k - Q, s, Q);
    return sig / std::pow(c, static_cast<int>(k)) * ex * trunc_dd(X, s, static_cast<int>(k), Q - 1);
}

// F (n, Q) and the lower band of B~ (Q, n)
template <int Q>
std::vector<double> factored_B(int64_t n, const double* X, double c, double* F) {
    for (int64_t k = 0; k < n; ++k) {
        F[k * Q] = k ? std::exp(-c * (X[k] - X[k - 1])) : 0.0;
        for (int j = 1; j < Q; ++j) F[k * Q + j] = k >= j ? c * (X[k] - X[k - j]) : 0.0;
    }
    double sig = std::sqrt(std::pow(2.0 * c, 2 * Q - 1) / FACT[2 * Q - 2]);
    std::vector<double> B = interval_band<Q>(n, [&](int64_t iv, double (&P)[Q][Q]) {
        double a = X[iv], bb = X[iv + 1], half = 0.5 * (bb - a), mid = 0.5 * (a + bb);
        int o = gl_order(6 + 2.0 * c * (bb - a));
        for (int p = 0; p < GL_ORDERS[o]; ++p) {
            double s = half * g_gl_x[o][p] + mid, g[Q]{};
            for (int r = 0; r < Q && iv + 1 + r < n; ++r) g[r] = g_factored<Q>(iv + 1 + r, X, c, sig, s);
            double wp = half * g_gl_w[o][p];
            for (int r = 0; r < Q; ++r)
                for (int rp = 0; rp <= r; ++rp) P[r][rp] += wp * g[r] * g[rp];
        }
    });
    // (-inf, x_0]: s = x_0 - t / (2c), Gauss-Laguerre in t (rows k < q)
    int mb = static_cast<int>(std::min<int64_t>(Q, n));
    for (int p = 0; p <= Q; ++p) {
        double t = g_lag_x[Q - 1][p], s = X[0] - t / (2.0 * c), g[Q]{};
        for (int k = 0; k < mb; ++k)
            g[k] = sig / std::pow(c, k) * std::exp(-c * (X[k] - X[0])) * trunc_dd(X, s, k, Q - 1);
        for (int k = 0; k < mb; ++k)
            for (int j = 0; j <= k; ++j) B[(k - j) * n + j] += g_lag_w[Q - 1][p] / (2.0 * c) * g[k] * g[j];
    }
    return B;
}

// F (n, Q) and L_B~ = chol(B~) as the lower band low[d * n + j] (Q, n): the prior sampler's factors
template <int Q>
void prior_setup(int64_t n, const double* X, double c, double* F, double* low) {
    band_chol<Q>(n, factored_B<Q>(n, X, c, F), low);
}

}  // namespace skp
