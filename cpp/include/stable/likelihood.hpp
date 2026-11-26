#pragma once
// Likelihood (as likelihood.py): K + tau2 I = C^{-1} G C^{-T}, G = L_B L_B^T + tau2 C C^T from the one-sided C and
// L_B of kernel_packets.hpp, factored G = R^T R by Givens rotations of the stacked rows [L_B^T; tau C^T] (G itself is
// never formed: cond(G) ~ cond(K + tau2 I) / sigma_min(C)^2 would lose it), with the forward substitution in one pass.
#include "kernel_packets.hpp"

namespace skp {

// log det G = 2 sum log |R_jj| and the Gram matrix S = U^T U (K x K) of R^T U = C Y for the K columns of Y (n x K,
// column-major); R kept as its active rows j..j+Q (upper band)
template <int Q>
double nll(int64_t n, const double* X, double c, double tau2, const double* Y, int K, double* S) {
    constexpr int M = Q + 1;
    std::vector<double> C(n * Q), low(Q * n);
    onesided_setup<Q>(n, X, c, C.data(), low.data());
    const double tau = std::sqrt(tau2);
    double Rw[M][M]{};                                     // Rw[t][d] = R_{j+t, j+t+d}
    double Rh[Q][M]{};                                     // rows j-1..j-Q of R
    std::vector<double> uh(Q * K, 0.0), u(K);              // u_{j-1..j-Q} of each column
    auto rotate = [&](double* m) {                         // m[d]: column j+d of a stacked row, into rows j..j+Q
        for (int t = 0; t < M; ++t) {
            if (m[0] != 0.0) {
                double r = std::hypot(Rw[t][0], m[0]), cs = Rw[t][0] / r, sn = m[0] / r;
                for (int d = 0; d < M; ++d) {
                    double a = Rw[t][d], b = m[d];
                    Rw[t][d] = cs * a + sn * b;
                    m[d] = cs * b - sn * a;
                }
            }
            for (int d = 0; d + 1 < M; ++d) m[d] = m[d + 1];
            m[M - 1] = 0.0;
        }
    };
    std::fill(S, S + K * K, 0.0);
    double ld = 0.0;
    for (int64_t j = 0; j < n; ++j) {
        double v[M]{};                                     // column j of L_B: L_B[j+d, j]
        for (int d = 0; d < Q && j + d < n; ++d) v[d] = low[d * n + j];
        rotate(v);
        if (tau2 != 0.0) {                                 // column j of tau C: C[j+m, j], coefficient Q - m of row j+m
            double w[M]{};
            for (int m = 0; m < M && j + m < n; ++m) w[m] = tau * (m == 0 ? 1.0 : C[(j + m) * Q + Q - m]);
            rotate(w);
        }
        for (int k = 0; k < K; ++k) {
            const double* y = Y + k * n;
            double e = y[j];                               // (C y)_j
            for (int l = 0; l < Q; ++l)
                if (j - Q + l >= 0) e += C[j * Q + l] * y[j - Q + l];
            for (int s = 1; s <= Q && s <= j; ++s) e -= Rh[s - 1][s] * uh[(s - 1) * K + k];
            u[k] = e / Rw[0][0];
        }
        for (int a = 0; a < K; ++a)
            for (int b = 0; b < K; ++b) S[a * K + b] += u[a] * u[b];
        ld += 2.0 * std::log(std::abs(Rw[0][0]));
        for (int s = Q - 1; s > 0; --s) std::memcpy(Rh[s], Rh[s - 1], sizeof(Rh[s]));
        std::memmove(uh.data() + K, uh.data(), (Q - 1) * K * sizeof(double));
        std::memcpy(Rh[0], Rw[0], sizeof(Rw[0]));
        std::copy(u.begin(), u.end(), uh.begin());
        for (int t = 0; t + 1 < M; ++t) std::memcpy(Rw[t], Rw[t + 1], sizeof(Rw[t]));
        std::fill(Rw[M - 1], Rw[M - 1] + M, 0.0);
    }
    return ld;
}

}  // namespace skp
