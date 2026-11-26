#pragma once
// Prior sampling (as sampling.py): F and L_B~ from the factored one-sided KP whitening of kernel_packets.hpp;
// y = C~^{-1} L_B~ z and its adjoint z = L_B~^T C~^{-T} y, C~ = L_q ... L_1.
#include "kernel_packets.hpp"

namespace skp {

// y = C~^{-1} L z on W columns c0..c0+w-1 of z (n, K): per k, e = (L z)_k, then
// u_{j-1,k} = s_kj u_{j,k} + e_k u_{j-1,k-1} (k >= j, else u_{j,k}) for j = q..1, u_{q,k} = e, y_k = u_{0,k}
template <int Q, int W>
void draw_cols(int64_t n, int64_t K, int64_t c0, int w, const double* F, const double* low, const double* Z,
               double* Y) {
    double u[Q][W]{};                              // u_{j,k-1}, j < q
    for (int64_t k = 0; k < n; ++k) {
        double top[W];
        for (int r = 0; r < W; ++r) top[r] = r < w ? low[k] * Z[k * K + c0 + r] : 0.0;
        for (int d = 1; d < Q && d <= k; ++d)
            for (int r = 0; r < w; ++r) top[r] += low[d * n + k - d] * Z[(k - d) * K + c0 + r];
        const double* Fk = F + k * Q;
        for (int j = Q; j >= 1; --j) {
            if (k >= j) {
                double sc = j < Q ? Fk[j] : 1.0;
                for (int r = 0; r < W; ++r) top[r] = sc * top[r] + Fk[0] * u[j - 1][r];
            }
            for (int r = 0; r < W; ++r) u[j - 1][r] = top[r];
        }
        for (int r = 0; r < w; ++r) Y[k * K + c0 + r] = top[r];
    }
}

// adjoint z = L^T C~^{-T} y: stage j = 1..q, w_{j,k} = v_k + e_{k+1} w_{j,k+1} (k+1 >= j), v <- s_kj w_{j,k} (k >= j);
// u = the last stage, z_k = sum_d L[k+d, k] u_{k+d}
template <int Q, int W>
void draw_cols_T(int64_t n, int64_t K, int64_t c0, int w, const double* F, const double* low, const double* Yin,
                 double* Z) {
    double wj[Q][W]{};                             // w_{j,k+1}
    double win[Q][W]{};                            // win[d] = u_{k+d}
    for (int64_t k = n - 1; k >= 0; --k) {
        double v[W];
        for (int r = 0; r < W; ++r) v[r] = r < w ? Yin[k * K + c0 + r] : 0.0;
        double e1 = k + 1 < n ? F[(k + 1) * Q] : 0.0;
        for (int j = 1; j <= Q; ++j) {
            double ej = k + 1 >= j ? e1 : 0.0, sc = k >= j && j < Q ? F[k * Q + j] : 1.0;
            for (int r = 0; r < W; ++r) {
                wj[j - 1][r] = v[r] + ej * wj[j - 1][r];
                v[r] = sc * wj[j - 1][r];
            }
        }
        for (int d = Q - 1; d > 0; --d)
            for (int r = 0; r < W; ++r) win[d][r] = win[d - 1][r];
        for (int r = 0; r < W; ++r) win[0][r] = v[r];
        for (int r = 0; r < w; ++r) {
            double z = low[k] * win[0][r];
            for (int d = 1; d < Q && k + d < n; ++d) z += low[d * n + k] * win[d][r];
            Z[k * K + c0 + r] = z;
        }
    }
}

template <int Q>
void prior_draw(int64_t n, int64_t K, const double* F, const double* low, const double* Z, double* Y, bool T) {
    if (K == 1) return T ? draw_cols_T<Q, 1>(n, 1, 0, 1, F, low, Z, Y) : draw_cols<Q, 1>(n, 1, 0, 1, F, low, Z, Y);
    constexpr int W = 4;
    int64_t nc = (K + W - 1) / W;
    parallel_for(nc, num_workers(nc * n, 1 << 16), [&](int64_t lo, int64_t hi, int) {
        for (int64_t ch = lo; ch < hi; ++ch) {
            int w = static_cast<int>(std::min<int64_t>(W, K - ch * W));
            T ? draw_cols_T<Q, W>(n, K, ch * W, w, F, low, Z, Y) : draw_cols<Q, W>(n, K, ch * W, w, F, low, Z, Y);
        }
    });
}

}  // namespace skp
