// Small dense LU / Cholesky and the block-tridiagonal solve of Phi_tau.
#pragma once
#include "common.hpp"

namespace skp {

// ----------------------------------------------------------------------------- small dense linear algebra
template <int M>
inline void lu_small(double A[M][M], uint8_t piv[M]) {
    for (int k = 0; k < M - 1; ++k) {
        int p = k;
        double mx = std::abs(A[k][k]);
        for (int i = k + 1; i < M; ++i)
            if (std::abs(A[i][k]) > mx) {
                mx = std::abs(A[i][k]);
                p = i;
            }
        piv[k] = static_cast<uint8_t>(p);
        if (p != k)
            for (int j = 0; j < M; ++j) std::swap(A[k][j], A[p][j]);
        for (int i = k + 1; i < M; ++i) {
            double lk = A[i][k] / A[k][k];
            A[i][k] = lk;
            for (int j = k + 1; j < M; ++j) A[i][j] -= lk * A[k][j];
        }
    }
}

template <int M, int R>
inline void lu_solve_small(const double LU[M][M], const uint8_t piv[M], double B[M][R]) {
    for (int k = 0; k < M - 1; ++k)
        if (piv[k] != k)
            for (int r = 0; r < R; ++r) std::swap(B[k][r], B[piv[k]][r]);
    for (int k = 0; k < M - 1; ++k)
        for (int i = k + 1; i < M; ++i)
            for (int r = 0; r < R; ++r) B[i][r] -= LU[i][k] * B[k][r];
    for (int k = M - 1; k >= 0; --k) {
        for (int j = k + 1; j < M; ++j)
            for (int r = 0; r < R; ++r) B[k][r] -= LU[k][j] * B[j][r];
        for (int r = 0; r < R; ++r) B[k][r] /= LU[k][k];
    }
}

template <int M, int R>
inline void lu_solve_T_small(const double LU[M][M], const uint8_t piv[M], double B[M][R]) {
    for (int k = 0; k < M; ++k) {
        for (int j = 0; j < k; ++j)
            for (int r = 0; r < R; ++r) B[k][r] -= LU[j][k] * B[j][r];
        for (int r = 0; r < R; ++r) B[k][r] /= LU[k][k];
    }
    for (int k = M - 2; k >= 0; --k)
        for (int j = k + 1; j < M; ++j)
            for (int r = 0; r < R; ++r) B[k][r] -= LU[j][k] * B[j][r];
    for (int k = M - 2; k >= 0; --k)
        if (piv[k] != k)
            for (int r = 0; r < R; ++r) std::swap(B[k][r], B[piv[k]][r]);
}

template <int M>
inline void inverse_small(double A[M][M], double Z[M][M]) {
    uint8_t pv[M]{};
    lu_small<M>(A, pv);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j) Z[i][j] = i == j;
    lu_solve_small<M, M>(A, pv, Z);
}

template <int M>
inline void chol_small(const double A[M][M], double L[M][M]) {
    for (int j = 0; j < M; ++j) {
        double s = A[j][j];
        for (int k = 0; k < j; ++k) s -= L[j][k] * L[j][k];
        L[j][j] = std::sqrt(s);
        for (int i = j + 1; i < M; ++i) {
            double v = A[i][j];
            for (int k = 0; k < j; ++k) v -= L[i][k] * L[j][k];
            L[i][j] = v / L[j][j];
        }
        for (int i = 0; i < j; ++i) L[i][j] = 0.0;
    }
}

// ----------------------------------------------------------------------------- block-tridiagonal solve
// Blocks of size M = 2q (last block padded with the identity); E_k = M_{k+1,k}, F_k = M_{k,k+1} non-zero in
// their H x H corners. lu/piv: LU of the Schur complements S_k = D_k - E_{k-1} S_{k-1}^{-1} F_{k-1}.
template <int M, int W>
void solve_cols(int64_t n, int64_t K, int64_t c0, int w, const double* lu, const uint8_t* piv,
                const double* Eh, const double* Fh, const double* b, double* out) {
    constexpr int H = M / 2;
    int64_t nb = (n + M - 1) / M;
    auto load = [&](int64_t k, double Y[M][W]) {
        for (int i = 0; i < M; ++i) {
            int64_t g = k * M + i;
            for (int r = 0; r < W; ++r) Y[i][r] = (g < n && r < w) ? b[g * K + c0 + r] : 0.0;
        }
    };
    auto store = [&](int64_t k, const double Y[M][W]) {
        for (int i = 0; i < M && k * M + i < n; ++i)
            for (int r = 0; r < w; ++r) out[(k * M + i) * K + c0 + r] = Y[i][r];
    };
    auto LU = [&](int64_t k) { return reinterpret_cast<const double(*)[M]>(lu + k * M * M); };
    double Y[M][W], U[M][W];
    load(0, Y);
    for (int64_t k = 0; k + 1 < nb; ++k) {        // forward: y_{k+1} = b_{k+1} - E_k S_k^{-1} y_k, y_k kept in out
        store(k, Y);
        std::memcpy(U, Y, sizeof(U));
        lu_solve_small<M, W>(LU(k), piv + k * M, U);
        load(k + 1, Y);
        const double* E = Eh + k * H * H;
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < H; ++j)
                for (int r = 0; r < W; ++r) Y[i][r] -= E[i * H + j] * U[M - H + j][r];
    }
    lu_solve_small<M, W>(LU(nb - 1), piv + (nb - 1) * M, Y);     // backward: x_k = S_k^{-1} (y_k - F_k x_{k+1})
    store(nb - 1, Y);
    for (int64_t k = nb - 2; k >= 0; --k) {
        for (int i = 0; i < M; ++i)
            for (int r = 0; r < W; ++r) U[i][r] = r < w ? out[(k * M + i) * K + c0 + r] : 0.0;
        const double* F = Fh + k * H * H;
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < H; ++j)
                for (int r = 0; r < W; ++r) U[M - H + i][r] -= F[i * H + j] * Y[j][r];
        lu_solve_small<M, W>(LU(k), piv + k * M, U);
        std::memcpy(Y, U, sizeof(Y));
        store(k, Y);
    }
}

// M^T x = b: U^T y = b (y_{k+1} = S_{k+1}^{-T} (b_{k+1} - F_k^T y_k)), then L^T x = y (x_k = y_k - S_k^{-T} E_k^T x_{k+1})
template <int M, int W>
void solve_cols_T(int64_t n, int64_t K, int64_t c0, int w, const double* lu, const uint8_t* piv,
                  const double* Eh, const double* Fh, const double* b, double* out) {
    constexpr int H = M / 2;
    int64_t nb = (n + M - 1) / M;
    auto load = [&](int64_t k, double Y[M][W]) {
        for (int i = 0; i < M; ++i) {
            int64_t g = k * M + i;
            for (int r = 0; r < W; ++r) Y[i][r] = (g < n && r < w) ? b[g * K + c0 + r] : 0.0;
        }
    };
    auto store = [&](int64_t k, const double Y[M][W]) {
        for (int i = 0; i < M && k * M + i < n; ++i)
            for (int r = 0; r < w; ++r) out[(k * M + i) * K + c0 + r] = Y[i][r];
    };
    auto LU = [&](int64_t k) { return reinterpret_cast<const double(*)[M]>(lu + k * M * M); };
    double Y[M][W], U[M][W];
    load(0, Y);
    lu_solve_T_small<M, W>(LU(0), piv, Y);
    store(0, Y);
    for (int64_t k = 0; k + 1 < nb; ++k) {          // forward, y_k kept in out
        load(k + 1, U);
        const double* F = Fh + k * H * H;
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < H; ++j)
                for (int r = 0; r < W; ++r) U[j][r] -= F[i * H + j] * Y[M - H + i][r];
        lu_solve_T_small<M, W>(LU(k + 1), piv + (k + 1) * M, U);
        std::memcpy(Y, U, sizeof(Y));
        store(k + 1, Y);
    }
    for (int64_t k = nb - 2; k >= 0; --k) {         // backward, Y holds x_{k+1}
        double T[M][W]{};
        const double* E = Eh + k * H * H;
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < H; ++j)
                for (int r = 0; r < W; ++r) T[M - H + j][r] += E[i * H + j] * Y[i][r];
        lu_solve_T_small<M, W>(LU(k), piv + k * M, T);
        for (int i = 0; i < M; ++i)
            for (int r = 0; r < W; ++r) Y[i][r] = (r < w ? out[(k * M + i) * K + c0 + r] : 0.0) - T[i][r];
        store(k, Y);
    }
}

template <int M>
void block_solve(int64_t n, int64_t K, const double* lu, const uint8_t* piv, const double* Eh, const double* Fh,
                 const double* b, double* out, bool T = false) {
    if (K == 1)
        return T ? solve_cols_T<M, 1>(n, 1, 0, 1, lu, piv, Eh, Fh, b, out)
                 : solve_cols<M, 1>(n, 1, 0, 1, lu, piv, Eh, Fh, b, out);
    constexpr int W = 4;
    int64_t nc = (K + W - 1) / W;
    parallel_for(nc, num_workers(nc * n, 1 << 16), [&](int64_t lo, int64_t hi, int) {
        for (int64_t ch = lo; ch < hi; ++ch) {
            int w = static_cast<int>(std::min<int64_t>(W, K - ch * W));
            T ? solve_cols_T<M, W>(n, K, ch * W, w, lu, piv, Eh, Fh, b, out)
              : solve_cols<M, W>(n, K, ch * W, w, lu, piv, Eh, Fh, b, out);
        }
    });
}

}  // namespace skp
