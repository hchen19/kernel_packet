// Likelihood (likelihood.py): profile log-likelihood -(N log sigma2 + log|K| + N) / 2 with
// beta = (F^T K^{-1} F)^{-1} F^T K^{-1} y and sigma2 = (y - F beta)^T K^{-1} (y - F beta) / N, from the Gram matrix
// S = [F y]^T K^{-1} [F y]: one solve of [F y] (original KPs, sparse grids), or the Givens rotations of the stable KPs.
#include "sparse_grid.hpp"
#include "kp_model.hpp"

#include "c_api.hpp"

namespace {

// S: (pf + 1) x (pf + 1), the last row and column y
double profile(int64_t N, std::vector<double> S, int pf, double logdet, double* beta, double* sigma2) {
    const int K = pf + 1;
    std::vector<double> G(S);                // Gauss-Jordan with partial pivoting on [S_FF | S_Fy]
    for (int c = 0; c < pf; ++c) {
        int p = c;
        for (int r = c + 1; r < pf; ++r)
            if (std::abs(G[r * K + c]) > std::abs(G[p * K + c])) p = r;
        for (int j = 0; j < K; ++j) std::swap(G[c * K + j], G[p * K + j]);
        for (int r = 0; r < pf; ++r) {
            if (r == c) continue;
            double l = G[r * K + c] / G[c * K + c];
            for (int j = c; j < K; ++j) G[r * K + j] -= l * G[c * K + j];
        }
    }
    double s = S[pf * K + pf];
    for (int a = 0; a < pf; ++a) beta[a] = G[a * K + pf] / G[a * K + a], s -= beta[a] * S[a * K + pf];
    *sigma2 = s / N;
    return -0.5 * (N * std::log(*sigma2) + logdet + N);
}

// [F y] (N x (pf + 1), column-major)
std::vector<double> stack(int64_t N, const double* y, const double* F, int pf) {
    std::vector<double> W(F, F + N * pf);
    W.insert(W.end(), y, y + N);
    return W;
}

// S = W^T V for the columns of W (N x K) and V = K^{-1} W
std::vector<double> inner(int64_t N, int K, const std::vector<double>& W, const std::vector<double>& V) {
    std::vector<double> S(K * K, 0.0);
    for (int a = 0; a < K; ++a)
        for (int b = 0; b < K; ++b)
            for (int64_t r = 0; r < N; ++r) S[a * K + b] += W[r + N * a] * V[r + N * b];
    return S;
}

}  // namespace

extern "C" {
double kp_loglike(void* h, const double* y, const double* F, int pf, double* beta, double* sigma2) {
    const auto& m = *static_cast<kp::Model*>(h);
    std::vector<double> W = stack(m.stable ? m.stable->n : m.original->n, y, F, pf), S((pf + 1) * (pf + 1));
    if (m.stable) {
        double ld = m.stable->gram(W.data(), pf + 1, S.data());
        return profile(m.stable->n, S, pf, ld, beta, sigma2);
    }
    const auto& K = *m.original;
    std::vector<double> V(W);
    K.solve(V.data(), pf + 1, true);
    return profile(K.n, inner(K.n, pf + 1, W, V), pf, K.logdet, beta, sigma2);
}

double sg_loglike(void* h, const double* y, const double* F, int pf, double* beta, double* sigma2) {
    const auto& g = *static_cast<kp::SparseGrid*>(h);
    std::vector<double> W = stack(g.N, y, F, pf), V(W.size());
    g.solve(W.data(), V.data(), pf + 1);
    return profile(g.N, inner(g.N, pf + 1, W, V), pf, g.logdet(), beta, sigma2);
}
}
