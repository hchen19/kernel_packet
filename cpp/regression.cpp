// Regression (regression.py): the 1-D posterior of the stable KPs (stable_kp.hpp) or, original, through the KP basis, phi(x)^T Phi^{-1} y and
// K(x, x') - phi(x)^T Phi^{-1} K(X, x') (O(n) memory per thread, no n x m matrix), and the kernel sum K(xs, X) w
// evaluated on the fly (no m x N matrix).
#include "kp_model.hpp"

#include "c_api.hpp"

using kp::KernelPacket;
using kp::Model;

extern "C" {
void kp_posterior(void* h, const double* xs, int64_t m, const double* y, double* mean, double* var) {
    const Model& md = *static_cast<Model*>(h);
    if (md.stable) return md.stable->posterior(xs, m, y, mean, var);
    const KernelPacket& K = *md.original;
    int64_t n = K.n, w = K.w;
    std::vector<int64_t> start(m);
    std::vector<double> V(m * w), u(y, y + n);
    K.basis(xs, m, start.data(), V.data());
    K.phi.solve(u.data());
    kp::parallel_for(m, var ? 16 : 4096, [&](int64_t j0, int64_t j1) {
        std::vector<double> g(n);
        for (int64_t j = j0; j < j1; ++j) {
            const double* v = V.data() + j * w;
            double s = 0.0;
            for (int t = 0; t < w; ++t) s += v[t] * u[start[j] + t];
            mean[j] = s;
            if (!var) continue;
            for (int64_t r = 0; r < n; ++r) g[r] = K.ker(K.x[r] - xs[j]);
            K.phi.solve(g.data());
            s = 0.0;
            for (int t = 0; t < w; ++t) s += v[t] * g[start[j] + t];
            var[j] = 1.0 - s;
        }
    });
}

void kp_posterior_cov(void* h, const double* xs, int64_t m, double* cov) {
    const Model& md = *static_cast<Model*>(h);
    if (md.stable) return md.stable->posterior_cov(xs, m, cov);
    const KernelPacket& K = *md.original;
    int64_t n = K.n, w = K.w;
    std::vector<int64_t> start(m);
    std::vector<double> V(m * w);
    K.basis(xs, m, start.data(), V.data());
    kp::parallel_for(m, 16, [&](int64_t b0, int64_t b1) {
        std::vector<double> g(n);
        for (int64_t b = b0; b < b1; ++b) {
            for (int64_t r = 0; r < n; ++r) g[r] = K.ker(K.x[r] - xs[b]);
            K.phi.solve(g.data());
            for (int64_t a = 0; a < m; ++a) {
                double s = 0.0;
                for (int t = 0; t < w; ++t) s += V[a * w + t] * g[start[a] + t];
                cov[a + m * b] = K.ker(xs[a] - xs[b]) - s;
            }
        }
    });
}

void kernel_predict(const double* xs, int64_t m, const double* X, int64_t N, int d, const double* w, int64_t p,
                    int k, const double* rho, double* out) {
    kp::Kernel ker((k - 1) / 2, rho, d);
    kp::parallel_for(m, 64, [&](int64_t i0, int64_t i1) {
        std::vector<double> acc(p);
        for (int64_t i = i0; i < i1; ++i) {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int64_t j = 0; j < N; ++j) {
                double kij = ker(xs + i * d, X + j * d);
                for (int64_t c = 0; c < p; ++c) acc[c] += kij * w[j + N * c];
            }
            for (int64_t c = 0; c < p; ++c) out[i + m * c] = acc[c];
        }
    });
}
}
