// Sampling (sampling.py): the stable KPs' samplers (stable_kp.hpp) or, original, prior draws Phi L^{-T} z on the nodes; 1-D posterior draws by conditioning prior draws on
// the union of the nodes and the test points, f(xs) + phi(xs)^T Phi^{-1} (y - f(x)).
#include "kp_model.hpp"

#include "c_api.hpp"

using kp::KernelPacket;
using kp::Model;

extern "C" {
void kp_sample(void* h, double* Z, int64_t p) {
    Model& md = *static_cast<Model*>(h);
    if (md.stable)
        md.stable->sample(Z, p);
    else
        md.original->sample(Z, p);
}

void kp_posterior_sample(void* h, const double* xs, int64_t m, const double* y, const double* Z, int64_t p,
                         double* out) {
    const Model& md = *static_cast<Model*>(h);
    if (md.stable) return md.stable->posterior_sample(xs, m, y, Z, p, out);
    const KernelPacket& K = *md.original;
    int64_t n = K.n;
    std::vector<double> u(K.x);
    u.insert(u.end(), xs, xs + m);
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end()), u.end());
    int64_t nu = u.size();
    auto at = [&](double v) { return std::lower_bound(u.begin(), u.end(), v) - u.begin(); };
    KernelPacket U(u.data(), nu, K.k, K.rho);
    std::vector<double> F(nu * p), R(n * p);
    for (int64_t c = 0; c < p; ++c) std::copy(Z + c * (n + m), Z + c * (n + m) + nu, F.begin() + c * nu);
    U.sample(F.data(), p);
    for (int64_t c = 0; c < p; ++c)
        for (int64_t r = 0; r < n; ++r) R[r + n * c] = y[r] - F[at(K.x[r]) + nu * c];
    K.solve(R.data(), p, false);
    std::vector<int64_t> start(m);
    std::vector<double> V(m * K.w);
    K.basis(xs, m, start.data(), V.data());
    kp::parallel_for(m, 1024, [&](int64_t j0, int64_t j1) {
        for (int64_t j = j0; j < j1; ++j) {
            int64_t ij = at(xs[j]);
            for (int64_t c = 0; c < p; ++c) {
                double s = F[ij + nu * c];
                for (int t = 0; t < K.w; ++t) s += V[j * K.w + t] * R[start[j] + t + n * c];
                out[j + m * c] = s;
            }
        }
    });
}
}
