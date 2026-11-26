// 1-D kernel packets (kernel_packets.py): the stable KPs (method 0, stable_kp.hpp; k <= 7, n >= k) or the original
// KernelPacket (method 1, noise-free; also otherwise); factorization, solves, logdet, basis at new points.
#include "kp_model.hpp"

#include "c_api.hpp"

using kp::Model;

namespace {
Model& model(void* h) { return *static_cast<Model*>(h); }
}  // namespace

extern "C" {
void* kp_new(const double* x, int64_t n, int k, double rho, double tau2, int method) {
    auto* m = new Model;
    if (method == 0 && k <= 7 && n > k - 1)
        m->stable = std::make_unique<kp::StableKP>(x, n, k, rho, tau2);
    else
        m->original = std::make_unique<kp::KernelPacket>(x, n, k, rho);
    return m;
}
void kp_free(void* h) { delete static_cast<Model*>(h); }
void kp_set_params(void* h, double rho, double tau2) {
    Model& m = model(h);
    if (m.stable)
        m.stable->set_params(rho, tau2);
    else
        m.original = std::make_unique<kp::KernelPacket>(m.original->x.data(), m.original->n, m.original->k, rho);
}
int kp_method(void* h) { return model(h).stable ? 0 : 1; }
double kp_logdet(void* h) {
    const Model& m = model(h);
    return m.stable ? m.stable->gram(nullptr, 0, nullptr) : m.original->logdet;
}
int kp_width(void* h) { return model(h).original->w; }
int64_t kp_bytes(void* h) { return model(h).stable ? model(h).stable->bytes() : model(h).original->bytes(); }
void kp_solve(void* h, double* B, int64_t p, int with_a) {
    Model& m = model(h);
    if (m.stable)
        m.stable->solve(B, p);
    else
        m.original->solve(B, p, with_a);
}
void kp_basis(void* h, const double* xs, int64_t m, int64_t* start, double* val) {
    model(h).original->basis(xs, m, start, val);
}
}
