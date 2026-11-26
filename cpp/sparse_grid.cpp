// Sparse grids (sparse_grid.py): design, K^{-1} by the combination technique, log|K|, tr(K^{-1} G).
#include "sparse_grid.hpp"

#include "c_api.hpp"

using kp::SparseGrid;

extern "C" {
void* sg_new(int d, int eta, const double* nodes, const int64_t* noff, int k, const double* rho) {
    return new SparseGrid(d, eta, nodes, noff, k, rho);
}
void sg_free(void* h) { delete static_cast<SparseGrid*>(h); }
int64_t sg_size(void* h) { return static_cast<SparseGrid*>(h)->N; }
void sg_points(void* h, double* X) {
    const auto& g = *static_cast<SparseGrid*>(h);
    std::copy(g.X.begin(), g.X.end(), X);
}
void sg_set_rho(void* h, const double* rho) { static_cast<SparseGrid*>(h)->set_rho(rho); }
void sg_solve(void* h, const double* Y, double* W, int64_t p) { static_cast<SparseGrid*>(h)->solve(Y, W, p); }
double sg_logdet(void* h) { return static_cast<SparseGrid*>(h)->logdet(); }
int64_t sg_bytes(void* h) { return static_cast<SparseGrid*>(h)->bytes(); }
double sg_trace_solve(void* h, const double* G) { return static_cast<SparseGrid*>(h)->trace_solve(G); }
}
