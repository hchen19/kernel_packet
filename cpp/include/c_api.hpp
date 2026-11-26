// C interface of the core (Python ctypes, Julia ccall, the MATLAB MEX gateway, C++ alone). Matrices are columns of
// contiguous values (column-major); handles are opaque pointers; indices are 0-based.
#pragma once
#include <cstdint>

#define KP_VERSION "1.0.0"

extern "C" {
// kernel_packets.cpp: 1-D kernel packets on sorted nodes x, noise ratio tau2; method 0: stable (k <= 7, n >= k),
// 1: original (tau2 = 0; also taken otherwise); kp_method: the one taken
void* kp_new(const double* x, int64_t n, int k, double rho, double tau2, int method);
void kp_free(void* h);
void kp_set_params(void* h, double rho, double tau2);
int kp_method(void* h);
double kp_logdet(void* h);
int kp_width(void* h);                                                            // original only
int64_t kp_bytes(void* h);                                                        // bytes held
void kp_solve(void* h, double* B, int64_t p, int with_a);   // B <- (K + tau2 I)^{-1} B (original, !with_a: Phi^{-1} B)
void kp_basis(void* h, const double* xs, int64_t m, int64_t* start, double* val);  // original; val: m x width

// regression.cpp
void kp_posterior(void* h, const double* xs, int64_t m, const double* y, double* mean, double* var);  // var may be null
void kp_posterior_cov(void* h, const double* xs, int64_t m, double* cov);                           // m x m
void kernel_predict(const double* xs, int64_t m, const double* X, int64_t N, int d, const double* w, int64_t p,
                    int k, const double* rho, double* out);   // out (m x p) = K(xs, X) w; xs, X row-major points

// sampling.cpp
void kp_sample(void* h, double* Z, int64_t p);                                     // Z <- C~^{-1} L_B Z (stable) or Phi L^{-T} Z (original)
void kp_posterior_sample(void* h, const double* xs, int64_t m, const double* y, const double* Z, int64_t p,
                         double* out);   // Z: (n + m) x p normals (2n + m with noise), out: m x p

// sparse_grid.cpp
void* sg_new(int d, int eta, const double* nodes, const int64_t* noff, int k, const double* rho);
void sg_free(void* h);
int64_t sg_size(void* h);
void sg_points(void* h, double* X);                                                // N x d, row-major
void sg_set_rho(void* h, const double* rho);
void sg_solve(void* h, const double* Y, double* W, int64_t p);
double sg_logdet(void* h);
int64_t sg_bytes(void* h);
double sg_trace_solve(void* h, const double* G);                               // tr(K^{-1} G), G: N x N

// likelihood.cpp: profile log-likelihood at the model's rho, beta and sigma2 at their MLEs (F: N x pf)
double kp_loglike(void* h, const double* y, const double* F, int pf, double* beta, double* sigma2);
double sg_loglike(void* h, const double* y, const double* F, int pf, double* beta, double* sigma2);

// cuda/: the same operations on device memory (pointers to GPU arrays), on the caller's CUDA stream
void* sg_cuda_new(void* h);                                                       // device copy of a sparse grid
void sg_cuda_free(void* hd);
void sg_cuda_solve(void* hd, const double* Y, double* W, int64_t p, void* stream);
double sg_cuda_trace_solve(void* hd, const double* G, void* stream);
void kernel_predict_cuda(const double* xs, int64_t m, const double* X, int64_t N, int d, const double* w, int64_t p,
                         int k, const double* rho, double* out, void* stream);   // rho: host pointer
}
