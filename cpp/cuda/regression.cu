// Kernel sums on the GPU (regression.cpp in CUDA): out = K(xs, X) w on device memory, one thread per (test point,
// column), the kernel evaluated on the fly in the CPU's order over X (no m x N matrix).
#include <cuda_runtime.h>

#include "c_api.hpp"
#include "common.hpp"

namespace {

constexpr int THREADS = 128;
constexpr int MAX_Q = 16, MAX_D = 64;

struct Params {
    int d, q;
    double c[MAX_D], P[MAX_Q];
};

__global__ void predict(const double* xs, int64_t m, const double* X, int64_t N, const double* w, int64_t p, Params K,
                        double* out) {
    int64_t g = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (g >= m * p) return;
    int64_t i = g % m, col = g / m;
    const double* xi = xs + i * K.d;
    double acc = 0.0;
    for (int64_t j = 0; j < N; ++j) {
        double prod = 1.0, sum = 0.0;
        for (int l = 0; l < K.d; ++l) {
            double z = K.c[l] * fabs(xi[l] - X[j * K.d + l]), h = 0.0;
            for (int t = K.q - 1; t >= 0; --t) h = h * z + K.P[t];
            prod *= h, sum += z;
        }
        acc += prod * exp(-sum) * w[j + N * col];
    }
    out[i + m * col] = acc;
}

}  // namespace

extern "C" void kernel_predict_cuda(const double* xs, int64_t m, const double* X, int64_t N, int d, const double* w,
                                    int64_t p, int k, const double* rho, double* out, void* stream) {
    Params K{d, (k - 1) / 2, {}, {}};
    auto P = kp::matern_poly(K.q);
    for (int t = 0; t < K.q; ++t) K.P[t] = P[t];
    for (int l = 0; l < d; ++l) K.c[l] = std::sqrt(2.0 * K.q - 1.0) / rho[l];
    predict<<<static_cast<unsigned>((m * p + THREADS - 1) / THREADS), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        xs, m, X, N, w, p, K, out);
}
