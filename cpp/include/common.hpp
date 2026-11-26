// Shared by the C++ core: threads over index ranges and the Matern-(q-1/2) kernels.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

namespace kp {

// fn(lo, hi) on contiguous ranges of [0, total), one std::thread per range of at least `grain` items
// Worker threads: KP_NUM_THREADS (set by scripts/ to the CPUs of the job), else the hardware threads
inline int num_threads() {
    static const int n = [] {
        const char* env = std::getenv("KP_NUM_THREADS");
        return env ? std::max(1, std::atoi(env)) : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }();
    return n;
}

template <typename Fn>
void parallel_for(int64_t total, int64_t grain, Fn&& fn) {
    const int hw = num_threads();
    int workers = static_cast<int>(std::clamp<int64_t>(total / std::max<int64_t>(grain, 1), 1, hw));
    if (workers == 1) return fn(int64_t(0), total);
    std::vector<std::thread> th;
    th.reserve(workers - 1);
    for (int t = 1; t < workers; ++t)
        th.emplace_back([&fn, total, workers, t] { fn(total * t / workers, total * (t + 1) / workers); });
    fn(int64_t(0), total / workers);
    for (auto& t : th) t.join();
}

// Coefficients of P in K(r) = P(z) e^{-z} (z = c |r|, degree q-1, P(0) = 1), lowest degree first
inline std::vector<double> matern_poly(int q) {
    std::vector<double> P(q, 1.0);
    for (int j = 0, p = q - 1; j < p; ++j) P[j + 1] = P[j] * 2.0 * (p - j) / ((2 * p - j) * (j + 1.0));
    return P;
}

inline double horner(const std::vector<double>& P, double z) {
    double s = 0.0;
    for (int j = static_cast<int>(P.size()) - 1; j >= 0; --j) s = s * z + P[j];
    return s;
}

// 1-D kernel K(r) = P(z) e^{-z}, z = c |r|, c = sqrt(2q-1) / rho
struct Matern {
    double c;
    std::vector<double> P;
    Matern(int q, double rho) : c(std::sqrt(2.0 * q - 1.0) / rho), P(matern_poly(q)) {}
    double operator()(double r) const {
        double z = c * std::abs(r);
        return horner(P, z) * std::exp(-z);
    }
};

// Product kernel on R^d: prod_i P(z_i) e^{-sum_i z_i}, z_i = c_i |a_i - b_i|
struct Kernel {
    int d;
    std::vector<double> c, P;
    Kernel(int q, const double* rho, int d) : d(d), c(d), P(matern_poly(q)) {
        for (int i = 0; i < d; ++i) c[i] = std::sqrt(2.0 * q - 1.0) / rho[i];
    }
    double operator()(const double* a, const double* b) const {
        double prod = 1.0, sum = 0.0;
        for (int i = 0; i < d; ++i) {
            double z = c[i] * std::abs(a[i] - b[i]);
            prod *= horner(P, z);
            sum += z;
        }
        return prod * std::exp(-sum);
    }
};

}  // namespace kp
