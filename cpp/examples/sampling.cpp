// Sampling in C++ alone: 16 prior draws by kernel packets at n = 10^2 .. 10^6 (time) on the jittered unit grid
// (Matern-3/2, rho = 20), and the sample variance of 4096 draws against the prior variance 1.
//   build/example_sampling
#include <chrono>
#include <cstdio>
#include <random>

#include "c_api.hpp"
#include "common.hpp"

int main() {
    std::mt19937_64 rng(0);
    std::uniform_real_distribution<double> U(0, 1);
    std::normal_distribution<double> N;
    std::printf("%10s %12s\n", "n", "time (s)");
    for (int64_t n : {100, 1000, 10000, 100000, 1000000}) {
        std::vector<double> x(n), Z(n * 16);
        for (int64_t i = 0; i < n; ++i) x[i] = i + 0.6 * U(rng) - 0.3;
        for (auto& v : Z) v = N(rng);
        auto t0 = std::chrono::steady_clock::now();
        void* h = kp_new(x.data(), n, 5, 20.0, 0.0, 0);
        kp_sample(h, Z.data(), 16);
        std::printf("%10lld %12.4g\n", static_cast<long long>(n),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        kp_free(h);
    }
    int64_t n = 200, p = 4096;
    std::vector<double> x(n), Z(n * p);
    for (int64_t i = 0; i < n; ++i) x[i] = i + 0.6 * U(rng) - 0.3;
    for (auto& v : Z) v = N(rng);
    void* h = kp_new(x.data(), n, 5, 20.0, 0.0, 0);
    kp_sample(h, Z.data(), p);
    double v = 0;
    for (int64_t c = 0; c < p; ++c) v += Z[n / 2 + n * c] * Z[n / 2 + n * c] / p;
    std::printf("sample variance at x[%lld] from %lld draws: %.4f (prior 1)\n", static_cast<long long>(n / 2),
                static_cast<long long>(p), v);
    kp_free(h);
}
