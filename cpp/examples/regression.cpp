// Regression in C++ alone: the exact 1-D posterior by kernel packets at n = 10^2 .. 10^6 (time, error against the
// truth), on the jittered unit grid of the timing experiments (y = sin(x / 50), Matern-3/2, rho = 20).
//   build/example_regression
#include <chrono>
#include <cstdio>
#include <random>

#include "c_api.hpp"
#include "common.hpp"

int main() {
    std::mt19937_64 rng(0);
    std::uniform_real_distribution<double> U(0, 1);
    std::printf("%10s %12s %12s\n", "n", "time (s)", "RMSE");
    for (int64_t n : {100, 1000, 10000, 100000, 1000000}) {
        std::vector<double> x(n), y(n), xs(1000), mean(1000);
        for (int64_t i = 0; i < n; ++i) x[i] = i + 0.6 * U(rng) - 0.3, y[i] = std::sin(x[i] / 50);
        for (auto& v : xs) v = (n - 1) * U(rng);
        auto t0 = std::chrono::steady_clock::now();
        void* h = kp_new(x.data(), n, 5, 20.0, 0.0, 0);
        kp_posterior(h, xs.data(), xs.size(), y.data(), mean.data(), nullptr);
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), e = 0;
        for (size_t j = 0; j < xs.size(); ++j) e += std::pow(mean[j] - std::sin(xs[j] / 50), 2) / xs.size();
        std::printf("%10lld %12.4g %12.3e\n", static_cast<long long>(n), t, std::sqrt(e));
        kp_free(h);
    }
}
