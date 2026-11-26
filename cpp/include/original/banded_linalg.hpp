// Band matrices: LU with partial pivoting and solves (log|det| for the likelihood), Cholesky, products.
#pragma once
#include "../common.hpp"

namespace kp {

// n x n band matrix of half-bandwidth b; LU with partial pivoting (as LAPACK gbtrf), row r keeps columns r-b .. r+2b
struct Band {
    int64_t n = 0;
    int b = 0, W = 1;
    std::vector<double> v;
    std::vector<int64_t> piv;
    Band() = default;
    Band(int64_t n, int b) : n(n), b(b), W(3 * b + 1), v(n * W, 0.0), piv(n) {}
    double& at(int64_t r, int64_t c) { return v[r * W + (c - r + b)]; }
    double at(int64_t r, int64_t c) const { return v[r * W + (c - r + b)]; }

    double factor() {  // returns log|det|
        double ld = 0.0;
        for (int64_t k = 0; k < n; ++k) {
            int64_t re = std::min(n, k + b + 1), ce = std::min(n, k + 2 * b + 1), p = k;
            for (int64_t r = k + 1; r < re; ++r)
                if (std::abs(at(r, k)) > std::abs(at(p, k))) p = r;
            piv[k] = p;
            if (p != k)
                for (int64_t c = k; c < ce; ++c) std::swap(at(k, c), at(p, c));
            double dk = at(k, k);
            ld += std::log(std::abs(dk));
            for (int64_t r = k + 1; r < re; ++r) {
                double l = at(r, k) /= dk;
                for (int64_t c = k + 1; c < ce; ++c) at(r, c) -= l * at(k, c);
            }
        }
        return ld;
    }

    void solve(double* y) const {
        for (int64_t k = 0; k < n; ++k) {
            if (piv[k] != k) std::swap(y[k], y[piv[k]]);
            for (int64_t r = k + 1, re = std::min(n, k + b + 1); r < re; ++r) y[r] -= at(r, k) * y[k];
        }
        for (int64_t k = n - 1; k >= 0; --k) {
            double s = y[k];
            for (int64_t c = k + 1, ce = std::min(n, k + 2 * b + 1); c < ce; ++c) s -= at(k, c) * y[c];
            y[k] = s / at(k, k);
        }
    }

    void cholesky() {  // symmetric positive definite: lower factor L in at(r, c), r - b <= c <= r
        for (int64_t j = 0; j < n; ++j) {
            double d = at(j, j);
            for (int64_t c = std::max<int64_t>(0, j - b); c < j; ++c) d -= at(j, c) * at(j, c);
            at(j, j) = std::sqrt(d);
            for (int64_t r = j + 1, re = std::min(n, j + b + 1); r < re; ++r) {
                double s = at(r, j);
                for (int64_t c = std::max<int64_t>(0, r - b); c < j; ++c) s -= at(r, c) * at(j, c);
                at(r, j) = s / at(j, j);
            }
        }
    }

    void ltsolve(double* z) const {  // z <- L^{-T} z after cholesky()
        for (int64_t r = n - 1; r >= 0; --r) {
            double s = z[r];
            for (int64_t c = r + 1, ce = std::min(n, r + b + 1); c < ce; ++c) s -= at(c, r) * z[c];
            z[r] = s / at(r, r);
        }
    }

    void mul(const double* z, double* y) const {  // y = M z before factoring
        for (int64_t r = 0; r < n; ++r) {
            double s = 0.0;
            for (int64_t c = std::max<int64_t>(0, r - b), ce = std::min(n, r + b + 1); c < ce; ++c) s += at(r, c) * z[c];
            y[r] = s;
        }
    }
};

}  // namespace kp
