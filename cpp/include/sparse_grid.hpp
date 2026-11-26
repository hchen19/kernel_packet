// Sparse grid of level eta >= d from nested 1-D designs (the inducing points of d-dimensional regression):
// K^{-1} Y = sum_c coef_c P_c^T (K_{c,1} x ... x K_{c,d})^{-1} P_c Y over the tensor-grid components c of levels
// |j| = max(d, eta-d+1) .. eta, coef_c = (-1)^(eta-|j|) C(d-1, eta-|j|), P_c the rows of component c (first coordinate
// slowest); the 1-D factors of axis i at level j are kernel packets on the level-j nodes (shared by axes of equal
// lengthscale), and log|K| = sum over |j| = d .. eta, axes i of (L_i(j_i) - L_i(j_i - 1)) prod_{l != i} (n(j_l) - n(j_l - 1)).
#pragma once
#include <numeric>

#include "original/kernel_packets.hpp"

namespace kp {

// fn(j) for every j of d positive integers summing to s
template <typename Fn>
void compositions(int d, int s, Fn&& fn) {
    std::vector<int> j(d);
    auto rec = [&](auto&& self, int i, int left) -> void {
        if (i == d - 1) {
            j[i] = left;
            return fn(j);
        }
        for (int v = 1; v <= left - (d - 1 - i); ++v) {
            j[i] = v;
            self(self, i + 1, left - v);
        }
    };
    rec(rec, 0, s);
}

struct SparseGrid {
    int d, eta, k, L;
    int64_t N = 0;
    std::vector<std::vector<double>> nodes;  // nodes[j]: level j, j = 0 .. L-1
    std::vector<int> J;                      // C x d levels of the components
    std::vector<double> coef, X;             // X: N x d unique points, sorted rows
    std::vector<int64_t> off, idx;           // rows of component c: idx[off[c] .. off[c+1])
    std::vector<std::shared_ptr<KernelPacket>> kps;  // kps[i * L + j], null for an empty level
    // tables of the GPU solve (cuda/: one launch per axis over all fibers of all components and columns):
    std::vector<double> cpos;                // coefficient of the component of each position
    std::vector<int64_t> inv_ptr, inv_pos;   // positions of row r: inv_pos[inv_ptr[r] .. inv_ptr[r+1]), increasing
    std::vector<std::vector<int64_t>> fbase; // fibers of axis i: start position, stride, level
    std::vector<std::vector<int>> fstride, flevel;

    // nodes of level j: nodes[noff[j] .. noff[j+1]), j = 0 .. eta-d+1
    SparseGrid(int d, int eta, const double* nd, const int64_t* noff, int k, const double* rho)
        : d(d), eta(eta), k(k), L(eta - d + 2), nodes(L) {
        for (int j = 0; j < L; ++j) nodes[j].assign(nd + noff[j], nd + noff[j + 1]);
        std::vector<double> P;  // points of all components, with repetition
        off.push_back(0);
        for (int s = std::max(d, eta - d + 1); s <= eta; ++s)
            compositions(d, s, [&](const std::vector<int>& j) {
                J.insert(J.end(), j.begin(), j.end());
                coef.push_back(((eta - s) % 2 ? -1.0 : 1.0) * binom(d - 1, eta - s));
                int64_t nc = 1;
                for (int i = 0; i < d; ++i) nc *= nodes[j[i]].size();
                for (int64_t r = 0; r < nc; ++r)
                    for (int i = 0, rem = 0; i < d; ++i) {  // first coordinate slowest
                        int64_t stride = 1;
                        for (int l = i + 1; l < d; ++l) stride *= nodes[j[l]].size();
                        rem = static_cast<int>((r / stride) % nodes[j[i]].size());
                        P.push_back(nodes[j[i]][rem]);
                    }
                off.push_back(off.back() + nc);
            });
        int64_t M = off.back();
        std::vector<int64_t> order(M);
        std::iota(order.begin(), order.end(), 0);
        auto row = [&](int64_t r) { return P.begin() + r * d; };
        std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return std::lexicographical_compare(row(a), row(a) + d, row(b), row(b) + d);
        });
        idx.resize(M);
        for (int64_t t = 0; t < M; ++t) {
            if (t == 0 || !std::equal(row(order[t]), row(order[t]) + d, row(order[t - 1]))) {
                X.insert(X.end(), row(order[t]), row(order[t]) + d);
                ++N;
            }
            idx[order[t]] = N - 1;
        }
        int64_t C = static_cast<int64_t>(coef.size());
        cpos.resize(M), inv_ptr.assign(N + 1, 0), inv_pos.resize(M);
        for (int64_t c = 0; c < C; ++c)
            for (int64_t r = off[c]; r < off[c + 1]; ++r) cpos[r] = coef[c], ++inv_ptr[idx[r] + 1];
        for (int64_t i = 0; i < N; ++i) inv_ptr[i + 1] += inv_ptr[i];
        std::vector<int64_t> fill(inv_ptr.begin(), inv_ptr.end() - 1);
        for (int64_t r = 0; r < M; ++r) inv_pos[fill[idx[r]]++] = r;
        fbase.resize(d), fstride.resize(d), flevel.resize(d);
        for (int i = 0; i < d; ++i)
            for (int64_t c = 0; c < C; ++c) {
                int64_t ni = nodes[J[c * d + i]].size(), stride = 1, nc = off[c + 1] - off[c];
                for (int l = i + 1; l < d; ++l) stride *= nodes[J[c * d + l]].size();
                for (int64_t o = 0; o < nc; o += ni * stride)
                    for (int64_t t = 0; t < stride; ++t)
                        fbase[i].push_back(off[c] + o + t), fstride[i].push_back(static_cast<int>(stride)),
                            flevel[i].push_back(J[c * d + i]);
            }
        set_rho(rho);
    }

    // bytes held: the design tables and the distinct 1-D factors
    int64_t bytes() const {
        int64_t s = int64_t((X.size() + coef.size() + cpos.size()) * sizeof(double) +
                            (off.size() + idx.size() + inv_ptr.size() + inv_pos.size()) * sizeof(int64_t));
        for (int i = 0; i < d; ++i) s += int64_t(fbase[i].size() * (sizeof(int64_t) + 2 * sizeof(int)));
        std::vector<const KernelPacket*> seen;
        for (auto& K : kps)
            if (K && std::find(seen.begin(), seen.end(), K.get()) == seen.end()) seen.push_back(K.get()), s += K->bytes();
        return s;
    }

    static double binom(int n, int r) {
        double b = 1.0;
        for (int i = 1; i <= r; ++i) b = b * (n - r + i) / i;
        return b;
    }

    void set_rho(const double* rho) {
        kps.assign(d * L, nullptr);
        parallel_for(d * L, 1, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) {
                int i = static_cast<int>(t / L), j = static_cast<int>(t % L), i0 = 0;
                while (rho[i0] != rho[i]) ++i0;
                if (i0 == i && !nodes[j].empty())
                    kps[t] = std::make_shared<KernelPacket>(nodes[j].data(), nodes[j].size(), k, rho[i]);
            }
        });
        for (int i = 0; i < d; ++i) {
            int i0 = 0;
            while (rho[i0] != rho[i]) ++i0;
            for (int j = 0; j < L; ++j) kps[i * L + j] = kps[i0 * L + j];
        }
    }

    double logdet() const {
        double ld = 0.0;
        for (int s = d; s <= eta; ++s)
            compositions(d, s, [&](const std::vector<int>& j) {
                for (int i = 0; i < d; ++i) {
                    double dL = kps[i * L + j[i]]->logdet - (j[i] > 1 ? kps[i * L + j[i] - 1]->logdet : 0.0), w = 1.0;
                    for (int l = 0; l < d; ++l)
                        if (l != i) w *= static_cast<double>(nodes[j[l]].size() - nodes[j[l] - 1].size());
                    ld += dL * w;
                }
            });
        return ld;
    }

    // w = K^{-1} y for one column; t, f, u: workspace of the largest component size
    void solve_col(const double* y, double* w, double* t, double* f, double* u) const {
        std::fill(w, w + N, 0.0);
        for (size_t c = 0; c + 1 < off.size(); ++c) {
            int64_t nc = off[c + 1] - off[c];
            const int64_t* rows = idx.data() + off[c];
            for (int64_t r = 0; r < nc; ++r) t[r] = y[rows[r]];
            for (int i = 0; i < d; ++i) {
                const KernelPacket& kp = *kps[i * L + J[c * d + i]];
                int64_t stride = 1;
                for (int l = i + 1; l < d; ++l) stride *= kps[l * L + J[c * d + l]]->n;
                int64_t ni = kp.n, block = ni * stride;
                if (ni == 1) continue;  // K = 1
                for (int64_t o = 0; o < nc; o += block)
                    for (int64_t s = 0; s < stride; ++s) {
                        double* base = t + o + s;
                        for (int64_t q = 0; q < ni; ++q) f[q] = base[q * stride];
                        kp.solve_col(f, u);
                        for (int64_t q = 0; q < ni; ++q) base[q * stride] = f[q];
                    }
            }
            for (int64_t r = 0; r < nc; ++r) w[rows[r]] += coef[c] * t[r];
        }
    }

    int64_t largest() const {
        int64_t big = 0;
        for (size_t c = 0; c + 1 < off.size(); ++c) big = std::max(big, off[c + 1] - off[c]);
        return big;
    }

    // W = K^{-1} Y for p columns of length N stored contiguously
    void solve(const double* Y, double* W, int64_t p) const {
        int64_t big = largest();
        parallel_for(p, 1, [&](int64_t c0, int64_t c1) {
            std::vector<double> t(big), f(big), u(big);
            for (int64_t col = c0; col < c1; ++col) solve_col(Y + col * N, W + col * N, t.data(), f.data(), u.data());
        });
    }

    // tr(K^{-1} G) for an N x N matrix G, one column at a time (O(N) memory per thread, no N x N result); the
    // diagonal entries are summed in a fixed order (the same result for any number of threads)
    double trace_solve(const double* G) const {
        int64_t big = largest();
        std::vector<double> diag(N);
        parallel_for(N, 1, [&](int64_t c0, int64_t c1) {
            std::vector<double> t(big), f(big), u(big), w(N);
            for (int64_t col = c0; col < c1; ++col) {
                solve_col(G + col * N, w.data(), t.data(), f.data(), u.data());
                diag[col] = w[col];
            }
        });
        return std::accumulate(diag.begin(), diag.end(), 0.0);
    }
};

}  // namespace kp
