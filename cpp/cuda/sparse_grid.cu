// Sparse grids on the GPU (sparse_grid.cpp in CUDA): K^{-1} Y and tr(K^{-1} G) on device memory, from the 1-D kernel
// packets and the fiber / inverse-map tables of the CPU SparseGrid (factored on the host, O(n) per level, then
// copied). Each axis is one launch over all fibers of all components and columns (one thread per fiber: banded LU
// solve and A product in place, the CPU's operations in the CPU's order; both compiled without FMA contraction, so
// the results equal the CPU's to the bit); the sum over components is a gather in a fixed order (no atomics).
// Columns run in chunks within WORKSPACE bytes.
#include <cuda_runtime.h>

#include "c_api.hpp"
#include "sparse_grid.hpp"

namespace {

constexpr int64_t WORKSPACE = int64_t(256) << 20;  // bytes of the two per-chunk buffers
constexpr int THREADS = 128;

struct Factor {  // a 1-D kernel packet factor: sizes and offsets into the flat device arrays
    int64_t n, v, piv, a, lo, len;
    int b, W, w, q, dense;
};

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

struct DeviceGrid {
    int d;
    int64_t N, M;
    Factor* factors;
    double *v, *a, *cpos;
    int64_t *piv, *lo, *idx, *inv_ptr, *inv_pos;
    int* len;
    std::vector<int64_t> nfib;            // fibers of axis i
    std::vector<int64_t*> fbase;          // fiber start (position in a column of the workspace)
    std::vector<int*> fstride, ffactor;   // fiber stride and factor

    explicit DeviceGrid(const kp::SparseGrid& g) : d(g.d), N(g.N), M(g.off.back()) {
        std::vector<Factor> F;
        std::vector<double> hv, ha;
        std::vector<int64_t> hpiv, hlo;
        std::vector<int> hlen, slot(g.kps.size());
        std::vector<const kp::KernelPacket*> seen;
        for (size_t t = 0; t < g.kps.size(); ++t) {  // distinct factors (axes of equal lengthscale share them)
            const kp::KernelPacket* K = g.kps[t].get();
            if (!K) continue;
            auto it = std::find(seen.begin(), seen.end(), K);
            if (it != seen.end()) {
                slot[t] = static_cast<int>(it - seen.begin());
                continue;
            }
            slot[t] = static_cast<int>(seen.size());
            seen.push_back(K);
            F.push_back({K->n, int64_t(hv.size()), int64_t(hpiv.size()), int64_t(ha.size()), int64_t(hlo.size()),
                         int64_t(hlen.size()), K->phi.b, K->phi.W, K->w, K->q, K->dense});
            hv.insert(hv.end(), K->phi.v.begin(), K->phi.v.end());
            hpiv.insert(hpiv.end(), K->phi.piv.begin(), K->phi.piv.end());
            ha.insert(ha.end(), K->a.begin(), K->a.end());
            hlo.insert(hlo.end(), K->lo.begin(), K->lo.end());
            hlen.insert(hlen.end(), K->len.begin(), K->len.end());
        }
        for (int i = 0; i < d; ++i) {
            std::vector<int> fac(g.flevel[i].size());
            for (size_t f = 0; f < fac.size(); ++f) fac[f] = slot[i * g.L + g.flevel[i][f]];
            nfib.push_back(fac.size());
            fbase.push_back(upload(g.fbase[i])), fstride.push_back(upload(g.fstride[i])), ffactor.push_back(upload(fac));
        }
        factors = upload(F), v = upload(hv), a = upload(ha), piv = upload(hpiv), lo = upload(hlo), len = upload(hlen);
        cpos = upload(g.cpos), idx = upload(g.idx), inv_ptr = upload(g.inv_ptr), inv_pos = upload(g.inv_pos);
    }

    ~DeviceGrid() {
        for (void* p : {(void*)factors, (void*)v, (void*)a, (void*)cpos, (void*)piv, (void*)lo, (void*)idx,
                        (void*)inv_ptr, (void*)inv_pos, (void*)len})
            cudaFree(p);
        for (int i = 0; i < d; ++i) cudaFree(fbase[i]), cudaFree(fstride[i]), cudaFree(ffactor[i]);
    }
};

__global__ void gather(const double* Y, int64_t N, const int64_t* idx, int64_t M, int64_t p, double* T) {
    int64_t g = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (g < M * p) T[g] = Y[(g / M) * N + idx[g % M]];
}

// one fiber per thread: x <- K^{-1} x (Phi^{-1} by the banded LU, then A), x strided in the workspace column
__global__ void axis(const Factor* factors, const double* V, const int64_t* PIV, const double* A, const int64_t* LO,
                     const int* LEN, int64_t F, const int64_t* fbase, const int* fstride, const int* ffactor,
                     int64_t M, int64_t p, double* T, double* U) {
    int64_t g = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (g >= F * p) return;
    int64_t f = g % F, col = g / F, s = fstride[f];
    const Factor K = factors[ffactor[f]];
    if (K.n == 1) return;  // K = 1
    double* x = T + col * M + fbase[f];
    double* u = U + col * M + fbase[f];
    const double* v = V + K.v;
    const int64_t* piv = PIV + K.piv;
    auto at = [&](int64_t r, int64_t c) { return v[r * K.W + (c - r + K.b)]; };
    for (int64_t k = 0; k < K.n; ++k) {
        if (piv[k] != k) {
            double t = x[k * s];
            x[k * s] = x[piv[k] * s], x[piv[k] * s] = t;
        }
        for (int64_t r = k + 1, re = min(K.n, k + K.b + 1); r < re; ++r) x[r * s] -= at(r, k) * x[k * s];
    }
    for (int64_t k = K.n - 1; k >= 0; --k) {
        double t = x[k * s];
        for (int64_t c = k + 1, ce = min(K.n, k + 2 * K.b + 1); c < ce; ++c) t -= at(k, c) * x[c * s];
        x[k * s] = t / at(k, k);
    }
    if (K.dense) return;
    for (int64_t r = 0; r < K.n; ++r) u[r * s] = x[r * s];
    for (int64_t r = 0; r < K.n; ++r) {  // (A u)_r = sum over the packets i covering r, in increasing i (as the CPU)
        double t = 0.0;
        for (int64_t i = max(int64_t(0), r - 2 * K.q - 1), ie = min(K.n, r + 2 * K.q + 2); i < ie; ++i) {
            int64_t l = LO[K.lo + i];
            if (l <= r && r < l + LEN[K.len + i]) t += A[K.a + i * K.w + (r - l)] * u[i * s];
        }
        x[r * s] = t;
    }
}

// W[row] = sum over the positions of row (increasing, as the CPU's order of components) of coef * T[position]
__global__ void scatter(const double* T, int64_t M, const int64_t* inv_ptr, const int64_t* inv_pos, const double* cpos,
                        int64_t N, int64_t p, double* W) {
    int64_t g = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (g >= N * p) return;
    int64_t row = g % N, col = g / N;
    double t = 0.0;
    for (int64_t k = inv_ptr[row]; k < inv_ptr[row + 1]; ++k) t += cpos[inv_pos[k]] * T[col * M + inv_pos[k]];
    W[g] = t;
}

// the diagonal entry of column c0 + col: W[c0 + col] of that column
__global__ void scatter_diag(const double* T, int64_t M, const int64_t* inv_ptr, const int64_t* inv_pos,
                             const double* cpos, int64_t c0, int64_t p, double* diag) {
    int64_t col = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (col >= p) return;
    int64_t row = c0 + col;
    double t = 0.0;
    for (int64_t k = inv_ptr[row]; k < inv_ptr[row + 1]; ++k) t += cpos[inv_pos[k]] * T[col * M + inv_pos[k]];
    diag[row] = t;
}

inline unsigned blocks(int64_t n) { return static_cast<unsigned>((n + THREADS - 1) / THREADS); }

// K^{-1} on columns c0 .. c0+pc of Y into the workspace T (U: scratch)
void solve_chunk(const DeviceGrid& G, const double* Y, int64_t pc, double* T, double* U, cudaStream_t st) {
    gather<<<blocks(G.M * pc), THREADS, 0, st>>>(Y, G.N, G.idx, G.M, pc, T);
    for (int i = 0; i < G.d; ++i)
        axis<<<blocks(G.nfib[i] * pc), THREADS, 0, st>>>(G.factors, G.v, G.piv, G.a, G.lo, G.len, G.nfib[i],
                                                           G.fbase[i], G.fstride[i], G.ffactor[i], G.M, pc, T, U);
}

int64_t chunk(const DeviceGrid& G) { return std::max<int64_t>(1, WORKSPACE / (2 * 8 * std::max<int64_t>(G.M, 1))); }

}  // namespace

extern "C" {
void* sg_cuda_new(void* h) { return new DeviceGrid(*static_cast<kp::SparseGrid*>(h)); }
void sg_cuda_free(void* hd) { delete static_cast<DeviceGrid*>(hd); }

void sg_cuda_solve(void* hd, const double* Y, double* W, int64_t p, void* stream) {
    const DeviceGrid& G = *static_cast<DeviceGrid*>(hd);
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    int64_t P = std::min(p, chunk(G));
    double *T, *U;
    cudaMallocAsync(&T, P * G.M * sizeof(double), st);
    cudaMallocAsync(&U, P * G.M * sizeof(double), st);
    for (int64_t c0 = 0; c0 < p; c0 += P) {
        int64_t pc = std::min(P, p - c0);
        solve_chunk(G, Y + c0 * G.N, pc, T, U, st);
        scatter<<<blocks(G.N * pc), THREADS, 0, st>>>(T, G.M, G.inv_ptr, G.inv_pos, G.cpos, G.N, pc, W + c0 * G.N);
    }
    cudaFreeAsync(T, st), cudaFreeAsync(U, st);
}

double sg_cuda_trace_solve(void* hd, const double* Gm, void* stream) {
    const DeviceGrid& G = *static_cast<DeviceGrid*>(hd);
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    int64_t P = std::min(G.N, chunk(G));
    double *T, *U, *diag;
    cudaMallocAsync(&T, P * G.M * sizeof(double), st);
    cudaMallocAsync(&U, P * G.M * sizeof(double), st);
    cudaMallocAsync(&diag, G.N * sizeof(double), st);
    for (int64_t c0 = 0; c0 < G.N; c0 += P) {
        int64_t pc = std::min(P, G.N - c0);
        solve_chunk(G, Gm + c0 * G.N, pc, T, U, st);
        scatter_diag<<<blocks(pc), THREADS, 0, st>>>(T, G.M, G.inv_ptr, G.inv_pos, G.cpos, c0, pc, diag);
    }
    std::vector<double> h(G.N);
    cudaMemcpyAsync(h.data(), diag, G.N * sizeof(double), cudaMemcpyDeviceToHost, st);
    cudaFreeAsync(T, st), cudaFreeAsync(U, st), cudaFreeAsync(diag, st);
    cudaStreamSynchronize(st);
    return std::accumulate(h.begin(), h.end(), 0.0);  // the CPU's fixed order
}
}
