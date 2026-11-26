// MATLAB gateway to the C API of cpp/include/c_api.hpp (handles are uint64 pointers, indices 1-based here):
//   [h, method] = kp_mex('kp_new', x, k, rho, tau2, method)            kp_mex('kp_free', h)            L = kp_mex('kp_logdet', h)
//   kp_mex('kp_set_params', h, rho, tau2)    b = kp_mex('kp_bytes', h)   b = kp_mex('sg_bytes', g)
//   W = kp_mex('kp_solve', h, Y, with_a)     F = kp_mex('kp_sample', h, Z)   [start, V] = kp_mex('kp_basis', h, xs)
//   [mu, v] = kp_mex('kp_posterior', h, xs, y)        C = kp_mex('kp_posterior_cov', h, xs)
//   F = kp_mex('kp_posterior_sample', h, xs, y, Z)    Y = kp_mex('kernel_predict', xs, X, w, k, rho)
//   g = kp_mex('sg_new', d, eta, nodes, noff, k, rho) kp_mex('sg_free', g)   X = kp_mex('sg_points', g)
//   kp_mex('sg_set_rho', g, rho)   W = kp_mex('sg_solve', g, Y)   L = kp_mex('sg_logdet', g)   t = kp_mex('sg_trace_solve', g, G)
//   [L, beta, sigma2] = kp_mex('kp_loglike' | 'sg_loglike', h, y, F)
#include <string>
#include <vector>

#include "c_api.hpp"
#include "mex.h"

static void* ptr(const mxArray* a) { return reinterpret_cast<void*>(*mxGetUint64s(a)); }

static mxArray* handle(void* p) {
    mxArray* a = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *mxGetUint64s(a) = reinterpret_cast<uint64_t>(p);
    return a;
}

static double* dbl(const mxArray* a) { return mxGetDoubles(a); }
static int64_t numel(const mxArray* a) { return static_cast<int64_t>(mxGetNumberOfElements(a)); }
static mxArray* matrix(int64_t m, int64_t n) { return mxCreateDoubleMatrix(m, n, mxREAL); }

void mexFunction(int nlhs, mxArray* plhs[], int, const mxArray* prhs[]) {
    char buf[32];
    mxGetString(prhs[0], buf, sizeof buf);
    std::string cmd(buf);
    if (cmd == "kp_new") {
        void* h = kp_new(dbl(prhs[1]), numel(prhs[1]), static_cast<int>(mxGetScalar(prhs[2])), mxGetScalar(prhs[3]),
                         mxGetScalar(prhs[4]), static_cast<int>(mxGetScalar(prhs[5])));
        plhs[0] = handle(h);
        if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(kp_method(h));
    } else if (cmd == "kp_free") {
        kp_free(ptr(prhs[1]));
    } else if (cmd == "kp_set_params") {
        kp_set_params(ptr(prhs[1]), mxGetScalar(prhs[2]), mxGetScalar(prhs[3]));
    } else if (cmd == "kp_logdet") {
        plhs[0] = mxCreateDoubleScalar(kp_logdet(ptr(prhs[1])));
    } else if (cmd == "kp_bytes") {
        plhs[0] = mxCreateDoubleScalar(static_cast<double>(kp_bytes(ptr(prhs[1]))));
    } else if (cmd == "sg_bytes") {
        plhs[0] = mxCreateDoubleScalar(static_cast<double>(sg_bytes(ptr(prhs[1]))));
    } else if (cmd == "kp_solve" || cmd == "kp_sample") {
        plhs[0] = mxDuplicateArray(prhs[2]);
        int64_t n = mxGetM(prhs[2]), p = mxGetN(prhs[2]);
        if (cmd == "kp_solve") kp_solve(ptr(prhs[1]), dbl(plhs[0]), p, mxGetScalar(prhs[3]) != 0);
        else kp_sample(ptr(prhs[1]), dbl(plhs[0]), p);
        (void)n;
    } else if (cmd == "kp_basis") {
        int64_t m = numel(prhs[2]), w = kp_width(ptr(prhs[1]));
        std::vector<int64_t> start(m);
        plhs[1] = matrix(w, m);
        kp_basis(ptr(prhs[1]), dbl(prhs[2]), m, start.data(), dbl(plhs[1]));
        plhs[0] = matrix(m, 1);
        for (int64_t j = 0; j < m; ++j) dbl(plhs[0])[j] = start[j] + 1.0;
    } else if (cmd == "kp_posterior") {
        int64_t m = numel(prhs[2]);
        plhs[0] = matrix(m, 1);
        if (nlhs > 1) plhs[1] = matrix(m, 1);
        kp_posterior(ptr(prhs[1]), dbl(prhs[2]), m, dbl(prhs[3]), dbl(plhs[0]), nlhs > 1 ? dbl(plhs[1]) : nullptr);
    } else if (cmd == "kp_posterior_cov") {
        int64_t m = numel(prhs[2]);
        plhs[0] = matrix(m, m);
        kp_posterior_cov(ptr(prhs[1]), dbl(prhs[2]), m, dbl(plhs[0]));
    } else if (cmd == "kp_posterior_sample") {
        int64_t m = numel(prhs[2]), p = mxGetN(prhs[4]);
        plhs[0] = matrix(m, p);
        kp_posterior_sample(ptr(prhs[1]), dbl(prhs[2]), m, dbl(prhs[3]), dbl(prhs[4]), p, dbl(plhs[0]));
    } else if (cmd == "kernel_predict") {  // xs (d x m) and X (d x N): points as columns
        int d = static_cast<int>(mxGetM(prhs[1]));
        int64_t m = mxGetN(prhs[1]), N = mxGetN(prhs[2]), p = mxGetN(prhs[3]);
        plhs[0] = matrix(m, p);
        kernel_predict(dbl(prhs[1]), m, dbl(prhs[2]), N, d, dbl(prhs[3]), p, static_cast<int>(mxGetScalar(prhs[4])),
                       dbl(prhs[5]), dbl(plhs[0]));
    } else if (cmd == "sg_new") {
        std::vector<int64_t> noff(dbl(prhs[4]), dbl(prhs[4]) + numel(prhs[4]));
        plhs[0] = handle(sg_new(static_cast<int>(mxGetScalar(prhs[1])), static_cast<int>(mxGetScalar(prhs[2])),
                                dbl(prhs[3]), noff.data(), static_cast<int>(mxGetScalar(prhs[5])), dbl(prhs[6])));
    } else if (cmd == "sg_free") {
        sg_free(ptr(prhs[1]));
    } else if (cmd == "sg_points") {  // d x N, points as columns
        int64_t N = sg_size(ptr(prhs[1])), d = static_cast<int64_t>(mxGetScalar(prhs[2]));
        plhs[0] = matrix(d, N);
        sg_points(ptr(prhs[1]), dbl(plhs[0]));
    } else if (cmd == "sg_set_rho") {
        sg_set_rho(ptr(prhs[1]), dbl(prhs[2]));
    } else if (cmd == "sg_solve") {
        plhs[0] = matrix(mxGetM(prhs[2]), mxGetN(prhs[2]));
        sg_solve(ptr(prhs[1]), dbl(prhs[2]), dbl(plhs[0]), mxGetN(prhs[2]));
    } else if (cmd == "sg_logdet") {
        plhs[0] = mxCreateDoubleScalar(sg_logdet(ptr(prhs[1])));
    } else if (cmd == "sg_trace_solve") {
        plhs[0] = mxCreateDoubleScalar(sg_trace_solve(ptr(prhs[1]), dbl(prhs[2])));
    } else {  // kp_loglike, sg_loglike
        int pf = static_cast<int>(mxGetN(prhs[3]));
        plhs[1] = matrix(pf, 1);
        plhs[2] = matrix(1, 1);
        double L = (cmd == "kp_loglike" ? kp_loglike : sg_loglike)(ptr(prhs[1]), dbl(prhs[2]), dbl(prhs[3]), pf,
                                                                   dbl(plhs[1]), dbl(plhs[2]));
        plhs[0] = mxCreateDoubleScalar(L);
    }
}
