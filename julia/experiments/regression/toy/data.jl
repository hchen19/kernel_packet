# Toy regression data: standard test functions of GP regression (Surjanovic & Bingham, Virtual Library
# of Simulation Experiments, https://www.sfu.ca/~ssurjano/).
#   1-D: the Gramacy & Lee (2012) function sin(10 pi t) / (2 t) + (t - 1)^4, t = 0.5 + 2x on x in [0, 1].
#   d-D: the Genz (1984) oscillatory function cos(2 pi w + sum_i a_i x_i) on [0, 1]^d, a_i = 9 / i^2, w = 0.5.
# Training and test inputs are random and drawn from different distributions: training from two Beta clusters,
# test from Beta(0.7, 0.7); the only structured points are the sparse grids, as inducing points.
using Random

const PARTS = ["posterior_1d", "accuracy_1d", "scaling_1d", "sparse_grid", "scaling_nd"]
const K = 5
const KS = (3, 5, 7)
const SEEDS = 5
const M = 1000
const N_1D = 60
const N_ACC = [100, 500, 10^3, 5 * 10^3, 10^4, 5 * 10^4, 10^5, 5 * 10^5, 10^6, 5 * 10^6, 10^7, 5 * 10^7, 10^8]
const M_W2 = 300
const SIZES = N_ACC
const BIG = 10^6
const SCALING_RHO, N_DENSE = 20.0, 10^4
const NOISE = 0.05
const SG_N, SG_DIMS = 2000, (2, 4)
const SG_LEVELS = Dict(2 => 2:6, 4 => 1:5)
const INIT = Dict("rho" => 0.5, "sigma2" => 1.0, "tau2" => NOISE^2)
const ND_SIZES = [100, 500, 10^3, 5 * 10^3, 10^4, 5 * 10^4, 10^5, 5 * 10^5, 10^6]

f(x) = (t = 0.5 .+ 2 .* x; sin.(10π .* t) ./ (2 .* t) .+ (t .- 1) .^ 4)

function train_inputs(n, d, rng)
    a = rand(rng, (5.0, 15.0), n, d)
    rand_beta(rng, a, 20.0 .- a)
end

test_inputs(m, d, rng) = rand_beta(rng, 0.7, 0.7, m, d)

function data_1d(n, rng; noise=0.0)
    X = sort(train_inputs(n, 1, rng)[:, 1])
    xs = test_inputs(M, 1, rng)[:, 1]
    X, f(X) .+ noise .* randn(rng, n), xs, f(xs)
end

function scaling_design(n, rng; m=M)
    X = collect(0:n-1) .+ (0.6 .* rand(rng, n) .- 0.3)
    xs = (n - 1) .* rand(rng, m)
    X, sin.(X ./ 50.0), xs, sin.(xs ./ 50.0)
end

f_nd(X) = cos.(π .+ X * (9.0 ./ (1:size(X, 2)) .^ 2))

function data_nd(d, n, rng)
    X, xs = train_inputs(n, d, rng), test_inputs(M, d, rng)
    X, f_nd(X) .+ NOISE .* randn(rng, n), xs, f_nd(xs)
end
