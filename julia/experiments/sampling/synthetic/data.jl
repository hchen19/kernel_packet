# Synthetic designs for prior and posterior sampling: data from the Griewank function (Griewank 1981,
# https://www.sfu.ca/~ssurjano/griewank.html) on [-5, 5]^d, rescaled to [0, 1]^d, with noise s.d. NOISE in d
# dimensions. Inputs are random: prior nodes from Beta(0.7, 0.7), training inputs from two clusters,
# test inputs from Beta(0.7, 0.7); the only structured points are the sparse grids, as inducing points.
using Random

const PARTS = ["prior_draws", "prior_accuracy", "prior_sensitivity", "prior_scaling", "posterior_draws",
               "posterior_accuracy", "posterior_sensitivity", "scaling_1d", "posterior_sparse_grid"]
const K = 5
const SEEDS = 5
const NOISE = 0.1
const KS = (3, 5, 7)
const PRIOR_RHO = 0.2
const PRIOR_ACC_N = [50, 100, 200, 500, 1000]
const POST_ACC_N = [100, 500, 10^3, 5 * 10^3, 10^4, 5 * 10^4, 10^5, 5 * 10^5, 10^6]
const P_ACC = 512
const SENS_N, SENS_KS = 100, (3, 5, 7)
const SENS_RHOS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
const POST_SENS_RHOS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
const POST_SENS_TAU2S = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
const SIZES = [100, 500, 10^3, 5 * 10^3, 10^4, 5 * 10^4, 10^5, 5 * 10^5, 10^6, 5 * 10^6, 10^7, 5 * 10^7, 10^8]
const BIG = 10^6
const SCALING_RHO, P_TIME, N_DENSE = 20.0, 16, 10^4
const N_POST, M_POST, P_POST = 20, 200, 4096
const SG_DIMS, SG_N, SG_M = (2, 4, 8), 2^10, 200

griewank(X) = (Z = 10 .* X .- 5; vec(sum(Z .^ 2; dims=2) ./ 4000 .- prod(cos.(Z ./ sqrt.((1:size(Z, 2))')); dims=2) .+ 1))
f(x) = griewank(reshape(x, :, 1))
nodes(n, rng) = sort(rand_beta(rng, 0.7, 0.7, n))
scaling_design(n, rng) = collect(0:n-1) .+ (0.6 .* rand(rng, n) .- 0.3)
scaling_data(n, rng; m=200) = (X = scaling_design(n, rng); (X, sin.(X ./ 50.0), sort((n - 1) .* rand(rng, m))))

function train_inputs(n, d, rng)
    a = rand(rng, (5.0, 15.0), n, d)
    rand_beta(rng, a, 20.0 .- a)
end

test_inputs(m, d, rng) = rand_beta(rng, 0.7, 0.7, m, d)
rho(d) = 0.25 * sqrt(d)
