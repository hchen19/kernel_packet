# Exact Gaussian process by dense Cholesky factorization: the reference of every experiment, O(n^3) time, O(n^2)
# memory (LAPACK) (as dense_gp.py)

using LinearAlgebra, Random
import ..KernelPackets: matern, predict, prior_sample, fit!, train!, nbytes, mle, batch, brent, nelder_mead

export DenseGP, sample

"Kernel sigma2 * matern(k, rho), noise tau2, data (X, y)."
mutable struct DenseGP
    X::Matrix{Float64}
    y::Vector{Float64}
    k::Int
    rho::Vector{Float64}
    sigma2::Float64
    tau2::Float64
    C::Cholesky{Float64,Matrix{Float64}}
    alpha::Vector{Float64}
    loglike::Float64
end

function DenseGP(X, y, k=5, rho=1.0; sigma2=1.0, tau2=0.0)
    Xm = Matrix{Float64}(reshape(X, size(X, 1), :))
    r = collect(Float64, rho .* ones(size(Xm, 2)))
    g = DenseGP(Xm, collect(Float64, y), k, r, Float64(sigma2), Float64(tau2),
                Cholesky(Matrix{Float64}(undef, 0, 0), :L, 0), Float64[], 0.0)
    fit!(g, rho, sigma2, tau2)
end

"Factor K + tau2 I and the log marginal likelihood `loglike` at the given parameters."
function fit!(g::DenseGP, rho, sigma2, tau2)
    g.rho = collect(Float64, rho .* ones(size(g.X, 2)))
    g.sigma2 = Float64(sigma2)
    g.tau2 = Float64(tau2)
    K = g.sigma2 .* matern(g.X, g.X, g.k, g.rho)
    for i in axes(K, 1)
        K[i, i] += g.tau2
    end
    g.C = cholesky!(Symmetric(K))
    g.alpha = g.C \ g.y
    n = length(g.y)
    g.loglike = -0.5 * (dot(g.y, g.alpha) + logdet(g.C) + n * log(2π))
    g
end

"Exact maximum likelihood over (rho, sigma2, tau2) in bounds (Nelder-Mead on their logarithms)."
function train!(g::DenseGP; bounds=((1e-2, 1e2), (1e-4, 1e4), (1e-8, 1e2)))
    function nll(z)
        try
            out = -fit!(g, clamp(exp(z[1]), bounds[1][1], bounds[1][2]),
                        clamp(exp(z[2]), bounds[2][1], bounds[2][2]),
                        clamp(exp(z[3]), bounds[3][1], bounds[3][2])).loglike
            isfinite(out) ? out : 1e30
        catch e
            e isa PosDefException ? 1e30 : rethrow(e)
        end
    end
    lo = [log(b[1]) for b in bounds]
    hi = [log(b[2]) for b in bounds]
    z0 = clamp.(log.([g.rho[1], g.sigma2, max(g.tau2, 1e-6)]), lo .+ 1e-9, hi .- 1e-9)
    z = nelder_mead(nll, z0, lo, hi; xtol=1e-3, ftol=1e-3).minimizer
    fit!(g, clamp(exp(z[1]), bounds[1][1], bounds[1][2]),
         clamp(exp(z[2]), bounds[2][1], bounds[2][2]),
         clamp(exp(z[3]), bounds[3][1], bounds[3][2]))
end

nbytes(g::DenseGP) = 8 * (length(g.C.factors) + length(g.X) + length(g.alpha))

"Maximum likelihood of rho (Brent on log rho), the constant mean and sigma2 profiled out, no noise."
function mle(::Type{DenseGP}, X, y, k; bounds=(1e-3, 1.0))
    Xm = Matrix{Float64}(reshape(X, size(X, 1), :))
    yv = collect(Float64, y)
    n = length(yv)
    ones_v = ones(n)
    function profile(rho)
        C = cholesky(Symmetric(matern(Xm, Xm, k, rho)))
        a = C.L \ ones_v
        c = C.L \ yv
        beta = dot(a, c) / dot(a, a)
        r = c .- beta .* a
        s2 = dot(r, r) / n
        out = 0.5 * (n * log(s2) + logdet(C) + n)
        beta, s2, out
    end
    function nll(z)
        try
            _, _, out = profile(exp(z))
            isfinite(out) ? out : 1e30
        catch e
            e isa PosDefException ? 1e30 : rethrow(e)
        end
    end
    rho = clamp(exp(brent(nll, log(bounds[1]), log(bounds[2]); tol=1e-3).minimizer), bounds[1], bounds[2])
    beta, s2, _ = profile(rho)
    DenseGP(Xm, yv .- beta, k, rho; sigma2=s2), beta
end

"Posterior mean and variance (or covariance if cov) of f at xnew; variance in batches."
function predict(g::DenseGP, xnew; cov=false)
    xs = reshape(xnew, size(xnew, 1), :)
    if cov
        Ks = g.sigma2 .* matern(xs, g.X, g.k, g.rho)
        V = g.C.L \ Matrix(Ks')
        return Ks * g.alpha, g.sigma2 .* matern(xs, xs, g.k, g.rho) - V' * V
    end
    m = size(xs, 1)
    means, vars = Vector{Float64}(undef, m), Vector{Float64}(undef, m)
    bs = batch(size(g.X, 1))
    for i in 1:bs:m
        r = i:min(i + bs - 1, m)
        Ks = g.sigma2 .* matern(view(xs, r, :), g.X, g.k, g.rho)
        V = g.C.L \ Matrix(Ks')
        means[r] .= Ks * g.alpha
        vars[r] .= g.sigma2 .- vec(sum(abs2, V, dims=1))
    end
    means, vars
end

"p posterior draws at xnew, m x p."
function sample(g::DenseGP, xnew, p=1; rng=Random.default_rng())
    mean, C = predict(g, xnew; cov=true)
    mean .+ cholesky(Symmetric(C + 1e-12I)).L * randn(rng, length(mean), p)
end

"p prior draws at X by the Cholesky factor of K (NaN if K is numerically non-PD)."
function prior_sample(::Type{DenseGP}, X, k, rho, p=1; rng=Random.default_rng())
    Xm = reshape(X, size(X, 1), :)
    C = cholesky(Symmetric(matern(Xm, Xm, k, rho)); check=false)
    issuccess(C) ? C.L * randn(rng, size(Xm, 1), p) : fill(NaN, size(Xm, 1), p)
end
