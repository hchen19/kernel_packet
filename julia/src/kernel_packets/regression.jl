# Regression: exact in 1-D through the KP basis (C++), scattered data in d dimensions with a sparse grid as inducing
# points (dense n_Z x n_Z algebra in BLAS/LAPACK, kernel-packet solves and kernel sums in C++) (as regression.py)

"""Posterior mean K(xnew, X) (K + tau2 I)^{-1} y of the KernelPacket kp (zero prior mean, unit variance, noise ratio
kp.tau2) and, if var, the variance, or, if cov, the covariance."""
function posterior(kp::KernelPacket, xnew::AbstractVector, y::AbstractVector; var=false, cov=false)
    xs, m = collect(Float64, xnew), length(xnew)
    mean, v = Vector{Float64}(undef, m), var ? Vector{Float64}(undef, m) : nothing
    ccall((:kp_posterior, LIB), Cvoid, (Ptr{Cvoid}, Ptr{Float64}, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
          kp.h, xs, m, collect(Float64, y), mean, var ? v : C_NULL)
    out = var ? (mean, v) : (mean,)
    if cov
        C = Matrix{Float64}(undef, m, m)
        ccall((:kp_posterior_cov, LIB), Cvoid, (Ptr{Cvoid}, Ptr{Float64}, Int64, Ptr{Float64}), kp.h, xs, m, C)
        out = (out..., C)
    end
    length(out) == 1 ? out[1] : out
end

"Posterior mean K(xnew, X) w, w = K^{-1} (y - mean), summed on the fly (no m x N matrix)."
function predict(xnew::AbstractVecOrMat, X::AbstractVecOrMat, w::AbstractVecOrMat, k::Integer, rho)
    xs, U = permutedims(Matrix{Float64}(reshape(xnew, size(xnew, 1), :))), permutedims(Matrix{Float64}(reshape(X, size(X, 1), :)))
    W = Matrix{Float64}(reshape(w, size(w, 1), :))
    d, m = size(xs)
    out = Matrix{Float64}(undef, m, size(W, 2))
    ccall((:kernel_predict, LIB), Cvoid,
          (Ptr{Float64}, Int64, Ptr{Float64}, Int64, Cint, Ptr{Float64}, Int64, Cint, Ptr{Float64}, Ptr{Float64}),
          xs, m, U, size(U, 2), d, W, size(W, 2), k, collect(Float64, rho .* ones(d)), out)
    w isa AbstractVector ? vec(out) : out
end

"""Scattered data (X, y) in d dimensions with the points Z of a SparseGrid as inducing points (Titsias VFE), kernel
sigma2 * matern and noise tau2.  With K = matern and Sigma = K_ZZ + sigma2 K_ZX K_XZ / tau2 = L L':
    mean(x) = sigma2 K(x, Z) alpha,  alpha = Sigma^{-1} K_ZX y / tau2,
    var(x) = sigma2 (1 - K(x, Z) K_ZZ^{-1} K(Z, x) + |L^{-1} K(Z, x)|^2).
K_ZZ^{-1} and log|K_ZZ| come from kernel packets (C++); K_ZX K_XZ, K_ZX y and the Cholesky factor of Sigma are dense BLAS
work, accumulated in chunks (memory of n_Z^2, not n n_Z)."""
mutable struct SparseGridRegression
    X::Matrix{Float64}
    y::Vector{Float64}
    sg::SparseGrid
    sigma2::Float64
    tau2::Float64
    L::LowerTriangular{Float64,Matrix{Float64}}
    alpha::Vector{Float64}
    elbo::Float64
    cache::Any   # (rho, G, b, K_ZZ, trace, log|K_ZZ|): they depend on rho only, reused across sigma2, tau2
    function SparseGridRegression(X, y, sg; sigma2=1.0, tau2=1e-4)
        m = new(Matrix{Float64}(reshape(X, size(X, 1), :)), collect(Float64, y), sg, sigma2, tau2)
        m.cache = nothing
        fit!(m)
    end
end

"Factor Sigma and the variational lower bound `elbo` of the log marginal likelihood at the current parameters."
function fit!(m::SparseGridRegression)
    Z, sg, s, t, y = m.sg.X, m.sg, m.sigma2, m.tau2, m.y
    n, n_z = length(y), length(sg)
    if m.cache === nothing || m.cache.rho != sg.rho
        G, b, c = zeros(n_z, n_z), zeros(n_z), batch(n_z)
        for i in 1:c:n
            r = i:min(i + c - 1, n)
            Kc = matern(Z, view(m.X, r, :), sg.k, sg.rho)
            mul!(G, Kc, Kc', 1.0, 1.0)
            mul!(b, Kc, view(y, r), 1.0, 1.0)
        end
        # tr(K_XZ K_ZZ^{-1} K_ZX) = tr(K_ZZ^{-1} G): n_Z columns, not n (C++)
        m.cache = (rho=copy(sg.rho), G=G, b=b, Kzz=matern(Z, Z, sg.k, sg.rho), trq=trace_solve(sg, G), ld_zz=logdet(sg))
    end
    G, b, trq = m.cache.G, m.cache.b, m.cache.trq
    A = m.cache.Kzz .+ (s / t) .* G
    # A is positive definite but cond(A) can exceed 1 / eps: a diagonal of n_Z eps times its mean keeps the factor defined
    # in every BLAS (CPU or GPU) at a relative change of ~1e-13
    A[diagind(A)] .+= n_z * eps(Float64) * mean(diag(A))
    m.L = cholesky!(Symmetric(A, :L)).L
    m.alpha = (m.L' \ (m.L \ b)) ./ t
    quad = (dot(y, y) - s * dot(b, m.alpha)) / t
    ld = 2 * sum(log, diag(m.L)) - m.cache.ld_zz + n * log(t)
    m.elbo = -0.5 * (quad + ld + n * log(2π)) - 0.5 * s * (n - trq) / t
    m
end

function nbytes(m::SparseGridRegression)
    c = m.cache === nothing ? 0 : (length(m.cache.G) + length(m.cache.Kzz))
    nbytes(m.sg) + 8 * (length(m.L) + length(m.alpha) + c + length(m.sg.X))
end

"Posterior mean (C++ kernel sum) and, if var and/or cov, variance and/or covariance at the rows of xnew."
function predict(m::SparseGridRegression, xnew::AbstractVecOrMat; var=false, cov=false)
    xs = reshape(xnew, size(xnew, 1), :)
    mean = m.sigma2 .* predict(xs, m.sg.X, m.alpha, m.sg.k, m.sg.rho)
    (var || cov) || return mean
    M = matern(xs, m.sg.X, m.sg.k, m.sg.rho)
    S = solve(m.sg, Matrix(M'))
    V = m.L \ Matrix(M')
    out = Any[mean]
    var && push!(out, m.sigma2 .* (1 .- vec(sum(M .* S', dims=2)) .+ vec(sum(abs2, V, dims=1))))
    cov && push!(out, m.sigma2 .* (matern(xs, xs, m.sg.k, m.sg.rho) - M * S + V' * V))
    Tuple(out)
end

"""Maximize the bound over (rho, sigma2, tau2) in bounds (Nelder-Mead in a box on their logarithms: the bound is noisy at
small tau2, finite-difference gradients are not reliable); rho is shared by the axes."""
function train!(m::SparseGridRegression; bounds=((1e-2, 1e2), (1e-4, 1e4), (1e-8, 1e2)))
    function nelbo(z)
        set_rho!(m.sg, clamp(exp(z[1]), bounds[1][1], bounds[1][2]))
        m.sigma2 = clamp(exp(z[2]), bounds[2][1], bounds[2][2])
        m.tau2 = clamp(exp(z[3]), bounds[3][1], bounds[3][2])
        -fit!(m).elbo
    end
    lo, hi = [log(b[1]) for b in bounds], [log(b[2]) for b in bounds]
    z0 = clamp.(log.([m.sg.rho[1], m.sigma2, m.tau2]), lo .+ 1e-9, hi .- 1e-9)
    r = nelder_mead(nelbo, z0, lo, hi; xtol=1e-3, ftol=1e-3)
    nelbo(r.minimizer)
    m
end
