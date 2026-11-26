# Sampling: 1-D prior draws C~^{-1} L_B z (stable) or Phi L^{-T} z (original) and posterior draws by Matheron's rule
# with noise ratio tau2 (C++); posterior draws from the sparse-grid inducing posterior in d dimensions (as sampling.py)

"p prior draws at the nodes of the KernelPacket kp, n x p."
prior_sample(kp::KernelPacket, p=1; rng=Random.default_rng()) = sample_(kp, randn(rng, length(kp), p))

"""p posterior draws at xnew given y at the nodes of the KernelPacket kp, m x p: Matheron's rule with noise ratio
tau2 (C++)."""
function posterior_sample(kp::KernelPacket, xnew::AbstractVector, y::AbstractVector, p=1; rng=Random.default_rng())
    xs, m = collect(Float64, xnew), length(xnew)
    Z, out = randn(rng, length(kp) + m + (kp.tau2 > 0 ? length(kp) : 0), p), Matrix{Float64}(undef, m, p)
    ccall((:kp_posterior_sample, LIB), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, Int64, Ptr{Float64}, Ptr{Float64}, Int64, Ptr{Float64}),
          kp.h, xs, m, collect(Float64, y), Z, p, out)
    out
end

"p posterior draws at the rows of xnew of a SparseGridRegression, m x p: mean + sqrt(sigma2) M(xnew, U) L^{-T} z."
function posterior_sample(m::SparseGridRegression, xnew::AbstractVecOrMat, p=1; rng=Random.default_rng())
    M = matern(reshape(xnew, size(xnew, 1), :), m.sg.X, m.sg.k, m.sg.rho)
    predict(m, xnew) .+ sqrt(m.sigma2) .* (M * (m.L' \ randn(rng, length(m.sg), p)))
end
