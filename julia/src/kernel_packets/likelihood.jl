# Profile likelihood of the lengthscale (C++) and its maximization, for KernelPacket and SparseGrid models
# (as likelihood.py)

for (T, f) in ((KernelPacket, :kp_loglike), (SparseGrid, :sg_loglike))
    @eval function loglike(model::$T, y::AbstractVector, F::AbstractVecOrMat=ones(length(y), 1))
        "Log-likelihood at the model's rho with mean F beta and variance sigma2 at their MLEs: (loglike, beta, sigma2)."
        Fm = Matrix{Float64}(reshape(F, size(F, 1), :))
        beta, sigma2 = Vector{Float64}(undef, size(Fm, 2)), Ref(0.0)
        L = ccall(($(QuoteNode(f)), LIB), Cdouble, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Ref{Cdouble}),
                  model.h, collect(Float64, y), Fm, size(Fm, 2), beta, sigma2)
        L, beta, sigma2[]
    end
end

"""Maximize the log-likelihood over a scalar rho in bounds (Brent on log rho), or, with tau2_bounds (a KernelPacket
of the stable method), over rho and tau2; returns (theta, L_hat, L_init), the model's rho (and tau2) set."""
function mle(model, y; F=ones(length(y), 1), bounds=(5e-2, 5.0), tau2_bounds=nothing)
    L_init = loglike(model, y, F)[1]
    if tau2_bounds === nothing
        function nll_rho(z)
            set_rho!(model, exp(z))
            L = loglike(model, y, F)[1]
            isfinite(L) ? -L : 1e30
        end
        rho = clamp(exp(brent(nll_rho, log(bounds[1]), log(bounds[2]); tol=1e-3).minimizer), bounds[1], bounds[2])
        set_rho!(model, rho)
    else
        function nll_joint(z)
            set_params!(model, clamp(exp(z[1]), bounds[1], bounds[2]), clamp(exp(z[2]), tau2_bounds[1], tau2_bounds[2]))
            L = loglike(model, y, F)[1]
            isfinite(L) ? -L : 1e30
        end
        lo = [log(bounds[1]), log(tau2_bounds[1])]
        hi = [log(bounds[2]), log(tau2_bounds[2])]
        z0 = 0.5 .* (lo .+ hi)
        z = nelder_mead(nll_joint, z0, lo, hi; xtol=1e-3, ftol=1e-6).minimizer
        set_params!(model, clamp(exp(z[1]), bounds[1], bounds[2]), clamp(exp(z[2]), tau2_bounds[1], tau2_bounds[2]))
    end
    L_hat, beta, sigma2 = loglike(model, y, F)
    tau2 = model isa KernelPacket ? model.tau2 : 0.0
    (rho=model.rho, tau2=tau2, beta=beta, sigma2=sigma2), L_hat, L_init
end
