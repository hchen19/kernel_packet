# Matern-(k-2)/2 kernel matrices (dense algebra) and kernel packets on sorted 1-D nodes (C++ core): method :stable
# (default, noise ratio tau2) or :original (noise-free: K^{-1} = A Phi^{-1}, log|K| = log|Phi| - log|A|).

"Coefficients of P in K(r) = P(z) e^{-z}, z = sqrt(k-2) |r| / rho, lowest degree first."
function matern_poly(k)
    p = (k - 3) ÷ 2
    c = ones(p + 1)
    for j in 0:p-1
        c[j+2] = c[j+1] * 2 * (p - j) / ((2p - j) * (j + 1))
    end
    Tuple(c)
end

"Kernel matrix between the rows of X1 (m x d) and X2 (n x d) (or 1-D points) with lengthscales rho, accumulated one
dimension at a time (memory of one m x n matrix)."
function matern(X1::AbstractVecOrMat, X2::AbstractVecOrMat, k::Integer, rho)
    P, d = matern_poly(k), size(X1, 2)
    c = sqrt(k - 2) ./ (rho .* ones(d))
    prod, s = fill!(similar(X1, size(X1, 1), size(X2, 1)), 1), fill!(similar(X1, size(X1, 1), size(X2, 1)), 0)
    for i in 1:d
        Z = abs.(view(X1, :, i) .- view(X2, :, i)') .* c[i]
        prod .*= evalpoly.(Z, Ref(P))
        s .+= Z
    end
    prod .* exp.(-s)
end

"""Kernel packets of the Matern-(k-2)/2 kernel with lengthscale rho on sorted nodes x, noise ratio tau2; method :stable
(default; k <= 7 and n >= k, else the original is taken): the stable KPs, or :original (noise-free)."""
mutable struct KernelPacket
    x::Vector{Float64}
    k::Int
    rho::Float64
    tau2::Float64
    method::Symbol
    h::Ptr{Cvoid}
    function KernelPacket(x::AbstractVector, k::Integer=5, rho::Real=sqrt(k - 2); tau2::Real=0.0, method::Symbol=:stable)
        kp = new(collect(Float64, x), k, rho, tau2, method, C_NULL)
        set_rho!(kp, rho)
        finalizer(free!, kp)
    end
end

Base.length(kp::KernelPacket) = length(kp.x)

function free!(kp::KernelPacket)
    kp.h == C_NULL || ccall((:kp_free, LIB), Cvoid, (Ptr{Cvoid},), kp.h)
    kp.h = C_NULL
end

function set_rho!(kp::KernelPacket, rho::Real)
    kp.rho = rho
    if kp.h == C_NULL
        kp.h = ccall((:kp_new, LIB), Ptr{Cvoid}, (Ptr{Float64}, Int64, Cint, Cdouble, Cdouble, Cint), kp.x, length(kp.x),
                     kp.k, rho, kp.tau2, kp.method === :stable ? 0 : 1)
        kp.method = ccall((:kp_method, LIB), Cint, (Ptr{Cvoid},), kp.h) == 0 ? :stable : :original
    else
        ccall((:kp_set_params, LIB), Cvoid, (Ptr{Cvoid}, Cdouble, Cdouble), kp.h, rho, kp.tau2)
    end
    kp
end

function set_params!(kp::KernelPacket, rho::Real, tau2::Real)
    kp.rho = Float64(rho)
    kp.tau2 = Float64(tau2)
    ccall((:kp_set_params, LIB), Cvoid, (Ptr{Cvoid}, Cdouble, Cdouble), kp.h, kp.rho, kp.tau2)
    kp
end

logdet(kp::KernelPacket) = ccall((:kp_logdet, LIB), Cdouble, (Ptr{Cvoid},), kp.h)
nbytes(kp::KernelPacket) = ccall((:kp_bytes, LIB), Int64, (Ptr{Cvoid},), kp.h)

"K^{-1} Y (Phi^{-1} Y if not a) for Y with n rows."
function solve(kp::KernelPacket, Y::AbstractVecOrMat; a=true)
    W = Matrix{Float64}(reshape(Y, size(Y, 1), :))
    ccall((:kp_solve, LIB), Cvoid, (Ptr{Cvoid}, Ptr{Float64}, Int64, Cint), kp.h, W, size(W, 2), a)
    Y isa AbstractVector ? vec(W) : W
end

"Phi L^{-T} Z (B = L L') for Z with n rows: N(0, K) from N(0, I)."
function sample_(kp::KernelPacket, Z::AbstractVecOrMat)
    W = Matrix{Float64}(reshape(Z, size(Z, 1), :))
    ccall((:kp_sample, LIB), Cvoid, (Ptr{Cvoid}, Ptr{Float64}, Int64), kp.h, W, size(W, 2))
    Z isa AbstractVector ? vec(W) : W
end

"KPs at xnew: phi_{start[j] + t}(xnew[j]) = V[t + 1, j], start 1-based."
function basis(kp::KernelPacket, xnew::AbstractVector)
    xs = collect(Float64, xnew)
    w = ccall((:kp_width, LIB), Cint, (Ptr{Cvoid},), kp.h)
    start, V = Vector{Int64}(undef, length(xs)), Matrix{Float64}(undef, w, length(xs))
    ccall((:kp_basis, LIB), Cvoid, (Ptr{Cvoid}, Ptr{Float64}, Int64, Ptr{Int64}, Ptr{Float64}), kp.h, xs, length(xs),
          start, V)
    start .+ 1, V
end
