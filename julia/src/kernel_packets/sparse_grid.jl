# Sparse grids from nested 1-D designs, the inducing points of d-dimensional regression (C++ core): design, K^{-1} by
# the combination technique over tensor grids of 1-D kernel packets, log|K| (as sparse_grid.py)

"Hyperbolic cross (bisection) design of level j: 2^j - 1 points in (lo, hi), nested in j."
hc(j, lo=0.0, hi=1.0) = lo .+ (hi - lo) .* (1:2^j-1) ./ 2^j

const PL = [0.5 0.5; 0.125 0.875; 0.25 0.75; 0 1; 0.375 0.625; 0.25-1/16 0.75+1/16; 1-1/16 1/16; 0.5-1/16 0.5+1/16;
            0.3125 0.6875; 0.25+1/32 0.75-1/32; 1/32 1-1/32; 0.875+1/32 0.125-1/32; 0.875-1/32 0.125+1/32;
            0.375+1/32 0.625-1/32; 0.5-1/32 0.5+1/32; 0.25-1/32 0.75+1/32]

"Plumlee's design of level j <= 16, nested in j."
pl(j, lo=0.0, hi=1.0) = lo .+ sort(unique(vec(PL[1:j, :]))) .* (hi - lo)

"""Sparse grid of level eta >= d in [lo, hi]^d: K^{-1} is the sum over |j| = max(d, eta-d+1) .. eta of
(-1)^(eta-|j|) binomial(d-1, eta-|j|) times the tensor-grid K^{-1} of level j; points X (N x d) with sorted rows."""
mutable struct SparseGrid
    d::Int
    eta::Int
    k::Int
    rho::Vector{Float64}
    X::Matrix{Float64}
    h::Ptr{Cvoid}
    function SparseGrid(d, eta; k=5, rho=sqrt(k - 2), design=hc, lo=0.0, hi=1.0)
        nodes = [collect(Float64, design(j, lo, hi)) for j in 0:eta-d+1]
        r = collect(Float64, rho .* ones(d))
        h = ccall((:sg_new, LIB), Ptr{Cvoid}, (Cint, Cint, Ptr{Float64}, Ptr{Int64}, Cint, Ptr{Float64}), d, eta,
                  reduce(vcat, nodes), cumsum([0; length.(nodes)]), k, r)
        X = Matrix{Float64}(undef, d, ccall((:sg_size, LIB), Int64, (Ptr{Cvoid},), h))
        ccall((:sg_points, LIB), Cvoid, (Ptr{Cvoid}, Ptr{Float64}), h, X)
        finalizer(g -> ccall((:sg_free, LIB), Cvoid, (Ptr{Cvoid},), g.h), new(d, eta, k, r, permutedims(X), h))
    end
end

Base.length(g::SparseGrid) = size(g.X, 1)

"""The sparse grid of the smallest level eta with tr(K_XX - Q_XX) / tau2 <= kl for the data X (n x d), within n_max
points (as Python SparseGrid.for_data)."""
function for_data(::Type{SparseGrid}, X::AbstractVecOrMat; k::Integer=5, rho=nothing, tau2::Real=1e-2,
                  n_max::Integer=4000, kl::Real=1.0, design=hc, lo::Real=0.0, hi::Real=1.0)
    Xm = Matrix{Float64}(reshape(X, size(X, 1), :))
    n, d = size(Xm)
    n_max = min(n_max, n)
    r = collect(Float64, (rho === nothing ? sqrt(k - 2.0) : rho) .* ones(d))
    L = max(1, floor(Int, log2(n_max + 1)))
    sizes = [length(design(l, lo, hi)) for l in 0:L]
    D = Array{Float64}(undef, d, L, n)
    for j in 1:d
        v0 = ones(n)
        for l in 1:L
            _, vl = posterior(KernelPacket(design(l, lo, hi), k, r[j]), view(Xm, :, j), zeros(sizes[l + 1]); var=true)
            D[j, l, :] .= v0 .- vl
            v0 = vl
        end
    end
    dn = Float64.(diff(sizes))
    S, C = ones(1, n), [1.0]
    for j in 1:d
        S2, C2 = zeros(size(S, 1) + L, n), zeros(length(C) + L)
        for l in 1:L
            S2[l+1:l+size(S, 1), :] .+= D[j, l, :]' .* S
            C2[l+1:l+length(C)] .+= dn[l] .* C
        end
        S, C = S2, C2
    end
    q, N = cumsum(vec(sum(S; dims=2))), cumsum(C)
    etas = [e for e in d:(length(N) - 1) if N[e + 1] <= n_max]
    isempty(etas) && (etas = [d])
    idx = findfirst(e -> (n - q[e + 1]) / tau2 <= kl, etas)
    eta = idx === nothing ? etas[end] : etas[idx]
    SparseGrid(d, eta; k=k, rho=r, design=design, lo=lo, hi=hi)
end

function set_rho!(g::SparseGrid, rho)
    g.rho = collect(Float64, rho .* ones(g.d))
    ccall((:sg_set_rho, LIB), Cvoid, (Ptr{Cvoid}, Ptr{Float64}), g.h, g.rho)
    g
end

"K^{-1} Y for Y with N rows."
function solve(g::SparseGrid, Y::AbstractVecOrMat)
    Z = Matrix{Float64}(reshape(Y, size(Y, 1), :))
    W = similar(Z)
    ccall((:sg_solve, LIB), Cvoid, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Int64), g.h, Z, W, size(Z, 2))
    Y isa AbstractVector ? vec(W) : W
end

logdet(g::SparseGrid) = ccall((:sg_logdet, LIB), Cdouble, (Ptr{Cvoid},), g.h)
nbytes(g::SparseGrid) = ccall((:sg_bytes, LIB), Int64, (Ptr{Cvoid},), g.h)

"tr(K^{-1} G) for an N x N matrix G (C++, O(N) memory per thread)."
trace_solve(g::SparseGrid, G::AbstractMatrix) =
    ccall((:sg_trace_solve, LIB), Cdouble, (Ptr{Cvoid}, Ptr{Float64}), g.h, Matrix{Float64}(G))
