"""Kernel packets for Matern-(k-2)/2 Gaussian processes: exact O(n) in 1-D, sparse-grid inducing points in d
dimensions; the kernel-packet work in the C++ core (cpp/) through ccall, the dense algebra in BLAS/LAPACK."""
module KernelPackets

using LinearAlgebra, Random, Statistics
import LinearAlgebra: logdet

include("kernel_packets/backend.jl")
include("kernel_packets/optimize.jl")
include("kernel_packets/kernel_packets.jl")
include("kernel_packets/sparse_grid.jl")
include("kernel_packets/regression.jl")
include("kernel_packets/sampling.jl")
include("kernel_packets/likelihood.jl")

__init__() = build()   # compile build/libkernel_packets.so when missing or older than a source

export KernelPacket, SparseGrid, SparseGridRegression, matern, solve, sample_, basis, logdet, set_rho!, set_params!,
       nbytes, for_data, posterior, predict, fit!, train!, prior_sample, posterior_sample, loglike, mle, hc, pl

module Benchmarks
include("benchmarks/dense_gp.jl")
end

module Evaluation
include("evaluation/metrics.jl")
include("evaluation/plotting.jl")
end

end
