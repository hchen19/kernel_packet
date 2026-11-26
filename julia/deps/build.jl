# Pkg.build hook: compile the C++ core (cpp/) into build/libkernel_packets.so (also done on `using KernelPackets` when
# missing or older than a source)
include(joinpath(@__DIR__, "..", "src", "kernel_packets", "backend.jl"))
build()
