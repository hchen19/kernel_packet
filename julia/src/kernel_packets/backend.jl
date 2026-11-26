# C++ core (cpp/) through ccall: build/libkernel_packets.so, compiled when missing or older than a source (as
# python/src/kernel_packets/backend.py); matrices are column-major (columns contiguous), as cpp/include/c_api.hpp

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const CPP = joinpath(ROOT, "cpp")
const LIB = joinpath(ROOT, "build", "libkernel_packets.so")

"Compiler and flags of cpp/toolchain.txt (pinned for identical binaries on every machine): the g++ of .gcc/ (installed
by scripts/setup_env.sh) or of the PATH, which must have the pinned version."
function toolchain()
    (cxx, version), flags = split.(readlines(joinpath(CPP, "toolchain.txt"))[1:2])
    local_cxx = joinpath(ROOT, ".gcc", "env", "bin", cxx)  # the pinned g++ in .gcc/ (scripts/setup_env.sh)
    isfile(local_cxx) && (cxx = local_cxx)
    found = strip(read(`$cxx -dumpfullversion`, String))
    (found == version || !isempty(get(ENV, "KP_ALLOW_ANY_CXX", ""))) ||
        error("cpp/toolchain.txt pins $cxx $version, found $found")
    [cxx; flags]
end

function build()
    src = [filter(endswith(".cpp"), readdir(CPP; join=true)); readdir(joinpath(CPP, "include"); join=true);
           readdir(joinpath(CPP, "include", "original"); join=true);
           readdir(joinpath(CPP, "include", "stable"); join=true); joinpath(CPP, "toolchain.txt")]
    isfile(LIB) && mtime(LIB) >= maximum(mtime, src) && return
    mkpath(dirname(LIB))
    run(`$(toolchain()) -shared -I$(joinpath(CPP, "include")) $(filter(endswith(".cpp"), src)) -o $LIB`)
end

batch(n; numbers=1 << 24) = max(1, numbers ÷ max(n, 1))
