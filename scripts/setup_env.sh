#!/usr/bin/env bash
# Environment of the package, in two uses:
#   source scripts/setup_env.sh          the run-time environment only (scripts/run_*.sh source it): the repository
#                                        root and the job's CPUs under any scheduler (or none), the threads of every
#                                        layer, the GPU or CPU devices, the caches
#   scripts/setup_env.sh <language>      also set up a language (below)
#
# Set up one language of the package and build the C++ core it calls with the compiler and flags pinned in
# cpp/toolchain.txt (g++ 14.1.0, C++20, -march=x86-64-v3: identical binaries on every machine; the pinned g++ is
# installed into .gcc/ from conda-forge when the system one has another version):
#   python: python/.venv (directory and prompt name) with Python 3.14.0 and the exact versions of python/uv.lock (uv pinned to
#           UV_VERSION, installed into ~/.local/bin if missing): pass --jax (JAX only, CUDA wheels on Linux x86-64 unless
#           --cpu), --torch (PyTorch 2.14.1 + CUDA 12.6 on Linux), --numpy (NumPy/SciPy core only), or
#           --all (default: both JAX and PyTorch); builds build/libkernel_packets.so, and on a GPU node with nvcc
#           build/libkernel_packets_cuda.so (on the first import)
#   matlab: MATLAB (on the PATH or `module load matlab`, matlab/resources/mpackage.json: >= R2023b, no toolbox),
#           build/kp_mex.<mexext> and the tests (matlab/buildfile.m)
#   cpp:    the C++ core alone: build/libkernel_packets.so, the tests build/test_kernel_packets (run) and the examples
#           build/example_regression, build/example_sampling (cpp/tests/, cpp/examples/)
#   julia:  Julia JULIA_VERSION, the official binary unpacked into .julia/ (no system or profile changes), the packages of
#           julia/Project.toml at the versions of julia/Manifest.toml in the depot
#           .julia/depot (JULIA_DEPOT_PATH); build/libkernel_packets.so (julia/deps/build.jl)
#
#   scripts/setup_env.sh cpp
#   scripts/setup_env.sh python [--jax|--torch|--numpy|--all] [--cpu]
#   scripts/setup_env.sh matlab
#   scripts/setup_env.sh julia
#   scripts/setup_env.sh all             # cpp, python, matlab, julia
#   scripts/setup_env.sh cleanup         # remove caches and build/
# Pinned: python (uv.lock: every wheel by hash), julia (the binary version, Manifest.toml), C++ (cpp/toolchain.txt:
# compiler version checked, standard, target); MATLAB is licensed: loaded, release checked by matlab/buildfile.m.

# Repository root: the submission directory under a scheduler (Slurm, LSF and Grid Engine may run a spooled copy of
# the script; submit from the root), else the parent of scripts/
for d in "${SLURM_SUBMIT_DIR:-}" "${PBS_O_WORKDIR:-}" "${LS_SUBCWD:-}" "${SGE_O_WORKDIR:-}" \
         "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"; do
    if [ -n "$d" ] && [ -f "$d/cpp/toolchain.txt" ]; then cd "$d"; break; fi
done

# CPUs of the job: Slurm, PBS, LSF, Grid Engine, else the CPUs this process may use (affinity and cgroup aware)
KP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${NCPUS:-${LSB_DJOB_NUMPROC:-${NSLOTS:-$(nproc)}}}}"
# Threads of every layer = the job's CPUs: the C++ core, BLAS / OpenMP (NumPy, JAX on the CPU, torch, MATLAB, Julia),
# Julia tasks; without this each layer starts one thread per core of the node and oversubscribes the job's CPUs.
export KP_NUM_THREADS OMP_NUM_THREADS="$KP_NUM_THREADS" MKL_NUM_THREADS="$KP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$KP_NUM_THREADS" JULIA_NUM_THREADS="$KP_NUM_THREADS"
export JULIA_PROJECT="$PWD/julia" JULIA_DEPOT_PATH="$PWD/.julia/depot" PYTHONDONTWRITEBYTECODE=1
export GKSwstype="${GKSwstype:-100}"
# the pinned g++ of cpp/toolchain.txt in .gcc/ (installed by check_cxx) before the system one
[ -x .gcc/env/bin/g++ ] && export PATH="$PWD/.gcc/env/bin:$PATH"

# Devices: JAX and torch run on the GPU when one is visible, otherwise on the CPU. GPU memory on demand: JAX does not
# preallocate 75% of the device; torch's caching allocator grows expandable segments instead of fragmenting.
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KP_DEVICE=gpu
nvidia-smi -L >/dev/null 2>&1 || { export JAX_PLATFORMS=cpu; KP_DEVICE=cpu; }
# JAX compiles each jitted function per input shape (XLA); the programs are cached on disk (python/.jax_cache, keyed
# by device) so compilation is paid once. torch runs eagerly here (no torch.compile), so it has nothing to cache.
export JAX_COMPILATION_CACHE_DIR="$PWD/python/.jax_cache" JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

# nvcc of cpp/toolchain.txt on a GPU node: from the PATH, CUDA_HOME or the module system (cuda/<major.minor>)
if [ "$KP_DEVICE" = gpu ] && ! command -v nvcc >/dev/null && [ -z "${CUDA_HOME:-}" ]; then
    v=$(sed -n 3p cpp/toolchain.txt | cut -d' ' -f2)
    source /etc/profile.d/modules.sh 2>/dev/null && module load "cuda/${v%.*}" 2>/dev/null || true
fi

# MATLAB from the PATH or the module system (lmod may reference unset variables under set -u)
matlab_cmd() {
    if ! command -v matlab >/dev/null; then
        local _u=0; [[ $- == *u* ]] && { _u=1; set +u; }
        source /etc/profile.d/modules.sh 2>/dev/null && module load matlab
        (( _u )) && set -u
    fi
    matlab "$@"
}

# Sourced: the environment only
(return 0 2>/dev/null) && return 0

set -euo pipefail

action="${1:-python}"
UV_VERSION=0.13.4
JULIA_VERSION=1.12.0
export PATH="$HOME/.local/bin:$PATH"
# compiler and flags of cpp/toolchain.txt (pinned for identical binaries on every machine)
read -r CXX CXX_VERSION < cpp/toolchain.txt
CXXFLAGS=$(sed -n 2p cpp/toolchain.txt)
check_cxx() {
    [ -n "${KP_ALLOW_ANY_CXX:-}" ] && return 0
    if [ "$("$CXX" -dumpfullversion 2>/dev/null)" != "$CXX_VERSION" ]; then
        # micromamba (one static binary) installs gxx of conda-forge into .gcc/env; the specs file of that package
        # adds an rpath to .gcc/env/lib, removed so the binaries use the system libstdc++
        [ -x .gcc/bin/micromamba ] || { mkdir -p .gcc; curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest |
            tar -xj -C .gcc bin/micromamba; }
        rm -rf .gcc/env
        MAMBA_ROOT_PREFIX="$PWD/.gcc/root" .gcc/bin/micromamba create -y -q -p .gcc/env -c conda-forge \
            --override-channels "gxx=$CXX_VERSION"
        rm -rf .gcc/root .gcc/env/lib/gcc/*/"$CXX_VERSION"/specs
        export PATH="$PWD/.gcc/env/bin:$PATH"
    fi
    found=$("$CXX" -dumpfullversion)
    [ "$found" = "$CXX_VERSION" ] || { echo "cpp/toolchain.txt pins $CXX $CXX_VERSION, found $found" >&2; exit 1; }
}
build_lib() {
    check_cxx
    mkdir -p build
    $CXX $CXXFLAGS -shared -Icpp/include cpp/*.cpp -o build/libkernel_packets.so
}

case "$action" in
    cpp)
        build_lib
        for f in cpp/tests/*.cpp cpp/examples/*.cpp; do
            b=$(basename "$f" .cpp); [ "${f#cpp/examples/}" != "$f" ] && b=example_$b
            $CXX $CXXFLAGS -Icpp/include "$f" cpp/*.cpp -o "build/$b"
        done
        build/test_kernel_packets ;;
    all)
        for l in cpp python matlab julia; do "$0" "$l" "${@:2}" || echo "$l: failed" >&2; done ;;
    python)
        device=cpu
        [ "$(uname -s)-$(uname -m)" = Linux-x86_64 ] && device=cuda
        framework=all
        for arg in "${@:2}"; do
            case "$arg" in
                --cpu) device=cpu ;;
                --jax) framework=jax ;;
                --torch) framework=torch ;;
                --numpy) framework=numpy ;;
                --all) framework=all ;;
                *) echo "usage: $0 python [--jax|--torch|--numpy|--all] [--cpu]" >&2; exit 2 ;;
            esac
        done
        if [ "$(uv --version 2>/dev/null | cut -d' ' -f2)" != "$UV_VERSION" ]; then
            curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh
        fi
        extras=(--extra test)
        case "$framework" in
            jax)
                extras+=(--extra jax)
                [ "$device" = cuda ] && extras+=(--extra cuda) ;;
            torch)
                extras+=(--extra torch) ;;
            numpy) ;;
            all)
                extras+=(--extra all)
                [ "$device" = cuda ] && extras+=(--extra cuda) ;;
        esac
        cd python
        [ -x .venv/bin/python ] && grep -qx "prompt = .venv" .venv/pyvenv.cfg ||
            uv venv --clear --python 3.14.0 --prompt .venv .venv
        uv sync --locked "${extras[@]}"
        cd ..
        build_lib
        python/.venv/bin/python -c '
import importlib.util, sys, kernel_packets
parts = [f"Python {sys.version.split()[0]}"]
if importlib.util.find_spec("jax"):
    import jax
    parts.append(f"JAX {jax.__version__}: {jax.devices()}")
if importlib.util.find_spec("torch"):
    import torch
    parts.append(f"torch {torch.__version__}: cuda={torch.cuda.is_available()}")
if len(parts) == 1:
    parts.append("NumPy-only core")
print("; ".join(parts))
' ;;
    matlab)
        (cd matlab && matlab_cmd -batch "maxNumCompThreads($KP_NUM_THREADS); disp(version); buildtool test") ;;
    julia)
        if [ ! -x ".julia/julia-$JULIA_VERSION/bin/julia" ]; then
            mkdir -p .julia
            curl -fsSL "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION%.*}/julia-$JULIA_VERSION-linux-x86_64.tar.gz" |
                tar -xz -C .julia
        fi
        build_lib
        JULIA_DEPOT_PATH="$PWD/.julia/depot" ".julia/julia-$JULIA_VERSION/bin/julia" --project=julia -e \
            'using Pkg; Pkg.instantiate(); Pkg.build(); using KernelPackets; println("Julia ", VERSION, "; threads ", Threads.nthreads())' ;;
    cleanup)
        find . -path ./python/.venv -prune -o -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
        rm -rf build python/.jax_cache .julia/depot/compiled
        echo "Removed caches and build/." ;;
    *) echo "usage: $0 <cpp|python [--jax|--torch|--numpy|--all] [--cpu]|matlab|julia|all|cleanup>" >&2; exit 2 ;;
esac
