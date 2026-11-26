"""C++ core through ctypes (cpp/, compiled into build/ on first use and whenever a source is newer; with a GPU and the
pinned nvcc the CUDA build, which also has the CPU functions) and unified lazy dispatch across NumPy, JAX, and PyTorch
(CPU or GPU): GPU arrays stay on device via DLPack/CUDA kernels, host arrays use the CPU core."""
import ctypes
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CPP = ROOT / "cpp"
D = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
I = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
H, I32, I64, F64, P = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_double, ctypes.c_void_p
API = {  # name: (result, arguments), as cpp/include/c_api.hpp
    "kp_new": (H, [D, I64, I32, F64, F64, I32]), "kp_free": (None, [H]), "kp_set_params": (None, [H, F64, F64]),
    "kp_method": (I32, [H]),
    "kp_logdet": (F64, [H]), "kp_width": (I32, [H]), "kp_bytes": (I64, [H]), "sg_bytes": (I64, [H]),
    "kp_solve": (None, [H, D, I64, I32]), "kp_basis": (None, [H, D, I64, I, D]),
    "kp_posterior": (None, [H, D, I64, D, D, P]), "kp_posterior_cov": (None, [H, D, I64, D]),
    "kernel_predict": (None, [D, I64, D, I64, I32, D, I64, I32, D, D]),
    "kp_sample": (None, [H, D, I64]), "kp_posterior_sample": (None, [H, D, I64, D, D, I64, D]),
    "sg_new": (H, [I32, I32, D, I, I32, D]), "sg_free": (None, [H]), "sg_size": (I64, [H]), "sg_points": (None, [H, D]),
    "sg_set_rho": (None, [H, D]), "sg_solve": (None, [H, D, D, I64]), "sg_logdet": (F64, [H]),
    "sg_trace_solve": (F64, [H, D]),
    "kp_loglike": (F64, [H, D, D, I32, D, D]), "sg_loglike": (F64, [H, D, D, I32, D, D]),
}
API_CUDA = {  # device pointers and the caller's CUDA stream
    "sg_cuda_new": (H, [H]), "sg_cuda_free": (None, [H]), "sg_cuda_solve": (None, [H, P, P, I64, P]),
    "sg_cuda_trace_solve": (F64, [H, P, P]), "kernel_predict_cuda": (None, [P, I64, P, I64, I32, P, I64, I32, D, P, P]),
}


class Backend:
    """Compiled C++ shared library handle plus lazy NumPy / JAX / PyTorch array dispatch."""

    def __init__(self):
        self._torch = None
        self._jax = None
        self.lib = self._load()

    def toolchain(self, line=0):
        """Compiler and flags of cpp/toolchain.txt (line 0: g++, line 2: nvcc)."""
        spec = (CPP / "toolchain.txt").read_text().splitlines()
        cxx, version = spec[line].split()
        if line:
            cxx = shutil.which(cxx) or str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / cxx)
            found = re.search(r"V([\d.]+)", subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout)[1]
        else:  # the pinned g++ in .gcc/ (scripts/setup_env.sh) before the PATH
            cxx = shutil.which(cxx, path=f"{ROOT / '.gcc' / 'env' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}") or cxx
            found = subprocess.run([cxx, "-dumpfullversion"], capture_output=True, text=True, check=True).stdout.strip()
        if found != version and not os.environ.get("KP_ALLOW_ANY_CXX"):
            raise RuntimeError(f"cpp/toolchain.txt pins {cxx} {version}, found {found}")
        return [cxx, *spec[line + 1].split()]

    def _build(self, lib_path, cuda):
        src = sorted(CPP.glob("*.cpp")) + sorted((CPP / "include").rglob("*.hpp")) + [CPP / "toolchain.txt"]
        src += sorted((CPP / "cuda").glob("*.cu")) if cuda else []
        if not lib_path.exists() or lib_path.stat().st_mtime < max(f.stat().st_mtime for f in src):
            lib_path.parent.mkdir(exist_ok=True)
            cmd = self.toolchain(2) + ["-ccbin", self.toolchain(0)[0]] if cuda else self.toolchain(0)
            subprocess.run([*cmd, "-shared", f"-I{CPP / 'include'}",
                            *map(str, (f for f in src if f.suffix in (".cpp", ".cu"))), "-o", str(lib_path)], check=True)

    def _gpu(self):
        if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1") or not os.path.exists("/dev/nvidiactl"):
            return False
        try:
            return self.torch().cuda.is_available() and bool(self.toolchain(2))
        except (ImportError, OSError, RuntimeError, TypeError):
            return False

    def _load(self):
        cuda = self._gpu()
        path = ROOT / "build" / ("libkernel_packets_cuda.so" if cuda else "libkernel_packets.so")
        self._build(path, cuda)
        lib = ctypes.CDLL(str(path))
        for name, (res, args) in (API | API_CUDA if cuda else API).items():
            f = getattr(lib, name)
            f.restype, f.argtypes = res, args
        lib.cuda = cuda
        return lib

    @staticmethod
    def is_torch(a):
        return type(a).__module__.startswith("torch")

    @staticmethod
    def is_jax(a):
        return type(a).__module__.startswith("jax")

    def torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    def jax(self):
        if self._jax is None:
            import jax
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
            self._jax = (jax, jnp)
        return self._jax

    def to_numpy(self, a):
        return np.asarray(a.detach().cpu() if self.is_torch(a) else a, dtype=np.float64)

    def like(self, a, ref):
        """Array `a` converted to the framework, dtype, and device of `ref`."""
        arr = np.asarray(a)
        if self.is_torch(ref):
            return self.torch().as_tensor(arr, dtype=ref.dtype, device=ref.device)
        if self.is_jax(ref):
            jax, jnp = self.jax()
            out = jnp.asarray(arr, dtype=ref.dtype)
            devs = list(ref.devices()) if hasattr(ref, "devices") else []
            return jax.device_put(out, devs[0]) if devs else out
        return np.asarray(arr, dtype=np.float64)

    @staticmethod
    def batch(n, numbers=1 << 24):
        """Rows per batch so a chunk holds at most `numbers` entries."""
        return max(1, numbers // max(n, 1))

    def xp(self, ref):
        """Array namespace of `ref`: torch, jax.numpy, or numpy."""
        if self.is_torch(ref):
            return self.torch()
        if self.is_jax(ref):
            return self.jax()[1]
        return np

    def cholesky(self, A):
        return self.xp(A).linalg.cholesky(A)

    def tri_solve(self, L, B, trans=False):
        """L^{-1} B (L^{-T} B if trans) for lower triangular L."""
        if self.is_torch(L):
            if B.ndim == 1:
                return self.tri_solve(L, B[:, None], trans)[:, 0]
            return self.torch().linalg.solve_triangular(L.mT if trans else L, B, upper=trans)
        if self.is_jax(L):
            self.jax()
            from jax.scipy.linalg import solve_triangular
            return solve_triangular(L, B, lower=True, trans=int(trans))
        from scipy.linalg import solve_triangular
        return solve_triangular(L, B, lower=True, trans=int(trans), check_finite=False)

    def add_diag(self, A, val):
        """Add scalar `val` to the diagonal of square matrix `A` (in-place on PyTorch/NumPy)."""
        if self.is_torch(A):
            A.diagonal().add_(val)
            return A
        if self.is_jax(A):
            _, jnp = self.jax()
            return A.at[jnp.diag_indices(A.shape[0])].add(val)
        A.flat[::A.shape[0] + 1] += val
        return A

    def sync(self, out):
        """Wait for asynchronous device work on any PyTorch (CUDA/MPS) or JAX outputs."""
        items = out if isinstance(out, (tuple, list)) else (out,)
        if any(getattr(a, "is_cuda", False) for a in items):
            self.torch().cuda.synchronize()
        if any(getattr(a, "is_mps", False) for a in items):
            self.torch().mps.synchronize()
        for a in items:
            if hasattr(a, "block_until_ready"):
                a.block_until_ready()
        return out

    def cols(self, a):
        """Host float64 copy of (n,) or (n, p) array with columns contiguous: shape (p, n), C order."""
        return np.array(self.to_numpy(a).reshape(len(a), -1).T, order="C")

    def on_gpu(self, a):
        if not self.lib.cuda:
            return False
        if self.is_torch(a):
            return a.is_cuda
        return self.is_jax(a) and hasattr(a, "devices") and next(iter(a.devices())).platform == "gpu"

    def device(self, a):
        """PyTorch CUDA float64 view of a device array (JAX via zero-copy DLPack)."""
        torch = self.torch()
        t = a if self.is_torch(a) else torch.from_dlpack(a)
        return t.to(torch.float64)

    def stream(self):
        return self.torch().cuda.current_stream().cuda_stream

    def back(self, t, ref):
        """Convert a PyTorch CUDA tensor back into the framework of `ref` (JAX via zero-copy DLPack)."""
        if self.is_torch(ref):
            return t
        import jax.dlpack
        return jax.dlpack.from_dlpack(t)


bk = Backend()
