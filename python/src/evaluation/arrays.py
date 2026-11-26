"""Experiment array framework (KP_FRAMEWORK: jax, torch, or numpy; auto-detected when unset) on the fastest device."""
import importlib.util
import os

import numpy as np

from kernel_packets.backend import bk


def _detect():
    env = os.environ.get("KP_FRAMEWORK")
    if env:
        return env.lower()
    return next((f for f in ("jax", "torch") if importlib.util.find_spec(f)), "numpy")


FRAMEWORK = _detect()


def set_framework(name):
    global FRAMEWORK
    FRAMEWORK = name.lower()


def array(a):
    arr = np.asarray(a, np.float64)
    if FRAMEWORK == "torch":
        t = bk.torch()
        return t.as_tensor(arr, dtype=t.float64, device="cuda" if t.cuda.is_available() else "cpu")
    return bk.jax()[1].asarray(arr, dtype=np.float64) if FRAMEWORK == "jax" else arr


def host(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a, dtype=np.float64)
