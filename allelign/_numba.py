"""Optional Numba support. Everything has a pure-NumPy/Python fallback."""

from __future__ import annotations

try:  # pragma: no cover - depends on the environment
    import numba as _numba
    from numba import config as numba_config
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - a missing or broken install
    _numba = None
    numba_config = None
    HAVE_NUMBA = False


def jit_available():
    """True when compiled kernels can actually run (not disabled by config)."""
    return HAVE_NUMBA and not getattr(numba_config, "DISABLE_JIT", False)


def jit_nogil(fn):
    """Compile ``fn`` in nopython, GIL-releasing mode with an on-disk cache."""
    return _numba.njit(cache=True, nogil=True)(fn)
