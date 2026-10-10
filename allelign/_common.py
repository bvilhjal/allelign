"""Small shared helpers: control validation, text input and atomic output."""

from __future__ import annotations

import gzip
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def finite_control(name, value, *, lower=None, upper=None):
    """Coerce a finite scalar control, rejecting booleans and vectors."""
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number") from None
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if lower is not None and value < lower:
        raise ValueError(f"{name} must be >= {lower}")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must be <= {upper}")
    return value


def integer_at_least(name, value, minimum):
    """Return an integer control, rejecting bools and silent coercion."""
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_boolean(**controls):
    """Reject truthy/falsy surrogates for controls that are genuinely boolean."""
    for name, value in controls.items():
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")


def validate_seed(seed):
    """Return a seed usable by ``numpy.random.SeedSequence``, or ``None``."""
    if seed is None:
        return None
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or not 0 <= int(seed) <= np.iinfo(np.uint32).max):
        raise ValueError("seed must be an integer in [0, 2**32 - 1] or None")
    return int(seed)


def open_text(path):
    """Open a plain or gzip/bgzip text file for reading.

    ``utf-8-sig`` strips a UTF-8 byte-order mark, which Excel and several GWAS
    portals emit; left in place it glues itself to the first header token,
    which then matches no alias, so the first column (usually the variant ID)
    is silently never mapped.
    """
    name = os.fspath(path)
    if name.endswith((".gz", ".bgz")):
        return gzip.open(name, "rt", encoding="utf-8-sig")
    return open(name, "r", encoding="utf-8-sig")


@contextmanager
def atomic_text(path):
    """Write text so that only a complete file is ever published at ``path``.

    A ``.gz`` destination is gzip-compressed. The temporary file lives on the
    destination filesystem, so the final ``os.replace`` is atomic.
    """
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.",
                                     dir=path.parent or ".")
    try:
        if path.name.endswith(".gz"):
            os.close(fd)
            with gzip.open(temporary, "wt", encoding="utf-8") as fh:
                yield fh
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                yield fh
                fh.flush()
                os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
