"""PLINK 1 ``.bed`` genotypes, read by variant, and the LD screen against them.

An LD reference is often a PLINK fileset rather than a precomputed matrix.
:class:`Bed` reads selected variants as ``int8`` allele counts (``-1`` for a
missing call) without touching the rest of the file, and
:func:`ld_consistency_screen_genotypes` runs the split-half screen one block at
a time, so a genome-wide reference never holds more than one block's LD.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from ._common import integer_at_least, validate_seed
from .warning_markers import LD_SCREEN_HIGH_DROP, LD_SCREEN_UNTESTED

__all__ = ["Bed", "ld_consistency_screen_genotypes", "read_sample_list"]

_MAGIC = b"\x6c\x1b\x01"                 # PLINK 1, variant-major
# 2-bit code -> count of the .bim's first allele (A1): 00 hom A1, 01 missing,
# 10 het, 11 hom A2. Sample k of a byte sits in bits 2k..2k+1.
_CODE_TO_COUNT = np.array([2, -1, 1, 0], dtype=np.int8)
_BYTE_TO_COUNTS = _CODE_TO_COUNT[(np.arange(256)[:, None] >> (2 * np.arange(4))) & 3]


def _prefix(path):
    path = os.fspath(path)
    for ext in (".bed", ".bim", ".fam"):
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def _read_fam_iids(path):
    iids = []
    with open(path) as fh:
        for line_no, line in enumerate(fh, start=1):
            f = line.split()
            if not f:
                continue
            if len(f) < 2:
                raise ValueError(f"{path}: line {line_no} has no IID column")
            iids.append(f[1])
    return np.array(iids, dtype=object)


def read_sample_list(path):
    """IIDs from a PLINK ``--keep``-style file: ``FID IID`` or one ``IID`` per line."""
    out = []
    with open(path) as fh:
        for line in fh:
            f = line.split()
            if f and not f[0].startswith("#"):
                out.append(f[1] if len(f) > 1 else f[0])
    return out


class Bed:
    """A PLINK 1 fileset: ``prefix.bed`` with its ``.bim`` and ``.fam``.

    ``variants`` (the ``.bim`` table) is read on construction unless given.
    ``samples`` restricts reads to some ``.fam`` rows: IIDs (strings), row
    indices, or a boolean mask -- for instance one ancestry of a mixed panel.
    Genotypes are read on demand.
    """

    def __init__(self, path, *, variants=None, samples=None):
        from .variants import read_bim
        prefix = _prefix(path)
        self.path = prefix + ".bed"
        self.variants = read_bim(prefix + ".bim") if variants is None else variants
        self.iid = _read_fam_iids(prefix + ".fam")
        self.n_variants = len(self.variants)
        n_fam = self.iid.size
        if n_fam == 0:
            raise ValueError(f"{prefix}.fam lists no samples")
        self._rows = self._sample_rows(samples, n_fam, prefix)
        self.n_samples = n_fam if self._rows is None else int(self._rows.size)
        self._n_fam = n_fam
        self._stride = (n_fam + 3) // 4
        with open(self.path, "rb") as fh:
            if fh.read(3) != _MAGIC:
                raise ValueError(f"{self.path}: not a variant-major PLINK 1 .bed")
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
        expected = 3 + self._stride * self.n_variants
        if size != expected:
            raise ValueError(
                f"{self.path}: {size - 3} genotype bytes, expected {expected - 3} "
                f"for {self.n_variants} variants x {n_fam} samples")

    def read(self, variant_idx=None):
        """Allele counts of the ``.bim``'s A1, ``(n_samples, n_selected)`` int8.

        ``variant_idx`` selects rows of the ``.bim`` (``None``: all of them);
        consecutive rows are read in one call. ``-1`` marks a missing call.
        """
        if variant_idx is None:
            idx = np.arange(self.n_variants, dtype=np.int64)
        else:
            idx = np.asarray(variant_idx, dtype=np.int64).reshape(-1)
            if idx.size and (idx.min() < 0 or idx.max() >= self.n_variants):
                raise ValueError("variant_idx is out of range")
        raw = np.empty((idx.size, self._stride), dtype=np.uint8)
        with open(self.path, "rb") as fh:
            start = 0
            for stop in np.append(np.flatnonzero(np.diff(idx) != 1) + 1, idx.size):
                if stop == start:
                    continue
                fh.seek(3 + int(idx[start]) * self._stride)
                count = int(stop - start) * self._stride
                chunk = np.fromfile(fh, dtype=np.uint8, count=count)
                if chunk.size != count:
                    raise ValueError(f"{self.path}: truncated genotype data")
                raw[start:stop] = chunk.reshape(-1, self._stride)
                start = stop
        counts = _BYTE_TO_COUNTS[raw].reshape(idx.size, 4 * self._stride)
        counts = counts[:, :self._n_fam]
        if self._rows is not None:
            counts = counts[:, self._rows]
        return np.ascontiguousarray(counts.T)

    def _sample_rows(self, samples, n_fam, prefix):
        if samples is None:
            return None
        samples = np.asarray(samples)
        if samples.dtype == bool:
            if samples.shape != (n_fam,):
                raise ValueError(f"a sample mask needs {n_fam} entries")
            rows = np.flatnonzero(samples)
        elif samples.dtype.kind in "iu":
            rows = np.unique(samples.astype(np.int64))
            if rows.size and (rows[0] < 0 or rows[-1] >= n_fam):
                raise ValueError("sample index out of range")
        else:
            wanted = {str(s) for s in samples.tolist()}
            rows = np.flatnonzero([iid in wanted for iid in self.iid])
            missing = len(wanted) - len({self.iid[r] for r in rows})
            if missing:
                warnings.warn(f"{missing:,} requested samples are not in "
                              f"{prefix}.fam", RuntimeWarning, stacklevel=3)
        if rows.size == 0:
            raise ValueError("the sample selection keeps no samples")
        return rows


def _block_seed(seed, block):
    """An independent 32-bit seed for one block, fixed by ``(seed, block)``."""
    return int(np.random.SeedSequence([seed, block]).generate_state(1)[0])


def ld_consistency_screen_genotypes(bed, var_index, z, *, max_block=2000,
                                    seed=0, progress=None, return_stats=False,
                                    **screen_options):
    """The split-half LD screen with LD computed from reference genotypes.

    ``z[i]`` is the z-score of ``bed`` row ``var_index[i]`` (each row at most
    once), on the ``.bim``'s A1 -- what :func:`allelign.harmonize` returns as
    ``var_index`` and ``beta / se``. Each chromosome, in ``.bim`` order, is
    cut into near-equal blocks of at most ``max_block`` variants; every block's
    correlation matrix is formed from the genotypes, screened with
    :func:`allelign.ld_consistency_screen` (``screen_options``: ``rounds``,
    ``window``, ``threshold``, ``eigenvalue_floor``) and released. Each block
    draws its splits from a stream fixed by ``(seed, block number)``.

    Returns the keep mask aligned to ``z`` (and, with ``return_stats``,
    ``n_tested``, ``n_untested``, ``n_dropped`` and ``n_blocks``). Drop-rate
    and untested-fraction warnings are issued once, for the whole screen.

    A small reference makes the screen drop more: with a few hundred samples
    the sampling noise in R alone can flag several percent of variants (see
    docs/guide.md). Prefer the LD the statistics will be fitted with.
    """
    from .ldcheck import (DEFAULT_DROP_FRACTION_WARN,
                          DEFAULT_UNTESTED_FRACTION_WARN, MIN_WINDOW,
                          chromosome_blocks, ld_blocks_from_genotypes,
                          ld_consistency_screen)
    if not isinstance(bed, Bed):
        raise TypeError("bed must be an allelign.Bed")
    var_index = np.asarray(var_index, dtype=np.int64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if z.shape != var_index.shape:
        raise ValueError("z and var_index must have the same length")
    if var_index.size and (var_index.min() < 0
                           or var_index.max() >= bed.n_variants):
        raise ValueError("var_index is out of range for the .bed")
    if np.unique(var_index).size != var_index.size:
        raise ValueError("var_index must name each .bed row at most once")
    max_block = integer_at_least("max_block", max_block, MIN_WINDOW)
    seed = validate_seed(seed)
    seed = 0 if seed is None else seed

    order = np.argsort(var_index, kind="stable")
    rows = var_index[order]
    sizes = chromosome_blocks(np.asarray(bed.variants.chrom, dtype=object)[rows],
                              max_size=max_block) if rows.size else []
    keep_sorted = np.ones(rows.size, dtype=bool)
    n_tested = n_untested = 0
    start = 0
    with warnings.catch_warnings():
        # Per-block rate warnings would repeat; one summary follows.
        for marker in (LD_SCREEN_HIGH_DROP, LD_SCREEN_UNTESTED):
            warnings.filterwarnings("ignore", message=f".*{marker}",
                                    category=RuntimeWarning)
        for b, size in enumerate(sizes):
            sl = slice(start, start + size)
            blocks = ld_blocks_from_genotypes(bed.read(rows[sl]), [size])
            keep, stats = ld_consistency_screen(
                blocks, z[order][sl], seed=_block_seed(seed, b),
                return_stats=True, **screen_options)
            keep_sorted[sl] = keep
            n_tested += stats["n_tested"]
            n_untested += stats["n_untested"]
            start += size
            if progress is not None:
                progress({"step": "LD consistency screen", "done": b + 1,
                          "total": len(sizes), "unit": "block"})
    keep = np.empty_like(keep_sorted)
    keep[order] = keep_sorted
    n_dropped = int((~keep).sum())
    total = int(keep.size)
    if n_tested and n_dropped / n_tested > DEFAULT_DROP_FRACTION_WARN:
        warnings.warn(
            f"{LD_SCREEN_HIGH_DROP} {n_dropped:,} of {n_tested:,} tested "
            f"variants ({n_dropped / n_tested:.0%}). With LD from reference "
            "genotypes, a small reference panel is the usual cause: sampling "
            "noise in R flags variants that are consistent with the true LD.",
            RuntimeWarning, stacklevel=2)
    if n_untested and (n_tested == 0
                       or n_untested / total >= DEFAULT_UNTESTED_FRACTION_WARN):
        warnings.warn(
            f"LD-consistency screen tested {n_tested:,} of {total:,} variants "
            f"({n_untested:,} {LD_SCREEN_UNTESTED} of size {MIN_WINDOW} and "
            "were kept unevaluated).", RuntimeWarning, stacklevel=2)
    if return_stats:
        return keep, {"n_tested": int(n_tested), "n_untested": int(n_untested),
                      "n_dropped": n_dropped, "n_blocks": len(sizes)}
    return keep
