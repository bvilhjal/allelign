"""Reference variant tables: the variants (and counted alleles) to align to.

A :class:`Variants` table is what summary statistics or weights are aligned to:
an LD reference, a genotype file, or another GWAS. ``a1`` is the counted
allele -- the allele a dosage counts and an aligned effect refers to. The
attribute names match ldpred3's ``VariantTable``, so either table works with
either package's harmoniser.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import warnings

import numpy as np

from ._common import open_text

__all__ = ["Variants", "read_bim", "read_pvar", "read_variants"]


@dataclass
class Variants:
    """Per-variant reference metadata as parallel arrays.

    Mutable by default. :meth:`freeze` marks the matching columns read-only so
    the harmoniser can cache its indices across calls (useful when many files
    are aligned to one reference).
    """

    chrom: np.ndarray
    id: np.ndarray
    pos: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    alternate_id: np.ndarray = None
    cm: np.ndarray = None      # genetic position, when the source has one

    def __len__(self):
        return len(self.id)

    def freeze(self):
        """Mark matching columns read-only, without copying; returns self."""
        was_writable = False
        for name in ("chrom", "id", "pos", "a1", "a2", "alternate_id"):
            column = getattr(self, name)
            if column is not None:
                column = np.asarray(column)
                was_writable |= column.flags.writeable
                column.flags.writeable = False
                setattr(self, name, column)
        if was_writable:
            from .align import _CACHE_ATTR, _INDEX_CACHE_LOCK
            with _INDEX_CACHE_LOCK:
                self.__dict__.pop(_CACHE_ATTR, None)
        return self

    def subset(self, idx):
        """A new (mutable) table keeping the rows in ``idx``."""
        idx = np.asarray(idx)
        return Variants(
            chrom=self.chrom[idx], id=self.id[idx], pos=self.pos[idx],
            a1=self.a1[idx], a2=self.a2[idx],
            alternate_id=(None if self.alternate_id is None
                          else self.alternate_id[idx]),
            cm=None if self.cm is None else self.cm[idx])


def read_bim(path):
    """Read a PLINK 1 ``.bim`` file; ``a1`` is column 5, PLINK's counted allele."""
    chrom, ids, cm, pos, a1, a2 = [], [], [], [], [], []
    pool = {}
    with open_text(path) as fh:
        for line_no, line in enumerate(fh, start=1):
            f = line.split()
            if not f:
                continue
            if len(f) < 6:
                raise ValueError(f"{path}: line {line_no} has {len(f)} fields, "
                                 "expected 6")
            chrom.append(pool.setdefault(f[0], f[0]))
            ids.append(f[1])
            cm.append(float(f[2]))
            pos.append(int(float(f[3])))
            x, y = f[4].upper(), f[5].upper()
            a1.append(pool.setdefault(x, x))
            a2.append(pool.setdefault(y, y))
    return Variants(chrom=np.array(chrom, dtype=object),
                    id=np.array(ids, dtype=object),
                    pos=np.array(pos, dtype=np.int64),
                    a1=np.array(a1, dtype=object),
                    a2=np.array(a2, dtype=object),
                    cm=np.array(cm, dtype=float))


def read_pvar(path):
    """Read a PLINK 2 ``.pvar`` (or VCF-like) variant file.

    ``a1`` is ``ALT``, the allele PLINK 2 dosages count by default, and ``a2``
    is ``REF``. Multi-allelic records (comma in ``ALT``) are skipped with a
    warning: a single dosage cannot represent them.
    """
    chrom, ids, pos, a1, a2 = [], [], [], [], []
    pool = {}
    n_multi = 0
    col = None
    with open_text(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = [h.strip().upper() for h in line[1:].split()]
                col = {name: i for i, name in enumerate(header)}
                continue
            f = line.split()
            if not f:
                continue
            if col is None:
                # A headerless .pvar follows the .bim column order.
                col = {name: i for i, name in enumerate(
                    ["CHROM", "ID", "CM", "POS", "ALT", "REF"])}
            try:
                c, vid = f[col["CHROM"]], f[col["ID"]]
                p, ref, alt = f[col["POS"]], f[col["REF"]], f[col["ALT"]]
            except KeyError as exc:
                raise ValueError(f"{path}: header lacks column {exc}") from None
            if "," in alt:
                n_multi += 1
                continue
            chrom.append(pool.setdefault(c, c))
            ids.append(vid)
            pos.append(int(float(p)))
            alt, ref = alt.upper(), ref.upper()
            a1.append(pool.setdefault(alt, alt))
            a2.append(pool.setdefault(ref, ref))
    if n_multi:
        warnings.warn(f"{path}: skipped {n_multi:,} multi-allelic records",
                      RuntimeWarning, stacklevel=2)
    return Variants(chrom=np.array(chrom, dtype=object),
                    id=np.array(ids, dtype=object),
                    pos=np.array(pos, dtype=np.int64),
                    a1=np.array(a1, dtype=object),
                    a2=np.array(a2, dtype=object))


def read_variants(path):
    """Read a ``.bim`` or ``.pvar`` (optionally gzipped) by its extension."""
    name = os.fspath(path)
    stem = name[:-3] if name.endswith(".gz") else name
    if stem.endswith(".bim"):
        return read_bim(name)
    if stem.endswith((".pvar", ".vcf")):
        return read_pvar(name)
    raise ValueError(f"{name}: expected a .bim or .pvar variant file")
