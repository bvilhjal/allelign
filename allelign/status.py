"""Per-variant terminal QC status: one code per input row.

Every filter and the harmoniser can stamp a caller-allocated ``int8`` array with
the reason a row was dropped (0 = kept). The codes are the vocabulary of
ldpred3's ``--qc-out`` table, kept numerically identical so tables written by
either package can be read by the other's consumers.

* file-level filters (:func:`allelign.qc.qc_sumstats`): codes 1-7;
* harmonisation (:func:`allelign.align.harmonize`): codes 8-13;
* reference-dependent checks: 14-20 (16-19 are
  :func:`~allelign.qc.af_concordance_mask` and
  :func:`~allelign.qc.sd_consistency_mask`, 20 the LD-consistency screen);
* 21: identifier re-anchoring could not anchor the row.
"""

from __future__ import annotations

import numpy as np

from ._common import atomic_text

__all__ = ["QC_REASON_NAMES", "new_status", "status_counts",
           "write_status_table", "STATUS_HEADER"]

NONFINITE = 1            # beta/se non-finite, se <= 0, or N non-finite
NONPOSITIVE_N = 2        # N finite and <= 0 (beta/se valid)
DUPLICATE = 3            # id (or chrom:pos:alleles) appears more than once
LOW_N = 4                # n_eff < min_n_ratio * anchor
LOW_MAF = 5              # minor allele frequency below the floor
LOW_INFO = 6             # imputation INFO below the floor
CHISQ_OUTLIER = 7        # (beta/se)^2 above max_chisq
UNMATCHED = 8            # no reference row at this id or coordinate
AMBIGUOUS_ID = 9         # multi-locus id, coordinates do not resolve it
AMBIGUOUS_ALLELES = 10   # palindromic allele pair
DUPLICATE_MATCH = 11     # every candidate reference row already matched
ALLELE_MISMATCH = 12     # no orientation maps (ea, oa) onto (a1, a2)
NONFINITE_BETA_SE = 13   # matched, but beta/se non-finite
ABSENT_LD_CACHE = 14     # harmonised, but absent from the LD cache
LD_REFERENCE_INTERSECT = 15  # outside the target and LD-reference intersection
AF_DIFF = 16             # GWAS/reference allele-frequency gap too large
SD_LOW = 17              # sumstats-implied dosage SD too small
SD_HIGH = 18             # sumstats-implied dosage SD too large
SD_UNTESTABLE = 19       # non-finite SD or near-monomorphic reference
LD_INCONSISTENT = 20     # LD-consistency (DENTIST-style) outlier
REANCHOR_DROPPED = 21    # identifier not held once by the reference

#: Code -> reason name. 0 is "no drop". Names match ldpred3's table.
QC_REASON_NAMES = {
    0: "",
    NONFINITE: "nonfinite",
    NONPOSITIVE_N: "nonpositive_n",
    DUPLICATE: "duplicate",
    LOW_N: "low_n",
    LOW_MAF: "low_maf",
    LOW_INFO: "low_info",
    CHISQ_OUTLIER: "chisq_outlier",
    UNMATCHED: "unmatched",
    AMBIGUOUS_ID: "ambiguous_id",
    AMBIGUOUS_ALLELES: "ambiguous_alleles",
    DUPLICATE_MATCH: "duplicate_match",
    ALLELE_MISMATCH: "allele_mismatch",
    NONFINITE_BETA_SE: "nonfinite_beta_se",
    ABSENT_LD_CACHE: "absent_ld_cache",
    LD_REFERENCE_INTERSECT: "ld_reference_intersect",
    AF_DIFF: "af_diff",
    SD_LOW: "sd_low",
    SD_HIGH: "sd_high",
    SD_UNTESTABLE: "sd_untestable",
    LD_INCONSISTENT: "dentist_outlier",
    REANCHOR_DROPPED: "reanchor_dropped",
}

#: Columns of the status table. ``EA``/``OA`` are the alleles as read.
STATUS_HEADER = ("ID", "CHR", "POS", "EA", "OA", "STATUS", "REASON")


def new_status(n):
    """A zeroed ``int8`` status array for ``n`` input rows."""
    return np.zeros(int(n), dtype=np.int8)


def check_status(status, n):
    """Validate a caller-allocated status array; returns it or ``None``."""
    if status is None:
        return None
    status = np.asarray(status)
    if status.shape != (n,) or status.dtype != np.int8:
        raise ValueError(f"status must be an int8 array of length {n}")
    return status


def mark(status, code, bad):
    """Stamp rows that are still unmarked and fail ``bad`` with ``code``."""
    if status is not None:
        status[(status == 0) & np.asarray(bad, dtype=bool)] = code


def status_counts(codes):
    """``{reason name or 'kept': count}`` over a status array."""
    codes = np.asarray(codes)
    out = {}
    values, counts = np.unique(codes, return_counts=True)
    for value, count in zip(values, counts):
        name = QC_REASON_NAMES.get(int(value))
        if name is None:
            raise ValueError(f"unknown QC status code {int(value)}")
        out[name or "kept"] = int(count)
    return out


def write_status_table(path, *, id, chrom, pos, ea, oa, codes):
    """Write one TSV row per input variant with its terminal status.

    Columns ``ID CHR POS EA OA STATUS REASON`` in input order; ``STATUS`` is
    ``kept`` or ``dropped`` and ``REASON`` names the stage that dropped the
    row. Published atomically; a ``.gz`` path is compressed. Returns ``path``.
    """
    n = len(np.asarray(codes))
    for name, values in {"id": id, "chrom": chrom, "pos": pos, "ea": ea,
                         "oa": oa}.items():
        if len(np.asarray(values)) != n:
            raise ValueError(f"{name} must have one value per input row ({n})")
    with atomic_text(path) as fh:
        fh.write("\t".join(STATUS_HEADER) + "\n")
        for vid, c, p, a1, a2, code in zip(id, chrom, pos, ea, oa, codes):
            code = int(code)
            status = "kept" if code == 0 else "dropped"
            reason = QC_REASON_NAMES[code] if code else "."
            fh.write(f"{vid}\t{c}\t{int(p)}\t{a1}\t{a2}\t{status}\t{reason}\n")
    return path
