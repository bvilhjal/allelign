"""Allele and chromosome primitives shared by every module.

The rules here decide whether two records describe the same variant and which
way round its effect points, so they are kept small, explicit and in one place.
Ported unchanged from ldpred3's harmonisation module.
"""

from __future__ import annotations

__all__ = ["normalize_chrom", "reverse_complement", "is_ambiguous",
           "allele_orientation", "CHROM_ALIASES", "AUTOSOMES"]

_COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}

#: PLINK numeric sex/mitochondrial codes and spellings -> canonical labels.
#: ``25`` (PLINK's pseudoautosomal code) and ``XY`` both map to ``X``: PAR
#: variants sit on X coordinates in PLINK filesets while GWAS summary
#: statistics label the same loci X, and matching is by (normalised chrom,
#: pos), so a separate XY label left every PAR variant silently unmatched.
CHROM_ALIASES = {"23": "X", "24": "Y", "25": "X", "26": "MT", "M": "MT",
                 "XY": "X"}

AUTOSOMES = frozenset(str(i) for i in range(1, 23))

_PALINDROMIC = frozenset((("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")))


def normalize_chrom(label):
    """Canonical chromosome label.

    Drops a ``chr`` prefix (any case) and zero padding (``01`` -> ``1``), and
    maps sex/mitochondrial codes (``23``/``25``/``XY`` -> ``X``, ``24`` -> ``Y``,
    ``26``/``M`` -> ``MT``). A ``chr1``/``1`` or ``X``/``23`` labelling clash is
    one of the commonest reasons a real-data run silently matches nothing.
    Sumstats that label PAR loci ``Y`` are not supported.
    """
    s = str(label).strip()
    if s[:3].lower() == "chr":
        s = s[3:]
    s = s.upper()
    if s.isdecimal():
        s = str(int(s))
    return CHROM_ALIASES.get(s, s)


def reverse_complement(allele):
    """Reverse complement of a (single- or multi-base) allele; ``None`` if not ACGT."""
    try:
        return "".join(_COMP[b] for b in reversed(allele))
    except KeyError:
        return None


def is_ambiguous(ea, oa):
    """True when a strand change exchanges the two alleles.

    Such a pair cannot be oriented from alleles alone, because changing strand
    also changes the effect sign. The single-base A/T and C/G case is the
    common one; the general rule also catches multi-base pairs such as an
    indel ``AA``/``TT``. Alleles that are individually self reverse
    complementary (``AT``/``CG``) are safe: a strand change preserves their
    order and therefore the sign. Inputs are expected upper-cased.
    """
    if (ea, oa) in _PALINDROMIC:
        return True
    if len(ea) == 1 and len(oa) == 1:
        return False
    cea, coa = reverse_complement(ea), reverse_complement(oa)
    return cea is not None and coa is not None and cea == oa and coa == ea


def allele_orientation(ea, oa, a1, a2):
    """How ``(ea, oa)`` maps onto ``(a1, a2)``.

    Returns ``(flip, strand)``: ``flip`` is True when the effect must be
    negated to count ``a1``, ``strand`` is True when the match needed the
    reverse complement. ``(None, False)`` means the allele pairs disagree.
    Inputs are expected upper-cased.
    """
    if ea == a1 and oa == a2:
        return False, False
    if ea == a2 and oa == a1:
        return True, False
    cea, coa = reverse_complement(ea), reverse_complement(oa)
    if cea is not None and coa is not None:
        if cea == a1 and coa == a2:
            return False, True
        if cea == a2 and coa == a1:
            return True, True
    return None, False
