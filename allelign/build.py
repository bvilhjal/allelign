"""Genome-build labels and the declared-build check.

A build mismatch does not look like an error: a GRCh38 file against a GRCh37
reference keeps about 1% of variants by coordinate and still yields numbers.
Declaring both builds turns that into a one-line refusal naming the cause.
"""

from __future__ import annotations

__all__ = ["normalize_build", "check_build", "BuildMismatchError", "BUILDS"]

BUILDS = ("GRCh37", "GRCh38")

_ALIASES = {
    "grch37": "GRCh37", "hg19": "GRCh37", "b37": "GRCh37", "build37": "GRCh37",
    "37": "GRCh37", "hs37d5": "GRCh37",
    "grch38": "GRCh38", "hg38": "GRCh38", "b38": "GRCh38", "build38": "GRCh38",
    "38": "GRCh38",
}


class BuildMismatchError(ValueError):
    """Two inputs are declared on different genome builds."""


def normalize_build(value, *, name="genome build"):
    """``'GRCh37'`` or ``'GRCh38'`` for any common spelling; ``None`` stays ``None``."""
    if value is None:
        return None
    key = str(value).strip().lower()
    try:
        return _ALIASES[key]
    except KeyError:
        raise ValueError(f"unknown {name} {value!r}; use GRCh37 (hg19, b37) "
                         "or GRCh38 (hg38, b38)") from None


def check_build(declared, reference, *, what="summary statistics",
                against="the reference"):
    """Refuse inputs declared on different builds.

    Returns ``{"declared", "reference", "status"}`` with status ``match`` or
    ``unchecked`` (either side unknown), and raises
    :class:`BuildMismatchError` on a disagreement, naming the remedies.
    """
    declared = normalize_build(declared, name=f"{what} build")
    reference = normalize_build(reference, name="reference build")
    if declared is None or reference is None:
        return {"declared": declared, "reference": reference,
                "status": "unchecked"}
    if declared != reference:
        raise BuildMismatchError(
            f"{what} are declared {declared} but {against} is {reference}. "
            f"Lift the statistics over to {reference} (UCSC liftOver, CrossMap "
            "or bcftools +liftover, keeping the chain and failed mappings), or "
            "re-anchor on identifiers (allelign.reanchor_on_identifier).")
    return {"declared": declared, "reference": reference, "status": "match"}
