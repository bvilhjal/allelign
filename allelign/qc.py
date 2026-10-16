"""Quality control for summary statistics.

Two kinds of check:

* **file-only** -- :func:`qc_sumstats` (validity, duplicates, relative N, MAF,
  INFO, chi-square cap), :func:`genomic_inflation`, :func:`in_long_range_ld`;
* **against a reference**, after harmonisation -- :func:`sd_consistency_mask`
  compares ``sd_ss ~ 1/sqrt(N se^2 + beta^2)`` with ``sd_ref = sqrt(2f(1-f))``,
  :func:`af_concordance_mask` compares reported and reference frequencies,
  :func:`impute_n_eff` recovers per-variant N (Prive et al. 2022),
  :func:`implied_sample_size` and :func:`genomic_control_factor` diagnose N and
  genomic control. The LD-consistency screen lives in :mod:`allelign.ldcheck`.

Reference disagreement can mean bad summary statistics *or* an ancestry
mismatch, so every check returns a keep mask plus diagnostics, and the AF check
refuses to drop an implausibly large fraction silently. Every function can
stamp a caller-allocated :mod:`allelign.status` array with per-row reasons.

Ported from ldpred3 (file and reference checks) and bipred (long-range LD).
"""

from __future__ import annotations

import numpy as np

from . import status as _st
from ._common import finite_control, validate_boolean
from .alleles import normalize_chrom
from .warning_markers import AF_SAFETY_LIMIT

__all__ = ["qc_sumstats", "sd_consistency_mask", "af_concordance_mask",
           "gwas_af_on_counted_allele", "should_impute_n", "impute_n_eff",
           "genomic_control_factor", "implied_sample_size",
           "genomic_inflation", "in_long_range_ld", "LONG_RANGE_LD_GRCH37",
           "APOE_GRCH37"]


def qc_sumstats(ss, *, min_n_ratio=0.7, min_n_anchor="median",
                min_maf=0.01, min_info=0.7, max_chisq=None,
                drop_duplicates=True, status=None):
    """File-only QC. Returns ``(keep, log)`` over ``len(ss)`` rows.

    Parameters
    ----------
    ss : Sumstats
    min_n_ratio : float, default 0.7
        Keep rows with ``n_eff >= min_n_ratio * anchor(n_eff)``; in (0, 1].
    min_n_anchor : {"median", "max"}, default "median"
        ``"max"`` is the LDpred2 convention; ``"median"`` is robust when a file
        pools cohorts of different sizes.
    min_maf : float, default 0.01
        Applied only where ``eaf`` is present.
    min_info : float, default 0.7
        Applied only where ``info`` is present.
    max_chisq : float, optional
        Drop rows with ``(beta/se)^2 > max_chisq``; ``None`` disables it.
    drop_duplicates : bool, default True
        Drop every row whose id (or chrom:pos:alleles when the id is missing)
        occurs more than once.
    status : int8 array, optional
        Stamped with the first filter that rejected each row. The log's
        ``n_drop_nonfinite`` also counts the non-positive-N rows the status
        array attributes to ``nonpositive_n``.
    """
    min_n_ratio = finite_control("min_n_ratio", min_n_ratio)
    if not 0.0 < min_n_ratio <= 1.0:
        raise ValueError("min_n_ratio must be in (0, 1]")
    min_maf = finite_control("min_maf", min_maf)
    if not 0.0 <= min_maf <= 0.5:
        raise ValueError("min_maf must be in [0, 0.5]")
    min_info = finite_control("min_info", min_info)
    if not 0.0 <= min_info <= 1.0:
        raise ValueError("min_info must be in [0, 1]")
    if min_n_anchor not in ("median", "max"):
        raise ValueError("min_n_anchor must be 'max' or 'median'")
    n = len(ss)
    status = _st.check_status(status, n)
    keep = np.ones(n, dtype=bool)
    log = {"n_input": n, "min_maf": min_maf, "min_info": min_info}
    if getattr(ss, "parse_log", None) is not None:
        log["sumstats_parse"] = dict(ss.parse_log)

    beta = np.asarray(ss.beta, float)
    se = np.asarray(ss.se, float)
    n_eff = np.asarray(ss.n_eff, float)
    eaf = np.asarray(ss.eaf, float)
    info = np.asarray(ss.info, float)

    valid_n = np.isfinite(n_eff) & (n_eff > 0)
    valid = np.isfinite(beta) & np.isfinite(se) & (se > 0) & valid_n
    log["n_drop_nonfinite"] = int((keep & ~valid).sum())
    log["n_drop_nonpositive_n"] = int(
        (keep & np.isfinite(n_eff) & (n_eff <= 0)).sum())
    _st.mark(status, _st.NONFINITE,
             ~(np.isfinite(beta) & np.isfinite(se) & (se > 0))
             | ~np.isfinite(n_eff))
    _st.mark(status, _st.NONPOSITIVE_N, np.isfinite(n_eff) & (n_eff <= 0))
    keep &= valid

    if drop_duplicates:
        ids = ss.id
        bad = [i for i, v in enumerate(ids) if not (v and v != ".")]
        if bad:
            keys = list(ids)
            for i in bad:
                keys[i] = (str(ss.chrom[i]), int(ss.pos[i]), ss.ea[i], ss.oa[i])
        else:
            keys = ids
        seen_once, repeated = set(), set()
        for k in keys:
            if k in seen_once:
                repeated.add(k)
            else:
                seen_once.add(k)
        dup = (np.array([k in repeated for k in keys], dtype=bool)
               if repeated else np.zeros(n, dtype=bool))
        log["n_drop_duplicate"] = int((keep & dup).sum())
        _st.mark(status, _st.DUPLICATE, dup)
        keep &= ~dup

    if np.any(valid_n):
        anchor = (float(np.median(n_eff[valid_n])) if min_n_anchor == "median"
                  else float(np.max(n_eff[valid_n])))
        bad = n_eff < (min_n_ratio * anchor)
        log["n_drop_lowN"] = int((keep & bad).sum())
        log["n_anchor"] = min_n_anchor
        log["n_anchor_value"] = anchor
        _st.mark(status, _st.LOW_N, bad)
        keep &= ~bad

    if np.any(np.isfinite(eaf)):
        maf = np.minimum(eaf, 1 - eaf)
        bad = np.isfinite(eaf) & (maf < min_maf)
        log["n_drop_lowMAF"] = int((keep & bad).sum())
        _st.mark(status, _st.LOW_MAF, bad)
        keep &= ~bad

    if np.any(np.isfinite(info)):
        bad = np.isfinite(info) & (info < min_info)
        log["n_drop_lowINFO"] = int((keep & bad).sum())
        _st.mark(status, _st.LOW_INFO, bad)
        keep &= ~bad

    if max_chisq is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            chisq = (beta / se) ** 2
        bad = np.isfinite(chisq) & (chisq > max_chisq)
        log["n_drop_chisq_outlier"] = int((keep & bad).sum())
        _st.mark(status, _st.CHISQ_OUTLIER, bad)
        keep &= ~bad

    log["n_kept"] = int(keep.sum())
    return keep, log


def _autosome_mask(chrom, n):
    labels = np.asarray(chrom, dtype=object)
    if labels.shape != (n,):
        raise ValueError("chrom must have one label per variant")
    memo = {}
    out = np.zeros(n, dtype=bool)
    for i, c in enumerate(labels):
        hit = memo.get(c)
        if hit is None:
            s = str(c).strip()
            if s[:3].lower() == "chr":
                s = s[3:]
            s = s.upper()
            hit = memo[c] = s.isdigit() and 1 <= int(s) <= 22
        out[i] = hit
    return out


def sd_consistency_mask(beta, se, n_eff, af_ref, *, info=None, chrom=None,
                        sd_ratio_bounds=(0.5, 2.0), min_sd_ref=0.05,
                        rule="ratio", min_sd_ss=0.1, sd_excess=0.1,
                        status=None):
    """SD-consistency check against reference allele frequencies.

    Compares ``sd_ss ~ 1/sqrt(N se^2 + beta^2)`` with ``sd_ref = sqrt(2f(1-f))``
    after anchoring their median ratio (the phenotype scale is unknown). A
    variant this check cannot test (non-finite ``sd_ss``/``sd_ref`` or
    ``sd_ref < min_sd_ref``) is *dropped*.

    Parameters
    ----------
    beta, se, n_eff : array_like
        Harmonised summary statistics.
    af_ref : array_like
        Frequency of the same counted allele in the reference.
    info : array_like, optional
        Divide ``sd_ss`` by ``sqrt(INFO)`` (Prive et al. 2022).
    chrom : array_like, optional
        Anchor autosomes and other chromosomes separately: sex-chromosome
        dosage variance differs from ``2f(1-f)`` by a constant that depends on
        the male fraction and coding. A group with fewer than 50 entries uses
        the other group's (then the global) anchor.
    rule : {"ratio", "prive"}, default "ratio"
        ``"ratio"`` keeps ``sd_ratio_bounds[0] <= sd_ss/sd_ref <=
        sd_ratio_bounds[1]``. ``"prive"`` is the published asymmetric rule:
        drop ``sd_ss < 0.5 sd_ref``, ``sd_ss > sd_ref + sd_excess`` or
        ``sd_ss < min_sd_ss``.

    Returns
    -------
    keep, log, diag
        ``diag`` holds the ``sd_ss`` and ``sd_ref`` arrays for plotting.
    """
    beta = np.asarray(beta, float); se = np.asarray(se, float)
    n_eff = np.asarray(n_eff, float); af_ref = np.asarray(af_ref, float)

    sd_ref = np.sqrt(2.0 * af_ref * (1.0 - af_ref))
    with np.errstate(divide="ignore", invalid="ignore"):
        sd_ss = 1.0 / np.sqrt(n_eff * se ** 2 + beta ** 2)
    if info is not None:
        with np.errstate(invalid="ignore"):
            fac = np.sqrt(np.clip(np.asarray(info, float), 0.0, 1.0))
        fac = np.where(np.isfinite(fac) & (fac > 0.0), fac, 1.0)
        sd_ss = sd_ss / fac
    finite = np.isfinite(sd_ss) & np.isfinite(sd_ref) & (sd_ref > 0)
    n_nonautosomal = 0
    nonauto_scale = None
    if chrom is None:
        if finite.any():
            scale = np.median(sd_ref[finite] / sd_ss[finite])
            sd_ss = sd_ss * scale
    else:
        auto = _autosome_mask(chrom, beta.size)
        n_nonautosomal = int((~auto).sum())

        def _group_scale(mask):
            sel = finite & mask
            if sel.sum() < 50:
                return None
            return float(np.median(sd_ref[sel] / sd_ss[sel]))

        global_scale = (float(np.median(sd_ref[finite] / sd_ss[finite]))
                        if finite.any() else None)
        auto_scale = _group_scale(auto)
        nonauto_scale = _group_scale(~auto)
        if auto_scale is None:
            auto_scale = nonauto_scale if nonauto_scale is not None else global_scale
        if nonauto_scale is None:
            nonauto_scale = auto_scale if auto_scale is not None else global_scale
        if auto_scale is not None:
            sd_ss = np.where(auto, sd_ss * auto_scale, sd_ss * nonauto_scale)

    if rule not in ("ratio", "prive"):
        raise ValueError("rule must be 'ratio' or 'prive'")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sd_ss / sd_ref
    testable = finite & (sd_ref >= min_sd_ref)
    if rule == "ratio":
        lo, hi = sd_ratio_bounds
        too_low = ratio < lo
        too_high = ratio > hi
    else:
        too_low = (sd_ss < 0.5 * sd_ref) | (sd_ss < min_sd_ss)
        too_high = sd_ss > sd_ref + sd_excess
    keep = testable & ~too_low & ~too_high

    dropped = testable & ~keep
    n_low = int((dropped & too_low).sum())
    n_high = int((dropped & too_high & ~too_low).sum())
    status = _st.check_status(status, beta.size)
    _st.mark(status, _st.SD_UNTESTABLE, ~testable)
    _st.mark(status, _st.SD_LOW, dropped & too_low)
    _st.mark(status, _st.SD_HIGH, dropped & too_high & ~too_low)
    if testable.any() and np.sum(sd_ref[testable] ** 2) > 0:
        slope = float(np.sum(sd_ss[testable] * sd_ref[testable])
                      / np.sum(sd_ref[testable] ** 2))
    else:
        slope = float("nan")

    log = {
        "n_input": int(beta.size),
        "n_drop_sd_inconsistent": int(dropped.sum()),
        "n_drop_sd_low": n_low,
        "n_drop_sd_high": n_high,
        "sd_slope": round(slope, 4),
        "rule": rule,
        "info_corrected": info is not None,
        "n_nonautosomal": n_nonautosomal,
        "nonautosomal_sd_scale": (round(nonauto_scale, 4)
                                  if nonauto_scale is not None
                                  and n_nonautosomal > 0 else None),
        "n_drop_nonfinite_or_mono": int((~finite | (sd_ref < min_sd_ref)).sum()),
        "n_kept": int(keep.sum()),
    }
    return keep, log, {"sd_ss": sd_ss, "sd_ref": sd_ref}


def af_concordance_mask(af_gwas, af_ref, *, max_af_diff=0.1,
                        max_drop_frac=0.2, min_testable_for_valve=50,
                        mirror_corr=-0.5, status=None):
    """Drop variants whose reported frequency disagrees with the reference.

    The Prive et al. (2022) ``qc2`` check. Both inputs must count the same
    allele; use :func:`gwas_af_on_counted_allele` after harmonisation.
    Non-finite entries are untestable and kept. Above ``max_drop_frac``
    failing, nothing is dropped (the signature of an ancestry mismatch or a
    mis-oriented column, not per-variant errors) unless fewer than
    ``min_testable_for_valve`` variants were testable.

    Returns ``(keep, log, diag)``; the log carries orientation diagnostics
    (``af_corr``, ``mirrored``, ``n_mirrored``) whether or not it fired.
    """
    af_gwas = np.asarray(af_gwas, dtype=float)
    af_ref = np.asarray(af_ref, dtype=float)
    if af_gwas.shape != af_ref.shape:
        raise ValueError("af_gwas and af_ref must have the same shape")
    max_af_diff = float(max_af_diff)
    if not np.isfinite(max_af_diff) or not 0.0 < max_af_diff <= 1.0:
        raise ValueError("max_af_diff must be in (0, 1]")
    max_drop_frac = float(max_drop_frac)
    if not np.isfinite(max_drop_frac) or not 0.0 < max_drop_frac <= 1.0:
        raise ValueError("max_drop_frac must be in (0, 1]")

    testable = np.isfinite(af_gwas) & np.isfinite(af_ref)
    with np.errstate(invalid="ignore"):
        abs_diff = np.abs(af_gwas - af_ref)
    fail = testable & (abs_diff > max_af_diff)
    n_testable = int(testable.sum())
    frac = float(fail.sum() / n_testable) if n_testable else 0.0

    if n_testable >= 2 and np.std(af_gwas[testable]) > 0 \
            and np.std(af_ref[testable]) > 0:
        af_corr = float(np.corrcoef(af_gwas[testable], af_ref[testable])[0, 1])
    else:
        af_corr = float("nan")
    n_mirrored = int((testable
                      & (np.abs(af_gwas - (1.0 - af_ref)) < abs_diff)).sum())

    applied = (frac <= max_drop_frac
               or n_testable < int(min_testable_for_valve))
    keep = np.ones(af_gwas.shape, dtype=bool)
    if applied:
        keep &= ~fail
        _st.mark(_st.check_status(status, af_gwas.size), _st.AF_DIFF, fail)

    log = {
        "n_input": int(af_gwas.size),
        "n_testable": n_testable,
        "max_af_diff": max_af_diff,
        "n_drop_af_diff": int(fail.sum()) if applied else 0,
        "n_flagged_af_diff": int(fail.sum()),
        "frac_failing": round(frac, 6),
        "applied": bool(applied),
        "af_corr": None if np.isnan(af_corr) else round(af_corr, 4),
        "mirrored": bool(np.isfinite(af_corr) and af_corr < mirror_corr),
        "n_mirrored": n_mirrored,
        "median_abs_diff": (round(float(np.median(abs_diff[testable])), 6)
                            if n_testable else None),
        "n_kept": int(keep.sum()),
    }
    log["valve_active"] = n_testable >= int(min_testable_for_valve)
    if not applied:
        log["not_applied_reason"] = (
            f"{frac:.1%} of testable variants differ by more than "
            f"{max_af_diff}, above the {max_drop_frac:.0%} {AF_SAFETY_LIMIT}; "
            "this is the signature of an ancestry-mismatched LD panel or a "
            "mis-oriented frequency column, not of per-variant errors. No "
            "variant was dropped on this check.")
    return keep, log, {"af_gwas": af_gwas, "af_ref": af_ref,
                       "abs_diff": abs_diff}


def gwas_af_on_counted_allele(eaf, src_index, flipped):
    """Align source GWAS frequency to the reference's ``a1``; swaps use ``1-f``."""
    eaf = np.asarray(eaf, dtype=float)
    src_index = np.asarray(src_index, dtype=np.int64)
    flipped = np.asarray(flipped, dtype=bool)
    if src_index.shape != flipped.shape:
        raise ValueError("src_index and flipped must have the same shape")
    af = eaf[src_index]
    return np.where(flipped, 1.0 - af, af)


def should_impute_n(impute_n, n_eff, *, n_from_caller=False):
    """Resolve a tri-state impute-N policy to ``(apply, reason)``.

    ``True`` always imputes, ``False`` never, ``None`` (auto) imputes only when
    the reported N is one constant -- the case a file-wide N describes -- and
    the constant did not come from the caller (``n_from_caller``), whose own
    arithmetic silently replaced would not be a repair.
    """
    if impute_n is False:
        return False, "disabled"
    if impute_n is True:
        return True, "requested"
    if impute_n is not None:
        raise ValueError("impute_n must be True, False, or None")
    n = np.asarray(n_eff, dtype=float)
    finite = n[np.isfinite(n)]
    if finite.size == 0:
        return False, "no_finite_n"
    if float(np.ptp(finite)) == 0.0:
        if n_from_caller:
            return False, "constant_n_supplied_by_caller"
        return True, "constant_n"
    return False, "ragged_n"


def impute_n_eff(se, af_ref, n_total, *, anchor_quantile=0.99,
                 anchor_bins=50, info=None):
    """Recover per-variant effective N from ``se`` and frequency.

    From Prive et al. (2022), ``N_j ~ 1 / (se_j^2 2 f_j (1-f_j))``. The
    unknown constant is set so that the median over equal-count AF bins of the
    per-bin ``anchor_quantile`` equals ``n_total`` (the best-typed variants in
    each frequency range carry the full sample). Optionally multiplied by
    INFO; non-finite values become ``n_total``; clipped at ``n_total``.

    Returns ``(n_imp, log)``.
    """
    se = np.asarray(se, dtype=float)
    af_ref = np.asarray(af_ref, dtype=float)
    n_total = float(n_total)
    sd_ref2 = 2.0 * af_ref * (1.0 - af_ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_raw = 1.0 / (se ** 2 * sd_ref2)
    finite = np.isfinite(n_raw) & (n_raw > 0)
    if not finite.any():
        raise ValueError("no variants with finite se and polymorphic af_ref to "
                         "impute N from")
    idx = np.flatnonzero(finite)
    n_bins = max(1, min(int(anchor_bins), idx.size // 10))
    if n_bins > 1:
        order = idx[np.argsort(af_ref[idx], kind="stable")]
        per_bin = np.array([np.quantile(n_raw[part], anchor_quantile)
                            for part in np.array_split(order, n_bins)])
        ref = float(np.median(per_bin))
        anchor_basis = f"median of per-bin q{anchor_quantile:g} over {n_bins} bins"
    else:
        ref = float(np.quantile(n_raw[idx], anchor_quantile))
        anchor_basis = "raw quantile (too few variants to bin)"
    scale = n_total / ref if ref > 0 else 1.0
    n_imp = n_raw * scale
    if info is not None:
        n_imp = n_imp * np.asarray(info, dtype=float)
    n_imp = np.where(np.isfinite(n_imp) & (n_imp > 0), n_imp, n_total)
    n_imp = np.minimum(n_imp, n_total)
    log = {
        "n_input": int(se.size),
        "n_total": n_total,
        "median_imputed_n": float(np.median(n_imp)),
        "frac_below_half": float(np.mean(n_imp < 0.5 * n_total)),
        "anchor_basis": anchor_basis,
        "anchor_ref": float(ref),
    }
    return n_imp, log


def genomic_control_factor(intercept, intercept_se, *, min_z=2.0,
                           n_variants=None, min_variants=50_000):
    """SE rescaling that undoes genomic control, or 1.0.

    Genomic control inflates SE by ``sqrt(lambda)`` while leaving beta fixed, so
    a GC-corrected file regresses to an LDSC intercept near ``1/lambda < 1``.
    When deflation is established (``(1 - intercept)/se >= min_z`` on at least
    ``min_variants`` variants) return ``sqrt(intercept)`` to multiply SE by.
    Intercepts above one are never rescaled (possible confounding).
    """
    intercept = float(intercept)
    intercept_se = float(intercept_se)
    if not (np.isfinite(intercept) and np.isfinite(intercept_se)):
        return 1.0
    if intercept <= 0 or intercept >= 1.0 or intercept_se <= 0:
        return 1.0
    if n_variants is not None and int(n_variants) < int(min_variants):
        return 1.0
    if (1.0 - intercept) / intercept_se < min_z:
        return 1.0
    return float(np.sqrt(intercept))


def implied_sample_size(beta, se, af, *, binary=False, reported_n=None,
                        strict=False):
    """Effective N the statistics behave as if they had (binary traits).

    Solving ``sd_ss == sd_ref`` gives ``n = (c^2/(2f(1-f)) - beta^2) / se^2``
    with ``c = 2`` for a case/control trait. The median is the sample size the
    file is internally consistent with. A quantitative trait's phenotype
    scale is unidentified, so ``median``/``ratio`` are NaN there.

    Returns ``{"median", "ratio", "consistent"}`` (ratio to ``reported_n``,
    consistent when within 0.9-1.1). ``reported_n`` may be a positive scalar
    or a vector matching beta; the ratio uses its median.

    By default, only finite positive implied estimates enter the median.
    ``strict=True`` preserves bipred's diagnostic: reject malformed inputs
    (including for quantitative traits), then take the median of *all*
    implied estimates, including nonpositive ones. This distinguishes input
    validation from deciding which estimates describe a plausible N.
    """
    validate_boolean(binary=binary, strict=strict)
    beta = np.asarray(beta, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    af = np.asarray(af, dtype=np.float64)
    if (beta.ndim != 1 or se.ndim != 1 or af.ndim != 1 or beta.size == 0
            or se.shape != beta.shape or af.shape != beta.shape):
        raise ValueError("beta, se and af must be non-empty equal-length vectors")
    if strict:
        if not np.all(np.isfinite(beta)):
            raise ValueError("beta contains non-finite values; filter them first")
        if not np.all(np.isfinite(se)) or np.any(se <= 0):
            raise ValueError("se must contain finite positive values")
        if not np.all(np.isfinite(af)) or np.any((af <= 0) | (af >= 1)):
            raise ValueError("af must contain finite values strictly between 0 and 1")
    reported = None
    if reported_n is not None:
        reported = np.asarray(reported_n, dtype=np.float64)
        if reported.ndim != 0 and reported.shape != beta.shape:
            raise ValueError("reported_n must be a scalar or a vector matching beta")
        if not np.all(np.isfinite(reported)) or np.any(reported <= 0):
            if strict:
                raise ValueError("reported_n must contain finite positive values")
            reported = None
    if not binary:
        return {"median": float("nan"), "ratio": float("nan"),
                "consistent": False}
    var = 2.0 * af * (1.0 - af)
    if strict:
        # Preserve the rounding of bipred's historical sd_ref ** 2 formula.
        var = np.sqrt(var) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = (4.0 / var - beta ** 2) / se ** 2
    estimates = implied if strict else implied[np.isfinite(implied) & (implied > 0)]
    if estimates.size == 0:
        return {"median": float("nan"), "ratio": float("nan"),
                "consistent": False}
    median = float(np.median(estimates))
    ratio = (median / float(np.median(reported)) if reported is not None
             else float("nan"))
    return {"median": median, "ratio": ratio,
            "consistent": bool(np.isfinite(ratio) and 0.9 <= ratio <= 1.1)}


#: Median of a chi-square with one degree of freedom.
_CHI2_1_MEDIAN = 0.45493642311957283


def genomic_inflation(z=None, *, chisq=None, pval=None):
    """Genomic-control lambda: ``median(chi^2) / 0.4549``.

    Pass z-scores (``beta/se``), chi-square statistics or p-values. Non-finite
    entries are ignored. Lambda mixes polygenic signal with confounding; use
    an LDSC intercept to separate them.
    """
    given = [x is not None for x in (z, chisq, pval)]
    if sum(given) != 1:
        raise ValueError("pass exactly one of z, chisq or pval")
    if z is not None:
        stat = np.asarray(z, dtype=float) ** 2
    elif chisq is not None:
        stat = np.asarray(chisq, dtype=float)
    else:
        from statistics import NormalDist
        inv = NormalDist().inv_cdf
        p = np.asarray(pval, dtype=float)
        ok = np.isfinite(p) & (p > 0) & (p <= 1)
        stat = np.full(p.shape, np.nan)
        stat[ok] = [inv(v / 2) ** 2 for v in p[ok].tolist()]
    stat = stat[np.isfinite(stat)]
    if stat.size == 0:
        return float("nan")
    return float(np.median(stat) / _CHI2_1_MEDIAN)


#: Long-range LD regions on GRCh37 as ``(chrom, start, end, label)``: the 24
#: regions of Price et al. 2008 (Am J Hum Genet 83:132-135), where correlation
#: extends far beyond the usual few hundred kilobases. Use the mask for
#: sensitivity analysis, not as a universal exclusion rule.
LONG_RANGE_LD_GRCH37 = (
    ("1", 48_000_000, 52_000_000, "1p13"),
    ("2", 86_000_000, 100_500_000, "2p11 / 2q11"),
    ("2", 134_500_000, 138_000_000, "2q21"),
    ("2", 183_000_000, 190_000_000, "2q31"),
    ("3", 47_500_000, 50_000_000, "3p21"),
    ("3", 83_500_000, 87_000_000, "3p12"),
    ("3", 89_000_000, 97_500_000, "3q11"),
    ("5", 44_500_000, 50_500_000, "5p12 / 5q11"),
    ("5", 98_000_000, 100_500_000, "5q21"),
    ("5", 129_000_000, 132_000_000, "5q31"),
    ("5", 135_500_000, 138_500_000, "5q31.2"),
    ("6", 25_000_000, 35_000_000, "MHC"),
    ("6", 57_000_000, 64_000_000, "6p11 / 6q11"),
    ("6", 140_000_000, 142_500_000, "6q24"),
    ("7", 55_000_000, 66_000_000, "7p11 / 7q11"),
    ("8", 7_000_000, 13_000_000, "8p23 inversion"),
    ("8", 43_000_000, 50_000_000, "8p11 / 8q11"),
    ("8", 112_000_000, 115_000_000, "8q23"),
    ("10", 37_000_000, 43_000_000, "10p11 / 10q11"),
    ("11", 46_000_000, 57_000_000, "11p11 / 11q11"),
    ("11", 87_500_000, 90_500_000, "11q14"),
    ("12", 33_000_000, 40_000_000, "12p11 / 12q11"),
    ("12", 109_500_000, 112_000_000, "12q24"),
    ("20", 32_000_000, 34_500_000, "20p11 / 20q11"),
)
#: APOE on GRCh37: not a long-range LD region, but a single locus whose effect
#: dominates lipid genetics. Optional.
APOE_GRCH37 = ("19", 44_912_079, 45_912_079, "APOE")


def in_long_range_ld(chrom, pos, *, genome_build="GRCh37", include_apoe=False,
                     regions=None):
    """Mask of variants inside a long-range LD region.

    The built-in table is GRCh37; ``genome_build`` must say so (any other
    build raises rather than silently mis-selecting). Chromosome labels are
    normalised, so ``chr6`` and ``6`` both match. Pass ``regions`` as
    ``(chrom, start, end, label)`` tuples to use another list on any build.
    """
    validate_boolean(include_apoe=include_apoe)
    if regions is None:
        from .build import normalize_build
        if normalize_build(genome_build) != "GRCh37":
            raise ValueError(
                "the built-in long-range LD table is GRCh37; lift positions "
                "over or pass regions= for another build")
        regions = list(LONG_RANGE_LD_GRCH37) + ([APOE_GRCH37]
                                                if include_apoe else [])
    labels = np.asarray(chrom, dtype=object)
    pos = np.asarray(pos, dtype=np.int64)
    if labels.shape != pos.shape:
        raise ValueError("chrom and pos must have the same shape")
    memo = {c: normalize_chrom(c) for c in set(labels.ravel())}
    norm = np.array([memo[c] for c in labels.ravel()],
                    dtype=object).reshape(labels.shape)
    inside = np.zeros(pos.shape, dtype=bool)
    for name, start, end, _label in regions:
        inside |= (norm == normalize_chrom(name)) & (pos >= start) & (pos <= end)
    return inside
