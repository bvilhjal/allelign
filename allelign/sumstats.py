"""Read GWAS summary statistics into one canonical table.

GWAS arrive as delimited text with wildly inconsistent column names. This
reader maps a large set of aliases onto canonical fields and lets the caller
override any column explicitly:

==============  ===========================================================
field           meaning
==============  ===========================================================
``id``          variant identifier (rsID), may be empty
``chrom``       chromosome label as read, may be empty
``pos``         base-pair position (int), 0 when missing
``ea``          effect allele: ``beta`` is per copy of this allele
``oa``          other (non-effect) allele
``beta``        per-allele effect (log-OR for binary traits)
``se``          standard error of ``beta``
``n_eff``       (effective) sample size per variant
``eaf``         effect-allele frequency (NaN when absent)
``info``        imputation quality (NaN when absent)
==============  ===========================================================

Odds ratios become ``beta = log(OR)``. A missing ``se`` is recovered from
``beta`` and the p-value (or ``-log10 p``) without underflow. Effective N comes,
in order of preference, from an explicit effective-N column, from per-variant
case and control counts (``4 / (1/N_case + 1/N_control)``), or from a total-N
column; :attr:`Sumstats.parse_log` records which.

Format-specific layouts are recognised from the header: SAIGE (effects for
Allele2), PLINK2 (``A1`` tested against ``REF``/``ALT``, or ``OMITTED``),
REGENIE (``GENPOS``), GWAS Catalog harmonised files (the ``hm_`` columns as one
consistent set), PGC ``daner`` files (``FRQ_U_*`` control frequency) and Neale
UK Biobank files (``chr:pos:ref:alt`` variant strings, effect on ALT).

Ported from ldpred3's reader, including its optional Numba block parser, whose
output is bit-identical to the reference row loop. On ldpred3's standard inputs
the result is identical to ``ldpred3.sumstats.read_sumstats``; the additions
above are documented differences.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import warnings

import numpy as np

from ._common import atomic_text, open_text

__all__ = ["Sumstats", "read_sumstats", "detect_columns", "inspect_columns",
           "write_sumstats", "rescale_n_eff", "anchor_n_eff",
           "n_eff_case_control", "COLUMN_ALIASES", "ColumnMappingError"]


class ColumnMappingError(ValueError):
    """A column override cannot be resolved, or a layout is inconsistent."""


#: Column aliases (lower-cased) -> canonical field, in priority order. The
#: first block of every list is ldpred3's, unchanged, so any file ldpred3 maps
#: maps identically here; later entries only add coverage. One exception: for
#: ``n_eff`` an effective-N name (``_EFFECTIVE_N_NAMES``) wins over a total-N
#: alias, so ``Effective_N`` beside ``N`` gives N from ``Effective_N``.
COLUMN_ALIASES = {
    "id": ["snp", "rsid", "rs", "id", "variant_id", "markername", "snpid",
           "marker", "rs_id",
           "rsids", "hm_rsid", "snp_id", "variant", "snpname"],
    "chrom": ["chr", "chrom", "chromosome", "#chrom", "hg19chr", "chr_name",
              "#chr", "hm_chrom", "hm_chr", "chr_id"],
    "pos": ["bp", "pos", "position", "base_pair_location", "bp_position",
            "pos_b37", "base_pair",
            "chr_position", "hm_pos", "genpos"],
    "ea": ["a1", "effect_allele", "ea", "allele1", "alt", "effectallele",
           "tested_allele", "inc_allele",
           "hm_effect_allele"],
    "oa": ["a2", "other_allele", "oa", "nea", "allele0", "allele2", "ref",
           "noneffect_allele", "dec_allele",
           "hm_other_allele", "omitted"],
    "beta": ["beta", "b", "effect", "effect_size", "effects", "log_odds",
             "hm_beta", "all_inv_var_meta_beta", "inv_var_meta_beta", "logor"],
    "or": ["or", "odds_ratio", "oddsratio",
           "hm_odds_ratio"],
    "se": ["se", "standard_error", "stderr", "standarderror", "sebeta",
           "se_beta", "logor_se", "std_err",
           "log(or)_se", "all_inv_var_meta_sebeta", "inv_var_meta_sebeta",
           "hm_standard_error"],
    "pval": ["p", "pval", "p_value", "pvalue", "p-value", "p.value",
             "p_bolt_lmm", "p_value_association", "p_wald", "pval_nominal",
             "all_inv_var_meta_p", "inv_var_meta_p"],
    "log10p": ["log10p", "mlogp", "neg_log_10_p_value", "neglog10p",
               "log10_p", "-log10p", "neg_log10_p"],
    "n_eff": ["n_eff", "neff", "n", "sample_size", "totalsamplesize",
              "n_samples", "n_total", "obs_ct", "n_complete_samples",
              "n_effective", "effective_n", "all_inv_var_meta_effective",
              "samplesize", "n_sample", "total_sample_size"],
    "n_cases": ["n_cases", "ncases", "n_case", "ncase", "nca", "cases",
                "num_cases", "n_cas"],
    "n_controls": ["n_controls", "ncontrols", "n_control", "ncontrol", "nco",
                   "controls", "num_controls", "n_ctrl", "n_con"],
    # MAF is deliberately absent: it does not say which allele is counted, so
    # treating it as EAF can turn a common effect allele (EAF=.9, MAF=.1) into
    # an apparent .8 frequency mismatch. A genuinely effect-allele column with
    # a non-standard name is available through the explicit ``eaf=`` override.
    "eaf": ["eaf", "freq", "frq", "effect_allele_frequency", "a1freq",
            "freq1", "af", "effect_allele_freq",
            "a1_freq", "af_alt", "hm_effect_allele_frequency"],
    "info": ["info", "imputation_info", "imp_info", "rsq", "r2", "info_score",
             "imputation_quality", "minimac_r2",
             "imputationinfo"],
}

#: N-column names that already are an effective N: preferred over a total-N
#: column and kept even when case and control counts are present.
_EFFECTIVE_N_NAMES = frozenset({"n_eff", "neff", "n_effective", "effective_n",
                                "all_inv_var_meta_effective"})

_SUMSTATS_FIELDS = ("id", "chrom", "pos", "ea", "oa", "beta", "se", "n_eff",
                    "eaf", "info")


@dataclass
class Sumstats:
    """Parsed GWAS summary statistics as parallel arrays.

    Field order and meaning match ldpred3's ``Sumstats``, so either package's
    functions accept the other's tables.
    """

    id: np.ndarray
    chrom: np.ndarray
    pos: np.ndarray
    ea: np.ndarray
    oa: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    n_eff: np.ndarray
    eaf: np.ndarray
    info: np.ndarray
    parse_log: dict = None       # source-file counts, preserved by subsetting

    def __len__(self):
        return len(self.beta)

    @property
    def z(self):
        """``beta / se`` (NaN where undefined)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(self.beta, float) / np.asarray(self.se, float)

    def subset(self, mask):
        """A new table keeping the rows selected by ``mask`` (bool or index)."""
        mask = np.asarray(mask)
        return Sumstats(*(getattr(self, f)[mask] for f in _SUMSTATS_FIELDS),
                        parse_log=(None if self.parse_log is None
                                   else dict(self.parse_log)))

    def as_variants(self):
        """These rows as a :class:`~allelign.variants.Variants` table.

        ``a1`` is the effect allele, so harmonising another file against it
        aligns that file's effects to this file's effect alleles.
        """
        from .variants import Variants
        return Variants(chrom=np.asarray(self.chrom, dtype=object),
                        id=np.asarray(self.id, dtype=object),
                        pos=np.asarray(self.pos, dtype=np.int64),
                        a1=np.asarray(self.ea, dtype=object),
                        a2=np.asarray(self.oa, dtype=object))

    def write(self, path):
        """Write the canonical table; see :func:`write_sumstats`."""
        return write_sumstats(path, self)


def n_eff_case_control(n_case, n_control):
    """Effective sample size of a case/control GWAS: ``4/(1/N_case + 1/N_control)``.

    Equals total N for a balanced study and tends to ``4 N_case`` with many
    more controls than cases. Scalars or arrays.

    For a meta-analysis the correct effective N is the *sum* of the
    per-cohort values; computing it from summed counts overstates it when
    case fractions differ between cohorts. Prefer a published ``Neff`` column.
    """
    n_case = np.asarray(n_case, dtype=float)
    n_control = np.asarray(n_control, dtype=float)
    if np.any(n_case <= 0) or np.any(n_control <= 0):
        raise ValueError("n_case and n_control must be positive")
    n = 4.0 / (1.0 / n_case + 1.0 / n_control)
    return float(n) if n.ndim == 0 else n


def _sniff_delimiter(header_line):
    if "\t" in header_line:
        return "\t"
    if "," in header_line:
        return ","
    return None      # whitespace; handled with str.split


def _read_header(fh):
    """``(delimiter, header columns)`` from the first line of ``fh``."""
    first = fh.readline().rstrip("\n")
    delim = _sniff_delimiter(first)
    return delim, (first.split(delim) if delim else first.split())


def _build_colmap(header, overrides):
    """``(colmap, layout)``: canonical field -> column index, plus format rules.

    ``layout`` records format-specific rules the header implied:
    ``saige``, ``gwas_catalog_harmonised``, ``plink2_omitted``,
    ``plink2_alt`` (index of ALT for per-row other-allele resolution),
    ``eaf_from`` (a pattern-matched frequency column) and ``variant_string``
    (index of a ``chr:pos:ref:alt`` column).
    """
    overrides = dict(overrides or {})
    lower = [h.strip().lower() for h in header]
    colmap = {}
    for field, aliases in COLUMN_ALIASES.items():
        if field == "n_eff":
            # Effective-N names first, so an ``Effective_N`` column is not
            # shadowed by a total ``N`` listed earlier in the aliases.
            aliases = sorted(aliases, key=lambda a: a not in _EFFECTIVE_N_NAMES)
        for a in aliases:
            if a in lower:
                colmap[field] = lower.index(a)
                break
    layout = {}
    # SAIGE reports effects and frequencies for Allele2. Allele1/Allele2 alone
    # are ambiguous across programs, so require a SAIGE-specific column.
    if {"allele1", "allele2"}.issubset(lower) and any(
            c in lower for c in ("af_allele2", "ac_allele2", "is.spa.converge")):
        colmap["ea"] = lower.index("allele2")
        colmap["oa"] = lower.index("allele1")
        if "af_allele2" in lower:
            colmap["eaf"] = lower.index("af_allele2")
        if "imputationinfo" in lower:
            colmap["info"] = lower.index("imputationinfo")
        layout["saige"] = True
    # GWAS Catalog harmonised files carry the submitted columns and the
    # harmonised ``hm_`` columns side by side, possibly on different builds and
    # with the effect allele swapped. Use one set consistently: the hm_ set
    # when it has a harmonised effect, the submitted set otherwise.
    if "hm_effect_allele" in lower and "hm_other_allele" in lower:
        if "hm_beta" in lower or "hm_odds_ratio" in lower:
            for field, name in (("ea", "hm_effect_allele"),
                                ("oa", "hm_other_allele"), ("id", "hm_rsid"),
                                ("chrom", "hm_chrom"), ("pos", "hm_pos"),
                                ("eaf", "hm_effect_allele_frequency")):
                if name in lower:
                    colmap[field] = lower.index(name)
            if "hm_beta" in lower:
                colmap["beta"] = lower.index("hm_beta")
            else:
                colmap.pop("beta", None)
                colmap["or"] = lower.index("hm_odds_ratio")
            # The submitted frequency counts the submitted effect allele, which
            # harmonisation may have swapped: without its hm_ counterpart it
            # would describe the other allele on those rows.
            if ("hm_effect_allele_frequency" not in lower
                    and colmap.get("eaf") is not None):
                del colmap["eaf"]
                layout["eaf_dropped"] = (
                    "hm_ columns without hm_effect_allele_frequency; the "
                    "submitted frequency may count the other allele")
            layout["gwas_catalog_harmonised"] = True
        else:
            for field in ("ea", "oa", "id", "chrom", "pos", "eaf"):
                if (colmap.get(field) is not None
                        and lower[colmap[field]].startswith("hm_")):
                    del colmap[field]
            layout["gwas_catalog_harmonised_ignored"] = (
                "no hm_beta or hm_odds_ratio; the submitted columns are used")
    # PLINK2 --glm: A1 is the tested allele, which may be REF or ALT.
    if (colmap.get("ea") is not None and lower[colmap["ea"]] == "a1"
            and colmap.get("oa") is not None and lower[colmap["oa"]] == "ref"):
        if "omitted" in lower:
            colmap["oa"] = lower.index("omitted")
            layout["plink2_omitted"] = True
        elif "alt" in lower:
            layout["plink2_alt"] = lower.index("alt")
    # PGC daner files: the A1 frequency in controls, FRQ_U_<n>.
    if "eaf" not in colmap:
        for i, h in enumerate(lower):
            if h.startswith("frq_u_"):
                colmap["eaf"] = i
                layout["eaf_from"] = header[i].strip()
                break
    # Neale UK Biobank: variant = chr:pos:ref:alt, effect on ALT.
    if "ea" not in colmap and "oa" not in colmap and "variant" in lower:
        layout["variant_string"] = lower.index("variant")

    # Explicit overrides win (a column name or a zero-based index).
    for field, col in overrides.items():
        if field not in COLUMN_ALIASES:
            raise ColumnMappingError(
                f"unknown sumstats field {field!r}; choose one of "
                + ", ".join(COLUMN_ALIASES))
        # A bare digit string (``--column beta=3``) is a zero-based index.
        if isinstance(col, str) and col.strip().isdigit():
            col = int(col.strip())
        if isinstance(col, (int, np.integer)) and not isinstance(col, bool):
            i = int(col)
            if not 0 <= i < len(header):
                raise ColumnMappingError(
                    f"sumstats column override {field}={i} is out of range for "
                    f"a {len(header)}-column file (indices are zero-based)")
            colmap[field] = i
        else:
            name = str(col).strip().lower()
            if name not in lower:
                raise ColumnMappingError(
                    f"sumstats column override {field}={col!r} was not found; "
                    "available columns: " + ", ".join(header))
            colmap[field] = lower.index(name)
    if "or" in overrides and "beta" not in overrides:
        colmap.pop("beta", None)  # explicit OR semantics override a BETA alias
    if "oa" in overrides or "ea" in overrides:
        layout.pop("plink2_alt", None)
        layout.pop("variant_string", None)
    if colmap.get("ea") is not None and colmap.get("ea") == colmap.get("oa"):
        raise ColumnMappingError(
            "ea and oa resolve to the same column; specify both allele columns")
    return colmap, layout


def _n_plan(colmap, lower, overrides, forced_n):
    """Where per-variant N comes from: ``(source, detail)``."""
    if forced_n is not None:
        return "scalar", None
    if "n_eff" in overrides:
        return "column", lower[colmap["n_eff"]]
    if "n_cases" in overrides or "n_controls" in overrides:
        if "n_cases" not in colmap or "n_controls" not in colmap:
            raise ColumnMappingError("explicit case/control columns require both counts")
        return "case_control", None
    if "n_eff" in colmap and lower[colmap["n_eff"]] in _EFFECTIVE_N_NAMES:
        return "column", lower[colmap["n_eff"]]
    if "n_cases" in colmap and "n_controls" in colmap:
        return "case_control", None
    if "n_eff" in colmap:
        return "column", lower[colmap["n_eff"]]
    return None, None


def detect_columns(path, **col_overrides):
    """Peek at the header and report the detected column mapping.

    Returns ``(header, mapping)`` with ``mapping = {field: column name}`` for
    every resolved field. Reads only the first line.
    """
    with open_text(path) as fh:
        _delim, header = _read_header(fh)
    colmap, _layout = _build_colmap(header, col_overrides)
    return header, {field: header[i] for field, i in colmap.items()}


def inspect_columns(path, **col_overrides):
    """Header-only preflight: mapping, recognised layout and N source.

    Returns a dict with ``header``, ``mapping``, ``layout``, ``n_source`` and
    ``missing`` (required fields that could not be resolved).
    """
    with open_text(path) as fh:
        delim, header = _read_header(fh)
    colmap, layout = _build_colmap(header, col_overrides)
    lower = [h.strip().lower() for h in header]
    source, detail = _n_plan(colmap, lower, col_overrides, None)
    missing = []
    if "variant_string" not in layout:
        missing += [f for f in ("ea", "oa") if f not in colmap]
    if "beta" not in colmap and "or" not in colmap:
        missing.append("beta/or")
    if "se" not in colmap and "pval" not in colmap and "log10p" not in colmap:
        missing.append("se/pval")
    return {"header": header, "delimiter": delim or "whitespace",
            "mapping": {field: header[i] for field, i in colmap.items()},
            "layout": {k: (header[v] if isinstance(v, int) and k != "saige"
                           and not isinstance(v, bool) else v)
                       for k, v in layout.items()},
            "n_source": source if detail is None else f"{source}:{detail}",
            "missing": missing}


def _to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return math.nan


def _z_from_log_tail(log_tail):
    """``z`` with ``log Phi(-z) = log_tail``, for tails beyond float range.

    Newton solve using Mills' expansion; used once ``z > 38``, where the
    neglected term is below 1e-12 in log probability.
    """
    z = math.sqrt(-2 * log_tail)
    for _ in range(5):
        t = 1 / (z * z)
        correction = math.log1p(-t + 3*t*t - 15*t**3 + 105*t**4)
        log_sf = -.5*z*z - math.log(z) - .5*math.log(2*math.pi) + correction
        z += (log_sf - log_tail) / (z + 1/z)
    return z


def _two_sided_z(token):
    """Two-sided normal z from a p-value token, without underflow."""
    from statistics import NormalDist
    p = _to_float(token)
    if 0 < p < 1 and p / 2 > 0:
        return -NormalDist().inv_cdf(p / 2)
    if p != 0 and not (0 < p < 1):
        return math.nan
    from decimal import Decimal, InvalidOperation
    try:
        value = Decimal(str(token))
        if not value.is_finite() or not 0 < value < 1:
            return math.nan
        log_tail = float(value.ln()) - math.log(2.)
    except (InvalidOperation, ValueError, OverflowError):
        return math.nan
    return _z_from_log_tail(log_tail)


def _z_from_log10p(token):
    """Two-sided normal z from a ``-log10 p`` token (REGENIE, FinnGen)."""
    from statistics import NormalDist
    x = _to_float(token)
    if not math.isfinite(x) or x <= 0:
        return math.nan
    if x < 300:
        return -NormalDist().inv_cdf(10.0 ** (-x) / 2)
    return _z_from_log_tail(-x * math.log(10.) - math.log(2.))


def read_sumstats(path, *, n_eff=None, n_cases=None, n_controls=None,
                  require_n=True, **col_overrides):
    """Read a GWAS summary-statistics file into a :class:`Sumstats`.

    Parameters
    ----------
    path : str or path-like
        Delimited text (tab, comma or whitespace, auto-detected), optionally
        gzip/bgzip compressed.
    n_eff : float or str, optional
        A number **overrides** any per-variant N and flattens every row to that
        constant (the low-level escape hatch; :func:`anchor_n_eff` instead keeps
        a column's pattern and rescales its median). A string names the
        per-variant N column (or its zero-based index).
    n_cases, n_controls : float or str, optional
        Numbers give a constant effective N ``4/(1/n_cases + 1/n_controls)``.
        Strings name per-variant case and control count columns.
    require_n : bool, default True
        Raise when no sample size can be found. With ``False`` the ``n_eff``
        column is NaN instead.
    **col_overrides
        Force a field to a column by name or zero-based index, e.g.
        ``read_sumstats(path, ea="A1", beta="effect")``.
    """
    for name, value in (("n_cases", n_cases), ("n_controls", n_controls)):
        if isinstance(value, str):
            col_overrides[name] = value
    if isinstance(n_eff, str):
        col_overrides["n_eff"] = n_eff
        n_eff = None
    scalar_counts = [v for v in (n_cases, n_controls)
                     if v is not None and not isinstance(v, str)]
    if scalar_counts:
        if len(scalar_counts) != 2:
            raise ValueError("n_cases and n_controls must be given together")
        if n_eff is not None:
            raise ValueError("pass either n_eff or n_cases/n_controls, not both")
        n_eff = n_eff_case_control(_n_eff_scalar_value(n_cases),
                                   _n_eff_scalar_value(n_controls))
    forced_n = _n_eff_scalar_value(n_eff)
    path = os.fspath(path)
    with open_text(path) as fh:
        delim, header = _read_header(fh)
        colmap, layout = _build_colmap(header, col_overrides)
        lower = [h.strip().lower() for h in header]

        if "variant_string" not in layout:
            for req in ("ea", "oa"):
                if req not in colmap:
                    raise ValueError(
                        f"could not find a '{req}' allele column in {path}; "
                        f"header was {header}. Pass it explicitly, e.g. {req}=...")
        if "beta" not in colmap and "or" not in colmap:
            raise ValueError(
                f"no effect column (beta/OR) found in {path}; header {header}")
        source, detail = _n_plan(colmap, lower, col_overrides, forced_n)
        if source is None and require_n:
            raise ValueError(
                f"no sample-size column found in {path}; pass n_eff=... or "
                "n_cases=/n_controls=")
        plan = {"source": source, "forced": forced_n,
                "i_n": colmap.get("n_eff") if source == "column" else None,
                "i_nca": colmap.get("n_cases") if source == "case_control" else None,
                "i_nco": colmap.get("n_controls") if source == "case_control" else None}

        def finish(ss):
            log = ss.parse_log
            log["columns"] = {field: header[i] for field, i in colmap.items()}
            log["n_eff_source"] = (source if detail is None
                                   else f"{source}:{detail}") or "missing"
            if layout:
                log["layout"] = {k: (header[v] if isinstance(v, int)
                                     and not isinstance(v, bool) else v)
                                 for k, v in layout.items()}
            if source == "case_control":
                warnings.warn(
                    "effective N computed per variant from case and control "
                    "counts as 4/(1/N_case + 1/N_control). For a meta-analysis "
                    "this overstates N_eff when case fractions differ between "
                    "cohorts; prefer a published effective-N column.",
                    RuntimeWarning, stacklevel=3)
            # A diagnostic, never an inferred transformation: positive effects
            # can be legitimate, and header semantics remain decisive.
            if "beta" in colmap:
                finite = ss.beta[np.isfinite(ss.beta)]
                if (len(finite) >= 20 and np.mean(finite > 0) >= .95
                        and .5 < np.median(finite) < 2):
                    log["suspected_odds_ratio_in_beta"] = True
                    warnings.warn(
                        "BETA values are almost all positive and centered near "
                        "one; check whether this column contains odds ratios. "
                        "Map the OR column explicitly if so; no transformation "
                        "has been inferred.", RuntimeWarning, stacklevel=3)
            return ss

        special = "plink2_alt" in layout or "variant_string" in layout
        kernels = None if special else _jit_kernels()
        # The replay below reopens ``path``, which only a regular file can
        # serve, so pipes keep the single-pass loop.
        if kernels is None or not os.path.isfile(path):
            return finish(_parse_rows_python(fh, delim, colmap, layout, plan))
        try:
            return finish(_parse_rows_jit(fh, delim, colmap, plan, kernels))
        except Exception:
            # Anything the compiled path cannot reproduce exactly replays the
            # whole body through the reference loop, which then returns -- or
            # raises -- what it always does.
            pass
    with open_text(path) as fh:
        fh.readline()
        return finish(_parse_rows_python(fh, delim, colmap, layout, plan))


def _case_control_n(nca, nco):
    """Per-row ``4/(1/nca + 1/nco)``; NaN unless both counts are positive."""
    nca = np.asarray(nca, dtype=float)
    nco = np.asarray(nco, dtype=float)
    ok = np.isfinite(nca) & np.isfinite(nco) & (nca > 0) & (nco > 0)
    out = np.full(nca.shape, math.nan)
    out[ok] = 4.0 / (1.0 / nca[ok] + 1.0 / nco[ok])
    return out


# ---------------------------------------------------------------------------
# Optional Numba tokenizer (from ldpred3).
#
# Without Numba the reference row loop runs. With Numba, ASCII blocks of the
# file are tokenized and their numeric cells parsed in compiled code, about 3x
# faster, and anything the kernels cannot reproduce bit for bit goes back to
# the exact Python rules, cell by cell or block by block.
#
# Numeric cells use Clinger's fast path: a strict ``[+-]digits[.digits]
# [(e|E)[+-]digits]`` token with at most 15 significant digits and a net
# decimal exponent within +-22 is one correctly rounded IEEE multiply or
# divide -- the same double CPython's ``float()`` returns. Every other token is
# flagged and parsed by ``_to_float``.
# ---------------------------------------------------------------------------

_BLOCK_CHARS = 1 << 22
_JIT_MIN_CHARS = 1 << 21
_INT64_LIMIT = float(2 ** 63)
_POW10 = np.array([float(10 ** k) for k in range(23)])   # all exact doubles


def _tokenize(b, delim, starts, ends):
    """Column-major field offsets of a rectangular block of lines.

    ``delim < 0`` means runs of ASCII whitespace (``str.split()``), else that
    single byte (``str.split(delim)``). Empty lines are skipped. Returns the
    number of lines filled, or -1 as soon as a line has another field count.
    """
    k = starts.shape[0]
    r = 0
    j = 0
    fs = 0
    ls = 0
    intok = False
    for i in range(len(b)):
        c = b[i]
        if delim >= 0:
            if c == delim or (c == 10 and i > ls):
                if j >= k:
                    return -1
                starts[j, r] = fs
                ends[j, r] = i
                j += 1
            if c == delim or c == 10:
                fs = i + 1
        elif c == 32 or 9 <= c <= 13 or 28 <= c <= 31:
            if intok:
                ends[j, r] = i
                j += 1
                intok = False
        elif not intok:
            if j >= k:
                return -1
            starts[j, r] = i
            intok = True
        if c == 10:
            if i > ls:
                if j != k:
                    return -1
                r += 1
                j = 0
            ls = i + 1
    return r


def _parse_decimal(b, starts, ends, pow10, out, ok):
    """Clinger fast-path parse of each token; ``ok[k]`` False = not handled."""
    for k in range(len(starts)):
        i = starts[k]
        e = ends[k]
        ok[k] = False
        neg = False
        if i < e and (b[i] == 43 or b[i] == 45):
            neg = b[i] == 45
            i += 1
        mant = 0
        nd = 0
        ndig = 0
        frac = 0
        bad = False
        while i < e and 48 <= b[i] <= 57:
            d = b[i] - 48
            ndig += 1
            if mant != 0 or d != 0:
                nd += 1
                mant = mant * 10 + d
            i += 1
        if i < e and b[i] == 46:
            i += 1
            while i < e and 48 <= b[i] <= 57:
                d = b[i] - 48
                ndig += 1
                frac += 1
                if mant != 0 or d != 0:
                    nd += 1
                    mant = mant * 10 + d
                i += 1
        if ndig == 0 or nd > 15:
            bad = True
        ex = 0
        if not bad and i < e and (b[i] == 101 or b[i] == 69):
            i += 1
            eneg = False
            if i < e and (b[i] == 43 or b[i] == 45):
                eneg = b[i] == 45
                i += 1
            nexp = 0
            while i < e and 48 <= b[i] <= 57:
                if nexp < 6:
                    ex = ex * 10 + (b[i] - 48)
                nexp += 1
                i += 1
            if nexp == 0 or nexp > 5:
                bad = True
            if eneg:
                ex = -ex
        if bad or i != e:
            continue
        e10 = ex - frac
        if mant == 0:
            v = 0.0
        elif 0 <= e10 <= 22:
            v = float(mant) * pow10[e10]
        elif -22 <= e10 < 0:
            v = float(mant) / pow10[-e10]
        else:
            continue
        out[k] = -v if neg else v
        ok[k] = True


def _pool_codes(b, starts, ends, codes, first):
    """Code each token of <= 7 bytes by its distinct value, first-seen order.

    ``codes[k]`` indexes the distinct tokens; a longer token gets -1. Returns
    the number of distinct tokens.
    """
    n = len(starts)
    size = 64
    while size < 2 * n:
        size *= 2
    table_key = np.full(size, -1, dtype=np.int64)
    table_code = np.empty(size, dtype=np.int64)
    ndistinct = 0
    for k in range(n):
        s = starts[k]
        ln = ends[k] - s
        if ln > 7:
            codes[k] = -1
            continue
        key = np.int64(ln)
        for j in range(ln):
            key |= np.int64(b[s + j]) << (8 * (j + 1))
        h = (key * np.int64(-7046029254386353131)) >> 40
        slot = h & (size - 1)
        while table_key[slot] != -1 and table_key[slot] != key:
            slot = (slot + 1) & (size - 1)
        if table_key[slot] == -1:
            table_key[slot] = key
            table_code[slot] = ndistinct
            first[ndistinct] = k
            ndistinct += 1
        codes[k] = table_code[slot]
    return ndistinct


_KERNELS = None


def _jit_kernels():
    """Compiled kernels, or ``None`` when Numba (or its JIT) is unavailable."""
    global _KERNELS
    if _KERNELS is None:
        try:
            from ._numba import jit_available, jit_nogil
            if not jit_available():
                _KERNELS = False
            else:
                _KERNELS = (jit_nogil(_tokenize), jit_nogil(_parse_decimal),
                            jit_nogil(_pool_codes))
        except Exception:                # a broken Numba install: no JIT
            _KERNELS = False
    return _KERNELS or None


def _empty_log(n, n_se):
    return {"n_input": int(n), "n_se_from_pval": int(n_se)}


def _parse_rows_jit(fh, delim, colmap, plan, kernels):
    """Block-wise equivalent of :func:`_parse_rows_python` using ``kernels``.

    A block goes through the compiled path only when it is ASCII and
    rectangular (every non-empty line has the same field count, at least the
    mapped columns); any other block is parsed by the reference loop itself,
    sharing the string pool.
    """
    tokenize, parse_decimal, pool_codes = kernels
    from statistics import NormalDist
    _inv_cdf = NormalDist().inv_cdf
    dcode = -1 if delim is None else ord(delim)
    i_id = colmap.get("id")
    i_chrom = colmap.get("chrom")
    i_pos = colmap.get("pos")
    i_beta = colmap.get("beta")
    i_se = colmap.get("se")
    i_pval = colmap.get("pval")
    i_log10p = colmap.get("log10p")
    i_eaf = colmap.get("eaf")
    i_info = colmap.get("info")
    nfields = max(colmap.values()) + 1
    pool_dict = {}
    pool = pool_dict.setdefault
    nan = math.nan

    def reference(text):
        return _parse_rows_python(text.split("\n"), delim, colmap, {}, plan,
                                  _pool=pool_dict)

    def block(text):
        if (len(text) < _JIT_MIN_CHARS or not text.endswith("\n")
                or not text.isascii()):
            return reference(text)
        b = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
        first_line = text[:text.index("\n")]
        k0 = len(first_line.split(delim) if delim else first_line.split())
        if not first_line or k0 < nfields:
            return reference(text)
        nlines = text.count("\n")
        starts = np.empty((k0, nlines), dtype=np.int64)
        ends = np.empty((k0, nlines), dtype=np.int64)
        m = tokenize(b, dcode, starts, ends)
        if m < 0:
            return reference(text)

        def floats(i):
            s = starts[i, :m]
            e = ends[i, :m]
            out = np.empty(m)
            ok = np.empty(m, dtype=np.bool_)
            parse_decimal(b, s, e, _POW10, out, ok)
            bad = np.flatnonzero(~ok)
            if len(bad):
                memo = {}
                for k, a, z in zip(bad.tolist(), s[bad].tolist(),
                                   e[bad].tolist()):
                    tok = text[a:z]
                    v = memo.get(tok)
                    if v is None:
                        v = memo[tok] = _to_float(tok)
                    out[k] = v
            return out

        def tokens(i, rows):
            s = starts[i, rows]
            e = ends[i, rows]
            return [text[a:z] for a, z in zip(s.tolist(), e.tolist())]

        def strings(i, transform):
            s = starts[i, :m]
            e = ends[i, :m]
            if transform is None:
                return np.array(list(map(text.__getitem__,
                                         map(slice, s.tolist(), e.tolist()))),
                                dtype=object)
            codes = np.empty(m, dtype=np.int64)
            first = np.empty(m, dtype=np.int64)
            u = pool_codes(b, s, e, codes, first)
            vals = np.empty(u + 1, dtype=object)       # [-1] -> None
            vals[:u] = [transform(text[a:z]) for a, z in
                        zip(s[first[:u]].tolist(), e[first[:u]].tolist())]
            out = vals[codes]
            for k in np.flatnonzero(codes < 0).tolist():
                out[k] = transform(text[s[k]:e[k]])
            return out

        if i_beta is not None:
            beta = floats(i_beta)
        else:
            orv = floats(colmap["or"])
            beta = np.full(m, nan)
            pos_or = orv > 0
            # math.log, not np.log: NumPy's SIMD log may differ from libm in
            # the last ulp, and the reference uses math.log.
            beta[pos_or] = np.fromiter(map(math.log, orv[pos_or].tolist()),
                                       dtype=float, count=int(pos_or.sum()))
        se = floats(i_se) if i_se is not None else np.full(m, nan)
        n_se_from_p = 0
        need = np.isnan(se) & ~np.isnan(beta)
        if need.any() and i_pval is not None:
            p = floats(i_pval)
            if np.any(p[need] < np.finfo(float).tiny):
                return reference(text)
            need &= (p > 0) & (p < 1) & (beta != 0)
            for k in np.flatnonzero(need).tolist():
                half = float(p[k]) / 2.0
                if half > 0.0:
                    z = -_inv_cdf(half)
                    if math.isfinite(z) and z > 0:
                        se[k] = abs(float(beta[k])) / z
                        n_se_from_p += 1
        elif need.any() and i_log10p is not None:
            rows = np.flatnonzero(need & (beta != 0))
            for k, tok in zip(rows.tolist(), tokens(i_log10p, rows)):
                z = _z_from_log10p(tok)
                if math.isfinite(z) and z > 0:
                    se[k] = abs(float(beta[k])) / z
                    n_se_from_p += 1
        if i_pos is not None:
            posf = floats(i_pos)
            fin = np.isfinite(posf)
            if ((posf[fin] >= _INT64_LIMIT)
                    | (posf[fin] < -_INT64_LIMIT)).any():
                raise OverflowError("position outside int64")
            pos = np.where(fin, posf, 0.0).astype(np.int64)
        else:
            pos = np.zeros(m, dtype=np.int64)

        if plan["forced"] is not None:
            n_eff = np.full(m, plan["forced"])
        elif plan["i_n"] is not None:
            n_eff = floats(plan["i_n"])
        elif plan["i_nca"] is not None:
            n_eff = _case_control_n(floats(plan["i_nca"]),
                                    floats(plan["i_nco"]))
        else:
            n_eff = np.full(m, nan)

        def pooled(t):
            return pool(t, t)

        def allele(t):
            t = t.upper()
            return pool(t, t)

        return Sumstats(
            id=(strings(i_id, None) if i_id is not None
                else np.full(m, "", dtype=object)),
            chrom=(strings(i_chrom, pooled) if i_chrom is not None
                   else np.full(m, pooled(""), dtype=object)),
            pos=pos,
            ea=strings(colmap["ea"], allele),
            oa=strings(colmap["oa"], allele),
            beta=beta, se=se, n_eff=n_eff,
            eaf=floats(i_eaf) if i_eaf is not None else np.full(m, nan),
            info=floats(i_info) if i_info is not None else np.full(m, nan),
            parse_log=_empty_log(m, n_se_from_p),
        )

    pieces = []
    carry = ""
    while True:
        text = fh.read(_BLOCK_CHARS)
        if not text:
            if carry:
                pieces.append(block(carry))
            break
        text = carry + text
        cut = text.rfind("\n") + 1
        carry = text[cut:]
        if cut:
            pieces.append(block(text[:cut]))
    if not pieces:
        return reference("")
    if len(pieces) == 1:
        return pieces[0]
    return Sumstats(*(np.concatenate([getattr(p, f) for p in pieces])
                      for f in _SUMSTATS_FIELDS),
                    parse_log=_empty_log(
                        sum(p.parse_log["n_input"] for p in pieces),
                        sum(p.parse_log["n_se_from_pval"] for p in pieces)))


def _parse_rows_python(fh, delim, colmap, layout, plan, _pool=None):
    """Reference row-at-a-time parser for the body lines of ``fh``.

    The semantic definition of a sumstats row, and the fallback for
    :func:`_parse_rows_jit`. Chromosome labels and alleles are pooled through
    one dict: they come from tiny alphabets, and a fresh ``str`` per cell
    retained 1.88 GB of duplicates at 10 million rows in ldpred3's measurement.
    """
    if _pool is None:
        _pool = {}
    ids, chroms, poss, eas, oas = [], [], [], [], []
    betas, ses, ns, ncas, ncos, eafs, infos = [], [], [], [], [], [], []
    n_se_from_p = 0

    i_id = colmap.get("id")
    i_chrom = colmap.get("chrom")
    i_pos = colmap.get("pos")
    i_ea = colmap.get("ea")
    i_oa = colmap.get("oa")
    i_beta = colmap.get("beta")
    i_or = colmap.get("or")
    i_se = colmap.get("se")
    i_pval = colmap.get("pval")
    i_log10p = colmap.get("log10p")
    i_eaf = colmap.get("eaf")
    i_info = colmap.get("info")
    i_n, i_nca, i_nco = plan["i_n"], plan["i_nca"], plan["i_nco"]
    forced_n = plan["forced"]
    i_alt = layout.get("plink2_alt")
    i_var = layout.get("variant_string")
    used = [i for i in (i_id, i_chrom, i_pos, i_ea, i_oa, i_beta, i_or, i_se,
                        i_pval, i_log10p, i_n, i_nca, i_nco, i_eaf, i_info,
                        i_alt, i_var) if i is not None]
    nfields = max(used) + 1

    id_add, chrom_add, pos_add = ids.append, chroms.append, poss.append
    ea_add, oa_add = eas.append, oas.append
    beta_add, se_add, n_add = betas.append, ses.append, ns.append
    eaf_add, info_add = eafs.append, infos.append
    pool = _pool.setdefault
    isnan, isfinite, nan = math.isnan, math.isfinite, math.nan

    for raw in fh:
        raw = raw.rstrip("\n")
        if not raw:
            continue
        f = raw.split(delim) if delim else raw.split()
        if len(f) < nfields:                 # pad once, not per field
            f = f + [""] * (nfields - len(f))

        if i_beta is not None:
            beta = _to_float(f[i_beta])
        else:
            orv = _to_float(f[i_or])
            beta = math.log(orv) if orv > 0 else nan

        se = _to_float(f[i_se]) if i_se is not None else nan
        if isnan(se) and not isnan(beta) and beta != 0:
            if i_pval is not None:
                z = _two_sided_z(f[i_pval])
            elif i_log10p is not None:
                z = _z_from_log10p(f[i_log10p])
            else:
                z = nan
            if isfinite(z) and z > 0:
                se = abs(beta) / z
                n_se_from_p += 1

        if forced_n is not None:
            n = forced_n
        elif i_n is not None:
            n = _to_float(f[i_n])
        else:
            n = nan
        if i_nca is not None:
            ncas.append(_to_float(f[i_nca]))
            ncos.append(_to_float(f[i_nco]))

        if i_var is not None:
            parts = f[i_var].split(":")
            if len(parts) >= 4:
                c, p_tok, ref, alt = parts[0], parts[1], parts[2], parts[3]
            else:
                c, p_tok, ref, alt = "", "", "", ""
            if i_chrom is None:
                chrom_add(pool(c, c))
            if i_pos is None:
                pos_v = _to_float(p_tok)
                pos_add(int(pos_v) if isfinite(pos_v) else 0)
            ea = alt.upper()
            oa = ref.upper()
        else:
            ea = f[i_ea].upper()
            oa = f[i_oa].upper()
            if i_alt is not None and ea == oa:
                oa = f[i_alt].upper()
        if i_var is None or i_chrom is not None:
            c = str(f[i_chrom]) if i_chrom is not None else ""
            chrom_add(pool(c, c))
        if i_var is None or i_pos is not None:
            # A blank / non-numeric position is missing (0), not an error.
            pos_v = _to_float(f[i_pos]) if i_pos is not None else 0.0
            pos_add(int(pos_v) if isfinite(pos_v) else 0)
        id_add(f[i_id] if i_id is not None else "")
        ea_add(pool(ea, ea))
        oa_add(pool(oa, oa))
        beta_add(beta); se_add(se); n_add(n)
        eaf_add(_to_float(f[i_eaf]) if i_eaf is not None else nan)
        info_add(_to_float(f[i_info]) if i_info is not None else nan)

    n_eff = np.array(ns, dtype=float)
    if i_nca is not None and forced_n is None:
        n_eff = _case_control_n(ncas, ncos)
    return Sumstats(
        id=np.array(ids, dtype=object),
        chrom=np.array(chroms, dtype=object),
        pos=np.array(poss, dtype=np.int64),
        ea=np.array(eas, dtype=object),
        oa=np.array(oas, dtype=object),
        beta=np.array(betas, dtype=float),
        se=np.array(ses, dtype=float),
        n_eff=n_eff,
        eaf=np.array(eafs, dtype=float),
        info=np.array(infos, dtype=float),
        parse_log=_empty_log(len(betas), n_se_from_p),
    )


def _n_eff_scalar_value(n_eff):
    """Float value of a scalar ``n_eff``, or ``None`` (absent or a column name)."""
    if n_eff is None or isinstance(n_eff, str):
        return None
    try:
        value = float(n_eff)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("n_eff must be a finite positive number") from None
    if (isinstance(n_eff, (bool, np.bool_)) or not math.isfinite(value)
            or value <= 0):
        raise ValueError("n_eff must be a finite positive number")
    return value


def rescale_n_eff(ss, target):
    """Anchor a per-variant N column's median at ``target``, downward only.

    ``ss.n_eff`` is rescaled **in place** by ``min(1, target / median)`` over
    its usable (finite, positive) values: the column's relative pattern is
    kept, values are never scaled up (a per-variant N below the target is a
    measurement, not a deficit) and unusable rows stay NaN. Returns the
    transform record, or ``None`` when no value is usable.
    """
    value = _n_eff_scalar_value(target)
    if value is None:
        raise ValueError("n_eff must be a finite positive number")
    n_col = np.asarray(ss.n_eff, dtype=float)
    usable = np.isfinite(n_col) & (n_col > 0)
    n_usable = int(usable.sum())
    if n_usable == 0:
        return None
    median = float(np.median(n_col[usable]))
    factor = min(1.0, value / median)
    applied = bool(factor < 1.0 - 1e-12)
    if applied:
        n_col[usable] = n_col[usable] * factor
        ss.n_eff = n_col
    return {
        "method": "downward median rescale",
        "source_median": median,
        "target_effective_n": value,
        "factor": factor,
        "applied": applied,
        "upward_scaling_refused": bool(value > median and factor == 1.0),
        "n_usable": n_usable,
    }


def anchor_n_eff(ss, target):
    """Apply a supplied scalar N: rescale the column, or set a constant.

    With at least one usable per-variant value this is :func:`rescale_n_eff`;
    with none, ``target`` becomes every row's N. Mutates ``ss`` and returns
    the transform record.
    """
    record = rescale_n_eff(ss, target)
    if record is not None:
        return record
    target = float(target)
    ss.n_eff = np.full(len(ss), target, dtype=float)
    return {"method": "constant scalar (no usable per-variant N)",
            "source_median": None, "target_effective_n": target,
            "factor": 1.0, "applied": True,
            "upward_scaling_refused": False, "n_usable": 0}


_WRITE_HEADER = ("ID", "CHR", "POS", "EA", "OA", "BETA", "SE", "N_EFF",
                 "EAF", "INFO")


def _fmt(value):
    # repr round-trips float64 exactly; NaN is written as NA.
    return "NA" if value != value else repr(float(value))


def write_sumstats(path, ss):
    """Write ``ss`` as the canonical tab-separated table; returns ``path``.

    Columns ``ID CHR POS EA OA BETA SE N_EFF EAF INFO``, floats written so they
    read back bit for bit, NaN as ``NA``; a ``.gz`` path is compressed. The
    file reads back with :func:`read_sumstats` without overrides.
    """
    with atomic_text(path) as fh:
        fh.write("\t".join(_WRITE_HEADER) + "\n")
        for row in zip(ss.id, ss.chrom, ss.pos, ss.ea, ss.oa, ss.beta, ss.se,
                       ss.n_eff, ss.eaf, ss.info):
            vid, c, p, ea, oa, b, s, n, f, q = row
            fh.write(f"{vid or '.'}\t{c}\t{int(p)}\t{ea}\t{oa}\t{_fmt(b)}\t"
                     f"{_fmt(s)}\t{_fmt(n)}\t{_fmt(f)}\t{_fmt(q)}\n")
    return path
