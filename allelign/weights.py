"""PGS effect sizes: ldpred3 weight tables and PGS Catalog scoring files.

Two on-disk conventions, kept apart because mixing them is a classic error:

* :class:`WeightsTable` -- ldpred3's ``ID CHR POS A1 A2 WEIGHT`` table, where
  ``WEIGHT`` is a *standardised* effect (per genotype SD), optionally with
  ``AF_REF SD_REF SD_SOURCE`` so the same standardisation can be reapplied to
  another cohort ("frozen" scaling).
* :class:`ScoringFile` -- a `PGS Catalog
  <https://www.pgscatalog.org/downloads/#dl_ftp_scoring>`_ scoring file, where
  ``effect_weight`` is per **allele count** and a ``#key=value`` header
  carries the metadata.

:func:`write_pgs_scoring_file` converts the first to the second
(``effect_weight = WEIGHT / SD_REF``). Both align to a reference with
:func:`allelign.harmonize_weights`. Ported from ldpred3 (tables, export) and
multipgs (Catalog reading).
"""

from __future__ import annotations

import math
import os
import warnings
from array import array
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._common import atomic_text, open_text
from .alleles import normalize_chrom

__all__ = ["WeightsTable", "write_weights", "read_weights", "ScoringFile",
           "read_scoring_file", "write_pgs_scoring_file", "SD_SOURCES",
           "PGS_BUILDS", "SCORING_COLUMNS"]

#: Values of ``SD_SOURCE``: observed in the fit cohort, or the HWE value
#: ``sqrt(2f(1-f))`` used by producers that never see a fit cohort.
SD_SOURCES = ("empirical", "hwe")
#: Builds accepted in a scoring file's ``#genome_build=`` header.
PGS_BUILDS = ("GRCh37", "GRCh38")


def _validate_columns(*, id, chrom, pos, effect_allele, other_allele, weight,
                      af=None, sd=None):
    """Validate and normalise the parallel arrays of a weights table."""
    named = {"id": id, "chrom": chrom, "pos": pos,
             "effect_allele": effect_allele, "other_allele": other_allele,
             "weight": weight}
    arrays = {}
    for name, values in named.items():
        arr = np.asarray(values)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        arrays[name] = arr
    lengths = {name: arr.size for name, arr in arrays.items()}
    if len(set(lengths.values())) != 1:
        detail = ", ".join(f"{name}={size}" for name, size in lengths.items())
        raise ValueError(f"weight columns must have equal lengths ({detail})")
    n = arrays["id"].size
    if n == 0:
        raise ValueError("weights must contain at least one variant")
    for name in ("id", "chrom", "effect_allele", "other_allele"):
        values = arrays[name]
        if values.dtype == object and any(value is None for value in values):
            raise ValueError(f"{name} must not contain missing or empty values")
        stripped = np.char.strip(values.astype(str))
        if np.any(stripped == ""):
            raise ValueError(f"{name} must not contain missing or empty values")
        arrays[name] = np.asarray(stripped, dtype=object)
    try:
        pos_float = np.asarray(arrays["pos"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("pos must contain finite non-negative integers") from exc
    if (not np.all(np.isfinite(pos_float)) or np.any(pos_float < 0)
            or np.any(pos_float != np.floor(pos_float))):
        raise ValueError("pos must contain finite non-negative integers")
    arrays["pos"] = pos_float.astype(np.int64)
    try:
        arrays["weight"] = np.asarray(arrays["weight"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("weight must contain finite numeric values") from exc
    if not np.all(np.isfinite(arrays["weight"])):
        raise ValueError("weight must contain finite numeric values")
    if (af is None) != (sd is None):
        raise ValueError("af and sd must be provided together")
    if af is not None:
        for name, values in (("af", af), ("sd", sd)):
            arr = np.asarray(values)
            if arr.ndim != 1 or arr.size != n:
                raise ValueError(f"{name} must be one-dimensional with length {n}")
            try:
                arr = np.asarray(arr, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain finite numeric values") from exc
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must contain finite numeric values")
            arrays[name] = arr
        if np.any((arrays["af"] < 0) | (arrays["af"] > 1)):
            raise ValueError("af must be in [0, 1]")
        if np.any(arrays["sd"] < 0):
            raise ValueError("sd must be >= 0")
    else:
        arrays["af"] = arrays["sd"] = None
    return arrays


@dataclass
class WeightsTable:
    """An ldpred3-format weights table (parallel arrays).

    ``weight`` counts ``a1`` on the standardised scale. ``sd_source`` says how
    ``sd_ref`` was obtained; ``None`` means the file predates the column.
    """

    id: np.ndarray
    chrom: np.ndarray
    pos: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    weight: np.ndarray
    af_ref: np.ndarray = None
    sd_ref: np.ndarray = None
    sd_source: str = None

    @property
    def has_scale(self):
        """True when ``AF_REF`` and ``SD_REF`` are present (frozen scaling)."""
        return self.af_ref is not None and self.sd_ref is not None

    @property
    def sd_is_hwe(self):
        return self.sd_source == "hwe"

    def __len__(self):
        return len(self.id)


def write_weights(path, *, id, chrom, pos, effect_allele, other_allele, weight,
                  af=None, sd=None, sd_source="empirical"):
    """Write an ldpred3 weights table; returns ``path``.

    ``af``/``sd`` (fit-cohort A1 frequency and dosage SD) are written as
    ``AF_REF``/``SD_REF`` with ``SD_SOURCE`` when both are given. Floats use
    ``repr``, so :func:`read_weights` reads them back bit for bit. Files
    written at 8 significant digits still parse.
    """
    if sd_source not in SD_SOURCES:
        raise ValueError(
            f"sd_source must be one of {SD_SOURCES}, got {sd_source!r}")
    cols = _validate_columns(
        id=id, chrom=chrom, pos=pos, effect_allele=effect_allele,
        other_allele=other_allele, weight=weight, af=af, sd=sd)
    have_scale = cols["af"] is not None
    with atomic_text(path) as fh:
        fh.write("ID\tCHR\tPOS\tA1\tA2\tWEIGHT"
                 + ("\tAF_REF\tSD_REF\tSD_SOURCE\n" if have_scale else "\n"))
        out_cols = [cols["id"], cols["chrom"], cols["pos"],
                    cols["effect_allele"], cols["other_allele"], cols["weight"]]
        if have_scale:
            out_cols += [cols["af"], cols["sd"]]
        for row in zip(*out_cols):
            if have_scale:
                vid, c, p, a1, a2, w, af_, sd_ = row
                fh.write(f"{vid}\t{c}\t{p}\t{a1}\t{a2}\t{repr(float(w))}"
                         f"\t{repr(float(af_))}\t{repr(float(sd_))}\t{sd_source}\n")
            else:
                vid, c, p, a1, a2, w = row
                fh.write(f"{vid}\t{c}\t{p}\t{a1}\t{a2}\t{repr(float(w))}\n")
    return path


def read_weights(path, *, sd_source=None, warn_missing_sd_source=True):
    """Parse an ldpred3 weights table (alleles upper-cased).

    A file with ``AF_REF``/``SD_REF`` but no ``SD_SOURCE`` warns, because
    bipred files written before 2026-08 carry the HWE approximation there;
    pass ``sd_source="hwe"`` to declare such a file. A value contradicting the
    file's own column is an error.
    """
    ids, chrom, pos, a1, a2, w, af_ref, sd_ref = [], [], [], [], [], [], [], []
    sd_sources = set()
    with open_text(path) as fh:
        header = fh.readline().rstrip("\n")
        if not header:
            raise ValueError("weights file is empty or has no header")
        cols = header.split("\t") if "\t" in header else header.split()
        if len(cols) != len(set(cols)):
            raise ValueError("weights header contains duplicate columns")
        required = ("ID", "CHR", "POS", "A1", "A2", "WEIGHT")
        missing = [name for name in required if name not in cols]
        if missing:
            raise ValueError("weights header is missing required columns: "
                             + ", ".join(missing))
        if ("AF_REF" in cols) != ("SD_REF" in cols):
            raise ValueError("weights header must contain both AF_REF and SD_REF")
        has_scale = "AF_REF" in cols
        index = {name: cols.index(name) for name in required}
        if has_scale:
            index.update(AF_REF=cols.index("AF_REF"), SD_REF=cols.index("SD_REF"))
        has_sd_source = has_scale and "SD_SOURCE" in cols
        if has_sd_source:
            index["SD_SOURCE"] = cols.index("SD_SOURCE")
        needed = max(index.values())
        for line_no, line in enumerate(fh, start=2):
            line = line.rstrip("\n")
            if not line:
                continue
            f = line.split("\t") if "\t" in line else line.split()
            if len(f) <= needed:
                raise ValueError(f"weights row {line_no} has too few fields")
            try:
                ids.append(f[index["ID"]]); chrom.append(str(f[index["CHR"]]))
                pos.append(float(f[index["POS"]]))
                a1.append(f[index["A1"]].upper())
                a2.append(f[index["A2"]].upper())
                w.append(float(f[index["WEIGHT"]]))
                if has_scale:
                    af_ref.append(float(f[index["AF_REF"]]))
                    sd_ref.append(float(f[index["SD_REF"]]))
                if has_sd_source:
                    sd_sources.add(f[index["SD_SOURCE"]].strip().lower())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid value in weights row {line_no}: {exc}") \
                    from exc
    if len(sd_sources) > 1:
        raise ValueError(
            "weights file mixes SD_SOURCE values: " + ", ".join(sorted(sd_sources)))
    if sd_sources and not sd_sources <= set(SD_SOURCES):
        raise ValueError(
            f"unknown SD_SOURCE {sd_sources.pop()!r}; expected one of {SD_SOURCES}")
    declared = sd_sources.pop() if sd_sources else None
    if sd_source is not None:
        if sd_source not in SD_SOURCES:
            raise ValueError(
                f"sd_source must be one of {SD_SOURCES}, got {sd_source!r}")
        if declared is not None and declared != sd_source:
            raise ValueError(
                f"sd_source={sd_source!r} contradicts the file's SD_SOURCE="
                f"{declared!r}")
        declared = sd_source
    elif declared is None and has_scale and warn_missing_sd_source:
        warnings.warn(
            "weights file has AF_REF/SD_REF but no SD_SOURCE column; assuming "
            "SD_REF is an observed fit-cohort SD. Files written by bipred "
            "before 2026-08 carry the HWE approximation sqrt(2f(1-f)) instead; "
            "pass sd_source='hwe' when reading those.", UserWarning, stacklevel=2)
    normalized = _validate_columns(
        id=ids, chrom=chrom, pos=pos, effect_allele=a1, other_allele=a2,
        weight=w, af=af_ref if has_scale else None,
        sd=sd_ref if has_scale else None)
    return WeightsTable(
        id=normalized["id"], chrom=normalized["chrom"], pos=normalized["pos"],
        a1=normalized["effect_allele"], a2=normalized["other_allele"],
        weight=normalized["weight"], af_ref=normalized["af"],
        sd_ref=normalized["sd"], sd_source=declared)


# ---------------------------------------------------------------------------
# PGS Catalog scoring files
# ---------------------------------------------------------------------------

#: Column name -> canonical field. Harmonised names win over the originals.
SCORING_COLUMNS = {
    "id": ("hm_rsid", "rsid", "rsids", "snpid", "variant_id"),
    "chrom": ("hm_chr", "chr_name", "chromosome", "chr"),
    "pos": ("hm_pos", "chr_position", "position", "bp"),
    "ea": ("effect_allele", "a1"),
    "oa": ("other_allele", "hm_inferotherallele", "a2"),
    "weight": ("effect_weight", "weight", "beta"),
}
# Compatibility for early consumers; new code uses SCORING_COLUMNS.
_SCORING_COLUMNS = SCORING_COLUMNS
_NON_ADDITIVE = ("is_dominant", "is_recessive", "is_haplotype", "is_diplotype",
                 "is_interaction")
_RATIO_WEIGHTS = ("or", "hr", "odds_ratio", "hazard_ratio")
_TRUE = ("true", "t", "yes", "y", "1")
_MISSING_ALLELE = ("", ".", "NA", "N/A", "NULL")


@dataclass
class ScoringFile:
    """One parsed PGS Catalog scoring file.

    ``weight`` counts ``ea`` per allele copy. ``meta`` holds the
    ``#key=value`` header (keys lower-cased); ``log`` records what parsing
    dropped or transformed.
    """

    id: np.ndarray
    chrom: np.ndarray
    pos: np.ndarray
    ea: np.ndarray
    oa: np.ndarray
    weight: np.ndarray
    meta: dict = field(default_factory=dict)
    log: dict = field(default_factory=dict)
    path: str = ""

    def __len__(self):
        return int(self.weight.size)

    @property
    def pgs_id(self):
        """``pgs_id`` from the header, falling back to the file name."""
        return self.meta.get("pgs_id") or scoring_file_id(self.path)

    @property
    def trait(self):
        return self.meta.get("trait_reported", "")

    @property
    def genome_build(self):
        """Build of the coordinates actually used (the harmonised build if any)."""
        return self.log.get("build_used") or self.meta.get("genome_build", "")

    def __str__(self):
        return (f"{self.pgs_id}: {len(self)} variants"
                + (f", {self.trait}" if self.trait else "")
                + (f", build {self.genome_build}" if self.genome_build else ""))


def scoring_file_id(path):
    """``PGS000123`` from ``.../PGS000123_hmPOS_GRCh37.txt.gz``."""
    base = os.path.basename(str(path))
    for suffix in (".txt.gz", ".tsv.gz", ".txt", ".tsv", ".gz"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.split("_")[0] if base.startswith("PGS") else base


def _pick(header, names):
    lower = {name.strip().lower(): i for i, name in enumerate(header)}
    for want in names:
        if want in lower:
            return lower[want], want
    return None, None


def _scoring_rows(fh, meta):
    """Stream rows; ``#key=value`` lines anywhere go to ``meta``."""
    delimiter, first = None, True
    for line in fh:
        if line.startswith("#"):
            stripped = line.lstrip("#").strip()
            if "=" in stripped:
                key, _, value = stripped.partition("=")
                meta[key.strip().lower()] = value.strip()
            continue
        line = line.rstrip("\n").rstrip("\r")
        if not line:
            continue
        if first:
            delimiter, first = ("\t" if "\t" in line else None), False
        yield line.split(delimiter)


def read_scoring_file(path, *, prefer_harmonized=True, drop_non_additive=True,
                      keep=None):
    """Parse a PGS Catalog scoring file (plain or gzipped).

    * ``hm_rsID``/``hm_chr``/``hm_pos`` are used when present (they are on
      the build the file name advertises); where ``other_allele`` is missing,
      ``hm_inferOtherAllele`` fills it row by row.
    * ``weight_type`` OR/HR (odds or hazard ratios) are log-transformed; a
      non-positive ratio is an error.
    * Dominant, recessive, haplotype, diplotype and interaction rows cannot be
      represented by an additive dosage score; they are dropped and counted
      (``drop_non_additive=False`` keeps them, which is wrong for every one).

    ``keep(id, position) -> bool`` prefilters rows a reference cannot match,
    so a genome-wide file aligned to a small panel never stores their strings.
    Rejected rows still count in the whole-file checks: the ratio test and
    ``log["weight_square_sum_all"]``, the denominator of the matched weight
    mass. ``log["n_reference_prefiltered"]`` counts them.
    """
    meta = {}
    with open_text(path) as fh:
        rows = _scoring_rows(fh, meta)
        header = next(rows, None)
        if header is None:
            raise ValueError(f"{path}: no column header found (only comment lines)")
        idx, used = {}, {}
        for field_name, names in SCORING_COLUMNS.items():
            candidates = names if prefer_harmonized else \
                tuple(n for n in names if not n.startswith("hm_")) + names
            i, chosen = _pick(header, candidates)
            idx[field_name] = i
            used[field_name] = chosen
        inferred_oa_idx, inferred_oa_name = _pick(header, ("hm_inferotherallele",))
        if idx["weight"] is None:
            raise ValueError(f"{path}: no effect_weight column in {header}")
        if idx["ea"] is None:
            raise ValueError(f"{path}: no effect_allele column in {header}")
        if idx["id"] is None and (idx["chrom"] is None or idx["pos"] is None):
            raise ValueError(f"{path}: needs either an rsID column or both "
                             f"chromosome and position, found {header}")
        flag_cols = [i for i, name in enumerate(header)
                     if name.strip().lower() in _NON_ADDITIVE]
        n_read = n_flagged = n_bad_weight = n_inferred_oa = n_bad_pos = 0
        ids, chroms, eas, oas = [], [], [], []
        poss, ws, retained = array("q"), array("d"), array("q")
        width = len(header)
        for fields in rows:
            n_read += 1
            if len(fields) < width:
                fields = fields + [""] * (width - len(fields))
            if flag_cols and any(fields[i].strip().lower() in _TRUE
                                 for i in flag_cols):
                n_flagged += 1
                if drop_non_additive:
                    continue
            try:
                w = float(fields[idx["weight"]])
            except ValueError:
                n_bad_weight += 1
                continue
            if not math.isfinite(w):
                n_bad_weight += 1
                continue
            sid = fields[idx["id"]].strip() if idx["id"] is not None else ""
            raw_pos = (fields[idx["pos"]].strip()
                       if idx["pos"] is not None else "0")
            if raw_pos.lstrip("-").isdigit():
                position = int(raw_pos)
            else:
                position = 0
                n_bad_pos += 1
            oa = (fields[idx["oa"]].strip().upper()
                  if idx["oa"] is not None else "")
            if (oa in _MISSING_ALLELE and inferred_oa_idx is not None
                    and inferred_oa_idx != idx["oa"]):
                inferred = fields[inferred_oa_idx].strip().upper()
                if inferred not in _MISSING_ALLELE:
                    oa = inferred
                    n_inferred_oa += 1
            ws.append(w)
            # Numeric weights are kept for every valid row, so the ratio test
            # and the weight-mass total read the whole file; only the costly
            # strings of rejected rows are skipped.
            if keep is not None:
                if not keep(sid, position):
                    continue
                retained.append(len(ws) - 1)
            ids.append(sid)
            chroms.append(fields[idx["chrom"]].strip()
                          if idx["chrom"] is not None else "")
            poss.append(position)
            eas.append(fields[idx["ea"]].strip().upper())
            oas.append(oa)

    pos = np.frombuffer(poss, dtype=np.int64)
    weight = np.frombuffer(ws, dtype=np.float64).copy()
    coordinate_used = [str(used[name]).startswith("hm_")
                       for name in ("id", "chrom", "pos") if used[name]]
    harmonized_columns = bool(coordinate_used) and all(coordinate_used)
    log = {"n_rows": n_read, "n_kept": int(weight.size),
           "n_non_additive": n_flagged, "n_unparsable_weight": n_bad_weight,
           "n_inferred_other_allele": n_inferred_oa,
           "n_unparsable_position": n_bad_pos,
           "columns_used": {k: v for k, v in used.items() if v},
           "harmonized_columns": harmonized_columns}
    if coordinate_used and not harmonized_columns:
        log["mixed_coordinate_warning"] = (
            "some but not all of the rsID/chromosome/position columns are "
            "harmonized; coordinates may mix genome builds")
    if harmonized_columns:
        log["build_used"] = (meta.get("hmpos_build")
                             or meta.get("harmonized_build")
                             or meta.get("genome_build", ""))
    if n_inferred_oa:
        log["columns_used"]["oa_fallback"] = inferred_oa_name
    multi = sum(1 for a in oas if "/" in a)
    if multi:
        log["n_multi_valued_other_allele"] = multi
    weight_type = meta.get("weight_type", "").strip().lower()
    if weight_type in _RATIO_WEIGHTS:
        if np.any(weight <= 0):
            raise ValueError(
                f"{path}: weight_type={weight_type!r} says these are ratios, "
                f"but {int(np.sum(weight <= 0))} are non-positive, so they "
                "cannot be. Fix the header or the file.")
        weight = np.log(weight)
        log["log_transformed"] = True
    if keep is not None:
        log["n_reference_prefiltered"] = int(weight.size - len(retained))
        log["weight_square_sum_all"] = float(weight @ weight)
        weight = weight[np.frombuffer(retained, dtype=np.int64)]
        log["n_kept"] = int(weight.size)
    return ScoringFile(
        id=np.array(ids, dtype=object), chrom=np.array(chroms, dtype=object),
        pos=pos, ea=np.array(eas, dtype=object),
        oa=np.array(oas, dtype=object), weight=weight, meta=meta, log=log,
        path=str(path))


#: An SD at or below this is inert in frozen scoring, so it exports as 0.
_PGS_SD_EPS = 1e-12
_PGS_CHROM_ALIASES = {"PAR": "X"}


def _pgs_chrom(label):
    """A chromosome label in the Catalog's ``chr_name`` style (1-22, X, Y, MT)."""
    s = normalize_chrom(label)
    return _PGS_CHROM_ALIASES.get(s, s)


def _pgs_default_id(path):
    base = Path(path).name
    for suffix in (".txt.gz", ".tsv.gz", ".txt", ".tsv", ".gz"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return "_".join(base.split()) or "ALLELIGN"


def _pgs_header_value(name, value):
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    if any(ch in text for ch in "\t\n\r#="):
        raise ValueError(f"{name} must not contain tabs, newlines, '#' or '=' "
                         f"(got {value!r}); it is embedded in a #key=value "
                         "header line")
    return text


def write_pgs_scoring_file(path, wt, *, genome_build, pgs_id=None,
                           trait_reported=None, source=None):
    """Write an ldpred3 weights table as a PGS Catalog scoring file.

    ``effect_weight = WEIGHT / SD_REF`` on the raw allele-count scale, so a
    plain dosage score ``sum(effect_weight * g)`` equals the frozen score
    ``sum((WEIGHT/SD_REF) (g - 2 AF_REF))`` up to the constant
    ``sum(effect_weight * 2 AF_REF)``. Rows with ``SD_REF <= 1e-12`` export as
    0. The header records ``#genome_build`` (required: nothing in a weights
    table records the build), ``#variants_number`` (the row count) and
    ``#sd_source`` -- ``unspecified`` when the table predates ``SD_SOURCE``,
    rather than claiming ``empirical``. ``#variants_genome_wide`` is omitted:
    a chromosome subset or a hits list would otherwise be labelled as the
    genome-wide score.

    plink2 ``--score`` reads one header line, not comment blocks: strip the
    ``#`` lines first (``grep -v '^#'``). ``pgsc_calc`` ingestion is untested.
    """
    if not isinstance(wt, WeightsTable):
        raise TypeError("wt must be a WeightsTable")
    if not wt.has_scale:
        raise ValueError(
            "a PGS Catalog export needs AF_REF/SD_REF to divide the weights by; "
            "this table has no scale columns")
    if genome_build not in PGS_BUILDS:
        raise ValueError(f"genome_build must be one of {PGS_BUILDS}, got "
                         f"{genome_build!r}")
    pgs_id = _pgs_header_value(
        "pgs_id", pgs_id if pgs_id is not None else _pgs_default_id(path))
    trait = _pgs_header_value(
        "trait_reported",
        trait_reported if trait_reported is not None else "unspecified")
    sd = np.asarray(wt.sd_ref, dtype=float)
    weight = np.asarray(wt.weight, dtype=float)
    inert = ~np.isfinite(sd) | (sd <= _PGS_SD_EPS)
    effect_weight = np.where(inert, 0.0, weight / np.where(inert, 1.0, sd))
    n = len(wt)
    seen, n_dup = set(), 0
    for vid in np.asarray(wt.id):
        key = str(vid)
        n_dup += key in seen
        seen.add(key)
    if n_dup:
        warnings.warn(f"{n_dup} duplicated variant IDs in the PGS export; "
                      "plink2 --score refuses them unless passed ignore-dup-ids",
                      RuntimeWarning, stacklevel=2)
    if wt.sd_source == "hwe":
        warnings.warn(
            "exporting weights with SD_SOURCE=hwe: effect_weight divides by "
            "the Hardy-Weinberg SD sqrt(2f(1-f)), not an observed dosage SD "
            "(misspecified for sex chromosomes under male hemizygosity).",
            RuntimeWarning, stacklevel=2)
    elif wt.sd_source is None:
        warnings.warn(
            "the weights table records no SD_SOURCE; the export says "
            "#sd_source=unspecified. Read the table with sd_source='hwe' or "
            "'empirical' to record it.", RuntimeWarning, stacklevel=2)
    from . import __version__
    header = [
        "###PGS CATALOG SCORING FILE",
        "#format_version=2",
        f"#pgs_id={pgs_id}",
        f"#trait_reported={trait}",
        f"#genome_build={genome_build}",
        f"#variants_number={n}",
        "#weight_type=beta",
        "#weight_type_description=standardized effect divided by the "
        "fit-cohort dosage SD (WEIGHT/SD_REF), per effect-allele count",
        f"#sd_source={wt.sd_source or 'unspecified'}",
        f"#source={_pgs_header_value('source', source) if source else 'allelign ' + __version__}",
    ]
    with atomic_text(path) as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write("rsID\tchr_name\tchr_position\teffect_allele\tother_allele"
                 "\teffect_weight\tAF\n")
        for vid, c, p, a1, a2, ew, af_ in zip(
                wt.id, wt.chrom, wt.pos, wt.a1, wt.a2, effect_weight,
                np.asarray(wt.af_ref, dtype=float)):
            fh.write(f"{vid}\t{_pgs_chrom(c)}\t{int(p)}\t{a1}\t{a2}"
                     f"\t{ew:.12g}\t{af_:.8g}\n")
    return path
