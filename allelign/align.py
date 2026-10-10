"""Align summary statistics or weights to a reference variant table.

The aligned ``beta`` counts the reference's ``a1`` allele, so it can be applied
directly to ``a1`` dosages. This is where a silent allele mix-up would flip
effect signs, so the rules are explicit:

1. *Locate.* A row whose identifier the reference holds (``id`` or
   ``alternate_id``) is matched by identifier, coordinate-checked: a unique
   identifier at a disagreeing chrom/pos is unmatched (a silent build mix-up),
   an identifier held at several loci is resolved by the row's coordinates or
   rejected, and neither falls through to a different variant at the row's
   coordinates. A row with no usable identifier matches by ``(chrom, pos)``.
   Chromosome labels are normalised (:func:`~allelign.alleles.normalize_chrom`)
   and ``pos <= 0`` means missing.
2. *Ambiguity.* With ``drop_ambiguous`` (default) a strand-ambiguous pair
   (A/T, C/G, and multi-base pairs such as ``AA``/``TT``) is dropped.
3. *Orient.* Alleles are compared upper-cased: same order, swapped,
   complemented, complemented and swapped; a swap negates ``beta``.
4. *Claim.* Each reference variant is taken by at most one row, the first in
   file order to reach it with compatible alleles and finite ``beta``/``se``.
   Several rows at one locus (multi-allelic sites) are tried in order.

Output rows are in reference order; ``src_index`` points back to the input.
Two implementations of one rule set -- a vectorised join for plain string and
numeric NumPy columns, and the reference row loop for anything else -- give
identical results. Ported from ldpred3 (its harmonisation was checked against an
independent naive implementation on 5,500 cases).
"""

from __future__ import annotations

import math
import threading
import warnings
from dataclasses import dataclass, field, replace
from itertools import repeat

import numpy as np

from . import status as _st
from .alleles import (allele_orientation as _allele_orientation,
                      is_ambiguous as _ambiguous_pair,
                      normalize_chrom as _norm_chrom,
                      reverse_complement as _complement)

__all__ = ["Harmonized", "harmonize", "diagnose_match", "reference_coverage",
           "coverage_message", "reanchor_on_identifier", "HarmonizedWeights",
           "harmonize_weights", "count_positional_only_matches",
           "variant_indices", "REFERENCE_COVERAGE_WARN",
           "SEVERE_REFERENCE_COVERAGE", "BUILD_MISMATCH_FRACTION"]

# Per-table matching indices, reused across calls on a frozen table: rebuilding
# the O(variants) dicts per file dominated runs that align many files to one
# reference. The cache lives on the table instance; only read-only columns are
# cached, and the key holds the column objects so replacement invalidates it.
_CACHE_ATTR = "_harmonize_index_cache"
_INDEX_CACHE_LOCK = threading.Lock()


def _readonly_column(column):
    """A read-only view over a writable owner is still mutable metadata."""
    if column is None:
        return True
    if not isinstance(column, np.ndarray):
        return False
    while isinstance(column, np.ndarray):
        if column.flags.writeable:
            return False
        column = column.base
    if column is None:
        return True
    try:
        return memoryview(column).readonly
    except TypeError:
        return False


def _table_cache_key(variants):
    columns = (variants.id, variants.chrom, variants.pos, variants.a1,
               variants.a2, getattr(variants, "alternate_id", None))
    if not all(_readonly_column(column) for column in columns):
        return None
    return (*columns, len(variants))


@dataclass
class Harmonized:
    """Summary statistics aligned to a reference's ``a1`` alleles.

    ``var_index`` indexes the reference table; ``beta``/``se``/``n_eff`` are
    parallel to it and count ``a1``. ``src_index`` is the input row each match
    came from, so extra source columns can be pulled positionally.
    """

    var_index: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    n_eff: np.ndarray
    flipped: np.ndarray
    src_index: np.ndarray = None
    log: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.var_index)


_TOLIST_KINDS = frozenset("biuUSO")


def _rows(column):
    """``column`` as a Python list when that is value-identical, else as is."""
    if isinstance(column, np.ndarray) and (
            column.dtype.kind in _TOLIST_KINDS
            or column.dtype == np.float64):
        return column.tolist()
    return column


def _build_id_index(variants):
    v_id = variants.id
    alternate_id = getattr(variants, "alternate_id", None)
    by_id = {}
    for i, vid in enumerate(v_id):
        if vid and vid != ".":
            by_id.setdefault(vid, []).append(i)
        if alternate_id is not None:
            aid = alternate_id[i]
            if aid and aid != "." and aid != vid:
                by_id.setdefault(aid, []).append(i)
    return by_id


_STR_TYPES = frozenset((str, np.str_))


def _str_rows(column):
    """``column`` as a list of ``str`` when it is a NumPy array of text only."""
    if not isinstance(column, np.ndarray) or column.ndim != 1:
        return None
    if column.dtype.kind == "U":
        return column.tolist()
    if column.dtype.kind != "O":
        return None
    rows = column.tolist()
    return rows if set(map(type, rows)) <= _STR_TYPES else None


def _query_locus(chrom, query_memo):
    """``(usable, normalised)`` for a sumstats chromosome label, memoised."""
    hit = query_memo.get(chrom)
    if hit is None:
        hit = query_memo[chrom] = (bool(str(chrom).strip()),
                                   _norm_chrom(chrom))
    return hit


def _coordinate_candidates(candidates, variants, chrom, pos, memo,
                           query_memo=None):
    """Disambiguate an identifier, and reject a unique ID at the wrong locus."""
    if query_memo is None:
        query_memo = {}
    loci = []
    for gi in candidates:
        raw = variants.chrom[gi]
        candidate_norm = memo.get(raw)
        if candidate_norm is None:
            candidate_norm = memo[raw] = _norm_chrom(raw)
        loci.append((candidate_norm, int(variants.pos[gi])))

    unique_loci = len(set(loci)) == 1
    try:
        pos = int(pos)
    except (TypeError, ValueError, OverflowError):
        pos = 0
    usable_query = False
    if pos > 0:
        usable_query, query_norm = _query_locus(chrom, query_memo)

    if unique_loci:
        if not usable_query:
            return candidates
        candidate_norm, candidate_pos = loci[0]
        if candidate_pos <= 0:
            return candidates
        if candidate_pos == pos and candidate_norm == query_norm:
            return candidates
        return None

    if not usable_query:
        return None
    hits = []
    for gi, (candidate_norm, candidate_pos) in zip(candidates, loci):
        if candidate_pos == pos and candidate_norm == query_norm:
            hits.append(gi)
    return hits or None


def variant_indices(variants):
    """``[key, by_id, by_pos, upper, chrom_memo, vector]`` for one table.

    The matching indices :func:`harmonize` builds once per frozen table and
    reuses; ``by_id`` maps an identifier to its row(s). Callers that look up
    many identifiers against a reference can borrow it instead of building
    their own (a 1.4M-variant dict costs about 0.6 GiB).
    """
    return _variant_indices(variants)


def _variant_indices(variants):
    """``[key, by_id, by_pos, upper, chrom_memo, vector]`` for one table."""
    key = _table_cache_key(variants)
    try:
        namespace = variants.__dict__
    except AttributeError:  # slotted / C-implemented variant tables
        return [key, _build_id_index(variants), None, {}, {}, {}]
    if key is None:
        with _INDEX_CACHE_LOCK:
            namespace.pop(_CACHE_ATTR, None)
        return [key, _build_id_index(variants), None, {}, {}, {}]
    with _INDEX_CACHE_LOCK:
        entry = namespace.get(_CACHE_ATTR)
        if (entry is not None and entry[0][-1] == key[-1]
                and all(old is new for old, new in zip(entry[0][:-1], key[:-1]))):
            return entry
        namespace.pop(_CACHE_ATTR, None)
    del entry
    entry = [key, _build_id_index(variants), None, {}, {}, {}]
    try:
        with _INDEX_CACHE_LOCK:
            namespace[_CACHE_ATTR] = entry
    except AttributeError:  # pragma: no cover - frozen table
        pass
    return entry


def _build_pos_index(variants, memo):
    v_chrom, v_pos = variants.chrom, variants.pos
    by_pos = {}
    for i in range(len(variants)):
        pos = int(v_pos[i])
        if pos <= 0:      # the missing-coordinate sentinel, not a locus
            continue
        c = v_chrom[i]
        nc = memo.get(c)
        if nc is None:
            nc = memo[c] = _norm_chrom(c)
        by_pos.setdefault((nc, pos), []).append(i)
    return by_pos


# ---------------------------------------------------------------------------
# Vectorised harmonisation: chromosome labels and alleles factorised to
# integer codes, the rsID join as one ``map`` over a dict, the position join
# as sorted ``chrom_code * span + pos`` keys. Only rows whose outcome depends on
# another row (multi-allelic sites, duplicate IDs, a reference row reached
# twice) go through the row loop.
# ---------------------------------------------------------------------------

_POS_LIMIT = 1 << 62


def _int_positions(column):
    """``column`` as int64 when ``int(value)`` of every element is exact."""
    if not isinstance(column, np.ndarray) or column.ndim != 1:
        return None
    kind = column.dtype.kind
    if kind not in "iuf" or (kind == "f" and column.dtype.itemsize > 8):
        return None
    if column.size:
        if kind == "f" and not np.isfinite(column).all():
            return None
        if kind != "i" and not (
                -_POS_LIMIT < column.min() and column.max() < _POS_LIMIT):
            return None
    return column.astype(np.int64)


def _numeric(column):
    return (isinstance(column, np.ndarray) and column.ndim == 1
            and column.dtype.kind in "iuf" and column.dtype.itemsize <= 8)


def _factorize(rows):
    """``(uniques, codes)``: first-appearance distinct values and int codes."""
    index = dict.fromkeys(rows)
    for code, value in enumerate(index):
        index[value] = code
    codes = np.fromiter(map(index.__getitem__, rows), dtype=np.int64,
                        count=len(rows))
    return list(index), codes


def _genotype_columns(variants, entry):
    """Per-table integer view of ``variants``, cached in ``entry[5]``."""
    cache = entry[5]
    cols = cache.get("columns")
    if cols is not None:
        return cols or None
    cols = False
    n = len(variants)
    chrom, a1, a2 = (_str_rows(variants.chrom), _str_rows(variants.a1),
                     _str_rows(variants.a2))
    pos = _int_positions(variants.pos)
    if (chrom is not None and a1 is not None and a2 is not None
            and pos is not None and len(variants.id) == n
            and len(chrom) == len(a1) == len(a2) == len(pos) == n):
        labels, label_codes = _factorize(chrom)
        norm_code = {}
        per_label = np.array(
            [norm_code.setdefault(_norm_chrom(c), len(norm_code))
             for c in labels], dtype=np.int64)
        vocab = {}
        allele_codes = []
        for column in (a1, a2):
            uniques, codes = _factorize(column)
            per_value = np.array(
                [vocab.setdefault(str(a).upper(), len(vocab))
                 for a in uniques], dtype=np.int64)
            allele_codes.append(per_value[codes])
        cols = {"pos": pos, "chrom": per_label[label_codes],
                "norm_code": norm_code, "vocab": vocab,
                "a1": allele_codes[0], "a2": allele_codes[1]}
    cache["columns"] = cols
    return cols or None


def _position_index(cols, cache):
    """Sorted ``(chrom, pos)`` join keys, built on first use and cached."""
    index = cache.get("pos_index")
    if index is not None:
        return index
    pos, chrom = cols["pos"], cols["chrom"]
    rows = np.flatnonzero(pos > 0)
    span = int(pos[rows].max()) + 1 if rows.size else 1
    if max(len(cols["norm_code"]), 1) * span >= _POS_LIMIT:
        index = False
    else:
        keys = chrom[rows] * span + pos[rows]
        order = np.argsort(keys, kind="stable")
        index = (keys[order], rows[order], span)
    cache["pos_index"] = index
    return index


def _vector_columns(sumstats, variants, entry):
    """Columns for the vectorised join, or ``None`` (defer to the row loop)."""
    try:
        n = len(sumstats)
        s_id, s_chrom = _str_rows(sumstats.id), _str_rows(sumstats.chrom)
        s_ea, s_oa = _str_rows(sumstats.ea), _str_rows(sumstats.oa)
        if s_id is None or s_chrom is None or s_ea is None or s_oa is None:
            return None
        s_pos = _int_positions(sumstats.pos)
        beta, se, n_eff = sumstats.beta, sumstats.se, sumstats.n_eff
        if (s_pos is None or not (_numeric(beta) and _numeric(se)
                                  and _numeric(n_eff))):
            return None
        if not (len(s_id) == len(s_chrom) == len(s_ea) == len(s_oa)
                == len(s_pos) == len(beta) == len(se) == len(n_eff) == n):
            return None
        genotype = _genotype_columns(variants, entry)
    except Exception:  # noqa: BLE001 - defer every oddity to the row loop
        return None
    if genotype is None:
        return None
    return {"id": s_id, "chrom": s_chrom, "pos": s_pos, "raw_pos":
            sumstats.pos, "ea": s_ea, "oa": s_oa, "beta": beta, "se": se,
            "n_eff": n_eff, "genotype": genotype}


def _sumstats_alleles(rows, vocab, local):
    """Upper-cased allele codes over the reference vocabulary (+ extras)."""
    uniques, codes = _factorize(rows)
    upper = [str(a).upper() for a in uniques]
    base = len(vocab)
    per_value = np.array(
        [vocab[a] if a in vocab else local.setdefault(a, base + len(local))
         for a in upper],
        dtype=np.int64)
    return upper, codes, per_value


def _complement_codes(upper, vocab, local):
    out = []
    for a in upper:
        c = _complement(a)
        if c is None:
            out.append(-1)
        else:
            code = vocab.get(c)
            out.append(code if code is not None else local.get(c, -2))
    return np.array(out, dtype=np.int64)


def _harmonize_vector(sumstats, variants, entry, columns, drop_ambiguous):
    """Vectorised :func:`harmonize`; identical output to the row loop."""
    g = columns["genotype"]
    cache = entry[5]
    n = len(sumstats)
    s_pos = columns["pos"]

    labels, label_codes = _factorize(columns["chrom"])
    norm_code = g["norm_code"]
    usable = np.array([bool(str(c).strip()) for c in labels],
                      dtype=bool)[label_codes]
    q_chrom = np.array([norm_code.get(_norm_chrom(c), -1) for c in labels],
                       dtype=np.int64)[label_codes]

    by_id = entry[1]
    none = (-1,)
    id_hit = np.array(
        [c[0] if len(c) == 1 else -2
         for c in map(by_id.get, columns["id"], repeat(none))],
        dtype=np.int64)

    cand = np.full(n, -1, dtype=np.int64)
    multi = {}
    n_unmatched = n_ambiguous_id = 0

    rows = np.flatnonzero(id_hit >= 0)
    gi = id_hit[rows]
    q = s_pos[rows]
    g_pos = g["pos"][gi]
    keep = (~((q > 0) & usable[rows]) | (g_pos <= 0)
            | ((g_pos == q) & (g["chrom"][gi] == q_chrom[rows])))
    cand[rows[keep]] = gi[keep]
    n_unmatched += int(rows.size - np.count_nonzero(keep))

    rows = np.flatnonzero(id_hit == -2)
    if rows.size:
        memo, query_memo = entry[4], {}
        s_id, s_chrom, raw_pos = (columns["id"], columns["chrom"],
                                  _rows(columns["raw_pos"]))
        for k in rows.tolist():
            hits = _coordinate_candidates(by_id[s_id[k]], variants,
                                          s_chrom[k], raw_pos[k], memo,
                                          query_memo)
            if not hits:
                n_ambiguous_id += 1
            elif len(hits) == 1:
                cand[k] = hits[0]
            else:
                multi[k] = hits

    rows = np.flatnonzero(id_hit == -1)
    if rows.size:
        index = _position_index(g, cache)
        if index is False:
            return None
        keys, key_rows, span = index
        q = s_pos[rows]
        c = q_chrom[rows]
        ok = (q > 0) & (q < span) & (c >= 0)
        rows_ok = rows[ok]
        qk = c[ok] * span + q[ok]
        lo = np.searchsorted(keys, qk, side="left")
        hi = np.searchsorted(keys, qk, side="right")
        width = hi - lo
        one = width == 1
        cand[rows_ok[one]] = key_rows[lo[one]]
        for k, a, b in zip(rows_ok[width > 1].tolist(),
                           lo[width > 1].tolist(), hi[width > 1].tolist()):
            multi[k] = key_rows[a:b].tolist()
        n_unmatched += int(rows.size - np.count_nonzero(width))

    vocab, local = g["vocab"], {}
    ea_up, ea_idx, ea_val = _sumstats_alleles(columns["ea"], vocab, local)
    oa_up, oa_idx, oa_val = _sumstats_alleles(columns["oa"], vocab, local)
    ea_comp = _complement_codes(ea_up, vocab, local)
    oa_comp = _complement_codes(oa_up, vocab, local)

    has = cand >= 0
    if multi:
        has[np.fromiter(multi, dtype=np.int64, count=len(multi))] = True
    stage = np.flatnonzero(has)
    e, o = ea_val[ea_idx[stage]], oa_val[oa_idx[stage]]
    ce, co = ea_comp[ea_idx[stage]], oa_comp[oa_idx[stage]]
    n_ambiguous = 0
    if drop_ambiguous:
        amb = (ce >= 0) & (co >= 0) & (ce == o) & (co == e)
        n_ambiguous = int(np.count_nonzero(amb))
        live = ~amb
        stage, e, o, ce, co = (stage[live], e[live], o[live], ce[live],
                               co[live])

    reach = [cand[stage[cand[stage] >= 0]]]
    reach += [np.asarray(multi[k], dtype=np.int64)
              for k in stage.tolist() if k in multi] if multi else []
    hits = np.bincount(np.concatenate(reach), minlength=len(variants))
    single = cand[stage] >= 0
    solo = single & (hits[np.where(single, cand[stage], 0)] == 1)

    beta, se, n_eff = columns["beta"], columns["se"], columns["n_eff"]
    finite = np.isfinite(beta) & np.isfinite(se)

    k_solo = stage[solo]
    gi = cand[k_solo]
    g1, g2 = g["a1"][gi], g["a2"][gi]
    e_, o_, ce_, co_ = e[solo], o[solo], ce[solo], co[solo]
    same = (e_ == g1) & (o_ == g2)
    swap = ~same & (e_ == g2) & (o_ == g1)
    strand_same = ~same & ~swap & (ce_ == g1) & (co_ == g2)
    strand_swap = ~same & ~swap & ~strand_same & (ce_ == g2) & (co_ == g1)
    compatible = same | swap | strand_same | strand_swap
    flip = swap | strand_swap
    n_mismatch = int(k_solo.size - np.count_nonzero(compatible))
    n_strand = int(np.count_nonzero(strand_same | strand_swap))
    fin = finite[k_solo]
    n_nonfinite = int(np.count_nonzero(compatible & ~fin))
    kept = compatible & fin
    n_flip = int(np.count_nonzero(kept & flip))
    idx_parts, src_parts, flip_parts = [gi[kept]], [k_solo[kept]], [flip[kept]]

    loop_rows = stage[~solo].tolist()
    if loop_rows:
        vocab_rows = list(vocab)
        ga1, ga2 = g["a1"], g["a2"]
        idx, srcs, flips = [], [], []
        n_duplicate = 0
        seen = set()
        for k in loop_rows:
            candidates = multi.get(k) or [int(cand[k])]
            ea, oa = ea_up[ea_idx[k]], oa_up[oa_idx[k]]
            gi_hit = None
            n_seen_here = 0
            for gi_k in candidates:
                if gi_k in seen:
                    n_seen_here += 1
                    continue
                f, strand = _allele_orientation(
                    ea, oa, vocab_rows[ga1[gi_k]], vocab_rows[ga2[gi_k]])
                if f is None:
                    continue
                if strand:
                    n_strand += 1
                gi_hit = gi_k
                break
            if gi_hit is None:
                if n_seen_here == len(candidates):
                    n_duplicate += 1
                else:
                    n_mismatch += 1
                continue
            if not finite[k]:
                n_nonfinite += 1
                continue
            if f:
                n_flip += 1
            seen.add(gi_hit)
            idx.append(gi_hit); srcs.append(k); flips.append(f)
        idx_parts.append(np.array(idx, dtype=np.int64))
        src_parts.append(np.array(srcs, dtype=np.int64))
        flip_parts.append(np.array(flips, dtype=bool))
    else:
        n_duplicate = 0

    idx = np.concatenate(idx_parts)
    src = np.concatenate(src_parts)
    flipped = np.concatenate(flip_parts)
    order = np.argsort(idx)
    idx, src, flipped = idx[order], src[order], flipped[order]
    # Negate after widening: exact, and no wrap-around for integer effects.
    out_beta = beta[src].astype(float)
    np.negative(out_beta, out=out_beta, where=flipped)
    if beta.dtype.kind in "iu":
        out_beta += 0.0
    log = {
        "n_sumstats": n,
        "n_genotype_variants": len(variants),
        "n_matched": len(idx),
        "n_flipped": n_flip,
        "n_strand_flipped": n_strand,
        "n_dropped_ambiguous": n_ambiguous,
        "n_dropped_mismatch": n_mismatch,
        "n_dropped_ambiguous_id": n_ambiguous_id,
        "n_dropped_nonfinite": n_nonfinite,
        "n_dropped_duplicate": n_duplicate,
        "n_unmatched": n_unmatched,
    }
    return Harmonized(var_index=idx, beta=out_beta, se=se[src].astype(float),
                      n_eff=n_eff[src].astype(float), flipped=flipped,
                      src_index=src, log=log)


def harmonize(sumstats, variants, *, drop_ambiguous=True, status=None):
    """Align ``sumstats`` to ``variants``; see the module docstring for the rules.

    ``sumstats`` is any object with ``id, chrom, pos, ea, oa, beta, se,
    n_eff`` arrays (a :class:`~allelign.sumstats.Sumstats`, or ldpred3's).
    ``variants`` has ``id, chrom, pos, a1, a2`` (and optionally
    ``alternate_id``). Frozen tables (:meth:`Variants.freeze`) cache their
    indices across calls.

    ``status`` is an optional ``int8`` array of ``len(sumstats)`` codes
    (:mod:`allelign.status`): every dropped row is stamped with its reason
    and matched rows stay 0. Such calls take the row loop -- the same rule
    set and the same result, at O(rows) Python.
    """
    entry = _variant_indices(variants)
    if status is not None:
        status = _st.check_status(status, len(sumstats))
        return _harmonize_rows(sumstats, variants, entry, drop_ambiguous,
                               status)
    columns = _vector_columns(sumstats, variants, entry)
    if columns is not None:
        out = _harmonize_vector(sumstats, variants, entry, columns,
                                drop_ambiguous)
        if out is not None:
            return out
    return _harmonize_rows(sumstats, variants, entry, drop_ambiguous)


def _harmonize_rows(sumstats, variants, entry, drop_ambiguous, status=None):
    """The reference row-by-row implementation of :func:`harmonize`."""

    def _drop(k, code):
        if status is not None:
            status[k] = code

    by_id = entry[1]
    by_pos = entry[2]
    _memo = entry[4]
    _upper = entry[3]

    s_id, s_chrom, s_pos = (_rows(sumstats.id), _rows(sumstats.chrom),
                            _rows(sumstats.pos))
    s_ea, s_oa = _rows(sumstats.ea), _rows(sumstats.oa)
    s_beta, s_se, s_n = (_rows(sumstats.beta), _rows(sumstats.se),
                         _rows(sumstats.n_eff))
    v_a1, v_a2 = variants.a1, variants.a2
    by_id_get = by_id.get
    isfinite = math.isfinite

    idx, betas, ses, ns, flips, srcs = [], [], [], [], [], []
    n_unmatched = n_ambiguous = n_mismatch = n_flip = n_strand = 0
    n_duplicate = n_ambiguous_id = n_nonfinite = 0
    seen = set()
    query_memo = {}
    ambiguous_memo = {}

    for k in range(len(sumstats)):
        sid = s_id[k]
        candidates = by_id_get(sid) if sid else None
        if candidates:
            n_id = len(candidates)
            candidates = _coordinate_candidates(
                candidates, variants, s_chrom[k], s_pos[k], _memo,
                query_memo)
            if not candidates:
                if n_id > 1:
                    n_ambiguous_id += 1
                    _drop(k, _st.AMBIGUOUS_ID)
                else:
                    n_unmatched += 1
                    _drop(k, _st.UNMATCHED)
                continue
        if not candidates:
            if by_pos is None:
                by_pos = _build_pos_index(variants, _memo)
                entry[2] = by_pos
            s_pos_k = int(s_pos[k])
            if s_pos_k <= 0:
                candidates = None
            else:
                c = s_chrom[k]
                nc = _memo.get(c)
                if nc is None:
                    nc = _memo[c] = _norm_chrom(c)
                candidates = by_pos.get((nc, s_pos_k))
        if not candidates:
            n_unmatched += 1
            _drop(k, _st.UNMATCHED)
            continue

        ea = _upper.get(s_ea[k])
        if ea is None:
            ea = _upper[s_ea[k]] = str(s_ea[k]).upper()
        oa = _upper.get(s_oa[k])
        if oa is None:
            oa = _upper[s_oa[k]] = str(s_oa[k]).upper()
        if drop_ambiguous:
            pair = (ea, oa)
            ambiguous = ambiguous_memo.get(pair)
            if ambiguous is None:
                ambiguous = ambiguous_memo[pair] = _ambiguous_pair(ea, oa)
            if ambiguous:
                n_ambiguous += 1
                _drop(k, _st.AMBIGUOUS_ALLELES)
                continue

        gi_hit = None
        flip = None
        n_seen_here = 0
        for gi in candidates:
            if gi in seen:
                n_seen_here += 1
                continue
            g1 = _upper.get(v_a1[gi])
            if g1 is None:
                g1 = _upper[v_a1[gi]] = str(v_a1[gi]).upper()
            g2 = _upper.get(v_a2[gi])
            if g2 is None:
                g2 = _upper[v_a2[gi]] = str(v_a2[gi]).upper()
            flip, strand = _allele_orientation(ea, oa, g1, g2)
            if flip is None:
                continue
            if strand:
                n_strand += 1
            gi_hit = gi
            break

        if gi_hit is None:
            if n_seen_here == len(candidates):
                n_duplicate += 1
                _drop(k, _st.DUPLICATE_MATCH)
            else:
                n_mismatch += 1
                _drop(k, _st.ALLELE_MISMATCH)
            continue

        beta = s_beta[k]
        if not isfinite(beta) or not isfinite(s_se[k]):
            n_nonfinite += 1
            _drop(k, _st.NONFINITE_BETA_SE)
            continue
        if flip:
            beta = -beta
            n_flip += 1

        seen.add(gi_hit)
        idx.append(gi_hit); betas.append(beta); ses.append(s_se[k])
        ns.append(s_n[k]); flips.append(flip); srcs.append(k)

    order = np.argsort(idx)
    idx = np.array(idx, dtype=np.int64)[order]
    log = {
        "n_sumstats": len(sumstats),
        "n_genotype_variants": len(variants),
        "n_matched": len(idx),
        "n_flipped": n_flip,
        "n_strand_flipped": n_strand,
        "n_dropped_ambiguous": n_ambiguous,
        "n_dropped_mismatch": n_mismatch,
        "n_dropped_ambiguous_id": n_ambiguous_id,
        "n_dropped_nonfinite": n_nonfinite,
        "n_dropped_duplicate": n_duplicate,
        "n_unmatched": n_unmatched,
    }
    return Harmonized(
        var_index=idx,
        beta=np.array(betas, dtype=float)[order],
        se=np.array(ses, dtype=float)[order],
        n_eff=np.array(ns, dtype=float)[order],
        flipped=np.array(flips, dtype=bool)[order],
        src_index=np.array(srcs, dtype=np.int64)[order],
        log=log,
    )


# ---------------------------------------------------------------------------
# Coverage and match diagnostics
# ---------------------------------------------------------------------------

#: Flag an alignment covering less than this fraction of the reference: LDSC's
#: M, the h2 denominator and the LD screen's windows are defined on the full
#: reference, so sparse coverage changes what those numbers mean.
REFERENCE_COVERAGE_WARN = 0.5
#: Below this the analysis is no longer genome-wide on that reference.
SEVERE_REFERENCE_COVERAGE = 0.1
#: Call it a build/coordinate mismatch when at least this fraction of the
#: unmatched rows carry an identifier the reference holds elsewhere.
BUILD_MISMATCH_FRACTION = 0.2


class _ReferenceLoci:
    """Loci carrying each reference identifier, over the cached index."""

    __slots__ = ("_by_id", "_chrom", "_pos", "_memo")

    def __init__(self, variants):
        self._by_id = _variant_indices(variants)[1]
        self._chrom = variants.chrom
        self._pos = variants.pos
        self._memo = {}

    def __len__(self):
        return len(self._by_id)

    def _norm(self, raw):
        norm = self._memo.get(raw)
        if norm is None:
            norm = self._memo[raw] = _norm_chrom(raw)
        return norm

    def loci(self, key):
        """Distinct normalised ``(chrom, pos)`` pairs for ``key``, or ``None``."""
        rows = self._by_id.get(key)
        if not rows:
            return None
        return {(self._norm(self._chrom[gi]), int(self._pos[gi]))
                for gi in rows}


def _diagnose_unmatched(sumstats, variants, harmonized):
    """Split unmatched rows into build-mismatch and absent-variant classes."""
    n_rows = len(sumstats.id)
    matched_rows = np.zeros(n_rows, dtype=bool)
    if harmonized.src_index is not None and len(harmonized):
        matched_rows[np.asarray(harmonized.src_index, dtype=np.int64)] = True
    unmatched = np.where(~matched_rows)[0]
    report = {"n_unmatched_rows": int(unmatched.size),
              "n_unmatched_id_elsewhere": 0,
              "n_unmatched_id_absent": 0,
              "n_unmatched_id_missing": 0,
              "examples": []}
    if unmatched.size == 0:
        return report
    index = _ReferenceLoci(variants)
    ss_id, ss_chrom, ss_pos = sumstats.id, sumstats.chrom, sumstats.pos
    memo = {}
    for k in unmatched:
        key = ss_id[k]
        if not key:
            report["n_unmatched_id_missing"] += 1
            continue
        loci = index.loci(key)
        if loci is None:
            report["n_unmatched_id_absent"] += 1
            continue
        raw = ss_chrom[k]
        norm = memo.get(raw)
        if norm is None:
            norm = memo[raw] = _norm_chrom(raw)
        try:
            here = int(ss_pos[k])
        except (TypeError, ValueError):
            here = 0
        if (norm, here) in loci:
            continue
        report["n_unmatched_id_elsewhere"] += 1
        if len(report["examples"]) < 3:
            ref_chrom, ref_pos = sorted(loci)[0]
            report["examples"].append(
                f"{key} GWAS {norm}:{here} vs reference "
                f"{ref_chrom}:{ref_pos}")
    return report


def reference_coverage(sumstats, variants, harmonized, *, n_reference=None):
    """How much of the LD reference the aligned rows cover, and why not more.

    Zero overlap is an error a caller catches elsewhere. What this reports is
    the case that still produces numbers: a fit running on a small,
    unrepresentative corner of the reference. Returns a dict with the two
    fractions and a ``reason`` from a fixed vocabulary --
    ``adequate_coverage``, ``no_rows_offered``, ``build_mismatch``,
    ``absent_identifiers``, ``rows_rejected_at_matched_locus``,
    ``sparse_sumstats`` -- plus the unmatched-row breakdown when it looked.

    Preparation uses the same report and adds validity, QC and re-anchoring
    counts, so CLI and web runs diagnose a thin alignment consistently.
    """
    return _reference_coverage(sumstats, variants, harmonized,
                               n_reference=n_reference)


def _reference_coverage(sumstats, variants, harmonized, *, n_reference=None,
                        n_matched=None, n_offered=None, n_invalid=0,
                        n_qc_dropped=0):
    """Shared coverage classification, including preparation's row filters."""
    n_rows = int(len(sumstats.id))
    n_cache = int(len(variants.id) if n_reference is None else n_reference)
    n_matched = int(len(harmonized) if n_matched is None else n_matched)
    n_offered = int(n_rows if n_offered is None else n_offered)
    out = {
        "n_sumstats_offered": n_offered,
        "n_cache": n_cache,
        "n_matched": n_matched,
        "frac_of_reference": (n_matched / n_cache) if n_cache else 0.0,
        "frac_of_sumstats": (n_matched / n_offered) if n_offered else 0.0,
        "diagnosed": False,
    }
    if out["frac_of_reference"] >= REFERENCE_COVERAGE_WARN:
        out["reason"] = "adequate_coverage"
        return out
    if n_rows == 0:
        # Nothing reached harmonization, so the reference says nothing about
        # why. Naming the build here would be a guess.
        out["reason"] = "no_rows_offered"
        return out
    diagnosis = _diagnose_unmatched(sumstats, variants, harmonized)
    out.update(diagnosis, diagnosed=True)
    n_unmatched = diagnosis["n_unmatched_rows"]
    elsewhere = diagnosis["n_unmatched_id_elsewhere"]
    absent = diagnosis["n_unmatched_id_absent"]
    build_mismatch = bool(
        n_unmatched and elsewhere / n_unmatched >= BUILD_MISMATCH_FRACTION)
    out["build_mismatch_suspected"] = build_mismatch
    out["severe"] = out["frac_of_reference"] < SEVERE_REFERENCE_COVERAGE
    out["reason"] = (
        "rows_failed_validity" if n_invalid and n_invalid >= max(n_matched, 1) else
        "qc_removed_rows" if n_qc_dropped and n_qc_dropped >= max(n_matched, 1) else
        "build_mismatch" if build_mismatch else
        "absent_identifiers" if absent and absent >= elsewhere else
        "rows_rejected_at_matched_locus" if n_unmatched else
        "sparse_sumstats")
    return out


def coverage_message(coverage, *, label="sumstats", n_invalid=0,
                     n_harmonized=0, n_qc_dropped=0):
    """One paragraph naming the coverage, the cause, and the remedy.

    Kept beside the thresholds it describes; the pipeline's alignment
    warnings use it.
    """
    reason = coverage.get("reason")
    if reason in (None, "adequate_coverage"):
        return None
    if reason == "reanchor_dropped_all":
        return (
            f"{label}: identifier re-anchoring dropped every one of "
            f"{coverage['n_sumstats_offered']:,} rows against the LD reference "
            f"({coverage['n_dropped_absent_identifier']:,} identifiers the reference "
            f"does not hold, {coverage['n_dropped_ambiguous_locus']:,} held at "
            "several loci). This is not a QC filter failure: the reference "
            "was consulted and none of the offered identifiers could be "
            "re-stamped onto it.")
    if reason == "no_rows_offered":
        return (f"{label}: QC removed every row before alignment to the LD "
                "reference, so no variant could match. Check the QC filters "
                "against the file's columns -- a missing or unparsed "
                "sample-size, allele-frequency, or standard-error column "
                "removes every row -- before suspecting the reference.")
    lines = [
        f"{label}: only {coverage['n_matched']:,} of {coverage['n_cache']:,} "
        f"LD-reference variants are covered "
        f"({coverage['frac_of_reference']:.1%}), from "
        f"{coverage['n_sumstats_offered']:,} rows offered to harmonisation."]
    if coverage.get("severe"):
        lines.append(
            "A fit on this few reference variants is not a genome-wide "
            "analysis of that reference: the LD blocks are nearly empty, the "
            "LD-consistency screen leaves most windows below its minimum "
            "size, and LDSC keeps the full reference M, so h2 and "
            "polygenicity are on a denominator the data do not populate.")
    else:
        lines.append(
            "LDSC keeps the full reference M and the LD blocks stay sparse, "
            "so h2, polygenicity, and screen coverage are all affected.")
    if reason == "build_mismatch" or coverage.get("build_mismatch_suspected"):
        lines.append(
            f"{coverage['n_unmatched_id_elsewhere']:,} of the "
            f"{coverage['n_unmatched_rows']:,} unmatched rows carry an identifier "
            "the reference holds at coordinates that disagree. That is the "
            "signature of a genome-build mismatch (GRCh38 summary statistics "
            "against a GRCh37 reference, or the reverse) or of CHR/BP read "
            "from the wrong columns. Check the column mapping; if the builds "
            "differ, lift the summary statistics over to the reference's build "
            "and keep the chain file and failed mappings with the results, "
            "or use a reference on the GWAS build. Do not drop coordinates "
            "to force an identifier-only match.")
        if coverage.get("examples"):
            lines.append("Examples: " + "; ".join(coverage["examples"]) + ".")
    elif reason == "absent_identifiers":
        lines.append(
            f"{coverage['n_unmatched_id_absent']:,} unmatched rows carry an "
            "identifier the reference does not hold at all, which points at a "
            "different variant set or ancestry panel rather than at "
            "coordinates.")
    elif reason == "rows_rejected_at_matched_locus":
        lines.append(
            "The unmatched rows are present in the reference at the same "
            "locus, so they were rejected on alleles, duplication, or a "
            "non-finite effect; check the effect/other allele columns.")
    elif reason == "rows_failed_validity":
        lines.append(
            f"{n_invalid:,} of the {n_harmonized:,} harmonized rows had a "
            "non-finite or non-positive standard error or sample size and "
            "were dropped after alignment. Check the SE and N columns -- a "
            "mis-mapped or all-zero standard-error column is a common cause "
            "-- before suspecting the deposition.")
    elif reason == "qc_removed_rows":
        lines.append(
            f"Sumstats QC removed {n_qc_dropped:,} rows before alignment, so "
            "only the survivors were offered to the reference. Relax the QC "
            "filters (min_n_ratio, min_maf, min_info, max_chisq) if they are "
            "tighter than intended, and check they point at the right columns.")
    else:
        lines.append(
            "Every offered row matched, so the GWAS itself covers only this "
            "part of the reference -- a significant-hits-only or "
            "array-restricted deposition rather than a misalignment. "
            "Genome-wide estimands cannot be read off such a subset.")
    return " ".join(lines)


def diagnose_match(sumstats, variants, *, sample=2000):
    """Explain *why* few or no variants matched (build, chr naming, alleles).

    Returns a dict with overlap counts and a human-readable ``message``.
    """
    gid = {v for v in variants.id if v and v != "."}
    sid = {s for s in sumstats.id if s}
    rsid_overlap = len(sid & gid)

    gpos = {(_norm_chrom(variants.chrom[i]), int(variants.pos[i]))
            for i in range(len(variants)) if int(variants.pos[i]) > 0}
    spos = {(_norm_chrom(sumstats.chrom[k]), int(sumstats.pos[k]))
            for k in range(len(sumstats)) if int(sumstats.pos[k]) > 0}
    pos_overlap = len(spos & gpos)

    def _prefixed(labels):
        return any(str(c)[:3].lower() == "chr" for c in labels[:sample])
    s_pref, g_pref = _prefixed(sumstats.chrom), _prefixed(variants.chrom)

    build_mismatch = False
    if rsid_overlap:
        g_at = {variants.id[i]: (_norm_chrom(variants.chrom[i]),
                                 int(variants.pos[i]))
                for i in range(len(variants)) if variants.id[i]}
        # sorted, not set order: set iteration varies with hash randomisation.
        shared = set(sorted(sid & gid)[:sample])
        disagree = tot = 0
        for k in range(len(sumstats)):
            r = sumstats.id[k]
            if r in g_at and r in shared:
                tot += 1
                if g_at[r] != (_norm_chrom(sumstats.chrom[k]),
                               int(sumstats.pos[k])):
                    disagree += 1
        build_mismatch = tot > 0 and disagree / tot > 0.5

    msgs = []
    if build_mismatch:
        msgs.append("rsIDs overlap but their chrom:pos disagree -- likely a "
                    "genome-build mismatch (e.g. hg19 vs hg38); lift over to a "
                    "common build")
    if rsid_overlap == 0 and pos_overlap == 0:
        if s_pref != g_pref:
            msgs.append("chromosome labels differ in style (one side uses a "
                        "'chr' prefix) -- normalisation handles this, so a "
                        "remaining 0-overlap points to different builds or "
                        "variant sets")
        else:
            msgs.append("no rsID and no chrom:pos overlap -- the sumstats and "
                        "reference look like different variant sets or builds")
    if rsid_overlap and pos_overlap and not build_mismatch:
        msgs.append("variants overlap but harmonisation dropped them -- likely "
                    "an allele/strand or effect-column problem (check ea/oa)")
    return {
        "n_sumstats": len(sumstats),
        "n_genotype_variants": len(variants),
        "rsid_overlap": rsid_overlap,
        "pos_overlap_normalized": pos_overlap,
        "sumstats_chr_prefixed": s_pref,
        "genotype_chr_prefixed": g_pref,
        "build_mismatch_suspected": build_mismatch,
        "message": "; ".join(msgs) if msgs else "no obvious cause detected",
    }


def count_positional_only_matches(sumstats, variants, harmonized):
    """Matched rows whose identifier the reference lacks (position-only match).

    A coordinate can land on a *different* variant that happens to carry a
    compatible allele pair, so these rows cannot be verified by identifier.
    """
    if not len(harmonized) or harmonized.src_index is None:
        return 0
    by_id = _variant_indices(variants)[1]
    ss_id = sumstats.id
    n = 0
    for k in np.asarray(harmonized.src_index, dtype=np.int64):
        key = ss_id[k]
        if key and key not in by_id:
            n += 1
    return n


def reanchor_on_identifier(sumstats, variants, *, label="sumstats",
                           warn=True, return_keep=False,
                           warning_stacklevel=2):
    """Move each row onto the reference's own coordinates for its identifier.

    The coordinate repair for a build mismatch when identifiers are shared and
    trustworthy (a GWAS Catalog harmonised file's ``hm_rsid`` against an
    rsID-keyed reference). It is *not* a chain-file liftover: no interval
    mapping is consulted and no coordinate is invented. A row whose identifier
    the reference holds at exactly one locus is re-stamped with that locus;
    alleles are still checked afterwards by :func:`harmonize`.

    Dropped rather than re-anchored: identifiers the reference does not hold
    (their coordinates are on the other build and could match a *different*
    variant by position), identifiers held at several loci, and rows with no
    identifier. Returns ``(anchored, log)`` -- plus the boolean keep mask over
    the input rows with ``return_keep=True``. Prefer a properly lifted-over
    file where one exists; the log is the audit trail.
    """
    ids = sumstats.id
    n_rows = len(ids)
    index = _ReferenceLoci(variants)
    chrom = np.array(sumstats.chrom, dtype=object)
    pos = np.array(sumstats.pos, dtype=np.int64)
    keep = np.ones(n_rows, dtype=bool)
    memo = {}
    n_anchored = n_moved = n_ambiguous = n_absent = n_missing_id = 0
    for k in range(n_rows):
        key = ids[k]
        if not key:
            n_missing_id += 1
            keep[k] = False
            continue
        loci = index.loci(key)
        if loci is None:
            n_absent += 1
            keep[k] = False
            continue
        if len(loci) > 1:
            n_ambiguous += 1
            keep[k] = False
            continue
        ref_chrom, ref_pos = next(iter(loci))
        raw = chrom[k]
        norm = memo.get(raw)
        if norm is None:
            norm = memo[raw] = _norm_chrom(raw)
        if norm != ref_chrom or int(pos[k]) != ref_pos:
            n_moved += 1
        chrom[k] = ref_chrom
        pos[k] = ref_pos
        n_anchored += 1
    log = {"applied": True, "n_rows": int(n_rows),
           "n_anchored": int(n_anchored), "n_moved": int(n_moved),
           "n_dropped_ambiguous_locus": int(n_ambiguous),
           "n_dropped_absent_identifier": int(n_absent),
           "n_dropped_missing_identifier": int(n_missing_id)}
    anchored = replace(sumstats, chrom=chrom, pos=pos).subset(keep)
    n_dropped = n_absent + n_ambiguous + n_missing_id
    # Warn on drops as well as moves: identifiers retired between builds can
    # lose a large fraction with nothing moved.
    if warn and (n_moved or n_dropped):
        actions = []
        if n_moved:
            actions.append(
                f"re-anchored {n_moved:,} of {n_rows:,} rows onto the LD "
                "reference's coordinates for their identifiers, because the "
                "two sides disagreed on where those variants are")
        if n_dropped:
            actions.append(
                f"dropped {n_dropped:,} of {n_rows:,} rows it could not anchor "
                f"by identifier ({n_absent:,} identifiers the reference does "
                f"not hold, {n_ambiguous:,} held at several loci, "
                f"{n_missing_id:,} with no identifier at all)")
        warnings.warn(
            f"{label}: identifier-keyed re-anchoring: " + "; ".join(actions)
            + ". The fit is therefore on the reference's genome build, keyed "
            "on identifier rather than on position. This is not a chain-file "
            "liftover: record it, and prefer a properly lifted-over input "
            "where one is available.",
            RuntimeWarning, stacklevel=warning_stacklevel)
    return (anchored, log, keep) if return_keep else (anchored, log)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

@dataclass
class HarmonizedWeights:
    """Weights aligned to a reference's ``a1`` alleles.

    ``af`` is the frequency of ``a1`` (``1 - af`` where the alleles swapped);
    ``sd`` is orientation-free. Both are ``None`` when the source had none.
    """

    var_index: np.ndarray
    weight: np.ndarray
    flipped: np.ndarray
    src_index: np.ndarray
    af: np.ndarray = None
    sd: np.ndarray = None
    log: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.var_index)


@dataclass
class _EffectRows:
    id: np.ndarray
    chrom: np.ndarray
    pos: np.ndarray
    ea: np.ndarray
    oa: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    n_eff: np.ndarray

    def __len__(self):
        return len(self.beta)


def harmonize_weights(weights, variants, *, drop_ambiguous=True, status=None):
    """Align a weights table or PGS Catalog score to ``variants``.

    ``weights`` needs ``id, chrom, pos, weight`` and the effect/other alleles
    as ``a1``/``a2`` (:class:`~allelign.weights.WeightsTable`) or
    ``ea``/``oa`` (:class:`~allelign.weights.ScoringFile`); ``af_ref`` and
    ``sd_ref`` are carried along when present. The same rules as
    :func:`harmonize` apply. The log adds ``weight_mass_matched``, the share
    of the total squared weight that matched (over the whole file when the
    score was read with a ``keep=`` prefilter).
    """
    ea = getattr(weights, "a1", None)
    oa = getattr(weights, "a2", None)
    if ea is None or oa is None:
        ea, oa = weights.ea, weights.oa
    w = np.asarray(weights.weight, dtype=float)
    ones = np.ones(w.size)
    rows = _EffectRows(id=np.asarray(weights.id, dtype=object),
                       chrom=np.asarray(weights.chrom, dtype=object),
                       pos=np.asarray(weights.pos, dtype=np.int64),
                       ea=np.asarray(ea, dtype=object),
                       oa=np.asarray(oa, dtype=object),
                       beta=w, se=ones, n_eff=ones)
    h = harmonize(rows, variants, drop_ambiguous=drop_ambiguous,
                  status=status)
    af = getattr(weights, "af_ref", None)
    sd = getattr(weights, "sd_ref", None)
    af_aligned = sd_aligned = None
    if af is not None:
        af = np.asarray(af, dtype=float)[h.src_index]
        af_aligned = np.where(h.flipped, 1.0 - af, af)
    if sd is not None:
        sd_aligned = np.asarray(sd, dtype=float)[h.src_index]
    # A score read with read_scoring_file(keep=...) lacks the rows the
    # prefilter rejected: they count as unmatched, and the weight mass is
    # taken over the whole file.
    parse_log = getattr(weights, "log", None) or {}
    removed = int(parse_log.get("n_reference_prefiltered", 0))
    total = parse_log.get("weight_square_sum_all")
    total = float(w @ w) if total is None else float(total)
    log = dict(h.log)
    if removed:
        log["n_sumstats"] += removed
        log["n_unmatched"] += removed
    log["weight_mass_matched"] = (float(h.beta @ h.beta) / total
                                  if total > 0 else float("nan"))
    return HarmonizedWeights(var_index=h.var_index, weight=h.beta,
                             flipped=h.flipped, src_index=h.src_index,
                             af=af_aligned, sd=sd_aligned, log=log)
