"""ldpred3 reads, aligns and screens through allelign: the seam between them.

Two kinds of check, both skipped when ldpred3 is not installed:

* names ldpred3 re-exports must *be* allelign's objects (an identity, not a
  bit-for-bit comparison of a function with itself). ldpred3's own suite no
  longer re-tests them, so a re-export that turns into a local copy fails here;
* ldpred3's own wrappers -- ``harmonize``/``qc_sumstats``/``sd_consistency_mask``
  with ``reasons=``, the split-half screen on ldpred3's float32 eigensolver and
  BLAS gate, ``read_weights``/``write_pgs_scoring_file`` and re-anchoring --
  must return what allelign returns on the same input, bit for bit.
"""

import importlib
import warnings

import numpy as np
import pytest

pytest.importorskip("ldpred3")
pytestmark = pytest.mark.parity

L_h = importlib.import_module("ldpred3.harmonize")
L_qc = importlib.import_module("ldpred3.qc")
L_w = importlib.import_module("ldpred3.weights")

import allelign.align as A_h                       # noqa: E402
import allelign.qc as A_qc                         # noqa: E402
import allelign.status as A_st                     # noqa: E402
import allelign.sumstats as A_ss                   # noqa: E402
import allelign.weights as A_w                     # noqa: E402
from allelign.ldcheck import ld_consistency_screen  # noqa: E402
from allelign.variants import Variants             # noqa: E402


REEXPORTS = [  # (ldpred3 module, its name, allelign module, allelign name)
    ("sumstats", "read_sumstats", "sumstats", "read_sumstats"),
    ("sumstats", "detect_columns", "sumstats", "detect_columns"),
    ("sumstats", "rescale_n_eff", "sumstats", "rescale_n_eff"),
    ("sumstats", "Sumstats", "sumstats", "Sumstats"),
    ("sumstats", "COLUMN_ALIASES", "sumstats", "COLUMN_ALIASES"),
    ("sumstats", "_ALIASES", "sumstats", "COLUMN_ALIASES"),
    ("sumstats", "_SUMSTATS_FIELDS", "sumstats", "_SUMSTATS_FIELDS"),
    ("sumstats", "ColumnMappingError", "sumstats", "ColumnMappingError"),
    ("sumstats", "_rescale_or_constant_n", "sumstats", "anchor_n_eff"),
    ("prepare", "n_eff_case_control", "sumstats", "n_eff_case_control"),
    ("harmonize", "Harmonized", "align", "Harmonized"),
    ("harmonize", "diagnose_match", "align", "diagnose_match"),
    ("harmonize", "reference_coverage", "align", "reference_coverage"),
    ("harmonize", "_reference_coverage", "align", "_reference_coverage"),
    ("harmonize", "coverage_message", "align", "coverage_message"),
    ("harmonize", "REFERENCE_COVERAGE_WARN", "align", "REFERENCE_COVERAGE_WARN"),
    ("harmonize", "SEVERE_REFERENCE_COVERAGE", "align", "SEVERE_REFERENCE_COVERAGE"),
    ("harmonize", "BUILD_MISMATCH_FRACTION", "align", "BUILD_MISMATCH_FRACTION"),
    ("harmonize", "variant_indices", "align", "_variant_indices"),
    ("harmonize", "_variant_indices", "align", "_variant_indices"),
    ("harmonize", "_vector_columns", "align", "_vector_columns"),
    ("harmonize", "_harmonize_vector", "align", "_harmonize_vector"),
    ("harmonize", "_harmonize_rows", "align", "_harmonize_rows"),
    ("harmonize", "_ReferenceLoci", "align", "_ReferenceLoci"),
    ("harmonize", "_CACHE_ATTR", "align", "_CACHE_ATTR"),
    ("harmonize", "_INDEX_CACHE_LOCK", "align", "_INDEX_CACHE_LOCK"),
    ("harmonize", "_norm_chrom", "alleles", "normalize_chrom"),
    ("harmonize", "_ambiguous_pair", "alleles", "is_ambiguous"),
    ("harmonize", "_allele_orientation", "alleles", "allele_orientation"),
    ("harmonize", "_complement", "alleles", "reverse_complement"),
    ("qc", "dentist_statistic", "ldcheck", "dentist_statistic"),
    ("qc", "_window_ld", "ldcheck", "_window_ld"),
    ("qc", "_precision_apply", "ldcheck", "_precision_apply"),
    ("qc", "_SplitHalfWindows", "ldcheck", "_SplitHalfWindows"),
    ("qc", "_split_half_tiles", "ldcheck", "_split_half_tiles"),
    ("qc", "af_concordance_mask", "qc", "af_concordance_mask"),
    ("qc", "gwas_af_on_counted_allele", "qc", "gwas_af_on_counted_allele"),
    ("qc", "genomic_control_factor", "qc", "genomic_control_factor"),
    ("qc", "implied_sample_size", "qc", "implied_sample_size"),
    ("qc", "impute_n_eff", "qc", "impute_n_eff"),
    ("qc", "should_impute_n", "qc", "should_impute_n"),
    ("weights", "write_weights", "weights", "write_weights"),
    ("weights", "WeightsTable", "weights", "WeightsTable"),
    ("weights", "PGS_BUILDS", "weights", "PGS_BUILDS"),
    ("weights", "SD_SOURCES", "weights", "SD_SOURCES"),
    ("_qc_status", "QC_REASON_NAMES", "status", "QC_REASON_NAMES"),
    ("_qc_status", "QC_OUT_HEADER", "status", "STATUS_HEADER"),
    ("_qc_status", "qc_status_counts", "status", "status_counts"),
    ("_qc_status", "write_qc_status_table", "status", "write_status_table"),
]


@pytest.mark.parametrize("ldpred3_module,name,allelign_module,allelign_name",
                         REEXPORTS, ids=[f"{m}.{n}" for m, n, _, _ in REEXPORTS])
def test_ldpred3_reexports_are_allelign_objects(ldpred3_module, name,
                                                allelign_module, allelign_name):
    ours = getattr(importlib.import_module(f"allelign.{allelign_module}"),
                   allelign_name)
    theirs = getattr(importlib.import_module(f"ldpred3.{ldpred3_module}"), name)
    assert theirs is ours


def _same_array(x, y, name=""):
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == y.shape, name
    if x.dtype.kind == "f" or y.dtype.kind == "f":
        assert x.dtype == y.dtype, name
        np.testing.assert_array_equal(x.view(np.int64), y.view(np.int64),
                                      err_msg=name)
    else:
        assert list(x) == list(y), name


def _random_grid(rng, nv=300, ns=500):
    alleles = ["A", "C", "G", "T", "AT", "AA", "TT", "AC"]
    comp = str.maketrans("ACGT", "TGCA")
    chrom = rng.choice(["1", "chr1", "2", "01", "X", "23"], nv)
    pos = rng.integers(0, 60, nv) * 10
    ids = np.array([f"rs{rng.integers(0, 200)}" if rng.random() < 0.8 else "."
                    for _ in range(nv)], dtype=object)
    a1 = rng.choice(alleles, nv)
    a2 = rng.choice(alleles, nv)
    variants = dict(chrom=np.asarray(chrom, dtype=object), id=ids,
                    pos=pos.astype(np.int64), a1=np.asarray(a1, dtype=object),
                    a2=np.asarray(a2, dtype=object))
    s_id = np.array([f"rs{rng.integers(0, 260)}" if rng.random() < 0.7 else ""
                     for _ in range(ns)], dtype=object)
    s_chrom = np.asarray(rng.choice(["1", "chr1", "2", "X", ""], ns), dtype=object)
    s_pos = (rng.integers(0, 60, ns) * 10).astype(np.int64)
    ea = np.asarray(rng.choice(alleles + ["a", "g"], ns), dtype=object)
    oa = np.asarray(rng.choice(alleles + ["t"], ns), dtype=object)
    for k in range(ns):
        if rng.random() < 0.6:
            gi = int(rng.integers(nv))
            e, o = a1[gi], a2[gi]
            if rng.random() < 0.5:
                e, o = o, e
            if rng.random() < 0.3:
                e, o = e.translate(comp)[::-1], o.translate(comp)[::-1]
            ea[k], oa[k] = e, o
            if rng.random() < 0.6:
                s_id[k] = ids[gi] if ids[gi] != "." else ""
            s_chrom[k] = chrom[gi]
            s_pos[k] = pos[gi]
    beta = rng.choice([0.1, -0.2, 0.0, np.nan], ns)
    se = rng.choice([0.01, 0.5, np.nan], ns)
    ss = A_ss.Sumstats(id=s_id, chrom=s_chrom, pos=s_pos, ea=ea, oa=oa,
                       beta=beta, se=se, n_eff=np.full(ns, 1000.0),
                       eaf=np.full(ns, np.nan), info=np.full(ns, np.nan))
    return ss, variants


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("drop", [True, False])
def test_harmonize_matches_ldpred3(seed, drop):
    ss, cols = _random_grid(np.random.default_rng(seed))
    va, vb = Variants(**cols), Variants(**{k: v.copy() for k, v in cols.items()})
    a = A_h.harmonize(ss, va, drop_ambiguous=drop)
    b = L_h.harmonize(ss, vb, drop_ambiguous=drop)
    for f in ("var_index", "beta", "se", "n_eff", "flipped", "src_index"):
        _same_array(getattr(a, f), getattr(b, f), f)
    assert a.log == b.log
    sa = A_st.new_status(len(ss))
    rb = np.zeros(len(ss), dtype=np.int8)
    A_h.harmonize(ss, Variants(**cols), drop_ambiguous=drop, status=sa)
    L_h.harmonize(ss, Variants(**cols), drop_ambiguous=drop, reasons=rb)
    np.testing.assert_array_equal(sa, rb)


def _random_sumstats(rng, n=3000):
    ids = np.array([f"rs{rng.integers(0, n)}" if rng.random() > 0.02 else "."
                    for _ in range(n)], dtype=object)
    beta = rng.normal(0, 0.02, n)
    se = np.abs(rng.normal(0.01, 0.003, n))
    n_eff = rng.choice([50_000.0, 30_000.0, 0.0, np.nan, 10_000.0], n,
                       p=[0.7, 0.15, 0.02, 0.03, 0.1])
    eaf = rng.uniform(0, 1, n)
    eaf[rng.random(n) < 0.05] = np.nan
    info = rng.uniform(0.4, 1, n)
    info[rng.random(n) < 0.3] = np.nan
    beta[rng.random(n) < 0.01] = np.nan
    return A_ss.Sumstats(
        id=ids, chrom=np.asarray(rng.choice(["1", "2", "X"], n), dtype=object),
        pos=rng.integers(1, 10 ** 6, n).astype(np.int64),
        ea=np.asarray(rng.choice(list("ACGT"), n), dtype=object),
        oa=np.asarray(rng.choice(list("ACGT"), n), dtype=object),
        beta=beta, se=se, n_eff=n_eff, eaf=eaf, info=info)


@pytest.mark.parametrize("anchor", ["median", "max"])
def test_qc_sumstats_matches_ldpred3(anchor):
    ss = _random_sumstats(np.random.default_rng(1))
    kw = dict(min_n_anchor=anchor, max_chisq=25.0, min_maf=0.02, min_info=0.6)
    ka, la = A_qc.qc_sumstats(ss, **kw)
    kb, lb = L_qc.qc_sumstats(ss, **kw)
    np.testing.assert_array_equal(ka, kb)
    assert la == lb
    sa = A_st.new_status(len(ss))
    rb = np.zeros(len(ss), dtype=np.int8)
    A_qc.qc_sumstats(ss, status=sa, **kw)
    L_qc.qc_sumstats(ss, reasons=rb, **kw)
    np.testing.assert_array_equal(sa, rb)


@pytest.mark.parametrize("rule", ["ratio", "prive"])
def test_sd_consistency_mask_matches_ldpred3(rule):
    rng = np.random.default_rng(2)
    m = 4000
    af = rng.uniform(0.01, 0.99, m)
    n = rng.choice([1e5, 5e4, np.nan], m, p=[0.8, 0.15, 0.05])
    sd = np.sqrt(2 * af * (1 - af))
    se = 1 / (sd * np.sqrt(1e5)) * rng.lognormal(0, 0.2, m)
    beta = rng.normal(0, 0.01, m)
    chrom = rng.choice(["1", "chr2", "X", "23"], m)
    info = rng.uniform(0.5, 1, m)
    for kwargs in ({}, {"info": info}, {"chrom": chrom}):
        a = A_qc.sd_consistency_mask(beta, se, n, af, rule=rule, **kwargs)
        b = L_qc.sd_consistency_mask(beta, se, n, af, rule=rule, **kwargs)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1]
        sa = A_st.new_status(m)
        rb = np.zeros(m, dtype=np.int8)
        A_qc.sd_consistency_mask(beta, se, n, af, rule=rule, status=sa, **kwargs)
        L_qc.sd_consistency_mask(beta, se, n, af, rule=rule, reasons=rb, **kwargs)
        np.testing.assert_array_equal(sa, rb)


def _ar1(m, rho):
    idx = np.arange(m)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _screen_inputs(seed):
    rng = np.random.default_rng(seed)
    blocks, zs = [], []
    start = 0
    for size, rho in ((400, 0.95), (250, 0.8), (40, 0.9)):
        R = _ar1(size, rho).astype(np.float32)
        L = np.linalg.cholesky(R.astype(np.float64) + 1e-9 * np.eye(size))
        beta = np.zeros(size)
        beta[rng.choice(size, 5, replace=False)] = rng.normal(0, 6, 5)
        z = R.astype(np.float64) @ beta + L @ rng.standard_normal(size)
        flip = rng.choice(size, 3, replace=False)
        z[flip] = -z[flip] - 8.0 * np.sign(z[flip] + 1e-9)
        blocks.append((R, np.arange(start, start + size)))
        zs.append(z)
        start += size
    return blocks, np.concatenate(zs)


@pytest.mark.parametrize("seed", [0, 1])
def test_ld_consistency_screen_matches_ldpred3(seed):
    blocks, z = _screen_inputs(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = ld_consistency_screen(blocks, z, seed=seed)
        b = L_qc.ld_consistency_screen(blocks, z, seed=seed)
        np.testing.assert_array_equal(a, b)
        assert (~a).sum() > 0
        q8 = [(np.round(R * 127).astype(np.int8), idx) for R, idx in blocks]
        np.testing.assert_array_equal(ld_consistency_screen(q8, z, seed=seed),
                                      L_qc.ld_consistency_screen(q8, z, seed=seed))
        zz = np.column_stack([z, -z[::-1]])
        np.testing.assert_array_equal(ld_consistency_screen(blocks, zz),
                                      L_qc.ld_consistency_screen(blocks, zz))


def test_weights_and_pgs_export_match_ldpred3(tmp_path):
    rng = np.random.default_rng(5)
    n = 50
    kw = dict(id=[f"rs{i}" for i in range(n)],
              chrom=list(rng.choice(["1", "23", "25", "chr2"], n)),
              pos=rng.integers(1, 10 ** 6, n), effect_allele=["A"] * n,
              other_allele=["G"] * n, weight=rng.normal(0, 0.01, n),
              af=rng.uniform(0.05, 0.95, n), sd=rng.uniform(0.2, 0.7, n),
              sd_source="hwe")
    pa = tmp_path / "a.tsv"
    A_w.write_weights(pa, **kw)
    wa, wb = A_w.read_weights(pa), L_w.read_weights(pa)
    for f in ("id", "chrom", "pos", "a1", "a2", "weight", "af_ref", "sd_ref"):
        _same_array(getattr(wa, f), getattr(wb, f), f)
    assert wa.sd_source == wb.sd_source == "hwe"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        A_w.write_pgs_scoring_file(tmp_path / "a.pgs", wa, genome_build="GRCh37")
        L_w.write_pgs_scoring_file(tmp_path / "b.pgs", wb, genome_build="GRCh37")

    def body(path):
        return [line for line in path.read_text().splitlines()
                if not line.startswith("#")]
    assert body(tmp_path / "a.pgs") == body(tmp_path / "b.pgs")


def test_selected_and_pooled_screens_match_ldpred3():
    blocks, z = _screen_inputs(3)
    rng = np.random.default_rng(3)
    selection = np.sort(rng.choice(z.size, z.size - 90, replace=False))
    from allelign.ldcheck import ld_consistency_screen_selected
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sa, sb = {}, {}
        a = ld_consistency_screen_selected(blocks, selection, z[selection],
                                           seed=4, stats_out=sa)
        b = L_qc._ld_consistency_screen_selected(blocks, selection,
                                                 z[selection], seed=4,
                                                 stats_out=sb)
        np.testing.assert_array_equal(a, b)
        assert sa == sb
        np.testing.assert_array_equal(
            ld_consistency_screen(blocks, z, seed=1, ncores=2),
            L_qc.ld_consistency_screen(blocks, z, seed=1))


def test_reanchoring_matches_ldpred3():
    L_prep = importlib.import_module("ldpred3.prepare")
    n = 300
    ids = np.array([f"rs{i}" for i in range(n)], dtype=object)
    cols = dict(chrom=np.full(n, "1", dtype=object), id=ids,
                pos=(np.arange(n) + 1) * 10, a1=np.full(n, "A", dtype=object),
                a2=np.full(n, "G", dtype=object))
    ss = A_ss.Sumstats(id=ids.copy(), chrom=np.full(n, "1", dtype=object),
                       pos=(np.arange(n) + 1) * 10 + 3,
                       ea=np.full(n, "A", dtype=object),
                       oa=np.full(n, "G", dtype=object),
                       beta=np.full(n, 0.1), se=np.full(n, 0.01),
                       n_eff=np.full(n, 1e4), eaf=np.full(n, np.nan),
                       info=np.full(n, np.nan))
    ss.id[:5] = ""
    va, vb = Variants(**cols), Variants(**{k: v.copy() for k, v in cols.items()})
    with warnings.catch_warnings(record=True) as wa:
        warnings.simplefilter("always")
        out_a, log_a, keep_a = A_h.reanchor_on_identifier(
            ss, va, label="x", return_keep=True)
    with warnings.catch_warnings(record=True) as wb:
        warnings.simplefilter("always")
        out_b, log_b, keep_b = L_prep._reanchor_on_identifier(
            ss, vb, label="x", warning_stacklevel=2, return_keep=True)
    assert log_a == log_b
    np.testing.assert_array_equal(keep_a, keep_b)
    np.testing.assert_array_equal(out_a.pos, out_b.pos)
    assert [str(w.message) for w in wa] == [str(w.message) for w in wb]
