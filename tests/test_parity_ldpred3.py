"""allelign reproduces ldpred3 on the inputs ldpred3 already handles.

This is the gate for moving ldpred3 (and its siblings) onto allelign: for the
same input, every ported function must return the same output, bit for bit.
Skipped when ldpred3 is not installed.
"""

import gzip
import importlib
import warnings

import numpy as np
import pytest

pytest.importorskip("ldpred3")
pytestmark = pytest.mark.parity

L_ss = importlib.import_module("ldpred3.sumstats")
L_h = importlib.import_module("ldpred3.harmonize")
L_qc = importlib.import_module("ldpred3.qc")
L_w = importlib.import_module("ldpred3.weights")
L_st = importlib.import_module("ldpred3._qc_status")

import allelign.align as A_h                       # noqa: E402
import allelign.qc as A_qc                         # noqa: E402
import allelign.status as A_st                     # noqa: E402
import allelign.sumstats as A_ss                   # noqa: E402
import allelign.weights as A_w                     # noqa: E402
from allelign.alleles import (allele_orientation, is_ambiguous,  # noqa: E402
                              normalize_chrom)
from allelign.ldcheck import ld_consistency_screen  # noqa: E402
from allelign.variants import Variants             # noqa: E402

FIELDS = ("id", "chrom", "pos", "ea", "oa", "beta", "se", "n_eff", "eaf",
          "info")


def _same_array(x, y, name=""):
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == y.shape, name
    if x.dtype.kind == "f" or y.dtype.kind == "f":
        assert x.dtype == y.dtype, name
        np.testing.assert_array_equal(x.view(np.int64), y.view(np.int64),
                                      err_msg=name)
    else:
        assert list(x) == list(y), name


def _same_sumstats(a, b):
    for f in FIELDS:
        _same_array(getattr(a, f), getattr(b, f), f)
    for key in ("n_input", "n_se_from_pval"):
        assert a.parse_log[key] == b.parse_log[key], key


def _messy_rows(rng, n):
    toks = ["NA", "", ".", "nan", "1e-300", "-0", "3.2E+04"]
    rows = []
    for i in range(n):
        def num(scale=1.0):
            if rng.random() < 0.05:
                return toks[rng.integers(len(toks))]
            return f"{rng.normal() * scale:.6g}"
        p = (f"{10 ** -rng.uniform(0, 330):.3g}" if rng.random() < 0.9
             else toks[rng.integers(len(toks))])
        rows.append([f"rs{i}" if rng.random() > 0.02 else "",
                     str(rng.integers(1, 24)), str(rng.integers(1, 10 ** 8)),
                     "ACGT"[rng.integers(4)], "acgt"[rng.integers(4)],
                     num(0.05), (num(0.01).lstrip("-") if rng.random() > 0.2
                                 else "NA"), p,
                     str(rng.integers(5000, 9000)),
                     f"{rng.uniform(0, 1):.4f}", f"{rng.uniform(0.3, 1):.3f}"])
    return rows


def _write(path, header, rows, delim="\t", gz=False):
    text = delim.join(header) + "\n" + "\n".join(delim.join(r) for r in rows) + "\n"
    if gz:
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        path.write_text(text)
    return str(path)


DIALECTS = {
    "plain": (["SNP", "CHR", "BP", "A1", "A2", "BETA", "SE", "P", "N", "EAF",
               "INFO"], "\t", {}),
    "or_whitespace": (["SNP", "CHR", "BP", "A1", "A2", "OR", "SE", "P", "N",
                       "FRQ", "INFO"], " ", {}),
    "comma_gwas_ssf": (["rsid", "chromosome", "base_pair_location",
                        "effect_allele", "other_allele", "beta",
                        "standard_error", "p_value", "n",
                        "effect_allele_frequency", "info"], ",", {}),
    "metal": (["MarkerName", "Chr", "Pos", "Allele1", "Allele2", "Effect",
               "StdErr", "P-value", "N", "Freq1", "Rsq"], "\t", {}),
}


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
@pytest.mark.parametrize("gz", [False, True])
def test_read_sumstats_matches_ldpred3(tmp_path, dialect, gz):
    header, delim, kw = DIALECTS[dialect]
    rows = _messy_rows(np.random.default_rng(len(dialect)), 400)
    if delim == " ":
        rows = [[t or "NA" for t in r] for r in rows]
    if header[5] == "OR":
        for r in rows:
            try:
                r[5] = f"{np.exp(min(float(r[5]), 50.0)):.6g}"
            except ValueError:
                pass
    path = _write(tmp_path / ("f.txt.gz" if gz else "f.txt"), header, rows,
                  delim, gz)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _same_sumstats(A_ss.read_sumstats(path, **kw),
                       L_ss.read_sumstats(path, **kw))
        _same_sumstats(A_ss.read_sumstats(path, n_eff=1234.0),
                       L_ss.read_sumstats(path, n_eff=1234.0))


@pytest.mark.numba
def test_jit_block_parser_matches_ldpred3(tmp_path):
    header, delim, _ = DIALECTS["plain"]
    rows = _messy_rows(np.random.default_rng(7), 60_000)   # > 2 MB of text
    path = _write(tmp_path / "big.tsv", header, rows, delim)
    _same_sumstats(A_ss.read_sumstats(path), L_ss.read_sumstats(path))


def test_column_aliases_extend_ldpred3s_in_order():
    for field, aliases in L_ss.COLUMN_ALIASES.items():
        assert A_ss.COLUMN_ALIASES[field][:len(aliases)] == aliases, field


def test_allele_primitives_match():
    labels = ["chr1", "01", "1", "X", "23", "25", "XY", "chrX", "24", "Y",
              "26", "M", "MT", "chrM", " 7 ", "chr", "", "Un"]
    for c in labels:
        assert normalize_chrom(c) == L_h._norm_chrom(c), c
    alleles = ["A", "C", "G", "T", "AT", "CG", "AA", "TT", "ACG", "CGT", "N",
               "AN", "I", "D"]
    for x in alleles:
        for y in alleles:
            assert is_ambiguous(x, y) == L_h._ambiguous_pair(x, y), (x, y)
            for g1 in ("A", "C", "AA", "TT"):
                for g2 in ("G", "T", "CGT"):
                    assert (allele_orientation(x, y, g1, g2)
                            == L_h._allele_orientation(x, y, g1, g2))


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
    cov_a = A_h.reference_coverage(ss, va, a)
    cov_b = L_h.reference_coverage(ss, vb, b)
    assert cov_a == cov_b


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
def test_reference_checks_match_ldpred3(rule):
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
    af_gwas = np.clip(af + rng.normal(0, 0.05, m), 0, 1)
    for gwas in (af_gwas, 1 - af):          # valve off, valve on (mirrored)
        a = A_qc.af_concordance_mask(gwas, af)
        b = L_qc.af_concordance_mask(gwas, af)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1]
    a = A_qc.impute_n_eff(se, af, 1e5, info=info)
    b = L_qc.impute_n_eff(se, af, 1e5, info=info)
    _same_array(a[0], b[0])
    assert a[1] == b[1]
    for args in ((0.9, 0.02), (0.99, 0.02), (1.1, 0.01), (0.8, 0.5)):
        assert (A_qc.genomic_control_factor(*args, n_variants=60_000)
                == L_qc.genomic_control_factor(*args, n_variants=60_000))
    assert (A_qc.implied_sample_size(beta, se, af, binary=True, reported_n=1e5)
            == L_qc.implied_sample_size(beta, se, af, binary=True,
                                        reported_n=1e5))
    for value in (True, False, None):
        for nvec in (np.full(5, 1e4), np.arange(5.0)):
            assert (A_qc.should_impute_n(value, nvec)
                    == L_qc.should_impute_n(value, nvec))


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
    pa, pb = tmp_path / "a.tsv", tmp_path / "b.tsv"
    A_w.write_weights(pa, **kw)
    L_w.write_weights(pb, **kw)
    assert pa.read_bytes() == pb.read_bytes()
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


def test_status_vocabulary_matches_ldpred3():
    assert A_st.QC_REASON_NAMES == L_st.QC_REASON_NAMES
