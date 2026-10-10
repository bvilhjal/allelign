"""Tests for the sumstats reader and harmonisation rules.

Two halves of one reader: the synthetic-format rules (aliases, OR->beta,
SE-from-p, external N, column overrides) and the harmonisation rules, then
the same reader against fixtures that mimic the header conventions of actual
tools and consortia (GWAS Catalog harmonised, BOLT-LMM, METAL, PGC
case-control OR files, SAIGE). No network / large files: small inline
fixtures stand in for the real multi-GB downloads.
"""


import gzip
import math

import numpy as np
import pytest


from allelign.sumstats import (read_sumstats, detect_columns,
                              rescale_n_eff)
from allelign.align import harmonize
from allelign.alleles import reverse_complement as _complement
from allelign.variants import Variants as VariantTable


from dataclasses import fields

from allelign.align import _variant_indices as variant_indices
from allelign.sumstats import Sumstats, _SUMSTATS_FIELDS


def _write(tmp_path, text, name="ss.txt"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_read_sumstats_bgz(tmp_path):
    path = tmp_path / "ss.tsv.bgz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tN\n"
                 "rs1\t1\t100\tA\tG\t0.10\t0.02\t1000\n")
    ss = read_sumstats(str(path))
    assert list(ss.id) == ["rs1"]
    np.testing.assert_allclose(ss.beta, [0.10])


def test_read_sumstats_tab_and_aliases(tmp_path):
    path = _write(tmp_path,
        "SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tP\tN\n"
        "rs1\t1\t100\tA\tG\t0.10\t0.02\t1e-6\t1000\n"
        "rs2\t1\t200\tC\tT\t-0.05\t0.01\t2e-7\t1000\n")
    ss = read_sumstats(path)
    assert len(ss) == 2
    np.testing.assert_array_equal(ss.id, ["rs1", "rs2"])
    np.testing.assert_allclose(ss.beta, [0.10, -0.05])
    np.testing.assert_allclose(ss.se, [0.02, 0.01])
    np.testing.assert_array_equal(ss.ea, ["A", "C"])


def test_read_sumstats_or_to_beta(tmp_path):
    path = _write(tmp_path,
        "SNP,A1,A2,OR,SE,N\n"
        "rs1,A,G,1.105170918,0.02,500\n")     # log(OR) = 0.1
    ss = read_sumstats(path)
    np.testing.assert_allclose(ss.beta, [0.1], atol=1e-6)


def test_read_sumstats_se_from_pvalue(tmp_path):
    # No SE column: recover from beta and p (z = 1.959964 at p=0.05).
    path = _write(tmp_path,
        "SNP A1 A2 BETA P N\n"
        "rs1 A G 0.1 0.05 1000\n")
    ss = read_sumstats(path)
    np.testing.assert_allclose(ss.se, [0.1 / 1.959964], rtol=1e-4)


def test_read_sumstats_se_from_tiny_pvalue(tmp_path):
    # Very small p-values (common in GWAS): 1 - p/2 rounds to 1.0 and the naive
    # inv_cdf(1.0) raises. The reader must still recover a finite, sane SE.
    path = _write(tmp_path,
        "SNP A1 A2 BETA P N\n"
        "rs1 A G 0.5 1e-300 100000\n"
        "rs2 A G 0.2 5e-20 100000\n")
    ss = read_sumstats(path)
    assert np.all(np.isfinite(ss.se)) and np.all(ss.se > 0)
    # larger z (smaller p) -> smaller SE for comparable beta scale
    assert ss.se[0] < ss.se[1]


def test_read_sumstats_external_n(tmp_path):
    path = _write(tmp_path, "SNP A1 A2 BETA SE\nrs1 A G 0.1 0.02\n")
    ss = read_sumstats(path, n_eff=2000)
    np.testing.assert_allclose(ss.n_eff, [2000])


def test_maf_is_not_automatically_treated_as_effect_allele_frequency(tmp_path):
    """MAF has no counted-allele orientation and therefore is not EAF."""
    path = _write(
        tmp_path,
        "SNP A1 A2 BETA SE N MAF\n"
        "rs1 A G 0.1 0.02 1000 0.1\n",
    )
    _, mapping = detect_columns(path)
    assert "eaf" not in mapping
    assert np.isnan(read_sumstats(path).eaf).all()

    # A producer may use an odd label for a genuinely effect-oriented column;
    # the explicit override remains available because the user then supplies
    # the missing semantic claim rather than the reader guessing it.
    ss = read_sumstats(path, eaf="MAF")
    np.testing.assert_allclose(ss.eaf, [0.1])


def test_numeric_n_eff_overrides_detected_column(tmp_path):
    # read_sumstats(n_eff=<number>) keeps its deliberate force-flatten
    # semantics: it is the low-level escape hatch for a file that reports
    # total N where effective N is required. Pipeline-level callers rescale
    # instead -- see rescale_n_eff below.
    path = _write(
        tmp_path,
        "SNP CHR BP A1 A2 BETA SE N\n"
        "rs1 1 10 A G 0.1 0.02 10000\n"
        "rs2 1 20 C T -0.2 0.03 12000\n",
    )
    ss = read_sumstats(path, n_eff=4321)
    np.testing.assert_array_equal(ss.n_eff, [4321.0, 4321.0])


def test_rescale_n_eff_anchors_median_downward(tmp_path):
    path = _write(
        tmp_path,
        "SNP CHR BP A1 A2 BETA SE N\n"
        "rs1 1 10 A G 0.1 0.02 8000\n"
        "rs2 1 20 C T -0.2 0.03 4000\n"
        "rs3 1 30 A T 0.05 0.03 6000\n",
    )
    ss = read_sumstats(path)
    rec = rescale_n_eff(ss, 3000)
    # median 6000 -> factor 0.5; the relative pattern is preserved.
    np.testing.assert_allclose(ss.n_eff, [4000.0, 2000.0, 3000.0])
    assert rec["method"] == "downward median rescale"
    assert rec["source_median"] == 6000.0
    assert rec["target_effective_n"] == 3000.0
    assert rec["factor"] == 0.5
    assert rec["applied"] is True
    assert rec["upward_scaling_refused"] is False
    assert rec["n_usable"] == 3


def test_rescale_n_eff_refuses_upward_and_keeps_missing(tmp_path):
    path = _write(
        tmp_path,
        "SNP CHR BP A1 A2 BETA SE N\n"
        "rs1 1 10 A G 0.1 0.02 4000\n"
        "rs2 1 20 C T -0.2 0.03 NA\n"
        "rs3 1 30 A T 0.05 0.03 2000\n",
    )
    ss = read_sumstats(path)
    rec = rescale_n_eff(ss, 9000)
    # Supplied scalar above the column median: the column is a measurement,
    # not a deficit, so it is left untouched and the refusal recorded.
    np.testing.assert_allclose(
        ss.n_eff, [4000.0, np.nan, 2000.0], equal_nan=True)
    assert rec["applied"] is False
    assert rec["factor"] == 1.0
    assert rec["upward_scaling_refused"] is True
    assert rec["n_usable"] == 2


def test_rescale_n_eff_returns_none_without_usable_values(tmp_path):
    path = _write(
        tmp_path,
        "SNP CHR BP A1 A2 BETA SE N\n"
        "rs1 1 10 A G 0.1 0.02 NA\n"
        "rs2 1 20 C T -0.2 0.03 0\n",
    )
    ss = read_sumstats(path)
    assert rescale_n_eff(ss, 5000) is None


def test_rescale_n_eff_validates_scalar(tmp_path):
    import pytest
    path = _write(
        tmp_path,
        "SNP CHR BP A1 A2 BETA SE N\n"
        "rs1 1 10 A G 0.1 0.02 4000\n",
    )
    ss = read_sumstats(path)
    for bad in (0, -5, float("nan"), float("inf"), True, np.bool_(True),
                "4000", None):
        with pytest.raises(ValueError, match="finite positive"):
            rescale_n_eff(ss, bad)


def test_read_sumstats_blank_or_na_position(tmp_path):
    # A blank / non-numeric position on one row must not abort the whole read
    # (regression: int(NaN) raised, killing files with any missing BP). The bad
    # positions fall back to 0; the rest of the row is preserved.
    path = _write(tmp_path,
        "SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tN\n"
        "rs1\t1\t100\tA\tG\t0.10\t0.02\t1000\n"
        "rs2\t1\t\tC\tT\t-0.05\t0.01\t1000\n"       # blank BP
        "rs3\t1\tNA\tA\tG\t0.07\t0.03\t1000\n")     # non-numeric BP
    ss = read_sumstats(path)
    assert len(ss) == 3
    np.testing.assert_array_equal(ss.id, ["rs1", "rs2", "rs3"])
    np.testing.assert_array_equal(ss.pos, [100, 0, 0])
    np.testing.assert_allclose(ss.beta, [0.10, -0.05, 0.07])


def _variants():
    return VariantTable(
        chrom=np.array(["1", "1", "1", "1", "1"], dtype=object),
        id=np.array(["rs1", "rs2", "rs3", "rs4", "rs5"], dtype=object),
        cm=np.zeros(5),
        pos=np.array([100, 200, 300, 400, 500], dtype=np.int64),
        a1=np.array(["A", "C", "A", "A", "G"], dtype=object),   # counted allele
        a2=np.array(["G", "T", "C", "G", "C"], dtype=object),
    )


def test_harmonize_alignment_rules(tmp_path):
    # rs1: ea==A1            -> keep beta
    # rs2: ea==A2 (swapped)  -> flip sign
    # rs3: strand flip (T/A complements A/T... actually use C/G via comp)
    # rs4: palindromic A/T   -> dropped (ambiguous)
    # rs5: allele mismatch   -> dropped
    path = _write(tmp_path,
        "SNP A1 A2 BETA SE N\n"
        "rs1 A G 0.20 0.01 1000\n"     # exact
        "rs2 T C -0.30 0.01 1000\n"    # swapped (A1=C,A2=T) -> ea=T==A2 -> flip
        "rs3 T G 0.40 0.01 1000\n"     # geno A/C; comp(T)=A,comp(G)=C -> A/C keep
        "rs4 A T 0.50 0.01 1000\n"     # geno A/G but ss A/T palindrome -> ambiguous
        "rs5 C A 0.60 0.01 1000\n")    # geno G/C, ss C/A -> mismatch
    ss = read_sumstats(path)
    h = harmonize(ss, _variants())

    kept = dict(zip(h.var_index.tolist(), h.beta.tolist()))
    assert 0 in kept and np.isclose(kept[0], 0.20)     # rs1 keep
    assert 1 in kept and np.isclose(kept[1], 0.30)     # rs2 flipped -0.30 -> +0.30
    assert 2 in kept and np.isclose(kept[2], 0.40)     # rs3 strand, same order
    assert 3 not in kept                               # rs4 dropped ambiguous
    assert 4 not in kept                               # rs5 dropped mismatch
    assert h.log["n_flipped"] == 1
    assert h.log["n_strand_flipped"] == 1
    assert h.log["n_dropped_ambiguous"] == 1
    assert h.log["n_dropped_mismatch"] == 1
    # var_index is sorted (genotype-column order)
    assert list(h.var_index) == sorted(h.var_index)


def test_harmonize_drops_multibase_palindromic_pairs(tmp_path):
    # A pair whose reverse complements reproduce the pair (an indel AA/TT) is
    # as strand-ambiguous as a single-base A/T: resolving it would orient the
    # alleles -- and the effect sign -- by the order the matcher tries, not by
    # evidence. A non-invariant pair still strand-matches, while a pair whose
    # alleles are each self reverse-complementary (AT/CG) preserves its order
    # and effect sign and is therefore safe.
    path = _write(tmp_path,
        "SNP A1 A2 BETA SE N\n"
        "rs1 AA TT 0.20 0.01 1000\n"    # multibase palindrome -> ambiguous
        "rs2 AA GG 0.40 0.01 1000\n"    # comp(AA)=TT, comp(GG)=CC -> orientable
        "rs3 AT CG 0.60 0.01 1000\n")   # each allele is self-RC -> safe
    ss = read_sumstats(path)
    variants = VariantTable(
        chrom=np.array(["1", "1", "1"], dtype=object),
        id=np.array(["rs1", "rs2", "rs3"], dtype=object),
        cm=np.zeros(3),
        pos=np.array([100, 200, 300], dtype=np.int64),
        a1=np.array(["TT", "TT", "AT"], dtype=object),  # rs2 opposite strand
        a2=np.array(["AA", "CC", "CG"], dtype=object))
    h = harmonize(ss, variants)
    assert h.log["n_dropped_ambiguous"] == 1
    assert h.log["n_matched"] == 2
    assert h.log["n_strand_flipped"] == 1
    assert list(h.var_index) == [1, 2]
    np.testing.assert_allclose(h.beta, [0.40, 0.60])


def test_read_sumstats_column_override_by_index(tmp_path):
    # Regression for D2: the guide advertises "override a column by name or
    # zero-based index", but a digit-string override was looked up as a column
    # *name* and failed. A digit-only override now resolves to a 0-based index.
    path = _write(tmp_path,
        "SNP A1 A2 BXYZ SE N\n"          # 'BXYZ' is not an auto-detected beta alias
        "rs1 A G 0.15 0.02 1000\n")
    ss_name = read_sumstats(path, beta="BXYZ")
    np.testing.assert_allclose(ss_name.beta, [0.15])
    ss_idx = read_sumstats(path, beta="3")   # column 3 (0-based) is BXYZ
    np.testing.assert_allclose(ss_idx.beta, [0.15])


def test_harmonize_counts_duplicate_matches(tmp_path):
    # Regression for C5: a second sumstats row matching an already-matched
    # genotype variant was dropped silently (in no log bucket), so the buckets did
    # not sum to n_sumstats. It is now counted as n_dropped_duplicate.
    path = _write(tmp_path,
        "SNP A1 A2 BETA SE N\n"
        "rs1 A G 0.20 0.01 1000\n"     # matches genotype rs1 (first wins)
        "rs1 A G 0.25 0.01 1000\n"     # duplicate of rs1 -> counted, not applied
        "rs2 C T -0.30 0.01 1000\n")   # matches genotype rs2
    ss = read_sumstats(path)
    h = harmonize(ss, _variants())
    assert h.log["n_matched"] == 2
    assert h.log["n_dropped_duplicate"] == 1
    # only the first rs1 beta is applied (0.20, not the duplicate's 0.25)
    kept = dict(zip(h.var_index.tolist(), h.beta.tolist()))
    assert np.isclose(kept[0], 0.20)
    # every source row now falls into exactly one bucket
    log = h.log
    total = (log["n_matched"] + log["n_unmatched"] + log["n_dropped_ambiguous"]
             + log["n_dropped_mismatch"] + log["n_dropped_ambiguous_id"]
             + log["n_dropped_nonfinite"] + log["n_dropped_duplicate"])
    assert total == log["n_sumstats"] == len(ss)


def test_harmonize_multiallelic_rsid_uses_allele_compatible_row():
    """Two BIM rows can share an rsID; keep the first allele-compatible one."""
    variants = VariantTable(
        chrom=np.array(["1", "1"], dtype=object),
        id=np.array(["rs1", "rs1"], dtype=object),
        cm=np.zeros(2),
        pos=np.array([100, 100], dtype=np.int64),
        a1=np.array(["A", "A"], dtype=object),
        a2=np.array(["G", "C"], dtype=object),
    )
    from allelign.sumstats import Sumstats
    ss = Sumstats(
        id=np.array(["rs1"], dtype=object),
        chrom=np.array(["1"], dtype=object),
        pos=np.array([100], dtype=np.int64),
        ea=np.array(["A"], dtype=object),
        oa=np.array(["C"], dtype=object),
        beta=np.array([0.20]), se=np.array([0.01]), n_eff=np.array([1000.0]),
        eaf=np.array([np.nan]), info=np.array([np.nan]))
    h = harmonize(ss, variants)
    assert list(h.var_index) == [1]
    np.testing.assert_allclose(h.beta, [0.20])
    assert h.log["n_matched"] == 1
    assert h.log["n_dropped_mismatch"] == 0

    # Multiple allele records at one locus do not require GWAS coordinates;
    # only an identifier duplicated across distinct loci does.
    ss.pos[0] = 0
    ss.chrom[0] = ""
    without_coordinate = harmonize(ss, variants)
    assert list(without_coordinate.var_index) == [1]


def test_harmonize_duplicate_rsid_uses_coordinate_or_rejects_ambiguity():
    variants = VariantTable(
        chrom=np.array(["1", "1"], dtype=object),
        id=np.array(["rs1", "rs1"], dtype=object),
        cm=np.zeros(2), pos=np.array([100, 200], dtype=np.int64),
        a1=np.array(["A", "A"], dtype=object),
        a2=np.array(["G", "G"], dtype=object),
    )
    from allelign.sumstats import Sumstats

    def one(pos):
        return Sumstats(
            id=np.array(["rs1"], dtype=object),
            chrom=np.array(["1"], dtype=object),
            pos=np.array([pos], dtype=np.int64),
            ea=np.array(["A"], dtype=object),
            oa=np.array(["G"], dtype=object), beta=np.array([0.2]),
            se=np.array([0.01]), n_eff=np.array([1000.0]),
            eaf=np.array([np.nan]), info=np.array([np.nan]),
        )

    matched = harmonize(one(200), variants)
    np.testing.assert_array_equal(matched.var_index, [1])

    for unresolved in (0, 300):
        rejected = harmonize(one(unresolved), variants)
        assert len(rejected) == 0
        assert rejected.log["n_dropped_ambiguous_id"] == 1
        assert rejected.log["n_dropped_mismatch"] == 0


def test_harmonize_unique_rsid_at_the_wrong_locus_is_unmatched():
    """A unique rsID whose chrom/pos disagrees must not match that ID."""
    variants = VariantTable(
        chrom=np.array(["1"], dtype=object),
        id=np.array(["rs1"], dtype=object),
        cm=np.zeros(1), pos=np.array([100], dtype=np.int64),
        a1=np.array(["A"], dtype=object),
        a2=np.array(["G"], dtype=object),
    )
    from allelign.sumstats import Sumstats
    ss = Sumstats(
        id=np.array(["rs1"], dtype=object),
        chrom=np.array(["1"], dtype=object),
        pos=np.array([99999], dtype=np.int64),
        ea=np.array(["A"], dtype=object),
        oa=np.array(["G"], dtype=object),
        beta=np.array([0.2]), se=np.array([0.01]), n_eff=np.array([1000.0]),
        eaf=np.array([np.nan]), info=np.array([np.nan]),
    )
    rejected = harmonize(ss, variants)
    assert len(rejected) == 0
    assert rejected.log["n_unmatched"] == 1
    assert rejected.log["n_dropped_ambiguous_id"] == 0

    # Missing coordinates still keep the unique rsID match.
    ss.pos[0] = 0
    ss.chrom[0] = ""
    kept = harmonize(ss, variants)
    np.testing.assert_array_equal(kept.var_index, [0])


def test_harmonize_uppercases_gwas_alleles():
    """Public ``harmonize`` must match lowercase GWAS alleles to BIM alleles."""
    from allelign.sumstats import Sumstats

    variants = _variants()
    ss = Sumstats(
        id=np.array(["rs1", "rs4"], dtype=object),
        chrom=np.array(["1", "1"], dtype=object),
        pos=np.array([100, 400], dtype=np.int64),
        ea=np.array(["a", "a"], dtype=object),
        oa=np.array(["g", "t"], dtype=object),
        beta=np.array([0.20, 0.50]), se=np.array([0.01, 0.01]),
        n_eff=np.array([1000.0, 1000.0]),
        eaf=np.array([np.nan, np.nan]), info=np.array([np.nan, np.nan]),
    )
    h = harmonize(ss, variants)
    assert list(h.var_index) == [0]
    np.testing.assert_allclose(h.beta, [0.20])
    assert h.log["n_dropped_ambiguous"] == 1
    assert h.log["n_dropped_mismatch"] == 0


def test_harmonize_counts_nonfinite_separately():
    from allelign.sumstats import Sumstats

    variants = _variants()
    ss = Sumstats(
        id=np.array(["rs1", "rs2"], dtype=object),
        chrom=np.array(["1", "1"], dtype=object),
        pos=np.array([100, 200], dtype=np.int64),
        ea=np.array(["A", "C"], dtype=object),
        oa=np.array(["G", "T"], dtype=object),
        beta=np.array([0.20, np.nan]), se=np.array([0.01, 0.01]),
        n_eff=np.array([1000.0, 1000.0]),
        eaf=np.array([np.nan, np.nan]), info=np.array([np.nan, np.nan]),
    )
    h = harmonize(ss, variants)
    assert list(h.var_index) == [0]
    assert h.log["n_dropped_nonfinite"] == 1
    assert h.log["n_dropped_mismatch"] == 0


def test_harmonize_match_by_position_when_no_id(tmp_path):
    path = _write(tmp_path,
        "CHR BP A1 A2 BETA SE N\n"
        "1 200 C T 0.7 0.01 1000\n")    # no rsID -> match by chrom:pos -> rs2
    ss = read_sumstats(path)
    h = harmonize(ss, _variants())
    assert list(h.var_index) == [1]
    np.testing.assert_allclose(h.beta, [0.7])


def test_complement_helper():
    assert _complement("AC") == "GT"
    assert _complement("N") is None


def test_harmonize_chr_prefix_and_sex_codes(tmp_path):
    # Genotypes labelled '1'; sumstats use 'chr1' (and 'chrX'/'23' style) — they
    # must still match by position after chromosome normalisation.
    from allelign.alleles import normalize_chrom as _norm_chrom
    assert _norm_chrom("chr1") == "1" and _norm_chrom("1") == "1"
    assert _norm_chrom("chrX") == "X" and _norm_chrom("23") == "X"
    assert _norm_chrom("MT") == "MT" and _norm_chrom("26") == "MT" and _norm_chrom("M") == "MT"

    path = _write(tmp_path,
        "CHR BP A1 A2 BETA SE N\n"
        "chr1 200 C T 0.7 0.01 1000\n")   # 'chr1' vs genotype '1'
    ss = read_sumstats(path)
    h = harmonize(ss, _variants())
    assert list(h.var_index) == [1]
    np.testing.assert_allclose(h.beta, [0.7])



def test_vectorised_harmonize_matches_the_row_loop():
    """The vectorised join must reproduce the row loop, field for field.

    Small pools force the collisions the rules are about: chr-prefixed and
    sex/MT labels, missing (0/negative) positions, IDs shared across loci or
    at the wrong locus, multi-allelic sites, palindromic and multi-base
    pairs, strand flips, lowercase and non-ACGT alleles, alternate IDs,
    non-finite and integer effects, and genotype rows several sumstats rows
    reach (the order-dependent ``seen`` rule). Each table is reused so the
    cached-index path runs too.
    """
    import allelign.align as hmod

    rng = np.random.default_rng(20260923)
    chroms = ["1", "chr1", "CHR1", " 1", "2", "23", "X", "chrX", "x", "MT",
              "M", "26", "25", "XY", "", " ", "chr"]
    ids = ["rs1", "rs2", "rs3", "rs4", "", ".", "RS1"]
    alleles = ["A", "C", "G", "T", "a", "t", "AA", "TT", "AT", "CG", "N",
               "ACG", "CGT", "AAC", "gtt", "-"]
    comp = str.maketrans("ACGTacgt", "TGCAtgca")

    def obj(values):
        out = np.empty(len(values), dtype=object)
        out[:] = values
        return out

    n_vector = 0
    for _case in range(300):
        nv, ns = int(rng.integers(0, 12)), int(rng.integers(0, 30))
        pick = lambda pool, n: [pool[i] for i in rng.integers(0, len(pool), n)]  # noqa: E731
        variants = VariantTable(
            chrom=obj(pick(chroms, nv)), id=obj(pick(ids, nv)),
            cm=np.zeros(nv), pos=rng.integers(-1, 5, nv),
            a1=obj(pick(alleles, nv)), a2=obj(pick(alleles, nv)),
            alternate_id=obj(pick(ids, nv)) if rng.random() < 0.3 else None)
        for _call in range(2):
            s_id, s_chrom = pick(ids, ns), pick(chroms, ns)
            s_pos = rng.integers(-1, 5, ns)
            ea, oa = pick(alleles, ns), pick(alleles, ns)
            for k in range(ns):
                if nv and rng.random() < 0.6:    # derived from a genotype row
                    gi = int(rng.integers(nv))
                    e, o = variants.a1[gi], variants.a2[gi]
                    if rng.random() < 0.5:
                        e, o = o, e
                    if rng.random() < 0.3:
                        e, o = e.translate(comp)[::-1], o.translate(comp)[::-1]
                    ea[k], oa[k] = e, o
                    s_id[k] = variants.id[gi] if rng.random() < 0.6 else ""
                    s_chrom[k] = variants.chrom[gi]
                    s_pos[k] = variants.pos[gi]
            beta = rng.choice([0.1, -0.2, 0.0, -0.0, np.nan, np.inf], ns)
            if rng.random() < 0.1:
                beta = np.nan_to_num(beta, posinf=3).astype(np.int64)
            ss = type("SS", (), {})()
            ss.id, ss.chrom, ss.pos = obj(s_id), obj(s_chrom), s_pos
            ss.ea, ss.oa, ss.beta = obj(ea), obj(oa), beta
            ss.se = rng.choice([0.01, 0.5, np.nan], ns)
            ss.n_eff = rng.choice([1000.0, 0.0], ns)
            type(ss).__len__ = lambda self: len(self.beta)
            for drop in (True, False):
                entry = hmod._variant_indices(variants)
                columns = hmod._vector_columns(ss, variants, entry)
                assert columns is not None
                got = hmod._harmonize_vector(ss, variants, entry, columns,
                                             drop)
                want = hmod._harmonize_rows(ss, variants, entry, drop)
                n_vector += 1
                assert list(got.log.items()) == list(want.log.items())
                for f in ("var_index", "beta", "se", "n_eff", "flipped",
                          "src_index"):
                    a, b = getattr(got, f), getattr(want, f)
                    assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), f
    assert n_vector == 1200


def test_harmonize_defers_unusual_columns_to_the_row_loop():
    """Non-text labels and NaN positions keep the row loop's semantics."""
    import allelign.align as hmod

    variants = VariantTable(
        chrom=np.array([1, "1"], dtype=object),
        id=np.array(["rs1", "rs2"], dtype=object), cm=np.zeros(2),
        pos=np.array([100, 200], dtype=np.int64),
        a1=np.array(["A", "C"], dtype=object),
        a2=np.array(["G", "T"], dtype=object))
    entry = hmod._variant_indices(variants)
    assert hmod._genotype_columns(variants, entry) is None
    ss = type("SS", (), {"__len__": lambda self: 1})()
    ss.id, ss.chrom = np.array([""], dtype=object), np.array(["1"], dtype=object)
    ss.pos = np.array([np.nan])
    ss.ea, ss.oa = np.array(["A"], dtype=object), np.array(["G"], dtype=object)
    ss.beta, ss.se, ss.n_eff = np.array([0.1]), np.array([0.01]), np.array([1.0])
    assert hmod._vector_columns(ss, _variants(), [None] * 5 + [{}]) is None


def test_diagnose_match_build_mismatch(tmp_path):
    # Same rsIDs, but shifted positions -> diagnosed as a build mismatch.
    from allelign.align import diagnose_match
    v = _variants()
    path = _write(tmp_path,
        "SNP CHR BP A1 A2 BETA SE N\n"
        "rs1 1 999100 A G 0.2 0.01 1000\n"
        "rs2 1 999200 C T 0.2 0.01 1000\n")
    ss = read_sumstats(path)
    d = diagnose_match(ss, v)
    assert d["rsid_overlap"] == 2
    assert d["build_mismatch_suspected"]
    assert "build" in d["message"]


def test_diagnose_match_disjoint(tmp_path):
    from allelign.align import diagnose_match
    v = _variants()
    path = _write(tmp_path,
        "SNP CHR BP A1 A2 BETA SE N\n"
        "rsX 9 10 A G 0.2 0.01 1000\n")
    ss = read_sumstats(path)
    d = diagnose_match(ss, v)
    assert d["rsid_overlap"] == 0 and d["pos_overlap_normalized"] == 0
    assert "different variant sets" in d["message"] or "build" in d["message"]


def test_harmonize_src_index_maps_back_to_source_rows():
    # src_index identifies each matched variant's *source* row so callers can pull
    # extra per-variant source columns positionally even when the match was by
    # chrom:pos (source ID != genotype ID). Regression for score_from_weights
    # frozen-scaling, which previously keyed AF/SD by genotype ID and KeyError'd.
    from allelign.sumstats import Sumstats
    # No rsIDs -> match by position; rows deliberately out of genomic order.
    ss = Sumstats(
        id=np.array(["", "", ""], dtype=object),
        chrom=np.array(["1", "1", "1"], dtype=object),
        pos=np.array([300, 100, 200], dtype=np.int64),   # rs3, rs1, rs2
        ea=np.array(["A", "A", "C"], dtype=object),       # match genotype A1 (no flip)
        oa=np.array(["C", "G", "T"], dtype=object),
        beta=np.array([0.30, 0.10, 0.20]),                # distinct per source row
        se=np.full(3, 0.01), n_eff=np.full(3, 1000.0),
        eaf=np.full(3, np.nan), info=np.full(3, np.nan))
    h = harmonize(ss, _variants())
    assert h.src_index is not None and len(h.src_index) == len(h)
    # source beta pulled positionally by src_index reproduces the harmonised beta
    np.testing.assert_allclose(ss.beta[h.src_index], h.beta)


def test_harmonize_index_cache_reuses_and_invalidates(tmp_path):
    # Panel workflows harmonise many files against one variant table; the
    # O(variants) Python index build must happen once per table, not once per
    # file, and must be dropped when the table's columns are replaced.
    import allelign.align as hmod

    path = _write(tmp_path,
        "SNP A1 A2 BETA SE N\n"
        "rs1 A G 0.20 0.01 1000\n"
        "rs2 T C -0.30 0.01 1000\n")
    ss = read_sumstats(path)
    variants = _variants().freeze()

    builds = []
    real_build = hmod._build_id_index

    def counting_build(v):
        builds.append(1)
        return real_build(v)

    hmod._build_id_index = counting_build
    try:
        first = harmonize(ss, variants)
        second = harmonize(ss, variants)
        assert builds, "index build was not routed through _build_id_index"
        assert len(builds) == 1, (
            f"repeated harmonize calls rebuilt the variant index "
            f"{len(builds)} times")

        # Identical results across calls (same table, cached indices).
        np.testing.assert_array_equal(first.var_index, second.var_index)
        np.testing.assert_array_equal(first.beta, second.beta)
        np.testing.assert_array_equal(first.flipped, second.flipped)
        assert first.log == second.log

        # Replacing a column invalidates the cache: different matches result.
        variants.id = np.array(["rs9", "rs2", "rs3", "rs4", "rs5"], dtype=object)
        third = harmonize(ss, variants)
        assert len(builds) == 2, "a replaced column must invalidate the cache"
        assert 0 not in third.var_index.tolist()   # rs1 no longer exists
        assert 1 in third.var_index.tolist()       # rs2 still matches

        # A fresh table (e.g. VariantTable.subset) builds its own indices.
        builds.clear()
        harmonize(ss, _variants())
        assert len(builds) == 1
    finally:
        hmod._build_id_index = real_build


# --- the same reader against real-world header conventions ---
def test_gwas_catalog_harmonised(tmp_path):
    # GWAS Catalog harmonised TSV: variant_id / base_pair_location / *_allele /
    # standard_error / p_value / effect_allele_frequency.
    path = _write(tmp_path,
        "variant_id\tchromosome\tbase_pair_location\teffect_allele\tother_allele"
        "\tbeta\tstandard_error\tp_value\teffect_allele_frequency\tn\n"
        "rs1\t1\t10000\tA\tG\t0.0321\t0.0067\t1.2e-6\t0.41\t250000\n"
        "rs2\t2\t20000\tT\tC\t-0.0150\t0.0070\t3.2e-2\t0.12\t250000\n")
    ss = read_sumstats(path)
    assert len(ss) == 2
    np.testing.assert_array_equal(ss.id, ["rs1", "rs2"])
    np.testing.assert_array_equal(ss.chrom, ["1", "2"])
    np.testing.assert_array_equal(ss.pos, [10000, 20000])
    np.testing.assert_array_equal(ss.ea, ["A", "T"])
    np.testing.assert_array_equal(ss.oa, ["G", "C"])
    np.testing.assert_allclose(ss.beta, [0.0321, -0.0150])
    np.testing.assert_allclose(ss.se, [0.0067, 0.0070])
    np.testing.assert_allclose(ss.eaf, [0.41, 0.12])
    np.testing.assert_allclose(ss.n_eff, [250000, 250000])


def test_bolt_lmm(tmp_path):
    # BOLT-LMM: ALLELE1 is the effect allele, ALLELE0 the other; P_BOLT_LMM;
    # A1FREQ; INFO; no per-variant N column (passed externally).
    path = _write(tmp_path,
        "SNP\tCHR\tBP\tGENPOS\tALLELE1\tALLELE0\tA1FREQ\tINFO\tBETA\tSE\tP_BOLT_LMM\n"
        "rs10\t1\t500\t0.0\tA\tG\t0.27\t0.98\t0.012\t0.003\t6.1e-5\n"
        "rs11\t1\t900\t0.0\tC\tT\t0.55\t0.91\t-0.008\t0.004\t4.5e-2\n")
    ss = read_sumstats(path, n_eff=400000)
    np.testing.assert_array_equal(ss.ea, ["A", "C"])     # ALLELE1 -> effect
    np.testing.assert_array_equal(ss.oa, ["G", "T"])     # ALLELE0 -> other
    np.testing.assert_allclose(ss.beta, [0.012, -0.008])
    np.testing.assert_allclose(ss.eaf, [0.27, 0.55])
    np.testing.assert_allclose(ss.info, [0.98, 0.91])
    np.testing.assert_allclose(ss.n_eff, [400000, 400000])


def test_metal_lowercase_alleles(tmp_path):
    # METAL: MarkerName / Allele1 / Allele2 / Effect / StdErr / P-value, with
    # lower-case alleles (METAL emits them lower-case) -> must be upper-cased.
    path = _write(tmp_path,
        "MarkerName\tAllele1\tAllele2\tEffect\tStdErr\tP-value\tTotalSampleSize\n"
        "rs5\ta\tg\t0.20\t0.05\t6e-5\t80000\n")
    ss = read_sumstats(path)
    np.testing.assert_array_equal(ss.ea, ["A"])
    np.testing.assert_array_equal(ss.oa, ["G"])
    np.testing.assert_allclose(ss.beta, [0.20])
    np.testing.assert_allclose(ss.n_eff, [80000])


def test_pgc_or_whitespace(tmp_path):
    # PGC-style case/control, whitespace-delimited, effect as an odds ratio.
    path = _write(tmp_path,
        "SNPID CHR POS A1 A2 OR SE P Neff\n"
        "rs7 6 3000 A G 2.0 0.08 1e-9 50000\n")
    ss = read_sumstats(path)
    np.testing.assert_allclose(ss.beta, [math.log(2.0)], atol=1e-9)   # OR -> log OR
    np.testing.assert_allclose(ss.n_eff, [50000])
    np.testing.assert_array_equal(ss.ea, ["A"])


def test_saige_se_from_pvalue(tmp_path):
    # No SE column: derive it from BETA and the p-value (z = Phi^-1(1-p/2)).
    p = 1e-8
    path = _write(tmp_path,
        "SNP,CHR,POS,Allele1,Allele2,BETA,p.value,N\n"
        f"rs9,3,400,A,G,0.10,{p},120000\n")
    ss = read_sumstats(path)
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - p / 2.0)
    np.testing.assert_allclose(ss.se, [0.10 / z], rtol=1e-6)


def test_gzipped_file(tmp_path):
    # Real sumstats are usually shipped gzipped.
    p = tmp_path / "gwas.txt.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("SNP\tA1\tA2\tBETA\tSE\tN\n"
                 "rs1\tA\tG\t0.1\t0.02\t1000\n"
                 "rs2\tC\tT\t-0.2\t0.03\t1000\n")
    ss = read_sumstats(str(p))
    assert len(ss) == 2
    np.testing.assert_allclose(ss.beta, [0.1, -0.2])


def test_detect_columns_on_bolt(tmp_path):
    # The preflight column detector should map a real BOLT header correctly.
    path = _write(tmp_path,
        "SNP\tCHR\tBP\tALLELE1\tALLELE0\tA1FREQ\tBETA\tSE\tP_BOLT_LMM\n"
        "rs1\t1\t1\tA\tG\t0.3\t0.01\t0.002\t1e-3\n")
    header, mapping = detect_columns(path)
    assert mapping["ea"] == "ALLELE1"
    assert mapping["oa"] == "ALLELE0"
    assert mapping["beta"] == "BETA"
    assert mapping["pval"] == "P_BOLT_LMM"
    assert mapping["eaf"] == "A1FREQ"


def test_explicit_column_override(tmp_path):
    # Ambiguous/non-standard header resolved by explicit overrides.
    path = _write(tmp_path,
        "id\tref\talt\tEFF\tSEBETA\tnsamp\n"          # 'ref'/'alt' swapped vs usual
        "rs1\tG\tA\t0.05\t0.01\t1000\n")
    ss = read_sumstats(path, ea="alt", oa="ref", beta="EFF",
                       se="SEBETA", n_eff="nsamp")
    np.testing.assert_array_equal(ss.ea, ["A"])
    np.testing.assert_array_equal(ss.oa, ["G"])
    np.testing.assert_allclose(ss.beta, [0.05])
    np.testing.assert_allclose(ss.n_eff, [1000])


def test_varying_per_variant_sample_size(tmp_path):
    # Meta-analysis sumstats: N differs per SNP and must parse per variant.
    path = _write(tmp_path,
        "SNP\tA1\tA2\tBETA\tSE\tN\n"
        "rs0\tA\tG\t0.05\t0.02\t10000\n"
        "rs1\tC\tT\t0.10\t0.02\t30000\n"
        "rs2\tA\tT\t0.15\t0.02\t60000\n")
    ss = read_sumstats(path)
    np.testing.assert_array_equal(ss.n_eff, [10000, 30000, 60000])
    assert ss.n_eff.min() != ss.n_eff.max()


def test_messy_rows_blank_and_extra_columns(tmp_path):
    # Blank lines tolerated; unknown trailing columns ignored.
    path = _write(tmp_path,
        "SNP\tA1\tA2\tBETA\tSE\tN\tEXTRA\tNOTE\n"
        "rs1\tA\tG\t0.1\t0.02\t1000\t99\thello\n"
        "\n"
        "rs2\tC\tT\t-0.2\t0.03\t1000\t98\tworld\n")
    ss = read_sumstats(path)
    assert len(ss) == 2
    np.testing.assert_array_equal(ss.id, ["rs1", "rs2"])
    np.testing.assert_allclose(ss.beta, [0.1, -0.2])


# --- a UTF-8 BOM must not eat the first column ----------------------------- #

def test_utf8_bom_does_not_disable_the_first_column(tmp_path):
    """Excel and several GWAS portals emit a BOM; it stuck to the first header.

    The token then matched no alias, so `id` was silently never mapped, and
    harmonisation quietly degraded to coordinate-only matching without
    `preflight_prs` flagging it (`id` is not in its required set).
    """
    body = ("SNP\tA1\tA2\tBETA\tSE\tN\n"
            "rs1\tA\tG\t0.01\t0.002\t10000\n"
            "rs2\tC\tT\t-0.02\t0.003\t10000\n")
    plain = tmp_path / "plain.tsv"
    plain.write_text(body, encoding="utf-8")
    bom = tmp_path / "bom.tsv"
    bom.write_text("﻿" + body, encoding="utf-8")
    gz = tmp_path / "bom.tsv.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        fh.write("﻿" + body)

    reference = read_sumstats(plain)
    for path in (bom, gz):
        _, mapping = detect_columns(path)
        assert mapping.get("id") == "SNP", f"{path.name}: id column lost to the BOM"
        got = read_sumstats(path)
        np.testing.assert_array_equal(got.id, reference.id)
        np.testing.assert_allclose(got.beta, reference.beta)


# --- the missing-position sentinel is not a coordinate --------------------- #

def _variants_at(ids, chroms, positions):
    n = len(ids)
    return VariantTable(
        chrom=np.array(chroms, dtype=object),
        id=np.array(ids, dtype=object),
        cm=np.zeros(n),
        pos=np.array(positions, dtype=np.int64),
        a1=np.array(["A"] * n, dtype=object),
        a2=np.array(["G"] * n, dtype=object))


def test_position_zero_is_not_matched_as_a_coordinate(tmp_path):
    """`read_sumstats` maps a blank BP to 0; harmonisation treated it as a locus.

    A sumstats row with no position whose rsID misses the genotype table could
    then be assigned to an unrelated pos-0 genotype variant, and counted as a
    good match -- while genotype_io, pipeline and bgen_io all guard `pos > 0`.
    """
    ss = _write(tmp_path, "SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tN\n"
                          "rsMISSING\t1\t\tA\tG\t0.01\t0.002\t10000\n")
    sumstats = read_sumstats(ss)
    assert int(sumstats.pos[0]) == 0, "fixture assumes the 0 sentinel"

    # A genotype variant that also carries no position, with a matching allele
    # pair and a different ID: the only thing that could join them is the 0.
    variants = _variants_at(["rsOTHER"], ["1"], [0])
    log = harmonize(sumstats, variants).log
    assert log["n_matched"] == 0, "a pos-0 sentinel was used as a real coordinate"
    assert log["n_unmatched"] == 1

def _coverage_fixture(n=500):
    """A reference and a factory for sumstats aligned to it."""
    from allelign.variants import Variants as VariantTable
    from allelign.sumstats import Sumstats

    ids = np.array([f"rs{i}" for i in range(n)])
    chrom = np.full(n, "1")
    pos = np.arange(1000, 1000 + n) * 10
    variants = VariantTable(chrom=chrom, id=ids, cm=np.zeros(n), pos=pos,
                            a1=np.full(n, "A"), a2=np.full(n, "G"),
                            alternate_id=None)

    def make(ids_, chrom_, pos_):
        k = len(ids_)
        return Sumstats(id=np.asarray(ids_), chrom=np.asarray(chrom_),
                        pos=np.asarray(pos_), ea=np.full(k, "A"),
                        oa=np.full(k, "G"), beta=np.full(k, 0.01),
                        se=np.full(k, 0.01), n_eff=np.full(k, 1e5),
                        eaf=np.full(k, 0.3), info=np.full(k, 1.0))

    return variants, ids, chrom, pos, make


def test_reference_coverage_names_the_cause_of_a_thin_alignment():
    """Low coverage still produces numbers, so it has to be diagnosed.

    LDSC keeps the full reference M whatever fraction of it the rows populate,
    so a fit on a sparse corner of the reference reports h2 on a denominator
    the data do not fill. The reason vocabulary is shared with bipred, so a
    CLI run and a SMARTpred run reach the same verdict on the same input.
    """
    from allelign.align import (coverage_message, harmonize,
                                   reference_coverage)

    variants, ids, chrom, pos, make = _coverage_fixture()

    # Same identifiers, every coordinate shifted: a build mismatch. harmonize
    # rejects a known id at a disagreeing locus rather than taking it, so this
    # collapses to zero matches with no other trace.
    shifted = make(ids, chrom, pos + 7)
    cov = reference_coverage(shifted, variants, harmonize(shifted, variants))
    assert cov["reason"] == "build_mismatch"
    assert cov["severe"] is True and cov["build_mismatch_suspected"] is True
    assert cov["n_unmatched_id_elsewhere"] == 500
    message = coverage_message(cov, label="alignment")
    assert "genome-build mismatch" in message and "lift" in message.lower()
    assert "wrong columns" in message
    assert "0 of 500" in message

    # Unknown identifiers *and* unknown loci -- harmonize falls back to
    # position, so changing only the identifiers would still match.
    absent = make([f"rsX{i}" for i in range(500)], np.full(500, "9"), pos * 3)
    cov = reference_coverage(absent, variants, harmonize(absent, variants))
    assert cov["reason"] == "absent_identifiers"
    assert cov["n_unmatched_id_absent"] == 500
    assert "different variant set" in coverage_message(cov)

    # A significant-hits-only deposit: every row correct, few of them.
    sparse = make(ids[:20], chrom[:20], pos[:20])
    cov = reference_coverage(sparse, variants, harmonize(sparse, variants))
    assert cov["reason"] == "sparse_sumstats"
    assert cov["n_matched"] == 20 and cov["frac_of_sumstats"] == 1.0
    assert "hits-only" in coverage_message(cov)

    # Full overlap is silent: no warning to ignore next time.
    full = make(ids, chrom, pos)
    cov = reference_coverage(full, variants, harmonize(full, variants))
    assert cov["reason"] == "adequate_coverage"
    assert cov["diagnosed"] is False
    assert coverage_message(cov) is None

    # Nothing offered cannot be blamed on the reference.
    empty = make([], [], [])
    cov = reference_coverage(empty, variants, harmonize(full, variants))
    cov["n_sumstats_offered"], cov["n_matched"] = 0, 0
    cov["frac_of_reference"] = 0.0
    cov = reference_coverage(empty, variants, harmonize(empty, variants))
    assert cov["reason"] == "no_rows_offered"
    assert "no rows reached alignment" in coverage_message(cov)


def test_coverage_thresholds_are_the_shared_ones():
    """These numbers are the contract bipred and SMARTpred also key on."""
    import allelign.align as H

    assert H.REFERENCE_COVERAGE_WARN == 0.5
    assert H.SEVERE_REFERENCE_COVERAGE == 0.1
    assert H.BUILD_MISMATCH_FRACTION == 0.2


# --- Numba tokenizer vs the reference row loop -----------------------------

import allelign.sumstats as _ss_mod                                # noqa: E402

_SS_FIELDS = ("id", "chrom", "pos", "ea", "oa", "beta", "se", "n_eff",
              "eaf", "info")


def _read_reference(monkeypatch, path, **kw):
    with monkeypatch.context() as m:
        m.setattr(_ss_mod, "_jit_kernels", lambda: None)
        return read_sumstats(path, **kw)


def _assert_identical(a, b):
    for f in _SS_FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        assert x.dtype == y.dtype and x.shape == y.shape, f
        if x.dtype == float:
            # bit-for-bit, NaN included
            np.testing.assert_array_equal(x.view(np.int64), y.view(np.int64),
                                          err_msg=f)
        else:
            assert list(x) == list(y), f
    # chrom/ea/oa share one pool: a distinct value is a single object.
    for ss in (a, b):
        objs = {}
        for f in ("chrom", "ea", "oa"):
            for v in getattr(ss, f):
                objs.setdefault(v, set()).add(id(v))
        assert all(len(s) == 1 for s in objs.values())


def _messy_body(rng, rows, delim, nl):
    toks = ["0.1", "-2.5e-3", "1e-300", "3.2E+04", "NA", "", ".", "nan",
            "-inf", "1_000", " 4.5", "0.12345678901234567", "+.5", "5.",
            "-0", "1e400", "123456789012345"]
    out = []
    for i in range(rows):
        num = [f"{rng.normal():.6g}" if rng.random() < .7
               else toks[rng.integers(len(toks))] for _ in range(5)]
        if delim == " ":
            num = [t.strip() or "NA" for t in num]
        out.append(delim.join(
            [f"rs{i}", str(rng.integers(1, 23)), str(rng.integers(1, 10**8)),
             "ACGTX"[rng.integers(5)], ["a", "g", "TTTTTTTTA"][i % 3]]
            + num + [str(rng.integers(1000, 9000))]))
    return nl.join(out) + nl


@pytest.mark.numba
@pytest.mark.parametrize("delim,nl,gz,effect", [
    ("\t", "\n", False, "BETA"), (",", "\r\n", True, "OR"),
    (" ", "\n", False, "BETA"), ("\t", "\r", True, "OR")])
def test_jit_parser_matches_reference_loop(tmp_path, monkeypatch, delim, nl,
                                           gz, effect):
    if _ss_mod._jit_kernels() is None:
        pytest.skip("Numba JIT unavailable")
    rng = np.random.default_rng(0)
    header = delim.join(["SNP", "CHR", "BP", "A1", "A2", effect, "SE", "P",
                         "EAF", "INFO", "N"])
    body = _messy_body(rng, 3000, delim, nl)
    # A ragged row, a whitespace-only line, a blank line and a non-ASCII
    # allele: their blocks must go through the reference loop.
    body += nl.join(["rsR" + delim + "1", "   ", "",
                     delim.join(["rsU", "1", "5", "ß", "A", "0.1", "NA",
                                 "0.5", "0.2", "1", "100"])]) + nl
    body += _messy_body(rng, 3000, delim, nl)
    text = "﻿" + header + nl + body
    path = tmp_path / ("ss.tsv.gz" if gz else "ss.tsv")
    data = text.encode("utf-8")
    path.write_bytes(gzip.compress(data) if gz else data)
    ref = _read_reference(monkeypatch, str(path))
    monkeypatch.setattr(_ss_mod, "_JIT_MIN_CHARS", 0)
    for block in (97, 4096, 1 << 22):
        monkeypatch.setattr(_ss_mod, "_BLOCK_CHARS", block)
        _assert_identical(ref, read_sumstats(str(path)))
        _assert_identical(_read_reference(monkeypatch, str(path), n_eff=77),
                          read_sumstats(str(path), n_eff=77))


@pytest.mark.numba
def test_jit_parser_defers_errors_to_reference_loop(tmp_path, monkeypatch):
    # An int64-overflowing position must raise exactly as the reference does.
    if _ss_mod._jit_kernels() is None:
        pytest.skip("Numba JIT unavailable")
    monkeypatch.setattr(_ss_mod, "_JIT_MIN_CHARS", 0)
    path = _write(tmp_path, "SNP\tBP\tA1\tA2\tBETA\tN\n"
                            "rs1\t100\tA\tG\t0.1\t10\n"
                            "rs2\t1e30\tA\tG\t0.1\t10\n")
    with pytest.raises(OverflowError):
        read_sumstats(path)


@pytest.mark.numba
def test_jit_decimal_kernel_is_bit_identical_to_float():
    kernels = _ss_mod._jit_kernels()
    if kernels is None:
        pytest.skip("Numba JIT unavailable")
    rng = np.random.default_rng(3)
    toks = []
    for _ in range(20000):
        t = rng.choice(["", "-", "+"]) + "".join(
            rng.choice(list("0123456789"), rng.integers(0, 18)))
        if rng.random() < .7:
            t += "." + "".join(rng.choice(list("0123456789"),
                                          rng.integers(0, 18)))
        if rng.random() < .5:
            t += rng.choice(["e", "E"]) + rng.choice(["", "-", "+"]) \
                + str(rng.integers(0, 40))
        toks.append(t)
    toks += ["1e", ".", "-", "e5", "1..2", "0x1", "1_0", "nan", " 1", "1e22",
             "1e23", "123456789012345", "1234567890123456", "0e999", "-0.0"]
    text = "\n".join(toks) + "\n"
    b = np.frombuffer(text.encode(), dtype=np.uint8)
    lens = np.array([len(t) for t in toks])
    ends = np.cumsum(lens + 1) - 1
    starts = ends - lens
    out = np.empty(len(toks))
    ok = np.empty(len(toks), dtype=np.bool_)
    kernels[1](b, starts, ends, _ss_mod._POW10, out, ok)
    ref = np.array([_ss_mod._to_float(t) for t in toks])
    assert ok.sum() > len(toks) // 4
    np.testing.assert_array_equal(out[ok].view(np.int64),
                                  ref[ok].view(np.int64))


def test_reader_without_numba_uses_reference_loop(tmp_path, monkeypatch):
    import allelign._numba as numba_shim
    monkeypatch.setattr(numba_shim, "HAVE_NUMBA", False)
    monkeypatch.setattr(_ss_mod, "_KERNELS", None)
    monkeypatch.setattr(_ss_mod, "_JIT_MIN_CHARS", 0)

    def boom(*a, **k):
        raise AssertionError("JIT path used without Numba")

    monkeypatch.setattr(_ss_mod, "_parse_rows_jit", boom)
    path = _write(tmp_path, "SNP\tA1\tA2\tBETA\tSE\tN\nrs1\ta\tg\t0.1\tNA\t9\n")
    ss = read_sumstats(path)
    assert list(ss.ea) == ["A"] and np.isnan(ss.se[0])
    assert _ss_mod._jit_kernels() is None


def test_sumstats_from_a_pipe_keeps_the_single_pass_loop(tmp_path, monkeypatch):
    # The compiled path replays a failed parse by reopening the file, which a
    # FIFO cannot serve; a stream must use the reference loop and read all rows.
    import os
    import threading
    import allelign.sumstats as ss
    if not hasattr(os, "mkfifo"):
        pytest.skip("no FIFOs on this platform")
    rows = ["SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tN"] + [
        f"rs{i}\t1\t{i + 1}\tA\tG\t0.01\t0.02\t5000" for i in range(20000)]
    text = "\n".join(rows) + "\n"
    plain = tmp_path / "plain.tsv"
    plain.write_text(text)
    fifo = tmp_path / "stream.tsv"
    monkeypatch.setattr(ss, "_JIT_MIN_CHARS", 1)
    calls = []
    real = ss._parse_rows_jit
    monkeypatch.setattr(ss, "_parse_rows_jit",
                        lambda *a, **k: calls.append(1) or real(*a, **k))

    def feed():
        with open(fifo, "w") as out:
            out.write(text)

    os.mkfifo(fifo)
    writer = threading.Thread(target=feed)
    writer.start()
    streamed = ss.read_sumstats(str(fifo))
    writer.join()
    assert not calls
    reference = ss.read_sumstats(str(plain))
    assert len(streamed) == len(reference) == 20000
    np.testing.assert_array_equal(streamed.beta, reference.beta)


def test_sumstats_array_contract_includes_every_column():
    assert {f.name for f in fields(Sumstats)} - {'parse_log'} == set(_SUMSTATS_FIELDS)


@pytest.mark.parametrize('jit', [False, True])
def test_se_reconstruction_provenance_survives_chunks_and_subset(tmp_path, monkeypatch, jit):
    import allelign.sumstats as ssmod
    from allelign.qc import qc_sumstats
    path = tmp_path / 'gwas.tsv'
    path.write_text('SNP\tA1\tA2\tBETA\tSE\tP\tN\n'
                    'a\tA\tG\t0.1\tNA\t0.05\t1000\n'
                    'b\tA\tG\t0.1\t0.01\t0.05\t1000\n'
                    'c\tA\tG\t0\tNA\t1\t1000\n')
    if not jit:
        monkeypatch.setattr(ssmod, '_jit_kernels', lambda: None)
    else:
        monkeypatch.setattr(ssmod, '_JIT_MIN_CHARS', 0)
        monkeypatch.setattr(ssmod, '_BLOCK_CHARS', 40)
    ss = ssmod.read_sumstats(path)
    assert ss.parse_log['n_input'] == 3 and ss.parse_log['n_se_from_pval'] == 1
    assert ss.parse_log['n_eff_source'] == 'column:n'
    sub = ss.subset(np.array([True, True, False]))
    assert sub.parse_log == ss.parse_log
    _, log = qc_sumstats(sub)
    assert log['sumstats_parse']['n_se_from_pval'] == 1


def _harmonize_tables():
    v = VariantTable(chrom=np.array(['1']), id=np.array(['rs1']),
                     cm=np.zeros(1), pos=np.array([100]), a1=np.array(['A']),
                     a2=np.array(['G']))
    s = Sumstats(id=np.array(['rs1']), chrom=np.array(['1']), pos=np.array([100]),
                 ea=np.array(['A']), oa=np.array(['G']), beta=np.array([.1]),
                 se=np.array([.02]), n_eff=np.array([1000.]),
                 eaf=np.array([.2]), info=np.array([1.]))
    return s, v


@pytest.mark.parametrize('field,value', [
    ('a1', 'C'), ('a2', 'C'), ('id', 'rs2'), ('pos', 101), ('chrom', '2'),
    ('alternate_id', 'rs2')])
def test_mutated_metadata_matches_a_fresh_table(field, value):
    s, v = _harmonize_tables()
    if field == 'alternate_id':
        v.alternate_id = v.id.copy()
        v.id[:] = 'ref'
    assert len(harmonize(s, v)) == 1
    getattr(v, field)[:] = value
    if field in ('id', 'alternate_id'):
        s.id[:] = value
        s.pos[:] = 0  # identifier lookup, without coordinate fallback
    actual, expected = harmonize(s, v), harmonize(s, v.subset(np.arange(1)))
    np.testing.assert_array_equal(actual.var_index, expected.var_index)
    np.testing.assert_array_equal(actual.beta, expected.beta)
    assert actual.log == expected.log


def test_swapping_mutable_alleles_reverses_the_effect():
    s, v = _harmonize_tables()
    assert harmonize(s, v).beta[0] == .1
    v.a1[:], v.a2[:] = 'G', 'A'
    result = harmonize(s, v)
    assert result.flipped[0] and result.beta[0] == -.1


def test_frozen_cache_reuses_storage_and_invalidates_on_thaw_or_replacement():
    s, v = _harmonize_tables()
    columns = (v.id, v.chrom, v.pos, v.a1, v.a2)
    assert v.freeze() is v
    assert all(a is b for a, b in zip(columns, (v.id, v.chrom, v.pos, v.a1, v.a2)))
    entry = variant_indices(v)
    assert variant_indices(v) is entry
    with pytest.raises(ValueError, match='read-only'):
        v.a1[:] = 'G'
    v.a1.flags.writeable = v.a2.flags.writeable = True
    v.a1[:], v.a2[:] = 'G', 'A'
    assert harmonize(s, v).beta[0] == -.1
    v.freeze()
    newer = variant_indices(v)
    assert newer is not entry and variant_indices(v) is newer
    v.pos = np.array([200])
    assert len(harmonize(s, v)) == 0


def test_readonly_view_of_writable_metadata_is_not_cached():
    s, v = _harmonize_tables()
    owner = v.a1
    v.a1 = owner.view()
    v.freeze()
    # Changing an ordinary writable owner must not bypass the view's flags.
    assert harmonize(s, v).beta[0] == .1
    owner[:] = 'C'
    assert len(harmonize(s, v)) == 0


def test_refreezing_edited_metadata_invalidates_without_an_intermediate_match():
    s, v = _harmonize_tables()
    v.freeze()
    assert harmonize(s, v).beta[0] == .1
    v.a1.flags.writeable = v.a2.flags.writeable = True
    v.a1[:], v.a2[:] = 'G', 'A'
    v.freeze()
    assert harmonize(s, v).beta[0] == -.1
    v.a1, v.a2 = np.array(['A']), np.array(['G'])
    v.a1.flags.writeable = v.a2.flags.writeable = False
    assert harmonize(s, v).beta[0] == .1


def test_saige_effect_and_frequency_are_for_allele2(tmp_path):
    from allelign.sumstats import read_sumstats
    path = tmp_path / "saige.tsv"
    path.write_text("CHR POS SNPID Allele1 Allele2 AF_Allele2 BETA SE N\n"
                    "01 101 rs1 A C .2 .3 .1 10000\n")
    ss = read_sumstats(path)
    assert (ss.ea[0], ss.oa[0], ss.eaf[0], ss.beta[0]) == ("C", "A", .2, .3)
    with pytest.raises(ValueError, match="same column"):
        read_sumstats(path, ea="Allele1")


@pytest.mark.parametrize("label, expected", [("01", "1"), ("chr002", "2"),
                                            ("023", "X"), ("026", "MT")])
def test_zero_padded_chromosomes(label, expected):
    from allelign.alleles import normalize_chrom as _norm_chrom
    assert _norm_chrom(label) == expected


@pytest.mark.parametrize("p", ["1e-320", "5e-324", "1e-1000"])
def test_extreme_p_values_reconstruct_se_without_underflow(tmp_path, p):
    from allelign.sumstats import read_sumstats
    from scipy.special import log_ndtr
    from decimal import Decimal
    path = tmp_path / "tiny.tsv"
    path.write_text(f"SNP A1 A2 BETA P N\nrs1 A C .2 {p} 10000\n")
    ss = read_sumstats(path)
    assert np.isfinite(ss.se[0]) and ss.se[0] > 0
    assert log_ndtr(-.2 / ss.se[0]) == pytest.approx(float(Decimal(p).ln()) - np.log(2), abs=1e-4)
    assert ss.parse_log["n_se_from_pval"] == 1


def test_odds_ratio_beta_warning_never_guesses_a_transformation(tmp_path):
    from allelign.sumstats import read_sumstats
    path = tmp_path / "odds.tsv"
    odds = np.linspace(.9, 1.1, 30)
    path.write_text("SNP A1 A2 BETA SE N\n" + "".join(
        f"rs{i} A C {x} .05 5000\n" for i, x in enumerate(odds)))
    with pytest.warns(RuntimeWarning, match="odds ratios"):
        original = read_sumstats(path)
    np.testing.assert_allclose(original.beta, odds)
    corrected = read_sumstats(path, **{"or": "BETA"})
    np.testing.assert_allclose(corrected.beta, np.log(odds))
