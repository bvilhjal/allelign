"""Formats and N rules beyond ldpred3's reader."""

import math
import warnings

import numpy as np
import pytest

import allelign.sumstats as S
from allelign import (ColumnMappingError, inspect_columns, n_eff_case_control,
                      read_sumstats, write_sumstats)


def _write(tmp_path, text, name="ss.txt"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


@pytest.fixture(params=[False, True], ids=["reference", "jit"])
def jit(request, monkeypatch):
    """Run each test through the reference loop and the Numba block parser."""
    if request.param:
        if S._jit_kernels() is None:
            pytest.skip("Numba JIT unavailable")
        monkeypatch.setattr(S, "_JIT_MIN_CHARS", 0)
    else:
        monkeypatch.setattr(S, "_jit_kernels", lambda: None)
    return request.param


def test_plink2_a1_is_resolved_against_ref_and_alt(tmp_path):
    path = _write(tmp_path,
                  "#CHROM\tPOS\tID\tREF\tALT\tA1\tA1_FREQ\tOBS_CT\tBETA\tSE\tP\n"
                  "1\t100\trs1\tA\tG\tG\t0.2\t1000\t0.1\t0.01\t1e-5\n"
                  "1\t200\trs2\tC\tT\tC\t0.7\t1000\t-0.2\t0.01\t1e-9\n")
    ss = read_sumstats(path)
    assert list(ss.ea) == ["G", "C"]
    assert list(ss.oa) == ["A", "T"]           # the non-tested allele per row
    np.testing.assert_allclose(ss.eaf, [0.2, 0.7])
    assert ss.parse_log["layout"]["plink2_alt"] == "ALT"


def test_plink2_omitted_column_is_the_other_allele(tmp_path, jit):
    path = _write(tmp_path,
                  "#CHROM\tPOS\tID\tREF\tALT\tA1\tOMITTED\tOBS_CT\tOR\tLOG(OR)_SE\tP\n"
                  "1\t100\trs1\tA\tG\tA\tG\t1000\t1.2\t0.05\t1e-5\n")
    ss = read_sumstats(path)
    assert list(ss.ea) == ["A"] and list(ss.oa) == ["G"]
    np.testing.assert_allclose(ss.beta, [math.log(1.2)])
    np.testing.assert_allclose(ss.se, [0.05])


def test_regenie_genpos_and_log10p(tmp_path, jit):
    path = _write(tmp_path,
                  "CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ INFO N TEST BETA SE CHISQ LOG10P\n"
                  "1 1000 rs1 A G 0.3 0.9 5000 ADD 0.05 NA 9 5\n"
                  "2 2000 rs2 C T 0.4 0.95 5000 ADD -0.02 0.004 25 6.2\n")
    ss = read_sumstats(path)
    assert list(ss.pos) == [1000, 2000]
    assert list(ss.ea) == ["G", "T"] and list(ss.oa) == ["A", "C"]
    z = S._z_from_log10p("5")
    assert ss.se[0] == pytest.approx(0.05 / z)
    assert ss.se[1] == 0.004
    assert ss.parse_log["n_se_from_pval"] == 1


def test_bolt_keeps_bp_over_genetic_genpos(tmp_path):
    path = _write(tmp_path,
                  "SNP CHR BP GENPOS ALLELE1 ALLELE0 A1FREQ BETA SE P_BOLT_LMM\n"
                  "rs1 1 12345 0.01 A G 0.3 0.1 0.01 1e-8\n")
    assert list(read_sumstats(path, n_eff=1000).pos) == [12345]


def test_huge_log10p_uses_the_tail_solver():
    assert S._z_from_log10p("500") == pytest.approx(S._two_sided_z("1e-500"),
                                                    rel=1e-12)
    assert S._z_from_log10p("0") != S._z_from_log10p("0")      # NaN
    assert S._z_from_log10p("NA") != S._z_from_log10p("NA")


def test_gwas_catalog_harmonised_uses_the_hm_set(tmp_path, jit):
    # Submitted columns on another build with the effect allele swapped.
    path = _write(tmp_path,
                  "chromosome\tbase_pair_location\teffect_allele\tother_allele\t"
                  "beta\tstandard_error\tp_value\tn\thm_rsid\thm_chrom\thm_pos\t"
                  "hm_effect_allele\thm_other_allele\thm_beta\t"
                  "hm_effect_allele_frequency\n"
                  "1\t752721\tG\tA\t0.1\t0.01\t1e-5\t900\trs3131972\t1\t817341\t"
                  "A\tG\t-0.1\t0.2\n")
    ss = read_sumstats(path)
    assert list(ss.id) == ["rs3131972"] and list(ss.pos) == [817341]
    assert list(ss.ea) == ["A"] and list(ss.oa) == ["G"]
    np.testing.assert_allclose(ss.beta, [-0.1])
    np.testing.assert_allclose(ss.eaf, [0.2])
    assert ss.parse_log["layout"]["gwas_catalog_harmonised"] is True
    # Without the harmonised frequency the submitted one may count the
    # submitted (here swapped) effect allele: it is dropped, not misread.
    no_hm_eaf = _write(tmp_path,
                       "chromosome\tbase_pair_location\teffect_allele\tother_allele\t"
                       "effect_allele_frequency\tbeta\tstandard_error\tn\thm_rsid\t"
                       "hm_pos\thm_effect_allele\thm_other_allele\thm_beta\n"
                       "1\t752721\tG\tA\t0.8\t0.1\t0.01\t900\trs3131972\t817341\t"
                       "A\tG\t-0.1\n", name="no_hm_eaf.tsv")
    ss = read_sumstats(no_hm_eaf)
    assert list(ss.ea) == ["A"] and np.isnan(ss.eaf[0])
    assert "eaf_dropped" in ss.parse_log["layout"]


def test_harmonised_alleles_without_harmonised_effect_use_the_submitted_set(tmp_path):
    # The submitted beta counts the submitted effect allele; pairing it with
    # the (possibly swapped) hm_ alleles would flip signs.
    path = _write(tmp_path,
                  "rsid\tchromosome\tbase_pair_location\teffect_allele\t"
                  "other_allele\tbeta\tse\tn\thm_rsid\thm_chrom\thm_pos\t"
                  "hm_effect_allele\thm_other_allele\n"
                  "rs1\t1\t752721\tG\tA\t0.1\t0.01\t900\trs1\t1\t817341\tA\tG\n")
    ss = read_sumstats(path)
    assert list(ss.ea) == ["G"] and list(ss.pos) == [752721]
    assert ss.beta[0] == 0.1
    assert "gwas_catalog_harmonised_ignored" in ss.parse_log["layout"]
    only_hm = _write(tmp_path, "hm_rsid\thm_effect_allele\thm_other_allele\t"
                     "beta\tse\tn\nrs1\tA\tG\t0.1\t0.01\t900\n", name="b.txt")
    with pytest.raises(ValueError, match="allele column"):
        read_sumstats(only_hm)
    with pytest.raises(ColumnMappingError):
        read_sumstats(only_hm, ea="nope")


def test_daner_control_frequency_and_neff(tmp_path, jit):
    path = _write(tmp_path,
                  "CHR SNP BP A1 A2 FRQ_A_3000 FRQ_U_5000 INFO OR SE P Nca Nco Neff\n"
                  "1 rs1 100 A G 0.31 0.30 0.95 1.1 0.02 0.01 3000 5000 7400\n")
    ss = read_sumstats(path)
    np.testing.assert_allclose(ss.eaf, [0.30])
    np.testing.assert_allclose(ss.n_eff, [7400])          # explicit Neff wins
    assert ss.parse_log["n_eff_source"] == "column:neff"
    assert ss.parse_log["layout"]["eaf_from"] == "FRQ_U_5000"


def test_case_control_columns_give_effective_n(tmp_path, jit):
    path = _write(tmp_path,
                  "SNP\tA1\tA2\tBETA\tSE\tN\tN_CASES\tN_CONTROLS\n"
                  "rs1\tA\tG\t0.1\t0.01\t11000\t1000\t10000\n"
                  "rs2\tA\tG\t0.1\t0.01\t11000\tNA\t10000\n")
    with pytest.warns(RuntimeWarning, match="case and control"):
        ss = read_sumstats(path)
    assert ss.n_eff[0] == pytest.approx(4 / (1 / 1000 + 1 / 10000))
    assert np.isnan(ss.n_eff[1])
    assert ss.parse_log["n_eff_source"] == "case_control"
    # An explicit N column override keeps the total.
    assert read_sumstats(path, n_eff="N").n_eff[0] == 11000


def test_scalar_case_control_counts(tmp_path):
    path = _write(tmp_path, "SNP\tA1\tA2\tBETA\tSE\nrs1\tA\tG\t0.1\t0.01\n")
    ss = read_sumstats(path, n_cases=2000, n_controls=8000)
    assert ss.n_eff[0] == pytest.approx(n_eff_case_control(2000, 8000))
    with pytest.raises(ValueError):
        read_sumstats(path, n_cases=2000)
    with pytest.raises(ValueError):
        read_sumstats(path)
    assert np.isnan(read_sumstats(path, require_n=False).n_eff[0])


def test_neale_variant_strings(tmp_path):
    path = _write(tmp_path,
                  "variant\tminor_allele\tminor_AF\tlow_confidence_variant\t"
                  "n_complete_samples\tAC\tytx\tbeta\tse\ttstat\tpval\n"
                  "1:15791:C:T\tT\t0.01\ttrue\t361194\t5\t3\t0.02\t0.01\t2\t0.04\n"
                  "X:2700157:G:A\tA\t0.2\tfalse\t361194\t5\t3\t-0.01\t0.003\t-3\t0.001\n")
    ss = read_sumstats(path)
    assert list(ss.chrom) == ["1", "X"] and list(ss.pos) == [15791, 2700157]
    assert list(ss.ea) == ["T", "A"] and list(ss.oa) == ["C", "G"]
    assert np.isnan(ss.eaf).all()              # minor_AF is not an EAF
    np.testing.assert_allclose(ss.n_eff, [361194, 361194])


def test_finngen_layout(tmp_path, jit):
    path = _write(tmp_path,
                  "#chrom\tpos\tref\talt\trsids\tnearest_genes\tpval\tmlogp\tbeta\t"
                  "sebeta\taf_alt\taf_alt_cases\taf_alt_controls\n"
                  "1\t13668\tG\tA\trs2691328\tOR4F5\t0.5\t0.3\t0.01\t0.02\t0.006\t"
                  "0.006\t0.006\n")
    ss = read_sumstats(path, n_cases=10000, n_controls=300000)
    assert list(ss.ea) == ["A"] and list(ss.oa) == ["G"]
    assert list(ss.id) == ["rs2691328"]
    np.testing.assert_allclose(ss.eaf, [0.006])


def test_write_sumstats_round_trips_bit_for_bit(tmp_path):
    path = _write(tmp_path,
                  "SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tN\tEAF\tINFO\n"
                  "rs1\t1\t100\ta\tg\t0.1234567890123456\t0.01\t1000\tNA\t0.9\n"
                  "\t2\t200\tC\tT\t-3e-300\t1e-5\t5e4\t0.25\tNA\n")
    ss = read_sumstats(path)
    out = tmp_path / "out.tsv.gz"
    write_sumstats(out, ss)
    back = read_sumstats(out)
    for f in ("pos", "ea", "oa", "chrom"):
        assert list(getattr(back, f)) == list(getattr(ss, f)), f
    assert list(back.id) == ["rs1", "."]
    for f in ("beta", "se", "n_eff", "eaf", "info"):
        np.testing.assert_array_equal(getattr(back, f).view(np.int64),
                                      getattr(ss, f).view(np.int64), err_msg=f)


def test_inspect_columns_reports_layout_and_gaps(tmp_path):
    path = _write(tmp_path, "SNP\tA1\tBETA\tN\nrs1\tA\t0.1\t9\n")
    info = inspect_columns(path)
    assert set(info["missing"]) == {"oa", "se/pval"}
    assert info["n_source"] == "column:n"


def test_or_beta_warning_still_fires(tmp_path):
    rows = "".join(f"rs{i}\tA\tG\t{1 + 0.01 * i}\t0.01\t100\n" for i in range(30))
    path = _write(tmp_path, "SNP\tA1\tA2\tBETA\tSE\tN\n" + rows)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ss = read_sumstats(path)
    assert ss.parse_log.get("suspected_odds_ratio_in_beta") is True
    assert any("odds ratios" in str(w.message) for w in caught)
