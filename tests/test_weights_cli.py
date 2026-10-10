"""Weights, PGS Catalog files, reference readers, build checks and the CLI."""

import json
import math
import warnings

import numpy as np
import pytest

from allelign import (BuildMismatchError, Variants, check_build, harmonize,
                      harmonize_weights, normalize_build, read_bim, read_pvar,
                      read_scoring_file, read_sumstats, read_variants,
                      read_weights, reanchor_on_identifier, WeightsTable,
                      write_pgs_scoring_file, write_weights)
from allelign.cli import main
from allelign.sumstats import Sumstats


def _weights(tmp_path, n=40, sd_source="empirical", seed=0):
    rng = np.random.default_rng(seed)
    path = tmp_path / "w.tsv"
    write_weights(path, id=[f"rs{i}" for i in range(n)], chrom=["1"] * n,
                  pos=np.arange(1, n + 1) * 100, effect_allele=["A"] * n,
                  other_allele=["G"] * n, weight=rng.normal(0, 0.05, n),
                  af=rng.uniform(0.05, 0.95, n), sd=rng.uniform(0.2, 0.7, n),
                  sd_source=sd_source)
    return path


def test_pgs_export_differs_from_frozen_score_by_one_constant(tmp_path):
    wt = read_weights(_weights(tmp_path))
    out = tmp_path / "s.txt"
    write_pgs_scoring_file(out, wt, genome_build="GRCh37", pgs_id="T1")
    sf = read_scoring_file(out)
    assert sf.meta["genome_build"] == "GRCh37" and sf.pgs_id == "T1"
    assert sf.meta["sd_source"] == "empirical"
    rng = np.random.default_rng(1)
    g = rng.binomial(2, wt.af_ref, (200, len(wt))).astype(float)
    frozen = ((g - 2 * wt.af_ref) / wt.sd_ref) @ wt.weight
    raw = g @ sf.weight
    diff = raw - frozen
    assert np.ptp(diff) < 1e-9 * np.abs(frozen).max()
    assert diff.mean() == pytest.approx(sf.weight @ (2 * wt.af_ref))


def test_pgs_export_without_sd_source_says_unspecified(tmp_path):
    path = _weights(tmp_path)
    text = path.read_text().replace("\tSD_SOURCE", "")
    lines = [line.rsplit("\t", 1)[0] if i else line
             for i, line in enumerate(text.splitlines())]
    path.write_text("\n".join(lines) + "\n")
    with pytest.warns(UserWarning, match="SD_SOURCE"):
        wt = read_weights(path)
    assert wt.sd_source is None
    with pytest.warns(RuntimeWarning, match="unspecified"):
        write_pgs_scoring_file(tmp_path / "s.txt", wt, genome_build="GRCh38")
    assert "#sd_source=unspecified" in (tmp_path / "s.txt").read_text()


def test_read_scoring_file_rules(tmp_path):
    path = tmp_path / "PGS000001_hmPOS_GRCh38.txt"
    path.write_text(
        "###PGS CATALOG SCORING FILE\n#pgs_id=PGS000001\n#weight_type=OR\n"
        "#genome_build=GRCh37\n#HmPOS_build=GRCh38\n"
        "rsID\tchr_name\tchr_position\teffect_allele\tother_allele\t"
        "effect_weight\tis_dominant\thm_rsID\thm_chr\thm_pos\thm_inferOtherAllele\n"
        "rs1\t1\t100\tA\tG\t1.5\tFALSE\trs1\t1\t1100\t\n"
        "rs2\t1\t200\tC\t\t0.8\tFALSE\trs2\t1\t1200\tT\n"
        "rs3\t1\t300\tA\tG\t2.0\tTRUE\trs3\t1\t1300\t\n"
        "rs4\t1\t400\tA\tG\tNA\tFALSE\trs4\t1\t1400\t\n")
    sf = read_scoring_file(path)
    assert list(sf.pos) == [1100, 1200]
    assert list(sf.oa) == ["G", "T"]
    np.testing.assert_allclose(sf.weight, [math.log(1.5), math.log(0.8)])
    assert sf.log["n_non_additive"] == 1 and sf.log["n_unparsable_weight"] == 1
    assert sf.log["harmonized_columns"] and sf.genome_build == "GRCh38"
    bad = tmp_path / "bad.txt"
    bad.write_text("#weight_type=OR\nrsID\teffect_allele\teffect_weight\nrs1\tA\t-1\n")
    with pytest.raises(ValueError, match="ratios"):
        read_scoring_file(bad)


def test_harmonize_weights_flips_frequencies():
    wt = WeightsTable(
        id=np.array(["rs1", "rs2"], dtype=object),
        chrom=np.array(["1", "1"], dtype=object), pos=np.array([100, 200]),
        a1=np.array(["A", "G"], dtype=object), a2=np.array(["G", "A"], dtype=object),
        weight=np.array([0.3, 0.4]), af_ref=np.array([0.2, 0.9]),
        sd_ref=np.array([0.5, 0.4]), sd_source="hwe")
    ref = Variants(chrom=np.array(["1", "1", "1"], dtype=object),
                   id=np.array(["rs1", "rs2", "rs9"], dtype=object),
                   pos=np.array([100, 200, 300]),
                   a1=np.array(["A", "A", "C"], dtype=object),
                   a2=np.array(["G", "G", "T"], dtype=object))
    h = harmonize_weights(wt, ref)
    np.testing.assert_array_equal(h.var_index, [0, 1])
    np.testing.assert_allclose(h.weight, [0.3, -0.4])
    np.testing.assert_allclose(h.af, [0.2, 0.1])
    np.testing.assert_allclose(h.sd, [0.5, 0.4])
    assert h.log["weight_mass_matched"] == pytest.approx(1.0)


def test_reference_readers(tmp_path):
    bim = tmp_path / "r.bim"
    bim.write_text("1\trs1\t0\t100\tA\tG\nchr2\trs2\t0.5\t200\tc\tt\n")
    v = read_bim(bim)
    assert list(v.a1) == ["A", "C"] and list(v.pos) == [100, 200]
    pvar = tmp_path / "r.pvar"
    pvar.write_text("##fileformat\n#CHROM\tPOS\tID\tREF\tALT\n"
                    "1\t100\trs1\tG\tA\n1\t150\trs5\tG\tA,T\n")
    with pytest.warns(RuntimeWarning, match="multi-allelic"):
        p = read_pvar(pvar)
    assert list(p.a1) == ["A"] and list(p.a2) == ["G"]
    assert len(read_variants(bim)) == 2
    with pytest.raises(ValueError):
        read_variants(tmp_path / "r.txt")


def test_build_check():
    assert normalize_build("hg19") == "GRCh37"
    assert check_build("b38", "GRCh38")["status"] == "match"
    assert check_build(None, "GRCh38")["status"] == "unchecked"
    with pytest.raises(BuildMismatchError, match="liftOver"):
        check_build("GRCh38", "hg19")
    with pytest.raises(ValueError):
        normalize_build("hg17")


def test_reanchor_on_identifier_moves_and_drops():
    ref = Variants(chrom=np.array(["1", "1", "2"], dtype=object),
                   id=np.array(["rs1", "rs2", "rs2"], dtype=object),
                   pos=np.array([752721, 900, 50]),
                   a1=np.array(["A", "A", "A"], dtype=object),
                   a2=np.array(["G", "G", "G"], dtype=object))
    ss = Sumstats(id=np.array(["rs1", "rs2", "rsX", ""], dtype=object),
                  chrom=np.array(["1"] * 4, dtype=object),
                  pos=np.array([817341, 1, 2, 3]),
                  ea=np.full(4, "A", dtype=object), oa=np.full(4, "G", dtype=object),
                  beta=np.full(4, 0.1), se=np.full(4, 0.01),
                  n_eff=np.full(4, 1e4), eaf=np.full(4, np.nan),
                  info=np.full(4, np.nan))
    assert len(harmonize(ss, ref)) == 0          # GRCh38 coordinates: no match
    with pytest.warns(RuntimeWarning, match="re-anchoring"):
        out, log, keep = reanchor_on_identifier(ss, ref, return_keep=True)
    assert list(keep) == [True, False, False, False]
    assert log["n_moved"] == 1 and log["n_dropped_ambiguous_locus"] == 1
    assert list(out.pos) == [752721]
    assert len(harmonize(out, ref)) == 1


def _gwas(tmp_path):
    path = tmp_path / "gwas.tsv"
    path.write_text("SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tN\tEAF\n"
                    "rs1\t1\t100\tG\tA\t0.1\t0.01\t1000\t0.8\n"
                    "rs2\t1\t200\tC\tT\t0.2\t0.01\t1000\t0.3\n"
                    "rs3\t1\t300\tA\tT\t0.3\t0.01\t1000\t0.3\n"
                    "rs4\t1\t400\tA\tG\tNA\t0.01\t1000\t0.3\n")
    bim = tmp_path / "ref.bim"
    bim.write_text("1\trs1\t0\t100\tA\tG\n1\trs2\t0\t200\tC\tT\n"
                   "1\trs3\t0\t300\tA\tT\n1\trs4\t0\t400\tA\tG\n")
    return path, bim


def test_cli_inspect_clean_align_export(tmp_path, capsys):
    gwas, bim = _gwas(tmp_path)
    assert main(["inspect", str(gwas)]) == 0
    assert json.loads(capsys.readouterr().out)["mapping"]["ea"] == "A1"

    clean = tmp_path / "clean.tsv.gz"
    status = tmp_path / "status.tsv"
    assert main(["clean", str(gwas), "-o", str(clean), "--min-maf", "0",
                 "--status-out", str(status)]) == 0
    assert len(read_sumstats(clean)) == 3
    assert "nonfinite" in status.read_text()

    aligned = tmp_path / "aligned.tsv"
    status2 = tmp_path / "status2.tsv"
    assert main(["align", str(gwas), "--reference", str(bim), "-o",
                 str(aligned), "--status-out", str(status2)]) == 0
    out = read_sumstats(aligned)
    assert list(out.id) == ["rs1", "rs2"]
    assert list(out.ea) == ["A", "C"]                  # reference a1
    np.testing.assert_allclose(out.beta, [-0.1, 0.2])
    np.testing.assert_allclose(out.eaf, [0.2, 0.3])
    rows = status2.read_text().splitlines()[1:]
    assert [r.split("\t")[-1] for r in rows] == [".", ".", "ambiguous_alleles",
                                                "nonfinite"]
    log = json.loads((tmp_path / "aligned.tsv.log.json").read_text())
    assert log["harmonize"]["n_matched"] == 2

    assert main(["align", str(gwas), "--reference", str(bim), "-o",
                 str(tmp_path / "x.tsv"), "--sumstats-build", "hg38",
                 "--reference-build", "hg19"]) == 2

    w = _weights(tmp_path, sd_source="hwe")
    pgs = tmp_path / "w.pgs.txt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert main(["pgs-export", str(w), "-o", str(pgs),
                     "--genome-build", "GRCh37"]) == 0
    assert read_scoring_file(pgs).meta["sd_source"] == "hwe"
