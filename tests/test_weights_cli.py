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
    text = out.read_text()
    assert f"#variants_number={len(wt)}" in text
    assert "variants_genome_wide" not in text
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
    assert str(sf) == "PGS000001: 2 variants, build GRCh38" and sf.trait == ""


def test_scoring_prefilter_keeps_whole_file_checks(tmp_path):
    path = tmp_path / "PGS000002.txt"
    path.write_text(
        "#pgs_id=PGS000002\n#trait_reported=Height\n#weight_type=beta\n"
        "rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\n"
        "rs1\t1\t100\tA\tG\t0.3\nrs2\t1\t200\tC\tT\t-0.4\n"
        "rs3\t1\t300\tA\tC\t0.5\nrs4\t1\t400\tG\tT\t0.1\n")
    ref = Variants(chrom=np.array(["1", "1"], dtype=object),
                   id=np.array(["rs1", "rs3"], dtype=object),
                   pos=np.array([100, 300]),
                   a1=np.array(["A", "C"], dtype=object),
                   a2=np.array(["G", "A"], dtype=object))
    known = set(ref.id)
    full = read_scoring_file(path)
    sub = read_scoring_file(path, keep=lambda rsid, pos: rsid in known)
    assert list(sub.id) == ["rs1", "rs3"] and len(sub) == sub.log["n_kept"] == 2
    np.testing.assert_array_equal(sub.weight, [0.3, 0.5])
    assert sub.log["n_reference_prefiltered"] == 2
    assert sub.log["weight_square_sum_all"] == pytest.approx(full.weight @ full.weight)
    assert str(sub) == "PGS000002: 2 variants, Height"
    hf, hs = harmonize_weights(full, ref), harmonize_weights(sub, ref)
    np.testing.assert_array_equal(hf.var_index, hs.var_index)
    np.testing.assert_array_equal(hf.weight, hs.weight)
    for key in ("n_sumstats", "n_unmatched", "weight_mass_matched"):
        assert hf.log[key] == pytest.approx(hs.log[key]), key
    # The ratio check reads rejected rows too.
    bad = tmp_path / "bad.txt"
    bad.write_text("#weight_type=OR\nrsID\teffect_allele\teffect_weight\n"
                   "rs1\tA\t1.2\nrs9\tA\t-1\n")
    with pytest.raises(ValueError, match="non-positive"):
        read_scoring_file(bad, keep=lambda rsid, pos: rsid == "rs1")


def test_variant_indices_is_the_cached_harmonize_index():
    from allelign import variant_indices
    ref = Variants(chrom=np.array(["1"], dtype=object), id=np.array(["rs1"], dtype=object),
                   pos=np.array([100]), a1=np.array(["A"], dtype=object),
                   a2=np.array(["G"], dtype=object)).freeze()
    first = variant_indices(ref)
    assert first[1]["rs1"] == [0] and variant_indices(ref)[1] is first[1]


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
    bare = tmp_path / "bare.pvar"
    bare.write_text("1\trs1\t0\t100\tA\tG\n")
    headerless = read_pvar(bare)
    assert list(headerless.a1) == ["A"] and list(headerless.a2) == ["G"]
    assert list(headerless.pos) == [100]
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


def test_normalize_build_accepts_common_spellings_and_refuses_the_rest():
    for text in ("GRCh37", "hg19", "b37", "37", " GRCH37 "):
        assert normalize_build(text) == "GRCh37"
    for text in ("GRCh38", "hg38", "b38", "38"):
        assert normalize_build(text) == "GRCh38"
    assert normalize_build(None) is None
    with pytest.raises(ValueError, match="unknown genome build"):
        normalize_build("hg18")


def test_weight_table_rejects_malformed_or_nonfinite_data_before_writing(tmp_path):
    base = dict(id=["rs1"], chrom=["1"], pos=[1], effect_allele=["A"],
                other_allele=["G"], weight=[0.1])
    path = tmp_path / "weights.tsv"
    with pytest.raises(ValueError, match="equal lengths"):
        write_weights(path, **{**base, "weight": [0.1, 0.2]})
    assert not path.exists()
    with pytest.raises(ValueError, match="finite"):
        write_weights(path, **{**base, "weight": [np.nan]})
    assert not path.exists()
    with pytest.raises(ValueError, match="provided together"):
        write_weights(path, **base, af=[0.2])
    assert not path.exists()
    with pytest.raises(ValueError, match="sd must be >= 0"):
        write_weights(path, **base, af=[0.2], sd=[-0.1])
    assert not path.exists()


def test_weight_write_failure_preserves_previous_file(tmp_path, monkeypatch):
    import os
    path = tmp_path / "weights.tsv"
    path.write_text("previous complete file\n")

    def fail(*args):
        raise OSError("simulated publication failure")
    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="publication"):
        write_weights(path, id=["rs1"], chrom=["1"], pos=[1],
                      effect_allele=["A"], other_allele=["C"], weight=[.1])
    assert path.read_text() == "previous complete file\n"
    assert list(tmp_path.iterdir()) == [path]


def test_weight_reader_validates_header_rows_and_scale(tmp_path):
    path = tmp_path / "weights.tsv"
    path.write_text("ID\tCHR\tPOS\tA1\tA2\tWEIGHT\nrs1\t1\t1\tA\tG\tnan\n")
    with pytest.raises(ValueError, match="finite"):
        read_weights(path)
    path.write_text("ID\tCHR\tPOS\tA1\tA2\tWEIGHT\tAF_REF\n"
                    "rs1\t1\t1\tA\tG\t0.1\t0.2\n")
    with pytest.raises(ValueError, match="both AF_REF and SD_REF"):
        read_weights(path)
    path.write_text("CHR ID POS A2 A1 WEIGHT SD_REF AF_REF\n"
                    "1 rs1 1 G A 0.1 1.2 0.2\n")
    with pytest.warns(UserWarning, match="no SD_SOURCE"):
        got = read_weights(path)
    assert got.id.tolist() == ["rs1"] and got.has_scale


def test_weights_round_trip_bit_for_bit(tmp_path):
    weight = np.array([0.1, 1.0 / 3.0, np.pi * 1e-8, -2.5e-3])
    af = np.array([1.0 / 3.0, 0.2, 0.7, 1e-6])
    sd = np.sqrt(2.0 * af * (1.0 - af))
    path = tmp_path / "exact.tsv"
    write_weights(path, id=[f"rs{i}" for i in range(4)], chrom=["1"] * 4,
                  pos=[1, 2, 3, 4], effect_allele=["A"] * 4,
                  other_allele=["G"] * 4, weight=weight, af=af, sd=sd)
    got = read_weights(path)
    np.testing.assert_array_equal(got.weight, weight)
    np.testing.assert_array_equal(got.af_ref, af)
    np.testing.assert_array_equal(got.sd_ref, sd)


def test_sd_source_round_trips_and_defaults_to_empirical(tmp_path):
    """SD_SOURCE records how SD_REF was obtained; old files stay readable.

    bipred fills SD_REF with the HWE approximation sqrt(2f(1-f)) behind a
    RuntimeWarning, because it fits summary statistics against an LD reference
    and never observes a fit cohort. That warning cannot survive serialisation,
    so before this column a frozen-scale scorer could not tell an approximated
    SD from an observed one.
    """
    written = tmp_path / "empirical.tsv"
    write_weights(written, id=["rs1", "rs2"], chrom=["1", "1"], pos=[100, 200],
                  effect_allele=["A", "A"], other_allele=["G", "G"],
                  weight=[0.2, -0.1], af=[0.4, 0.3], sd=[0.5, 0.4])
    assert "SD_SOURCE" in written.read_text().splitlines()[0]
    wt = read_weights(written)
    assert wt.sd_source == "empirical" and not wt.sd_is_hwe

    hwe = tmp_path / "hwe.tsv"
    write_weights(hwe, id=["rs1"], chrom=["1"], pos=[100],
                  effect_allele=["A"], other_allele=["G"], weight=[0.2],
                  af=[0.4], sd=[0.5], sd_source="hwe")
    assert read_weights(hwe).sd_is_hwe

    # Pre-2026-08-22 files carry no such column and must still parse, reporting
    # None rather than guessing which kind of SD they hold -- but the
    # empirical assumption is now audible, and a reader who knows the file
    # (e.g. a legacy bipred output) can declare it instead.
    legacy = tmp_path / "legacy.tsv"
    legacy.write_text("ID\tCHR\tPOS\tA1\tA2\tWEIGHT\tAF_REF\tSD_REF\n"
                      "rs1\t1\t100\tA\tG\t0.2\t0.4\t0.5\n")
    with pytest.warns(UserWarning, match="no SD_SOURCE"):
        old = read_weights(legacy)
    assert old.has_scale and old.sd_source is None and not old.sd_is_hwe
    declared = read_weights(legacy, sd_source="hwe")
    assert declared.sd_source == "hwe" and declared.sd_is_hwe
    with pytest.raises(ValueError, match="contradicts"):
        read_weights(written, sd_source="hwe")
    with pytest.raises(ValueError, match="sd_source must be one of"):
        read_weights(legacy, sd_source="guessed")

    with pytest.raises(ValueError, match="sd_source must be one of"):
        write_weights(tmp_path / "bad.tsv", id=["rs1"], chrom=["1"], pos=[100],
                      effect_allele=["A"], other_allele=["G"], weight=[0.2],
                      af=[0.4], sd=[0.5], sd_source="guessed")

    mixed = tmp_path / "mixed.tsv"
    mixed.write_text("ID\tCHR\tPOS\tA1\tA2\tWEIGHT\tAF_REF\tSD_REF\tSD_SOURCE\n"
                     "rs1\t1\t100\tA\tG\t0.2\t0.4\t0.5\tempirical\n"
                     "rs2\t1\t200\tA\tG\t0.1\t0.3\t0.4\thwe\n")
    with pytest.raises(ValueError, match="mixes SD_SOURCE"):
        read_weights(mixed)


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


@pytest.mark.parametrize("options", [
    ["--n-eff", "1000"],
    ["--n-cases", "2000", "--n-controls", "8000"],
])
def test_cli_inspect_accepts_scalar_n(tmp_path, capsys, options):
    path = tmp_path / "no_n.tsv"
    path.write_text("SNP A1 A2 BETA SE\nrs1 A G 0.1 0.01\n")
    assert main(["inspect", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["missing"] == ["n_eff"]
    assert main(["inspect", str(path), *options]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["missing"] == [] and info["n_source"] == "scalar"
    out = tmp_path / "clean.tsv"
    assert main(["clean", str(path), "-o", str(out), *options]) == 0
    log = json.loads((tmp_path / "clean.tsv.log.json").read_text())
    assert log["parse"]["n_eff_source"] == info["n_source"]
    assert main(["inspect", str(path), "--n-cases", "2000"]) == 2
    assert "given together" in capsys.readouterr().err


@pytest.mark.parametrize("existing", [False, True])
def test_cli_log_write_failure_preserves_destination(tmp_path, monkeypatch, existing):
    from allelign import cli
    out = tmp_path / "clean.tsv"
    path = tmp_path / "clean.tsv.log.json"
    previous = '{"previous": true}\n'
    if existing:
        path.write_text(previous)

    def fail(log, fh, **kwargs):
        fh.write('{"partial":')
        raise OSError("simulated log write failure")

    monkeypatch.setattr(cli.json, "dump", fail)
    with pytest.raises(OSError, match="log write failure"):
        cli._write_log(out, {"parse": {"n_input": 1}})
    if existing:
        assert path.read_text() == previous
    else:
        assert not path.exists()
    assert list(tmp_path.iterdir()) == ([path] if existing else [])
