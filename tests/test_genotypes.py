"""PLINK .bed reading and the LD screen with LD from reference genotypes."""

import warnings

import numpy as np
import pytest

from allelign import (Bed, chromosome_blocks, ld_blocks_from_genotypes,
                      ld_consistency_screen, ld_consistency_screen_genotypes)
from allelign.cli import main
from allelign.genotypes import _block_seed


def _pack(G):
    """Independent .bed encoder: counts of A1 (-1 missing) -> variant-major bytes."""
    code = {2: 0b00, -1: 0b01, 1: 0b10, 0: 0b11}
    n, m = G.shape
    stride = (n + 3) // 4
    out = bytearray(b"\x6c\x1b\x01")
    for j in range(m):
        row = bytearray(stride)
        for i in range(n):
            row[i // 4] |= code[int(G[i, j])] << (2 * (i % 4))
        out += row
    return bytes(out)


def _fileset(tmp_path, G, chrom=None, name="ref", a1="A", a2="G"):
    n, m = G.shape
    chrom = ["1"] * m if chrom is None else chrom
    prefix = tmp_path / name
    (prefix.with_suffix(".bed")).write_bytes(_pack(G))
    with open(prefix.with_suffix(".bim"), "w") as fh:
        for j in range(m):
            fh.write(f"{chrom[j]}\trs{j}\t0\t{(j + 1) * 100}\t{a1}\t{a2}\n")
    with open(prefix.with_suffix(".fam"), "w") as fh:
        for i in range(n):
            fh.write(f"f{i}\ti{i}\t0\t0\t0\t-9\n")
    return prefix


def _ld_genotypes(n=1500, m=240, rho=0.92, seed=0):
    """Allele counts with AR(1)-like LD: two thresholded latent haplotypes."""
    rng = np.random.default_rng(seed)

    def haplotype():
        x = np.empty((n, m))
        x[:, 0] = rng.standard_normal(n)
        for j in range(1, m):
            x[:, j] = rho * x[:, j - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
        cut = rng.uniform(-0.8, 0.8, m)
        return (x > cut).astype(np.int8)

    return haplotype() + haplotype()


@pytest.mark.parametrize("n", [7, 8, 13])
def test_bed_reads_counts_missing_calls_and_selected_rows(tmp_path, n):
    rng = np.random.default_rng(n)
    G = rng.integers(-1, 3, (n, 11)).astype(np.int8)
    bed = Bed(_fileset(tmp_path, G))
    assert (bed.n_samples, bed.n_variants) == (n, 11)
    np.testing.assert_array_equal(bed.read(), G)
    pick = [10, 0, 1, 2, 7]
    np.testing.assert_array_equal(bed.read(pick), G[:, pick])
    assert bed.read([]).shape == (n, 0)
    with pytest.raises(ValueError, match="out of range"):
        bed.read([11])


def test_bed_rejects_a_bad_header_or_size(tmp_path):
    prefix = _fileset(tmp_path, np.zeros((5, 3), dtype=np.int8))
    bed_path = prefix.with_suffix(".bed")
    good = bed_path.read_bytes()
    bed_path.write_bytes(good[:-1])
    with pytest.raises(ValueError, match="genotype bytes"):
        Bed(prefix)
    bed_path.write_bytes(b"\x6c\x1b\x00" + good[3:])
    with pytest.raises(ValueError, match="variant-major"):
        Bed(str(prefix) + ".bed")


@pytest.mark.parity
def test_bed_matches_ldpred3_reader(tmp_path):
    genotype_io = pytest.importorskip("ldpred3.genotype_io")
    G = _ld_genotypes(n=37, m=50)
    G[3, 5] = -1
    prefix = _fileset(tmp_path, G)
    idx = np.array([0, 1, 2, 9, 30, 31, 49])
    ours = Bed(prefix).read(idx)
    theirs = genotype_io.read_bed(str(prefix) + ".bed", 37, 50, idx)
    np.testing.assert_array_equal(ours, theirs)


def test_genotype_screen_equals_the_block_screen_and_catches_a_flip(tmp_path):
    G = _ld_genotypes()
    m = G.shape[1]
    chrom = ["1"] * 140 + ["2"] * (m - 140)
    bed = Bed(_fileset(tmp_path, G, chrom=chrom))
    rng = np.random.default_rng(5)
    R = np.corrcoef(G.T.astype(float))
    z = np.linalg.cholesky(R + 1e-6 * np.eye(m)) @ rng.standard_normal(m)
    victim = 60
    z[victim] = -z[victim] - 8.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        keep, stats = ld_consistency_screen_genotypes(
            bed, np.arange(m), z, max_block=100, seed=3, return_stats=True)
    assert not keep[victim] and keep.mean() > 0.95
    assert stats["n_blocks"] == len(chromosome_blocks(chrom, max_size=100)) == 3

    # Block by block, each with its own derived seed: the same mask.
    expected, start = [], 0
    for b, size in enumerate(chromosome_blocks(chrom, max_size=100)):
        blocks = ld_blocks_from_genotypes(G[:, start:start + size], [size])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected.append(ld_consistency_screen(
                blocks, z[start:start + size], seed=_block_seed(3, b)))
        start += size
    np.testing.assert_array_equal(keep, np.concatenate(expected))

    # Rows given in any order map back to their own z.
    perm = rng.permutation(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        shuffled = ld_consistency_screen_genotypes(
            bed, perm, z[perm], max_block=100, seed=3)
    np.testing.assert_array_equal(shuffled, keep[perm])


def test_genotype_screen_validates_its_inputs(tmp_path):
    bed = Bed(_fileset(tmp_path, _ld_genotypes(n=40, m=60)))
    with pytest.raises(TypeError):
        ld_consistency_screen_genotypes(object(), [0], [1.0])
    with pytest.raises(ValueError, match="same length"):
        ld_consistency_screen_genotypes(bed, [0, 1], [1.0])
    with pytest.raises(ValueError, match="at most once"):
        ld_consistency_screen_genotypes(bed, [0, 0], [1.0, 2.0])
    with pytest.raises(ValueError, match="out of range"):
        ld_consistency_screen_genotypes(bed, [60], [1.0])


def test_cli_align_ld_screen_marks_the_flipped_variant(tmp_path, capsys):
    G = _ld_genotypes(m=200, seed=2)
    m = G.shape[1]
    prefix = _fileset(tmp_path, G)
    rng = np.random.default_rng(9)
    R = np.corrcoef(G.T.astype(float))
    z = np.linalg.cholesky(R + 1e-6 * np.eye(m)) @ rng.standard_normal(m)
    z[80] = -z[80] - 8.0
    se = np.full(m, 0.01)
    gwas = tmp_path / "gwas.tsv"
    with open(gwas, "w") as fh:
        fh.write("SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tN\n")
        for j in range(m):
            fh.write(f"rs{j}\t1\t{(j + 1) * 100}\tA\tG\t{z[j] * se[j]:.8g}"
                     f"\t{se[j]}\t50000\n")
    out, status = tmp_path / "aligned.tsv", tmp_path / "status.tsv"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert main(["align", str(gwas), "--reference", f"{prefix}.bim",
                     "-o", str(out), "--min-maf", "0", "--ld-screen",
                     "--ld-block-size", "100", "--status-out", str(status)]) == 0
    rows = [line.split("\t") for line in status.read_text().splitlines()[1:]]
    reasons = {r[0]: r[6] for r in rows}
    assert reasons["rs80"] == "dentist_outlier"
    n_out = len(out.read_text().splitlines()) - 1
    assert n_out == sum(r[5] == "kept" for r in rows) < m
    assert "LD screen dropped" in capsys.readouterr().err


def test_cli_ld_screen_needs_a_bed_reference(tmp_path, capsys):
    pvar = tmp_path / "ref.pvar"
    pvar.write_text("#CHROM\tPOS\tID\tREF\tALT\n1\t100\trs0\tG\tA\n")
    gwas = tmp_path / "gwas.tsv"
    gwas.write_text("SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tN\nrs0\t1\t100\tA\tG\t0.1\t0.01\t1000\n")
    assert main(["align", str(gwas), "--reference", str(pvar), "-o",
                 str(tmp_path / "o.tsv"), "--ld-screen"]) == 2
    assert "PLINK 1" in capsys.readouterr().err


def test_bed_sample_selection_by_iid_index_or_mask(tmp_path):
    rng = np.random.default_rng(4)
    G = rng.integers(-1, 3, (9, 5)).astype(np.int8)
    prefix = _fileset(tmp_path, G)
    np.testing.assert_array_equal(Bed(prefix, samples=["i2", "i7"]).read(), G[[2, 7]])
    np.testing.assert_array_equal(Bed(prefix, samples=[7, 2]).read([4]), G[[2, 7]][:, [4]])
    mask = np.zeros(9, dtype=bool)
    mask[[0, 8]] = True
    bed = Bed(prefix, samples=mask)
    assert bed.n_samples == 2
    np.testing.assert_array_equal(bed.read(), G[[0, 8]])
    with pytest.warns(RuntimeWarning, match="1 requested samples"):
        Bed(prefix, samples=["i1", "nobody"])
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="keeps no samples"):
        Bed(prefix, samples=["nobody"])
