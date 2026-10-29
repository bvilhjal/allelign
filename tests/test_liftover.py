"""Chain-file liftover, build detection and the automatic check in harmonize.

Synthetic chain files stand in for UCSC's, and a fake opener serves the
download, so the suite stays offline.
"""
import hashlib
import io
import warnings
from contextlib import contextmanager

import numpy as np
import pytest

import allelign
from allelign import lift as lo
from allelign.sumstats import Sumstats
from allelign.variants import Variants

# Source chr1 [100, 200) -> target chr1 [150, 250), gap, [250, 350) -> [300, 400);
# source chr2 [0, 100) -> target chr3 reverse strand at [500, 600).
CHAIN_TEXT = (
    "chain 1000 chr1 1000 + 100 350 chr1 1200 + 150 400 1\n"
    "100 50 50\n"
    "100\n"
    "\n"
    "chain 500 chr2 1000 + 0 100 chr3 2000 - 500 600 2\n"
    "100\n")


def _chain_file(tmp_path, text=CHAIN_TEXT, name="test.over.chain"):
    path = tmp_path / name
    path.write_text(text)
    return path


def _shift_chain(tmp_path, shift, name):
    """chr1 [0, 1e7) moved by ``shift`` (a stand-in for a GRCh37/GRCh38 chain)."""
    start = max(0, shift)
    text = (f"chain 1 chr1 100000000 + {max(0, -shift)} {max(0, -shift) + 10_000_000} "
            f"chr1 100000000 + {start} {start + 10_000_000} 1\n10000000\n")
    return lo.read_chain(_chain_file(tmp_path, text, name))


def _sumstats(ids, chrom, pos, ea, oa, beta=None):
    n = len(ids)
    beta = np.full(n, 0.1) if beta is None else np.asarray(beta, dtype=float)
    return Sumstats(np.asarray(ids, dtype=str), np.asarray(chrom, dtype=str),
                    np.asarray(pos, dtype=np.int64), np.asarray(ea, dtype=str),
                    np.asarray(oa, dtype=str), beta, np.full(n, 0.01),
                    np.full(n, 10_000.0), np.full(n, 0.3), np.ones(n))


def test_lift_positions_follows_blocks_gaps_and_reverse_strand(tmp_path):
    chain = lo.read_chain(_chain_file(tmp_path))
    chrom = np.array(["1", "chr1", "1", "1", "2", "2", "5", "1"])
    pos = np.array([101, 200, 201, 251, 1, 100, 10, 0])
    out_chrom, out_pos, reverse, mapped = lo.lift_positions(chain, chrom, pos)
    assert out_chrom.tolist() == ["1", "1", "", "1", "3", "3", "", ""]
    assert out_pos.tolist() == [151, 250, 0, 301, 1500, 1401, 0, 0]
    assert reverse.tolist() == [False, False, False, False, True, True, False, False]
    assert mapped.tolist() == [True, True, False, True, True, True, False, False]


def test_lift_positions_drops_non_primary_contigs(tmp_path):
    text = "chain 1 chr1 1000 + 0 100 chr1_KI270706v1_random 500 + 0 100 1\n100\n"
    chain = lo.read_chain(_chain_file(tmp_path, text))
    _, out_pos, _, mapped = lo.lift_positions(chain, ["1"], [10])
    assert not mapped[0] and out_pos[0] == 0


def test_read_chain_refuses_overlapping_blocks(tmp_path):
    text = ("chain 2 chr1 1000 + 0 100 chr1 1000 + 0 100 1\n100\n"
            "chain 1 chr1 1000 + 50 150 chr1 1000 + 500 600 2\n100\n")
    with pytest.raises(ValueError, match="overlapping"):
        lo.read_chain(_chain_file(tmp_path, text))


def test_liftover_moves_rows_in_place(tmp_path):
    chain = lo.read_chain(_chain_file(tmp_path))
    ss = _sumstats(["rs1", "rs2", "rs3", "rs4"], ["1", "2", "1", ""], [101, 1, 201, 0],
                   ["A", "AC", "G", "T"], ["G", "T", "C", "C"])
    lifted, log = lo.liftover(ss, "GRCh37", "GRCh38", chain=chain)
    assert len(lifted.id) == 4
    assert lifted.chrom.tolist() == ["1", "3", "", ""]
    assert lifted.pos.tolist() == [151, 1500, 0, 0]
    # Reverse strand: alleles reverse-complemented, so beta counts the same allele.
    assert lifted.ea.tolist() == ["A", "GT", "G", "T"]
    assert lifted.oa.tolist() == ["G", "A", "C", "C"]
    np.testing.assert_array_equal(lifted.beta, ss.beta)
    # The unmappable row can no longer match; the row without coordinates is kept.
    assert lifted.id.tolist() == ["rs1", "rs2", "", "rs4"]
    assert log["n_lifted"] == 2 and log["n_reverse_strand"] == 1
    assert log["n_unmapped"] == 1 and log["n_without_coordinates"] == 1


class _FakeResponse(io.BytesIO):
    pass


def _fake_opener(payload, calls):
    @contextmanager
    def opener(url, method="GET", timeout=None):
        calls.append(url)
        yield _FakeResponse(payload)
    return opener


def test_fetch_chain_downloads_once_and_checks_the_checksum(tmp_path, monkeypatch):
    payload = CHAIN_TEXT.encode()
    url = "https://example.org/fake.over.chain"
    monkeypatch.setitem(lo.CHAIN_FILES, ("GRCh37", "GRCh38"),
                        (url, hashlib.sha256(payload).hexdigest()))
    calls = []
    first = lo.fetch_chain("hg19", "hg38", cache=tmp_path, opener=_fake_opener(payload, calls))
    second = lo.fetch_chain("GRCh37", "GRCh38", cache=tmp_path, opener=_fake_opener(payload, calls))
    assert first == second == tmp_path / "liftover" / "fake.over.chain"
    assert calls == [url]
    first.unlink()
    with pytest.raises(ValueError, match="SHA-256"):
        lo.fetch_chain("GRCh37", "GRCh38", cache=tmp_path,
                       opener=_fake_opener(b"not a chain", []))
    assert not first.exists()


def _install_shift_chains(monkeypatch, tmp_path, shift=50):
    """GRCh37 -> GRCh38 adds ``shift`` on chr1; GRCh38 -> GRCh37 subtracts it."""
    chains = {("GRCh37", "GRCh38"): _shift_chain(tmp_path, shift, "up.chain"),
              ("GRCh38", "GRCh37"): _shift_chain(tmp_path, -shift, "down.chain")}
    monkeypatch.setattr(lo, "load_chain", lambda s, t, **_: chains[(s, t)])


def _reference(n, step=1000):
    ids = np.array([f"rs{i}" for i in range(n)])
    pos = np.arange(1, n + 1, dtype=np.int64) * step
    return Variants(chrom=np.full(n, "1"), id=ids, pos=pos,
                    a1=np.full(n, "A"), a2=np.full(n, "G")).freeze()


def test_detect_build_names_both_builds(tmp_path, monkeypatch):
    _install_shift_chains(monkeypatch, tmp_path)
    ref = _reference(3000)
    same = _sumstats(ref.id, ref.chrom, ref.pos, ref.a1, ref.a2)
    assert lo.detect_build(same, ref)["relation"] == "same"
    moved = _sumstats(ref.id, ref.chrom, ref.pos + 50, ref.a1, ref.a2)
    found = lo.detect_build(moved, ref)
    assert found["relation"] == "differ"
    assert (found["sumstats_build"], found["reference_build"]) == ("GRCh38", "GRCh37")
    assert found["agreement"]["GRCh38->GRCh37"] == pytest.approx(1.0)
    assert found["agreement"]["as_is"] == 0.0


def test_harmonize_lifts_a_build_mismatch_automatically(tmp_path, monkeypatch):
    _install_shift_chains(monkeypatch, tmp_path)
    ref = _reference(3000)
    rng = np.random.default_rng(3)
    beta = rng.normal(size=3000)
    ss = _sumstats(ref.id, ref.chrom, ref.pos + 50, ref.a1, ref.a2, beta)
    plain = allelign.harmonize(ss, ref, liftover=False)
    assert len(plain) == 0
    with pytest.warns(RuntimeWarning, match="lifted 3,000 of 3,000 rows"):
        h = allelign.harmonize(ss, ref)
    assert len(h) == 3000
    np.testing.assert_array_equal(h.src_index, np.arange(3000))
    np.testing.assert_allclose(h.beta, beta)
    record = h.log["liftover"]
    assert record["applied"] and record["n_matched_before"] == 0
    assert (record["source"], record["target"]) == ("GRCh38", "GRCh37")


def test_harmonize_lifts_an_identifier_less_file(tmp_path, monkeypatch):
    _install_shift_chains(monkeypatch, tmp_path)
    ref = _reference(12_000, step=500)
    ids = np.array([f"1:{p + 50}" for p in ref.pos])
    ss = _sumstats(ids, ref.chrom, ref.pos + 50, ref.a1, ref.a2)
    with pytest.warns(RuntimeWarning, match="GRCh38 and the reference on GRCh37"):
        h = allelign.harmonize(ss, ref)
    assert len(h) == 12_000


def test_harmonize_leaves_well_matched_and_small_inputs_alone(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the build check must not run")
    monkeypatch.setattr(lo, "detect_build", refuse)
    ref = _reference(3000)
    same = _sumstats(ref.id, ref.chrom, ref.pos, ref.a1, ref.a2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h = allelign.harmonize(same, ref)
    assert len(h) == 3000 and "liftover" not in h.log
    small = _sumstats(ref.id[:500], ref.chrom[:500], ref.pos[:500] + 50, ref.a1[:500], ref.a2[:500])
    assert len(allelign.harmonize(small, ref)) == 0


def test_harmonize_keeps_the_alignment_when_the_build_check_fails(monkeypatch):
    def offline(*args, **kwargs):
        raise ValueError("network unreachable")
    monkeypatch.setattr(lo, "load_chain", offline)
    ref = _reference(3000)
    ss = _sumstats(ref.id, ref.chrom, ref.pos + 50, ref.a1, ref.a2)
    with pytest.warns(RuntimeWarning, match="build check failed"):
        h = allelign.harmonize(ss, ref)
    assert len(h) == 0 and h.log["liftover"]["applied"] is False


def test_harmonize_rejects_an_unknown_liftover_mode():
    ref = _reference(10)
    with pytest.raises(ValueError, match="liftover"):
        allelign.harmonize(_sumstats(ref.id, ref.chrom, ref.pos, ref.a1, ref.a2), ref,
                           liftover="always")
