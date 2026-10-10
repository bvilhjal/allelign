"""LD-consistency screen and QC behaviour (beyond the ldpred3 parity checks)."""

import warnings
from dataclasses import dataclass

import numpy as np
import pytest

from allelign import (chromosome_blocks, dense_blocks, dentist_statistic,
                      genomic_inflation, in_long_range_ld,
                      ld_blocks_from_genotypes, ld_consistency_screen,
                      loo_statistic, new_status, qc_sumstats, status_counts)
from allelign.sumstats import Sumstats


def _ar1(m, rho):
    idx = np.arange(m)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _planted(seed, m=600, rho=0.95, n_flip=5):
    rng = np.random.default_rng(seed)
    R = _ar1(m, rho)
    L = np.linalg.cholesky(R + 1e-10 * np.eye(m))
    beta = np.zeros(m)
    causal = rng.choice(m, 20, replace=False)
    beta[causal] = rng.normal(0, 5, 20)
    z = R @ beta + L @ rng.standard_normal(m)
    # Flip the sign of strong, well-tagged associations: LD-inconsistent.
    strong = np.argsort(-np.abs(z))
    flip = [j for j in strong if 5 < j < m - 5][:n_flip]
    z[flip] = -z[flip]
    return R.astype(np.float32), z, np.array(flip)


def test_dentist_statistic_matches_the_definition():
    rng = np.random.default_rng(0)
    R = _ar1(60, 0.7)
    z = rng.standard_normal(60)
    a, b = np.arange(0, 60, 2), np.arange(1, 60, 2)
    got = dentist_statistic(R, z, b, a, eigenvalue_floor=0.0)
    inv = np.linalg.inv(R[np.ix_(b, b)])
    pred = R[np.ix_(a, b)] @ inv @ z[b]
    var = 1 - np.einsum("ij,jk,ik->i", R[np.ix_(a, b)], inv, R[np.ix_(a, b)])
    np.testing.assert_allclose(got, (z[a] - pred) ** 2 / var, rtol=1e-6)


def test_loo_statistic_is_the_conditional_residual():
    rng = np.random.default_rng(1)
    R = _ar1(40, 0.6)
    z = rng.standard_normal(40)
    stat, _ = loo_statistic(R, z, ridge=0.01, eigenvalue_floor=0.0)
    A = R + 0.01 * np.eye(40)
    for j in (0, 7, 39):
        rest = np.delete(np.arange(40), j)
        w = np.linalg.solve(A[np.ix_(rest, rest)], A[rest, j])
        zhat = w @ z[rest]
        var = A[j, j] - A[j, rest] @ w
        assert stat[j] == pytest.approx((z[j] - zhat) ** 2 / var, rel=2e-3)


def test_screen_finds_planted_flips_and_spares_the_rest():
    found = false = tested = 0
    for seed in range(3):
        R, z, flip = _planted(seed)
        keep, stats = ld_consistency_screen([(R, np.arange(len(z)))], z,
                                            seed=seed, return_stats=True)
        found += np.isin(flip, np.flatnonzero(~keep)).sum()
        false += np.setdiff1d(np.flatnonzero(~keep), flip).size
        tested += stats["n_tested"]
    assert found >= 12                       # of 15 planted
    assert false <= 0.01 * tested


def test_clean_z_drops_nothing():
    rng = np.random.default_rng(4)
    R = _ar1(500, 0.9)
    z = np.linalg.cholesky(R + 1e-10 * np.eye(500)) @ rng.standard_normal(500)
    keep = ld_consistency_screen(dense_blocks(R.astype(np.float32), [250, 250]), z)
    assert keep.all()


@dataclass
class _LowRank:
    U: np.ndarray
    residual_diag: np.ndarray
    scale: float = 1.0


class _Provider:
    def __init__(self, R):
        self.R = R
        self.size = R.shape[0]

    def window(self, rows):
        return self.R[np.ix_(rows, rows)]


def test_ld_forms_give_the_same_mask():
    R, z, _ = _planted(7, m=400)
    idx = np.arange(400)
    dense = ld_consistency_screen([(R, idx)], z, seed=3)
    q8 = np.round(R * 127).astype(np.int8)
    from_q8 = ld_consistency_screen([(q8, idx)], z, seed=3)
    deq = ld_consistency_screen([(q8.astype(np.float32) / np.float32(127), idx)],
                                z, seed=3)
    np.testing.assert_array_equal(from_q8, deq)
    w, V = np.linalg.eigh(R.astype(np.float64))
    U = (V[:, -60:] * np.sqrt(w[-60:])).astype(np.float32)
    resid = (1.0 - (U.astype(np.float64) ** 2).sum(1)).astype(np.float32)
    lowrank = _LowRank(U, resid)
    dense_lr = (U @ U.T) + np.diag(resid)
    with warnings.catch_warnings():
        # A rank-60 truncation of this block is not the LD z was drawn from.
        warnings.simplefilter("ignore", RuntimeWarning)
        np.testing.assert_array_equal(
            ld_consistency_screen([(lowrank, idx)], z, seed=3),
            ld_consistency_screen([(dense_lr.astype(np.float32), idx)], z,
                                  seed=3))
    np.testing.assert_array_equal(
        ld_consistency_screen([(_Provider(R), idx)], z, seed=3), dense)


def test_traits_screen_independently():
    R, z, _ = _planted(9, m=300)
    blocks = [(R, np.arange(300))]
    z2 = -z[::-1].copy()
    both = ld_consistency_screen(blocks, np.column_stack([z, z2]), seed=1)
    np.testing.assert_array_equal(both[:, 0], ld_consistency_screen(blocks, z, seed=1))
    np.testing.assert_array_equal(both[:, 1], ld_consistency_screen(blocks, z2, seed=1))


def test_small_blocks_are_untested_and_warn():
    R = _ar1(30, 0.5).astype(np.float32)
    with pytest.warns(RuntimeWarning, match="never entered a window"):
        keep, stats = ld_consistency_screen([(R, np.arange(30))], np.zeros(30),
                                            return_stats=True)
    assert keep.all() and stats["n_untested"] == 30


def test_block_validation():
    R = _ar1(60, 0.5)
    with pytest.raises(ValueError, match="span 50 variants"):
        ld_consistency_screen([(R[:50, :50], np.arange(50))], np.zeros(60))
    with pytest.raises(ValueError, match="overlap"):
        ld_consistency_screen([(R[:30, :30], np.arange(30))] * 2, np.zeros(60))
    with pytest.raises(ValueError, match="size"):
        ld_consistency_screen([(R, np.arange(50))], np.zeros(50))
    with pytest.raises(ValueError, match="non-finite"):
        ld_consistency_screen([(R, np.arange(60))], np.full(60, np.nan))


def test_ld_blocks_from_genotypes_and_chromosome_blocks():
    rng = np.random.default_rng(2)
    G = rng.integers(0, 3, (500, 120)).astype(np.int8)
    G[:, 5] = 1                                   # monomorphic
    G[3, 7] = -1                                  # missing call
    sizes = chromosome_blocks(["1"] * 70 + ["chr2"] * 50, max_size=60)
    assert sizes == [35, 35, 50]
    blocks = ld_blocks_from_genotypes(G, sizes)
    R0 = blocks[0][0]
    ref = np.corrcoef(G[:, 10:20].astype(float), rowvar=False)
    np.testing.assert_allclose(R0[10:20, 10:20], ref, atol=1e-5)
    assert np.all(R0[5, np.arange(35) != 5] == 0) and R0[5, 5] == 1
    assert np.isfinite(R0[7]).all()
    with pytest.raises(ValueError, match="contiguous"):
        chromosome_blocks(["1", "2", "1"])


def test_genomic_inflation():
    rng = np.random.default_rng(3)
    z = rng.standard_normal(200_000)
    assert genomic_inflation(z) == pytest.approx(1.0, abs=0.02)
    assert genomic_inflation(z * np.sqrt(1.2)) == pytest.approx(1.2, abs=0.03)
    from math import erfc, sqrt
    p = np.array([erfc(abs(v) / sqrt(2)) for v in z[:20_000]])
    assert genomic_inflation(pval=p) == pytest.approx(genomic_inflation(z[:20_000]),
                                                      rel=1e-6)
    with pytest.raises(ValueError):
        genomic_inflation()


def test_long_range_ld_mask_normalises_labels_and_refuses_other_builds():
    mask = in_long_range_ld(["chr6", "6", "06", "7", "19"],
                            [30_000_000, 30_000_000, 30_000_000, 30_000_000,
                             45_000_000])
    assert list(mask) == [True, True, True, False, False]
    assert in_long_range_ld(["19"], [45_000_000], include_apoe=True)[0]
    with pytest.raises(ValueError, match="GRCh37"):
        in_long_range_ld(["6"], [30_000_000], genome_build="GRCh38")
    custom = in_long_range_ld(["6"], [1], regions=[("6", 0, 10, "x")],
                              genome_build="GRCh38")
    assert custom[0]


def test_qc_status_is_the_first_failing_filter():
    n = 6
    ss = Sumstats(id=np.array(["a", "b", "c", "c", "e", "f"], dtype=object),
                  chrom=np.full(n, "1", dtype=object),
                  pos=np.arange(1, n + 1),
                  ea=np.full(n, "A", dtype=object),
                  oa=np.full(n, "G", dtype=object),
                  beta=np.array([0.1, np.nan, 0.1, 0.1, 0.1, 0.1]),
                  se=np.full(n, 0.01),
                  n_eff=np.array([1e4, 1e4, 1e4, 1e4, 10.0, 1e4]),
                  eaf=np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.001]),
                  info=np.full(n, np.nan))
    status = new_status(n)
    keep, log = qc_sumstats(ss, status=status)
    assert list(keep) == [True, False, False, False, False, False]
    assert status_counts(status) == {"kept": 1, "nonfinite": 1, "duplicate": 2,
                                     "low_n": 1, "low_maf": 1}
    assert log["n_kept"] == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        qc_sumstats(ss)


def test_selected_screen_equals_subsetting_first():
    from allelign import ld_consistency_screen_selected
    R1, z1, _ = _planted(11, m=300)
    R2, z2, _ = _planted(12, m=200)
    blocks = [(R1, np.arange(300)), (R2, np.arange(300, 500))]
    rng = np.random.default_rng(0)
    selection = np.sort(rng.choice(500, 380, replace=False))
    z = np.concatenate([z1, z2])[selection]
    sub_blocks, start = [], 0
    for R, idx in blocks:
        rows = selection[(selection >= idx[0]) & (selection <= idx[-1])] - idx[0]
        if rows.size:
            sub_blocks.append((R[np.ix_(rows, rows)],
                               np.arange(start, start + rows.size)))
            start += rows.size
    events = []
    keep = ld_consistency_screen_selected(blocks, selection, z, seed=5,
                                          progress=events.append)
    np.testing.assert_array_equal(keep, ld_consistency_screen(sub_blocks, z,
                                                              seed=5))
    assert [e["done"] for e in events] == [1, 2]
    assert events[-1] == {"step": "LD consistency screen", "done": 2,
                          "total": 2, "unit": "block"}
    with pytest.raises(ValueError, match="strictly increasing"):
        ld_consistency_screen_selected(blocks, selection[::-1], z)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_pooled_screen_equals_serial_and_fills_stats_out():
    blocks, zs = [], []
    for b in range(4):
        R, z, _ = _planted(20 + b, m=150)
        blocks.append((R, np.arange(150 * b, 150 * (b + 1))))
        zs.append(z)
    z = np.concatenate(zs)
    stats = {}
    serial = ld_consistency_screen(blocks, z, seed=2, stats_out=stats)
    pooled = ld_consistency_screen(blocks, z, seed=2, ncores=3)
    np.testing.assert_array_equal(serial, pooled)
    assert stats["n_tested"] == 600 and stats["n_dropped"] == (~serial).sum()
    calls = []

    def eigh(a):
        calls.append(a.shape)
        return np.linalg.eigh(a)
    ld_consistency_screen(blocks, z, seed=2, _eigh=eigh)
    assert calls                                   # the hook is used
