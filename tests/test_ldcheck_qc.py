"""LD-consistency screen and QC behaviour (beyond the ldpred3 parity checks)."""

import warnings
from dataclasses import dataclass

import numpy as np
import pytest

from allelign import (af_concordance_mask, chromosome_blocks, dense_blocks,
                      dentist_statistic, genomic_control_factor,
                      genomic_inflation, gwas_af_on_counted_allele,
                      impute_n_eff, in_long_range_ld,
                      ld_blocks_from_genotypes, ld_consistency_screen,
                      loo_statistic, new_status, qc_sumstats,
                      sd_consistency_mask, should_impute_n, status_counts)
from allelign.ldcheck import _window_ld
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


def test_ld_constant_fractional_calls_have_no_ld_but_tiny_variation_survives():
    G = np.full((100, 4), .3)
    G[:20, 1] = np.nan
    G[20:40, 1] = -1
    G[:, 2] += (np.arange(100) % 2) * 1e-10
    G[:, 3] = G[:, 2]
    R = ld_blocks_from_genotypes(G, [4])[0][0]
    expected = np.eye(4)
    expected[2, 3] = expected[3, 2] = 1
    np.testing.assert_allclose(R, expected, atol=1e-7)


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
    zz = np.column_stack([z, -z])
    with pytest.raises(ValueError, match="return_stats"):
        ld_consistency_screen(blocks, zz, seed=2, stats_out={})
    _keep, per_trait = ld_consistency_screen(blocks, zz, seed=2,
                                             return_stats=True)
    assert len(per_trait) == 2 and per_trait[0]["n_tested"] == 600
    calls = []

    def eigh(a):
        calls.append(a.shape)
        return np.linalg.eigh(a)
    ld_consistency_screen(blocks, z, seed=2, _eigh=eigh)
    assert calls                                   # the hook is used


def _ss(n, **over):
    d = dict(
        id=np.array([f"rs{i}" for i in range(n)], dtype=object),
        chrom=np.array(["1"] * n, dtype=object),
        pos=np.arange(1, n + 1, dtype=np.int64),
        ea=np.array(["A"] * n, dtype=object),
        oa=np.array(["G"] * n, dtype=object),
        beta=np.full(n, 0.1),
        se=np.full(n, 0.02),
        n_eff=np.full(n, 1000.0),
        eaf=np.full(n, 0.3),
        info=np.full(n, 0.99),
    )
    d.update(over)
    return Sumstats(**d)

def test_qc_drops_nonfinite_and_bad_se():
    ss = _ss(5)
    ss.beta[0] = np.nan
    ss.se[1] = 0.0
    ss.se[2] = -1.0
    keep, log = qc_sumstats(ss)
    assert log["n_drop_nonfinite"] == 3
    assert list(np.where(keep)[0]) == [3, 4]


def test_qc_low_n_filter():
    ss = _ss(4, n_eff=np.array([1000., 1000., 600., 1000.]))  # max 1000, 0.7->700
    keep, log = qc_sumstats(ss)
    assert log["n_drop_lowN"] == 1
    assert not keep[2]


def test_qc_drops_nonfinite_n_eff():
    # A NaN per-variant sample size must be dropped as non-finite (regression:
    # the low-N filter uses ``n_eff < thresh`` which is False for NaN, so it
    # slipped through and reached the sampler, which requires N > 0).
    ss = _ss(4, n_eff=np.array([1000., np.nan, 1000., 1000.]))
    keep, log = qc_sumstats(ss)
    assert log["n_drop_nonfinite"] == 1
    assert not keep[1]
    assert np.all(np.isfinite(ss.n_eff[keep]))


def test_qc_drops_nonpositive_n_eff_before_anchoring():
    ss = _ss(4, n_eff=np.array([1000.0, 0.0, -5.0, 900.0]))
    keep, log = qc_sumstats(ss)
    np.testing.assert_array_equal(keep, [True, False, False, True])
    assert log["n_drop_nonpositive_n"] == 2
    assert log["n_anchor_value"] == 950.0
    assert np.all(ss.n_eff[keep] > 0)


def test_qc_maf_and_info():
    ss = _ss(4, eaf=np.array([0.3, 0.005, 0.3, 0.3]),
             info=np.array([0.99, 0.99, 0.5, 0.99]))
    keep, log = qc_sumstats(ss, min_maf=0.01, min_info=0.7)
    assert log["n_drop_lowMAF"] == 1 and not keep[1]
    assert log["n_drop_lowINFO"] == 1 and not keep[2]
    assert log["min_maf"] == 0.01 and log["min_info"] == 0.7


@pytest.mark.parametrize("name,value,match", [
    ("min_maf", np.nan, "finite"),
    ("min_maf", -0.01, r"\[0, 0\.5\]"),
    ("min_maf", 0.51, r"\[0, 0\.5\]"),
    ("min_info", np.inf, "finite"),
    ("min_info", -0.01, r"\[0, 1\]"),
    ("min_info", 1.01, r"\[0, 1\]"),
])
def test_qc_rejects_invalid_maf_and_info_controls(name, value, match):
    with pytest.raises(ValueError, match=match):
        qc_sumstats(_ss(3), **{name: value})


def test_qc_duplicates_dropped():
    ss = _ss(4)
    ss.id[2] = "rs0"          # duplicate of index 0
    keep, log = qc_sumstats(ss)
    assert log["n_drop_duplicate"] == 2   # both copies dropped
    assert not keep[0] and not keep[2]


def test_qc_chisq_outlier():
    ss = _ss(3, beta=np.array([0.1, 5.0, 0.1]), se=np.full(3, 0.02))
    # z = beta/se -> (5, 250, 5); chisq (25, 62500, 25)
    keep, log = qc_sumstats(ss, max_chisq=1000)
    assert log["n_drop_chisq_outlier"] == 1 and not keep[1]


def test_qc_columns_absent_are_skipped():
    # No eaf/info available -> those filters are no-ops, nothing dropped.
    ss = _ss(3, eaf=np.full(3, np.nan), info=np.full(3, np.nan))
    keep, log = qc_sumstats(ss)
    assert keep.all()
    assert "n_drop_lowMAF" not in log and "n_drop_lowINFO" not in log


def test_sd_consistency_flags_wrong_n():
    rng = np.random.default_rng(0)
    n = 200
    af = rng.uniform(0.1, 0.9, n)
    # Consistent variants: se ~ 1/sqrt(N * 2f(1-f)) so sd_ss ~ sd_ref.
    N = np.full(n, 10000.0)
    se = 1.0 / np.sqrt(N * 2 * af * (1 - af))
    beta = rng.normal(0, 0.01, n)
    # Corrupt 20 variants with a 100x-too-small N (inflated se mismatch).
    bad = rng.choice(n, 20, replace=False)
    se[bad] *= 10
    keep, log, diag = sd_consistency_mask(beta, se, N, af)
    # Most corrupted variants should be flagged, most good ones kept.
    assert log["n_drop_sd_inconsistent"] >= 10
    assert keep.sum() > n - 40
    assert keep[~np.isin(np.arange(n), bad)].mean() > 0.8
    # Effective-N diagnostic: se was inflated -> sd_ss too small -> drops land on
    # the LOW side; a consistent bulk gives slope ~1.
    assert log["n_drop_sd_low"] > log["n_drop_sd_high"]
    assert 0.8 < log["sd_slope"] < 1.25


def test_sd_consistency_info_correction_avoids_false_flags():
    # A well-imputed but low-INFO variant has a genuinely reduced dosage SD, hence a
    # smaller sd_ss; without the INFO correction it is falsely flagged, with it kept.
    rng = np.random.default_rng(2)
    n = 300
    af = rng.uniform(0.1, 0.9, n)
    N = np.full(n, 10000.0)
    se = 1.0 / np.sqrt(N * 2 * af * (1 - af))       # consistent variants
    beta = rng.normal(0, 0.005, n)
    info = np.ones(n)
    low = rng.choice(n, 40, replace=False)
    info[low] = 0.2                                 # poorly imputed
    se[low] = se[low] / np.sqrt(0.2)                # dosage SD reduced by sqrt(INFO)

    keep_no, log_no, _ = sd_consistency_mask(beta, se, N, af)               # no INFO
    keep_yes, log_yes, _ = sd_consistency_mask(beta, se, N, af, info=info)  # INFO-aware
    lowmask = np.isin(np.arange(n), low)
    # Without correction the low-INFO variants are largely dropped; with it, kept.
    assert keep_no[lowmask].mean() < 0.3
    assert keep_yes[lowmask].mean() > 0.85
    assert log_yes["info_corrected"] and not log_no["info_corrected"]
    assert log_yes["n_drop_sd_inconsistent"] < log_no["n_drop_sd_inconsistent"]

def test_impute_n_eff_recovers_constant_n():
    # se generated from a known N and frequency: se = 1/sqrt(N * 2f(1-f)) on the
    # standardized scale -> the imputed N should recover N (anchored to n_total).
    rng = np.random.default_rng(0)
    m, N = 2000, 50000
    af = rng.uniform(0.05, 0.95, m)
    se = 1.0 / np.sqrt(N * 2 * af * (1 - af))
    n_imp, log = impute_n_eff(se, af, N)
    # recovered within a few percent for the bulk
    assert np.median(np.abs(n_imp - N) / N) < 0.05
    assert (n_imp <= N + 1e-6).all()           # clipped at the total


def test_impute_n_eff_recovers_varying_n():
    # Half the variants are typed at full N, half at N/2 (meta-analysis style):
    # the imputed N must track the two-tier structure, not the single global N.
    rng = np.random.default_rng(1)
    m, N = 4000, 80000
    af = rng.uniform(0.05, 0.95, m)
    n_true = np.where(rng.random(m) < 0.5, N, N // 2).astype(float)
    se = 1.0 / np.sqrt(n_true * 2 * af * (1 - af))
    n_imp, log = impute_n_eff(se, af, N)
    full = n_true == N
    half = n_true == N // 2
    # the two groups separate cleanly around their true sizes
    assert abs(np.median(n_imp[full]) - N) / N < 0.05
    assert abs(np.median(n_imp[half]) - N / 2) / (N / 2) < 0.08
    assert np.median(n_imp[full]) > 1.5 * np.median(n_imp[half])


def test_impute_n_eff_handles_degenerate_entries():
    af = np.array([0.3, 0.0, 0.5, 1.0])       # two monomorphic
    se = np.array([0.01, 0.01, 0.0, 0.02])    # one zero-se
    n_imp, log = impute_n_eff(se, af, 10000)
    assert np.all(np.isfinite(n_imp)) and np.all(n_imp > 0)
    with pytest.raises(ValueError, match="anchor_quantile"):
        impute_n_eff(se, af, 10000, anchor_quantile=1.5)
    with pytest.raises(ValueError, match="anchor_bins"):
        impute_n_eff(se, af, 10000, anchor_bins=1.5)
    assert np.all(n_imp <= 10000 + 1e-6)


def test_dot_rsid_not_treated_as_duplicate():
    # The "." missing-rsID placeholder must fall back to chrom:pos:alleles, not
    # collapse every "."-labelled variant to one key and drop them all as
    # spurious duplicates (regression: distinct imputed variants wiped out).
    n = 5
    ss = _ss(n, id=np.array(["."] * n, dtype=object),
             pos=np.arange(1, n + 1, dtype=np.int64))
    keep, log = qc_sumstats(ss)
    assert keep.sum() == n
    assert log["n_drop_duplicate"] == 0
    # a genuine duplicate (same chrom:pos:alleles) is still caught
    ss2 = _ss(3, id=np.array([".", ".", "."], dtype=object),
              pos=np.array([10, 10, 20], dtype=np.int64))
    keep2, log2 = qc_sumstats(ss2)
    assert log2["n_drop_duplicate"] == 2 and keep2.sum() == 1


# --------------------------------------------------------------------------- #
# Allele-frequency concordance (Prive et al. 2022 "qc2")
# --------------------------------------------------------------------------- #
def _af_pair(m=2000, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    af_ref = rng.uniform(0.05, 0.95, m)
    af_gwas = np.clip(af_ref + rng.normal(0, noise, m), 0.0, 1.0)
    return af_gwas, af_ref, rng


def test_af_concordance_catches_allele_inversions():
    # An inverted allele reports 1-f. The check must catch it wherever the two
    # are actually distinguishable, i.e. away from f = 0.5.
    af_gwas, af_ref, rng = _af_pair()
    bad = rng.choice(af_ref.size, 40, replace=False)
    af_gwas[bad] = 1.0 - af_gwas[bad]
    status = new_status(af_ref.size)
    keep, log, _ = af_concordance_mask(af_gwas, af_ref, status=status)

    assert log["applied"] is True
    assert (status[~keep] != 0).all() and (status[keep] == 0).all()
    # Every dropped variant is one we planted: no false positives.
    assert set(np.flatnonzero(~keep)) <= set(bad)
    # Everything with |f - 0.5| comfortably beyond the threshold is caught.
    detectable = bad[np.abs(af_ref[bad] - 0.5) > 0.1]
    assert detectable.size > 0
    assert not keep[detectable].any()


def test_af_concordance_blind_near_half_is_a_stated_limit():
    # f = 0.5 inverts to itself, so the check cannot see it. Pinned so the
    # docstring's claim stays honest rather than aspirational.
    af_ref = np.array([0.5, 0.52, 0.9])
    af_gwas = 1.0 - af_ref                    # all three "inverted"
    keep, _log, _ = af_concordance_mask(af_gwas, af_ref)
    assert keep[0] and keep[1]                # invisible near 0.5
    assert not keep[2]                        # visible out in the tail


def test_af_concordance_refuses_to_apply_a_wholesale_drop():
    # A mirrored frequency column fails nearly everything. Dropping the genome
    # silently would be the worst outcome; report instead.
    af_gwas, af_ref, _ = _af_pair()
    status = new_status(af_ref.size)
    keep, log, _ = af_concordance_mask(1.0 - af_ref, af_ref, status=status)
    assert log["applied"] is False
    assert keep.all() and (status == 0).all()  # nothing dropped or stamped
    with pytest.raises(ValueError, match="status"):
        af_concordance_mask(1.0 - af_ref, af_ref,
                            status=np.zeros(af_ref.size))
    assert log["n_drop_af_diff"] == 0
    assert log["n_flagged_af_diff"] > 0        # but the count is still reported
    assert log["mirrored"] is True
    assert log["af_corr"] == pytest.approx(-1.0, abs=1e-6)
    assert "safety limit" in log["not_applied_reason"]

    # ... unless the caller insists.
    keep2, log2, _ = af_concordance_mask(1.0 - af_ref, af_ref, max_drop_frac=1.0)
    assert log2["applied"] is True
    assert log2["n_drop_af_diff"] > 0
    assert not keep2.all()


def test_af_concordance_keeps_untestable_and_validates():
    af_ref = np.array([0.3, 0.4, 0.5])
    af_gwas = np.array([np.nan, 0.4, 0.5])
    keep, log, _ = af_concordance_mask(af_gwas, af_ref)
    assert keep.all()                          # a missing frequency is not a fault
    assert log["n_testable"] == 2
    with pytest.raises(ValueError, match="same shape"):
        af_concordance_mask(af_gwas[:2], af_ref)
    for kwargs in ({"max_af_diff": 0.0}, {"max_af_diff": 1.5},
                   {"max_drop_frac": 0.0}, {"max_drop_frac": 2.0}):
        with pytest.raises(ValueError):
            af_concordance_mask(af_gwas, af_ref, **kwargs)


def test_gwas_af_is_reoriented_to_the_counted_allele():
    # Harmonisation flips beta for swapped alleles; the frequency must follow.
    # Getting this wrong makes exactly the flipped variants look discordant,
    # which is why it is a separate, separately-tested step.
    eaf = np.array([0.1, 0.25, 0.8, 0.4])
    src = np.array([3, 1, 0])
    flipped = np.array([False, True, True])
    got = gwas_af_on_counted_allele(eaf, src, flipped)
    np.testing.assert_allclose(got, [0.4, 0.75, 0.9])

    # And end to end: without the flip, flipped variants fail concordance.
    rng = np.random.default_rng(3)
    af_ref = rng.uniform(0.05, 0.45, 500)      # away from 0.5 so flips are visible
    eaf = af_ref.copy()
    src = np.arange(500)
    flipped = np.zeros(500, dtype=bool)
    flipped[:200] = True
    eaf[flipped] = 1.0 - eaf[flipped]          # source reports the other allele
    oriented = gwas_af_on_counted_allele(eaf, src, flipped)
    keep_ok, log_ok, _ = af_concordance_mask(oriented, af_ref)
    assert keep_ok.all() and log_ok["n_drop_af_diff"] == 0
    keep_bad, log_bad, _ = af_concordance_mask(eaf[src], af_ref)
    assert log_bad["n_flagged_af_diff"] >= 190     # the un-flipped 200


# --------------------------------------------------------------------------- #
# The published (bigsnpr) SD rejection region
# --------------------------------------------------------------------------- #
def _consistent_sumstats(m=4000, seed=1, n=100_000.0):
    rng = np.random.default_rng(seed)
    af = rng.uniform(0.05, 0.95, m)
    sd_ref = np.sqrt(2 * af * (1 - af))
    n_eff = np.full(m, n)
    se = 1.0 / (sd_ref * np.sqrt(n_eff))
    beta = rng.normal(0, 0.005, m)
    return beta, se, n_eff, af, rng


@pytest.mark.parametrize("rule", ["ratio", "prive"])
def test_sd_rules_agree_on_clean_data(rule):
    beta, se, n_eff, af, _ = _consistent_sumstats()
    keep, log, _ = sd_consistency_mask(beta, se, n_eff, af, rule=rule)
    assert keep.all()
    assert log["n_drop_sd_inconsistent"] == 0
    assert log["rule"] == rule


def test_prive_rule_catches_moderate_sd_inflation_that_ratio_misses():
    # The whole reason to offer the published rule: its absolute upper cut
    # (sd_ss > sd_ref + 0.1) sees a uniform ~1.5x inflation, which a symmetric
    # band with hi=2.0 cannot. Understating N by 2.25x produces exactly that.
    beta, se, n_eff, af, rng = _consistent_sumstats()
    bad = rng.choice(af.size, 400, replace=False)
    n_bad = n_eff.copy()
    n_bad[bad] /= 2.25

    keep_ratio, log_ratio, _ = sd_consistency_mask(beta, se, n_bad, af,
                                                   rule="ratio")
    keep_prive, log_prive, _ = sd_consistency_mask(beta, se, n_bad, af,
                                                   rule="prive")
    assert log_ratio["n_drop_sd_inconsistent"] == 0        # invisible
    assert not keep_prive[bad].any()                       # all 400 caught
    # and no collateral damage among the untouched variants
    good = np.setdiff1d(np.arange(af.size), bad)
    assert keep_prive[good].all()


def test_both_sd_rules_catch_gross_n_inflation():
    beta, se, n_eff, af, rng = _consistent_sumstats()
    bad = rng.choice(af.size, 60, replace=False)
    n_bad = n_eff.copy()
    n_bad[bad] *= 16
    for rule in ("ratio", "prive"):
        keep, _log, _ = sd_consistency_mask(beta, se, n_bad, af, rule=rule)
        assert not keep[bad].any(), rule


def test_sd_rule_is_validated():
    beta, se, n_eff, af, _ = _consistent_sumstats(m=50)
    with pytest.raises(ValueError, match="rule must be"):
        sd_consistency_mask(beta, se, n_eff, af, rule="bigsnpr")


def test_min_n_anchor_median_survives_a_mixed_arm_cohort():
    """The low-N ratio can be taken against the median, not only the max.

    A file that pools arms of very different size defeats a max-anchored
    threshold: the GLGC lipid releases carry a Metabochip arm at roughly twice
    the GWAS arm's N, so `0.7 * max` discards the entire GWAS arm. The samplers
    consume a per-variant n throughout, so those variants were usable; the
    filter is a convention, and the anchor is what makes it robust.
    """
    m = 1000
    ss = _ss(m)
    ss.n_eff[:] = 90_000.0
    ss.n_eff[:100] = 180_000.0                # a small high-N "Metabochip" arm

    keep_max, log_max = qc_sumstats(ss, min_n_anchor="max")
    keep_med, log_med = qc_sumstats(ss, min_n_anchor="median")

    # max anchor: 0.7 * 180,000 = 126,000 -> the whole 90k arm is discarded
    assert log_max["n_anchor"] == "max"
    assert log_max["n_anchor_value"] == 180_000.0
    assert log_max["n_drop_lowN"] == 900
    assert keep_max.sum() == 100

    # median anchor: 0.7 * 90,000 = 63,000 -> every variant survives
    assert log_med["n_anchor"] == "median"
    assert log_med["n_anchor_value"] == 90_000.0
    assert log_med["n_drop_lowN"] == 0
    assert keep_med.sum() == m


def test_min_n_anchor_rejects_an_unknown_rule():
    ss = _ss(10)
    with pytest.raises(ValueError, match="min_n_anchor"):
        qc_sumstats(ss, min_n_anchor="mean")


def test_min_n_ratio_must_be_in_unit_interval():
    ss = _ss(4)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="min_n_ratio"):
            qc_sumstats(ss, min_n_ratio=bad)
    keep, log = qc_sumstats(ss, min_n_ratio=1.0)
    assert keep.all()
    assert log["n_drop_lowN"] == 0


def test_impute_n_anchor_survives_one_corrupt_standard_error():
    """A single near-zero se must not rescale every other variant to zero.

    The anchor matches a high quantile of `1/(se^2 * 2f(1-f))` to the reported
    total. That quantity explodes as se -> 0, so one corrupt entry becomes the
    anchor and drives everything else toward zero -- measured at median 50,000
    -> 0 with a single se=1e-6 among 90 clean variants. Published sumstats
    routinely carry a few such entries. The plot of sd_ss against sd_af shows
    this immediately by eye, which does not scale to a sweep of hundreds of
    traits, so the anchor excludes implausible values itself and counts them.
    """

    rng = np.random.default_rng(0)
    m = 90
    af = rng.uniform(0.1, 0.5, m)
    se = 1.0 / np.sqrt(50_000 * 2 * af * (1 - af))
    clean, clean_log = impute_n_eff(se, af, 50_000.0)

    corrupt = se.copy()
    corrupt[0] = 1e-6
    imputed, log = impute_n_eff(corrupt, af, 50_000.0)

    # the anchor is a quantile of per-bin medians, so one corrupt value cannot
    # move the bin it lands in, let alone the scale
    assert log["anchor_basis"].startswith("median of")
    assert log["anchor_ref"] == pytest.approx(clean_log["anchor_ref"], rel=1e-9)
    # every other variant keeps the N it had before the corruption
    np.testing.assert_allclose(imputed[1:], clean[1:], rtol=1e-9)
    assert np.median(imputed) == pytest.approx(50_000.0, rel=1e-6)


def test_impute_n_anchor_survives_widespread_corruption():
    """Half a percent of corrupt standard errors must still not move the scale.

    A per-bin median has a 50% breakdown point, so the anchor holds until a
    bin is majority-corrupt -- unlike a raw order statistic, which one bad
    entry owns.
    """

    rng = np.random.default_rng(3)
    m = 20_000
    af = rng.uniform(0.05, 0.5, m)
    n_true = np.where(rng.random(m) < 0.3, 20_000.0, 50_000.0)
    se = 1.0 / np.sqrt(n_true * 2 * af * (1 - af))
    clean, _ = impute_n_eff(se, af, 50_000.0)

    corrupt = se.copy()
    corrupt[rng.choice(m, m // 200, replace=False)] = 1e-6      # 0.5%
    imputed, log = impute_n_eff(corrupt, af, 50_000.0)

    assert log["anchor_basis"].startswith("median of per-bin")
    assert np.median(imputed) == pytest.approx(np.median(clean), rel=1e-6)
    # and the per-variant pattern is still recovered
    assert np.corrcoef(clean, n_true)[0, 1] > 0.99


def test_impute_n_anchor_is_unbiased_on_a_genuinely_ragged_n():
    """Robustness must not shift what the anchor estimates.

    The scale is set so the best-typed variants carry the reported total.
    Binning per your neighbouring-values suggestion makes that resistant, but
    only if the high quantile is taken *within* each bin and the median
    *across* bins. Taking bin medians first anchors the TYPICAL variant to
    n_total instead, inflating every imputed N -- on a distribution whose
    median is 0.39 of its maximum that is a 2.5x error, invisible to a
    corruption test because both versions survive corruption.
    """

    rng = np.random.default_rng(0)
    m = 200_000
    af = rng.uniform(0.05, 0.5, m)
    n_true = np.clip(50_000 * rng.beta(2.0, 3.0, m), 2_000, None)
    n_true[rng.random(m) < 0.02] = 50_000.0          # a small fully-typed tail
    se = 1.0 / np.sqrt(n_true * 2 * af * (1 - af))

    imputed, log = impute_n_eff(se, af, 50_000.0)

    assert np.median(n_true) < 0.5 * n_true.max()    # the case that discriminates
    assert np.median(imputed) == pytest.approx(np.median(n_true), rel=0.02)
    assert imputed.max() == pytest.approx(50_000.0, rel=0.02)
    assert np.corrcoef(imputed, n_true)[0, 1] > 0.99


def test_gc_correction_declines_on_a_panel_too_small_to_fit():
    """An LD-score regression on a handful of variants is not a measurement.

    Sibling packages bridge into ldpred3 with toy fixtures -- gwfm's bridge
    tests fit four variants on identity LD -- where the intercept is
    meaningless but a rescale of the standard errors would be very real.
    """
    # the same intercept that legitimately fires on a genome-wide panel
    assert genomic_control_factor(0.6494, 0.0126, n_variants=917_785) < 1.0
    assert genomic_control_factor(0.6494, 0.0126, n_variants=4) == 1.0
    assert genomic_control_factor(0.6494, 0.0126, n_variants=49_999) == 1.0
    # unknown count keeps the historical behaviour
    assert genomic_control_factor(0.6494, 0.0126) < 1.0


def test_should_impute_n_tri_state():
    constant = np.full(6, 20000.0)
    ragged = np.array([5000.0, 20000.0, 20000.0])
    assert should_impute_n(None, constant) == (True, "constant_n")
    assert should_impute_n(None, ragged) == (False, "ragged_n")
    assert should_impute_n(True, ragged) == (True, "requested")
    assert should_impute_n(False, constant) == (False, "disabled")
    with pytest.raises(ValueError, match="impute_n"):
        should_impute_n("yes", constant)


def test_impute_n_auto_declines_a_caller_supplied_scalar():
    """Auto refines a file-wide N; it does not overwrite the caller's own.

    The commonest invocation passes a scalar -- ``--n-eff``, or case/control
    counts turned into 4/(1/Nca+1/Nco) -- and that number is the caller's
    arithmetic, not a placeholder to be repaired.
    """
    constant = np.full(100, 250_000.0)
    assert should_impute_n(None, constant) == (True, "constant_n")
    assert should_impute_n(None, constant, n_from_caller=True) == (
        False, "constant_n_supplied_by_caller")
    # An explicit request still wins in both directions.
    assert should_impute_n(True, constant, n_from_caller=True) == (
        True, "requested")
    assert should_impute_n(False, constant) == (False, "disabled")
    # A real per-variant column is never replaced, whatever the flag's origin.
    ragged = np.arange(1e5, 1e5 + 100)
    assert should_impute_n(None, ragged) == (False, "ragged_n")
    assert should_impute_n(None, ragged, n_from_caller=True) == (
        False, "ragged_n")
    assert should_impute_n(None, np.full(4, np.nan)) == (False, "no_finite_n")


def test_split_half_statistic_is_calibrated_under_the_null():
    """T should look like chi2_1 when the z really are consistent with the LD."""
    rng = np.random.default_rng(3)
    k = 300
    R = _ar1(k, 0.6)
    chol = np.linalg.cholesky(R + 1e-8 * np.eye(k))
    stats = []
    for _ in range(30):
        z = chol @ rng.standard_normal(k)
        order = rng.permutation(k)
        stats.append(dentist_statistic(R, z, order[:k // 2], order[k // 2:]))
    stats = np.concatenate(stats)
    # chi2_1 has median 0.455; allow generous slack for a finite, LD-correlated
    # sample -- this catches a statistic that is wrong by a factor, not by noise.
    assert 0.2 < np.median(stats) < 1.2, np.median(stats)


def test_dentist_statistic_validates_floor_indices_and_shapes():
    R = np.eye(4)
    for floor in (np.nan, -1e-3, 1.0):
        with pytest.raises(ValueError, match="eigenvalue_floor"):
            dentist_statistic(R, np.zeros(4), [0, 1], [2, 3],
                              eigenvalue_floor=floor)
    with pytest.raises(ValueError, match="disjoint"):
        dentist_statistic(R, np.zeros(4), [0, 1], [1, 2])
    with pytest.raises(ValueError, match="matching finite vector"):
        dentist_statistic(R, np.zeros((4, 1)), [0, 1], [2, 3])


def test_dense_float_window_matches_the_float32_ld_values():
    R = _ar1(8, 0.37).astype(np.float64)
    R[1, 3] = R[3, 1] = np.nextafter(R[1, 3], 1.0)
    local = np.array([1, 3, 6])
    expected = R.astype(np.float32)[np.ix_(local, local)].astype(np.float64)
    np.testing.assert_array_equal(_window_ld(R, local), expected)
