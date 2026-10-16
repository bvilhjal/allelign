"""Contracts needed by the family without importing a method package."""

import numpy as np
import pytest

from allelign import implied_sample_size, in_long_range_ld, resolve_gwas_catalog_columns


@pytest.mark.parametrize("strict", [False, True])
def test_implied_n_compares_medians_for_variant_reported_n(strict):
    af = np.array([0.1, 0.3, 0.7])
    beta = np.array([0.01, -0.02, 0.03])
    truth = np.array([1000., 2000., 8000.])
    se = np.sqrt((4 / (2 * af * (1 - af)) - beta**2) / truth)
    result = implied_sample_size(beta, se, af, binary=True,
                                 reported_n=[500., 2000., 16000.], strict=strict)
    assert result["median"] == pytest.approx(2000.)
    assert result["ratio"] == pytest.approx(1.)
    assert result["consistent"]


def test_implied_n_preserves_both_nonpositive_estimate_policies():
    args = ([4., 4., 0.], [1., 1., 1.], [.5, .5, .5])
    assert implied_sample_size(*args, binary=True)["median"] == 8.
    assert implied_sample_size(*args, binary=True, strict=True)["median"] == pytest.approx(-8.)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("field,value", [
    ("beta", [np.nan, 0.]), ("se", [0., 1.]), ("af", [0., .5]),
    ("reported_n", [1., np.inf]), ("reported_n", [1.]),
])
def test_strict_implied_n_validates_before_reporting_unidentified(binary, field, value):
    args = dict(beta=[0., 0.], se=[1., 1.], af=[.5, .5], reported_n=[10., 10.])
    args[field] = value
    with pytest.raises(ValueError):
        implied_sample_size(**args, binary=binary, strict=True)


def test_region_mask_preserves_array_shape_and_inclusive_boundaries():
    chrom = [["chr6", "06"], ["6", "chr19"]]
    pos = [[25_000_000, 35_000_000], [35_000_001, 45_000_000]]
    np.testing.assert_array_equal(in_long_range_ld(chrom, pos),
                                  [[True, True], [False, False]])
    assert in_long_range_ld("chr6", 30_000_000).item()


def test_catalog_schema_keeps_harmonised_frequency_with_its_allele():
    header = ["hm_rsid", "hm_effect_allele", "hm_other_allele", "hm_beta",
              "standard_error", "effect_allele_frequency", "N"]
    mapping, odds, z = resolve_gwas_catalog_columns(header)
    assert mapping["effect_allele"] == "hm_effect_allele"
    assert mapping["n"] == "N"
    assert "eaf" not in mapping
    mapping, _, _ = resolve_gwas_catalog_columns(header + ["hm_effect_allele_frequency"])
    assert mapping["eaf"] == "hm_effect_allele_frequency"
    assert odds is None and z is None


@pytest.mark.parametrize("effect", ["hm_odds_ratio", "Zscore"])
def test_catalog_schema_preserves_fallback_columns_and_provider_priority(effect):
    header = ["rsid", "effect_allele", "other_allele", "se", effect,
              "meta_se", "N", "effective_n"]
    mapping, odds, z = resolve_gwas_catalog_columns(
        header, {"se": ("meta_se",), "n": ("effective_n",)})
    assert mapping["se"] == "meta_se"
    assert mapping["n"] == "effective_n"
    assert (odds, z) == ((effect, None) if effect == "hm_odds_ratio" else (None, effect))


def test_catalog_schema_rejects_an_effect_without_direction():
    with pytest.raises(ValueError, match="beta/odds_ratio/z_score"):
        resolve_gwas_catalog_columns(["rsid", "effect_allele", "other_allele", "se", "p"])
