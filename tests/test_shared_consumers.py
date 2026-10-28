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


def _binary_sumstats(n_eff, n_variants=5000, seed=0):
    """Case/control sumstats whose implied N is ``n_eff`` by construction."""
    rng = np.random.default_rng(seed)
    af = rng.uniform(0.05, 0.95, n_variants)
    sd_ref = np.sqrt(2 * af * (1 - af))
    beta = rng.standard_normal(n_variants) * 1e-3
    se = np.sqrt(np.maximum(4.0 / sd_ref**2 - beta**2, 1e-12) / n_eff)
    return beta, se, af


@pytest.mark.parametrize("strict", [False, True])
def test_implied_n_recovers_a_scalar_n_and_flags_an_overstated_one(strict):
    """The absolute N is recoverable for case/control (bipred uses strict=True)."""
    beta, se, af = _binary_sumstats(80_000.0, seed=3)
    out = implied_sample_size(beta, se, af, binary=True, reported_n=80_000.0, strict=strict)
    assert abs(out["median"] - 80_000.0) / 80_000.0 < 0.02
    assert out["consistent"] is True
    # CARDIoGRAM CAD: reported 162,973, implied 92,966, ratio 0.570.
    beta, se, af = _binary_sumstats(92_966.0, seed=4)
    out = implied_sample_size(beta, se, af, binary=True, reported_n=162_973.0, strict=strict)
    assert out["median"] == pytest.approx(92_966.0, rel=1e-3)
    assert out["ratio"] == pytest.approx(0.570, abs=0.01)
    assert out["consistent"] is False


@pytest.mark.parametrize("strict", [False, True])
def test_quantitative_implied_n_is_unidentified(strict):
    """A quantitative trait's phenotype scale is unknown: calibrating it from
    reported_n would force the ratio to one, so nothing is reported."""
    beta, se, af = _binary_sumstats(80_000.0, seed=5)
    for reported in (80_000.0, 720_000.0):
        out = implied_sample_size(beta, se, af, reported_n=reported, strict=strict)
        assert np.isnan(out["median"]) and np.isnan(out["ratio"])
        assert out["consistent"] is False


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("field,value,match", [
    ("beta", [np.nan, 0.], "beta contains non-finite"),
    ("se", [0., 1.], "se must contain finite positive"),
    ("se", [np.nan, 1.], "se must contain finite positive"),
    ("af", [0., .5], "af must contain finite values"),
    ("af", [np.nan, .5], "af must contain finite values"),
    ("af", [1., .5], "af must contain finite values"),
    ("af", [2., .5], "af must contain finite values"),
    ("beta", [[0.], [0.]], "equal-length vectors"),
    ("se", [[1.], [1.]], "equal-length vectors"),
    ("af", [[.5], [.5]], "equal-length vectors"),
    ("reported_n", [1., np.inf], "finite positive"),
    ("reported_n", [1.], "scalar or a vector matching beta"),
    ("reported_n", 0., "finite positive"),
    ("reported_n", -1., "finite positive"),
    ("reported_n", np.nan, "finite positive"),
    ("reported_n", np.inf, "finite positive"),
])
def test_strict_implied_n_validates_before_reporting_unidentified(binary, field, value, match):
    args = dict(beta=[0., 0.], se=[1., 1.], af=[.5, .5], reported_n=[10., 10.])
    args[field] = value
    with pytest.raises(ValueError, match=match):
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


def test_catalog_schema_keeps_a_submitted_frequency_beside_submitted_alleles():
    """Without hm_ alleles the submitted frequency describes its own allele."""
    mapping, _, _ = resolve_gwas_catalog_columns(
        ["variant_id", "chromosome", "base_pair_location", "effect_allele",
         "other_allele", "beta", "standard_error", "p_value",
         "effect_allele_frequency"])
    assert mapping["eaf"] == "effect_allele_frequency"


@pytest.mark.parametrize("effect", ["hm_odds_ratio", "Zscore"])
def test_catalog_schema_preserves_fallback_columns_and_provider_priority(effect):
    prefix = "hm_" if effect.startswith("hm_") else ""
    header = ["rsid", prefix + "effect_allele", prefix + "other_allele", "se", effect,
              "meta_se", "N", "effective_n"]
    mapping, odds, z = resolve_gwas_catalog_columns(
        header, {"se": ("meta_se",), "n": ("effective_n",)})
    assert mapping["se"] == "meta_se"
    assert mapping["n"] == "effective_n"
    assert (odds, z) == ((effect, None) if effect == "hm_odds_ratio" else (None, effect))


@pytest.mark.parametrize("hm_effect", ["hm_beta", "hm_odds_ratio", "hm_z_score"])
def test_catalog_effects_and_every_fallback_share_the_allele_orientation(hm_effect):
    header = ["rsid", "effect_allele", "other_allele", "beta", "odds_ratio",
              "z_score", "effect_allele_frequency", "se", "hm_effect_allele",
              "hm_other_allele", hm_effect]
    row = dict(zip(header, ["rs1", "G", "A", "0.1", str(np.exp(.1)), "10",
                            ".2", ".01", "A", "G",
                            {"hm_beta": "-.1", "hm_odds_ratio": str(np.exp(-.1)),
                             "hm_z_score": "-10"}[hm_effect]]))
    mapping, odds, zscore = resolve_gwas_catalog_columns(header)
    assert row[mapping["effect_allele"]] == "A"
    assert row[mapping["other_allele"]] == "G"
    assert "eaf" not in mapping
    # Follow the consumer's beta -> OR/z fallback using an independently
    # specified allele swap, not another copy of the resolver's rules.
    effects = ([float(row[mapping["beta"]])] if "beta" in mapping else [])
    if odds:
        effects.append(np.log(float(row[odds])))
    if zscore:
        effects.append(float(row[zscore]) * float(row[mapping["se"]]))
    np.testing.assert_allclose(effects, [-.1])


def test_catalog_uses_submitted_set_without_a_harmonised_effect():
    mapping, odds, zscore = resolve_gwas_catalog_columns(
        ["rsid", "effect_allele", "other_allele", "beta", "se",
         "effect_allele_frequency", "hm_effect_allele", "hm_other_allele",
         "hm_effect_allele_frequency"])
    assert mapping["effect_allele"] == "effect_allele"
    assert mapping["other_allele"] == "other_allele"
    assert mapping["eaf"] == "effect_allele_frequency"
    assert odds is None and zscore is None


@pytest.mark.parametrize("fields", [
    ["hm_effect_allele", "hm_other_allele", "beta"],
    ["effect_allele", "other_allele", "hm_beta"],
    ["hm_effect_allele", "other_allele", "hm_odds_ratio"],
])
def test_catalog_rejects_incomplete_orientation_sets(fields):
    with pytest.raises(ValueError, match="unmappable columns"):
        resolve_gwas_catalog_columns(["rsid", "se"] + fields)


def test_catalog_provider_effect_override_keeps_submitted_alleles():
    mapping, odds, zscore = resolve_gwas_catalog_columns(
        ["rsid", "se", "effect_allele", "other_allele", "beta", "meta_beta",
         "hm_effect_allele", "hm_other_allele", "hm_beta", "hm_odds_ratio"],
        {"beta": ("meta_beta",)})
    assert mapping["beta"] == "meta_beta"
    assert mapping["effect_allele"] == "effect_allele"
    assert mapping["other_allele"] == "other_allele"
    assert odds is None and zscore is None


def test_catalog_schema_rejects_an_effect_without_direction():
    with pytest.raises(ValueError, match="beta/odds_ratio/z_score"):
        resolve_gwas_catalog_columns(["rsid", "effect_allele", "other_allele", "se", "p"])
