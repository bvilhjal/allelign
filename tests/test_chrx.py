"""Chromosome-X handling in allelign: PAR aliases, PAR matching, SD-check anchoring.

PLINK labels pseudoautosomal variants 25/XY while sumstats label them X (so
they would silently never match), and the SD-consistency check must not anchor
every chromosome on the autosomal sqrt(2f(1-f)) scale. Moved from ldpred3's
test_chrx.py, where they were the only coverage of this allelign code.
"""

import numpy as np
import pytest

from allelign.align import harmonize
from allelign.alleles import normalize_chrom
from allelign.qc import sd_consistency_mask
from allelign.sumstats import Sumstats
from allelign.variants import Variants as VariantTable


def test_norm_chrom_par_aliases():
    # PLINK's pseudoautosomal code 25 (and the XY spelling) normalises to X:
    # PAR loci carry X coordinates in PLINK filesets and are labelled X in
    # sumstats, so this is the direction that matches.
    assert normalize_chrom("25") == "X" and normalize_chrom("XY") == "X"
    assert normalize_chrom("chrX") == "X" and normalize_chrom("23") == "X"
    assert normalize_chrom("24") == "Y" and normalize_chrom("26") == "MT"


def test_par_variants_match_x_sumstats():
    # Genotypes carry PLINK PAR labels; the GWAS labels the same loci X. One
    # variant shares its rsID (coordinate-validated id match), the other does
    # not (position match) -- both must land.
    variants = VariantTable(
        chrom=np.array(["25", "XY"], dtype=object),
        id=np.array(["rsPAR1", "rsPAR2"], dtype=object),
        cm=np.zeros(2),
        pos=np.array([100, 200], dtype=np.int64),
        a1=np.array(["A", "C"], dtype=object),
        a2=np.array(["G", "T"], dtype=object))
    ss = Sumstats(
        id=np.array(["rsPAR1", "gwasOnly"], dtype=object),
        chrom=np.array(["X", "X"], dtype=object),
        pos=np.array([100, 200], dtype=np.int64),
        ea=np.array(["A", "C"], dtype=object),
        oa=np.array(["G", "T"], dtype=object),
        beta=np.array([0.10, -0.05]), se=np.array([0.01, 0.01]),
        n_eff=np.array([1000.0, 1000.0]),
        eaf=np.array([np.nan, np.nan]), info=np.array([np.nan, np.nan]))
    h = harmonize(ss, variants)
    assert list(h.var_index) == [0, 1]
    assert h.log["n_matched"] == 2
    np.testing.assert_allclose(h.beta, [0.10, -0.05])


def test_sd_check_anchors_sex_chromosomes_separately():
    # X dosages in a mixed-sex GWAS deviate from sqrt(2f(1-f)) by a constant
    # factor (here 1.3); per-group median anchoring absorbs it, one global
    # anchor cannot.
    rng = np.random.default_rng(0)
    n_auto, n_x = 200, 200
    af = rng.uniform(0.1, 0.9, n_auto + n_x)
    chrom = np.array(["1"] * n_auto + ["X"] * n_x, dtype=object)
    scale = np.concatenate([np.ones(n_auto), np.full(n_x, 1.3)])
    sd_ss_target = np.sqrt(2 * af * (1 - af)) * scale
    n_eff = np.full(n_auto + n_x, 10000.0)
    beta = np.zeros(n_auto + n_x)
    se = 1.0 / (sd_ss_target * np.sqrt(n_eff))   # sd_ss == sd_ss_target exactly
    bounds = (0.9, 1.1)

    keep_no, log_no, _ = sd_consistency_mask(
        beta, se, n_eff, af, rule="ratio", sd_ratio_bounds=bounds)
    assert not keep_no[n_auto:].any()             # X rows dropped wholesale
    assert log_no["n_nonautosomal"] == 0
    assert log_no["nonautosomal_sd_scale"] is None

    keep, log, _ = sd_consistency_mask(
        beta, se, n_eff, af, chrom=chrom, rule="ratio", sd_ratio_bounds=bounds)
    assert keep.all()
    assert log["n_nonautosomal"] == n_x
    assert log["nonautosomal_sd_scale"] == pytest.approx(1 / 1.3, abs=1e-4)
