"""allelign: read, align and quality-control GWAS summary statistics and PGS weights.

One place for the steps every downstream method repeats -- parsing the many
summary-statistics dialects, getting effect alleles right against a reference,
and the checks that catch bad input before it becomes a confident wrong
number. NumPy-only; Numba and SciPy speed up the parser and the LD screen when
installed.
"""

__version__ = "0.1.0.dev4"

from .alleles import (allele_orientation, is_ambiguous, normalize_chrom,
                      reverse_complement)
from .build import BuildMismatchError, check_build, normalize_build
from .genotypes import Bed, ld_consistency_screen_genotypes
from .align import (Harmonized, HarmonizedWeights,
                    count_positional_only_matches, coverage_message,
                    diagnose_match, harmonize, harmonize_weights,
                    reanchor_on_identifier, reference_coverage,
                    variant_indices)
from .ldcheck import (chromosome_blocks, dense_blocks, dentist_statistic,
                      ld_consistency_screen_selected,
                      ld_blocks_from_genotypes, ld_consistency_screen,
                      loo_statistic)
from .qc import (af_concordance_mask, genomic_control_factor,
                 genomic_inflation, gwas_af_on_counted_allele,
                 implied_sample_size, impute_n_eff, in_long_range_ld,
                 qc_sumstats, sd_consistency_mask, should_impute_n)
from .status import (QC_REASON_NAMES, new_status, status_counts,
                     write_status_table)
from .sumstats import (ColumnMappingError, Sumstats, anchor_n_eff,
                       detect_columns, inspect_columns, n_eff_case_control,
                       read_sumstats, rescale_n_eff, write_sumstats)
from .variants import Variants, read_bim, read_pvar, read_variants
from .weights import (ScoringFile, WeightsTable, read_scoring_file,
                      read_weights, scoring_file_id, write_pgs_scoring_file,
                      write_weights)

__all__ = [
    "__version__",
    # reading and writing
    "Sumstats", "read_sumstats", "write_sumstats", "detect_columns",
    "inspect_columns", "ColumnMappingError", "n_eff_case_control",
    "rescale_n_eff", "anchor_n_eff",
    "Variants", "read_bim", "read_pvar", "read_variants", "Bed",
    "WeightsTable", "read_weights", "write_weights", "ScoringFile",
    "read_scoring_file", "scoring_file_id", "write_pgs_scoring_file",
    # alignment
    "Harmonized", "harmonize", "HarmonizedWeights", "harmonize_weights",
    "reanchor_on_identifier", "reference_coverage", "coverage_message",
    "diagnose_match", "count_positional_only_matches", "variant_indices",
    "normalize_chrom", "reverse_complement", "is_ambiguous",
    "allele_orientation", "normalize_build", "check_build",
    "BuildMismatchError",
    # quality control
    "qc_sumstats", "sd_consistency_mask", "af_concordance_mask",
    "gwas_af_on_counted_allele", "should_impute_n", "impute_n_eff",
    "genomic_control_factor", "implied_sample_size", "genomic_inflation",
    "in_long_range_ld", "ld_consistency_screen",
    "ld_consistency_screen_selected", "dentist_statistic",
    "loo_statistic", "ld_blocks_from_genotypes", "dense_blocks",
    "chromosome_blocks", "ld_consistency_screen_genotypes",
    # per-variant status
    "QC_REASON_NAMES", "new_status", "status_counts", "write_status_table",
]
