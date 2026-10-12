# Changelog

All notable changes are recorded here, in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) style.

## [Unreleased]

## [0.1.0.dev4] - 2026-10-12

### Fixed

- `read_sumstats`, `detect_columns` and `inspect_columns` take N from an effective-N
  column (`Effective_N`, `N_effective`, `all_inv_var_meta_effective`) beside a total-N
  column such as `N`, as documented. They used to read the total, ~2.8x too high at 10% cases.

## [0.1.0.dev3] - 2026-10-12

### Changed

- Gzip weight and status tables use compression level 1 to reduce writing
  time. Decompressed text and atomic publication remain unchanged.

## [0.1.0.dev2] - 2026-10-10

### Added (0.1.0.dev2)

- `read_scoring_file(keep=...)` prefilters rows a reference cannot match, as multipgs's private `_reference_filter` did. `harmonize_weights` takes the weight mass over the whole file and counts the skipped rows as unmatched. `ScoringFile` gains `trait` and `__str__`, and `scoring_file_id` is public: multipgs now reads PGS Catalog files through allelign.
- Public `variant_indices`, the identifier index `harmonize` caches per table, for callers that look up identifiers against a reference.

### Fixed (0.1.0.dev2)

- A GWAS Catalog file read through its `hm_` columns without `hm_effect_allele_frequency` no longer pairs the submitted frequency with the harmonised alleles. The frequency is dropped and `parse_log["layout"]["eaf_dropped"]` says why.

### Added (0.1.0.dev1)

- First version, extracted from ldpred3 0.9.x with pieces of multipgs and bipred:
  - summary-statistics reader with the alias table, OR-to-beta, SE from p, and the optional Numba block parser;
  - harmoniser with both the vectorised join and the row loop, coverage and match diagnostics, and identifier re-anchoring;
  - file-level and reference QC (SD consistency, allele-frequency concordance with its safety valve, imputed N, genomic-control undo, implied N);
  - the split-half LD-consistency screen with full-window confirmation;
  - ldpred3 weight tables, PGS Catalog reading (from multipgs) and writing;
  - the per-variant status vocabulary;
  - the GRCh37 long-range LD regions (from bipred).
- Beyond ldpred3:
  - effective N from per-variant case and control counts;
  - PLINK2 `A1` resolved against `REF`/`ALT` (or `OMITTED`) row by row;
  - REGENIE `GENPOS` as position;
  - SE from `-log10 p` (`LOG10P`, `mlogp`);
  - GWAS Catalog `hm_` columns used as one consistent set;
  - PGC `FRQ_U_*` as the effect-allele frequency;
  - Neale `chr:pos:ref:alt` variant strings.
- `.bim`/`.pvar` readers, `harmonize_weights` (aligns frequencies and reports matched weight mass), `genomic_inflation`, LD blocks from reference genotypes, and a CLI (`inspect`, `clean`, `align`, `pgs-export`).
- `Bed`, a NumPy-only PLINK 1 `.bed` reader. It reads selected variants with coalesced seeks and can restrict to some samples (IIDs, indices or a mask).
- `ld_consistency_screen_genotypes` screens with LD formed block by block from a `.bed`, so a genome-wide reference never holds more than one block of LD. It is also available as `allelign align --ld-screen [--ld-keep FILE]`.
- Hooks ldpred3 uses to consume the screen: `ld_consistency_screen_selected` (screen a sparse selection without materialising it), `ncores` block pooling, `stats_out`, and a private eigensolver hook.
- CI (lint; py3.9 with numpy 1.20 and no Numba; py3.12 with Numba).

### Consumers

- ldpred3 0.9.9 imports allelign for reading, harmonisation, QC, weights and the split-half screen.
- ppb 0.0.1.dev10 takes its chromosome, reverse-complement and strand-ambiguity rules from `allelign.alleles`.
- bipred, gwfm, multipgs and SMARTpred (from 0.1.0.dev2) import reading, alignment, QC and weight-file I/O directly; multipgs's PGS Catalog reader is allelign's.

### Fixed (relative to the ldpred3 originals)

- The PGS Catalog export no longer claims `#sd_source=empirical` for a weights table that records no SD source; it writes `unspecified` and warns.
- The long-range LD mask normalises chromosome labels, so `chr6` matches.
