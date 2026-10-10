# Changelog

All notable changes are recorded here, in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) style.

## [Unreleased]

### Added

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

### Fixed (relative to the ldpred3 originals)

- The PGS Catalog export no longer claims `#sd_source=empirical` for a weights table that records no SD source; it writes `unspecified` and warns.
- The long-range LD mask normalises chromosome labels, so `chr6` matches.
