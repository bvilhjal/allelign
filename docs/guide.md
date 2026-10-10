# allelign guide

## 1. Reading summary statistics

`read_sumstats` maps header names onto the fields `id chrom pos ea oa beta se n_eff eaf info`, case-insensitively, using `allelign.sumstats.COLUMN_ALIASES`. Any field can be forced to a column by name or zero-based index, e.g. `read_sumstats(path, ea="A1", beta=6)`. Use `inspect_columns(path)` (or `allelign inspect`) to see the mapping before a long read.

- **Effects.** An OR column becomes `beta = log(OR)`. A missing SE is recovered as `|beta| / z`, with z from the p-value or from `-log10 p`; tiny p-values are handled in log space. A BETA column that looks like odds ratios (almost all positive, centred near 1) warns, but nothing is inferred.
- **Frequencies.** Only effect-allele frequencies are mapped automatically. MAF is not, because it does not say which allele it counts.
- **Missing values.** A blank or non-numeric position becomes 0 (missing). Empty lines are skipped.

**Table 1. Layouts recognised from the header.**

| Layout | Rule |
|---|---|
| SAIGE | `Allele2` is the effect allele (`AF_Allele2`, `imputationInfo`) |
| PLINK2 | `A1` is tested: the other allele is `OMITTED`, or row by row `ALT` where `A1 == REF`, else `REF` |
| REGENIE | `GENPOS` is the position when there is no `BP`; `LOG10P` recovers SE |
| BOLT | `BP` wins over `GENPOS` (genetic position) |
| GWAS Catalog harmonised | when `hm_effect_allele` and `hm_other_allele` exist, the `hm_` columns are used as one set; harmonised alleles without `hm_beta`/`hm_odds_ratio` refuse |
| PGC daner | `FRQ_U_<n>` (control frequency of A1) is the EAF when no EAF column exists |
| Neale UKB | `variant` = `chr:pos:ref:alt`, effect on ALT |
| FinnGen | `alt` is the effect allele, `af_alt` the EAF |

### Sample size

Per-variant N comes from, in order:

1. an explicit override (`n_eff="col"`) or a scalar (`n_eff=…`, or `n_cases=…, n_controls=…`, which flatten every row);
2. an explicit effective-N column (`Neff`, `n_eff`);
3. per-variant case and control counts, giving `4/(1/N_case + 1/N_control)` (warns);
4. a total-N column.

`parse_log["n_eff_source"]` records which. For a meta-analysis, effective N from summed counts overstates the true value when case fractions differ between cohorts; prefer a published `Neff`. `anchor_n_eff(ss, target)` rescales a per-variant column's median down to a supplied N, and never scales it up.

## 2. Alignment

`harmonize(ss, variants)` aligns effects to the reference's counted allele `a1`. The rules, in order:

1. **Locate.** Rows match by identifier. A unique identifier whose coordinates disagree with the reference is unmatched, which catches a build mix-up. An identifier held at several loci is resolved by coordinates or rejected. A row with no usable identifier matches by normalised `(chrom, pos)`.
2. **Drop ambiguous pairs.** A/T, C/G and multi-base pairs such as `AA`/`TT` are dropped by default.
3. **Orient.** Alleles are compared as given, swapped, complemented, then complemented and swapped. A swap negates beta.
4. **Claim.** Each reference variant is taken by at most one row, the first in file order.

Chromosome labels normalise as follows: the `chr` prefix and zero padding are dropped, `23`/`25`/`XY` become `X`, `24` becomes `Y`, and `26`/`M` become `MT`.

`reference_coverage` classifies a thin alignment as build mismatch, absent identifiers, rows rejected at a matched locus, or a sparse (hits-only) deposit. `coverage_message` states the cause and the remedy.

For a known build mismatch:

- `check_build` refuses mismatched declared builds.
- `reanchor_on_identifier` moves rows onto the reference's coordinates for their identifiers. It drops rows whose identifier is absent, held at several loci, or missing. It is not a liftover.

`harmonize_weights` applies the same rules to an ldpred3 weights table or a PGS Catalog score, aligns `AF_REF` (`1 - f` on a swap), and reports the matched share of squared weight.

## 3. Quality control

**Table 2. Checks and defaults.**

| Check | Default | Drops |
|---|---|---|
| `qc_sumstats` validity | beta, SE > 0 and N > 0 finite | invalid rows |
| duplicates | on | every copy of a repeated id (or chrom:pos:alleles) |
| relative N | `n_eff >= 0.7 × median` | low-N rows |
| MAF / INFO | 0.01 / 0.7, where present | rare / poorly imputed |
| chi-square cap | off | `(beta/se)^2 > max_chisq` |
| `sd_consistency_mask` | ratio rule, 0.5-2 after median anchoring | inconsistent and untestable rows |
| `af_concordance_mask` | `|Δf| > 0.1` | nothing if more than 20% fail (ancestry mismatch or orientation) |
| `ld_consistency_screen` | window 1000, χ² 29.72, 4 rounds | confirmed LD-inconsistent z |

Diagnostics that drop nothing:

- `impute_n_eff`: per-variant N (Privé et al. 2022).
- `genomic_control_factor`: undoes genomic control when the LDSC intercept is established below 1.
- `implied_sample_size`: binary traits only.
- `genomic_inflation`: λGC.
- `in_long_range_ld`: GRCh37 regions; any other build refuses unless you pass `regions=`.

Every check accepts `status=`, an int8 array that records the first stage dropping each row. The codes match ldpred3's `--qc-out` vocabulary.

### LD-consistency screen

The screen takes LD as `[(R, idx), …]` blocks partitioning the variants. `R` can be any of:

- dense float;
- int8 on ldpred3's `round(127 r)` scale;
- low-rank (`U`, `residual_diag`, `scale`);
- any object with a `window(rows)` method.

`ld_blocks_from_genotypes` and `chromosome_blocks` build blocks from a genotype matrix in memory. For a PLINK reference, `ld_consistency_screen_genotypes(Bed(...), var_index, z)` reads one block of genotypes at a time, forms its LD, screens it and releases it. `Bed(..., samples=...)` restricts LD to one ancestry of a mixed panel. Each block's random splits come from a stream fixed by `(seed, block number)`. The CLI exposes this as `allelign align --ld-screen [--ld-keep FILE]`; dropped rows get the `dentist_outlier` status.

**How it works.** Each window is split at random into halves. Each half predicts the other through the LD (DENTIST's statistic, Chen et al. 2021). A flag is confirmed by the full-window leave-one-out statistic. The variant is dropped only if it has an LD neighbour and is not a private effect consistent with `R`.

**Limits.** It is not the complete published DENTIST procedure, so validate the mask for your study. A drop rate of several percent usually means the LD is not the matrix the GWAS was drawn against.

Reference size matters. The null calibration assumes LD sampling noise is small relative to the z-scores' precision. On chr22 of Yengo 2022 EUR height (N about 1.6M), aligned to 1000G with LD from its 503 EUR samples, the screen dropped 1,300 of 14,582 variants (9%); the CLI path gives the same count (8 s, 0.57 GB). That mostly measures the small reference, not bad statistics. With a very large GWAS, use a reference of comparable quality (for example UK Biobank LD), or read the mask as a diagnostic.

## 4. PGS effect sizes

- **ldpred3 tables** (`read_weights`/`write_weights`) hold standardised weights, with optional `AF_REF SD_REF SD_SOURCE`.
- **PGS Catalog files** (`read_scoring_file`):
  - weights are per allele count;
  - `hm_` columns are preferred, and `hm_inferOtherAllele` fills a missing other allele;
  - OR/HR weights are logged;
  - non-additive rows are dropped.
- **`write_pgs_scoring_file`** converts an ldpred3 table to a Catalog file with `effect_weight = WEIGHT/SD_REF`. The genome build is required, and `#sd_source` is `unspecified` when the table does not record it. For plink2 `--score`, strip the `#` lines first.

## 5. Deliberate differences from ldpred3

ldpred3 0.9.9 reads and aligns through allelign, so these now hold for ldpred3 too. They are listed against ldpred3 0.9.8, the last release with its own reader.

- Case and control count columns give effective N when no explicit effective-N column exists. ldpred3 would use a total-N column.
- New aliases sit after ldpred3's in every list, so files ldpred3 maps are mapped identically. Files ldpred3 could not map (PLINK2 with REF/ALT, REGENIE, Neale, `hm_` sets) now map.
- `write_pgs_scoring_file` writes `#sd_source=unspecified` instead of `empirical` when the source is unknown.
- `in_long_range_ld` normalises chromosome labels. APOE is off by default.
