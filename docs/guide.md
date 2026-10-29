# allelign guide

Parts of the code and documentation were generated with AI assistance and
reviewed by the maintainer.

## 1. Reading summary statistics

`read_sumstats` maps header names onto the fields `id chrom pos ea oa beta se n_eff eaf info`, case-insensitively, using `allelign.sumstats.COLUMN_ALIASES`. Any field can be forced to a column by name or zero-based index, e.g. `read_sumstats(path, ea="A1", beta=6)`. Use `inspect_columns(path)` (or `allelign inspect`) to see the mapping before a long read. Files are split on tabs, commas or whitespace, with no CSV quoting: a comma inside a field shifts the later columns.

- **Effects.** An OR column becomes `beta = log(OR)`. A missing SE is recovered as `|beta| / z`, with z from the p-value or from `-log10 p`; tiny p-values are handled in log space. A BETA column that looks like odds ratios (almost all positive, centred near 1) warns, but nothing is inferred.
- **Frequencies.** Only effect-allele frequencies are mapped automatically. MAF is not, because it does not say which allele it counts.
- **Missing values.** A blank or non-numeric position becomes 0 (missing). Empty lines are skipped.
- **Association tests.** With a `TEST` column, only `ADD` and REGENIE's
  covariate-adjusted marginal `ADD-CONDTL` are read. Covariate, interaction,
  non-additive, blank and unknown tests are dropped with a warning.
  `parse_log` records `n_input`, `n_kept`, `n_dropped_test` and
  `accepted_test_values`; later per-row status tables cover the retained rows.
  If both additive models occur for one variant, duplicate QC still applies.

**Table 1. Layouts recognised from the header.**

| Layout | Rule |
|---|---|
| SAIGE | `Allele2` is the effect allele (`AF_Allele2`, `imputationInfo`) |
| PLINK2 | `A1` is tested: the other allele is `OMITTED`, or row by row `ALT` where `A1 == REF`, else `REF` |
| REGENIE | `GENPOS` is the position when there is no `BP`; `LOG10P` recovers SE |
| BOLT | `BP` wins over `GENPOS` (genetic position) |
| GWAS Catalog harmonised | with `hm_beta` or `hm_odds_ratio`, the `hm_` columns are used as one set, and the submitted frequency is dropped if there is no `hm_effect_allele_frequency` (it may count the other allele); without a harmonised effect, the submitted columns are used |
| PGC daner | `FRQ_U_<n>` (control frequency of A1) is the EAF when no EAF column exists |
| Neale UKB | `variant` = `chr:pos:ref:alt`, effect on ALT |
| FinnGen | `alt` is the effect allele, `af_alt` the EAF |

`resolve_gwas_catalog_columns(header, extra_aliases=None)` serves streamed
Catalog normalisers. It returns the normalised-name mapping and the OR and
z-score fallback columns; provider aliases take priority. Its deposit schema
keeps reported total N distinct from effective N. The normaliser owns row
filtering, effect recovery and the study's sample-size policy. Alleles, beta,
OR/z-score fallbacks and frequency share one orientation: a complete harmonised
set is preferred, otherwise the submitted set is used. Incomplete sets are
rejected. A provider allele/effect override selects its own orientation;
fallback effects never cross to the other set.

### Sample size

Per-variant N comes from, in order:

1. a scalar (`n_eff=…`, or numeric `n_cases=…, n_controls=…`, which flatten every row), then an explicit `n_eff="col"` mapping, then explicit case/control column mappings (both counts must resolve);
2. an explicit effective-N column (`Neff`, `n_eff`, `Effective_N`, `N_effective`), even beside a total-N column;
3. per-variant case and control counts, giving `4/(1/N_case + 1/N_control)` (warns);
4. a total-N column.

`parse_log["n_eff_source"]` records which. For a meta-analysis, effective N from summed counts overstates the true value when case fractions differ between cohorts; prefer a published `Neff`. `anchor_n_eff(ss, target)` rescales a per-variant column's median down to a supplied N, and never scales it up.

`inspect_columns(path, n_eff=...)` uses the same sample-size arguments and
precedence as `read_sumstats`, without reading data rows. It reports `n_eff`
in `missing` when no N source resolves. On the CLI, supply constants with
`allelign inspect gwas.tsv --n-eff 10000` or `--n-cases 2000 --n-controls 8000`;
use `--column n_eff=Neff` to select a column.

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

The CLI permits mismatched declarations only with `--reanchor-on-identifier`.
Its build log retains `declared` and `reference`, records `status="reanchored"`
for that mismatch, and identifies the `output` build as the reference build.
The usual allele checks still apply after reanchoring.

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
- `implied_sample_size`: binary traits only; scalar or per-variant `reported_n`,
  with a ratio of medians. The default median excludes nonpositive/nonfinite
  estimates. `strict=True` preserves bipred's policy: reject invalid input
  rows, then take the median of all estimates, including nonpositive ones.
  Quantitative-trait absolute N remains unidentified under either policy.
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

Reference size matters. The null calibration assumes LD sampling noise is small relative to the z-scores' precision. On chr22 of Yengo et al. 2022 EUR height (N about 1.6M), aligned to 1000G with LD from its 503 EUR samples, the screen dropped 1,300 of 14,582 variants (9%); the CLI path gives the same count (8 s, 0.57 GB). That mostly measures the small reference, not bad statistics. With a very large GWAS, use a reference of comparable quality (for example UK Biobank LD), or read the mask as a diagnostic.

## 4. PGS effect sizes

- **ldpred3 tables** (`read_weights`/`write_weights`) hold standardised weights, with optional `AF_REF SD_REF SD_SOURCE`. Floats are written so they read back bit for bit.
- **PGS Catalog files** (`read_scoring_file`; public header vocabulary
  `SCORING_COLUMNS` for download validators):
  - weights are per allele count;
  - `hm_` columns are preferred, and `hm_inferOtherAllele` fills a missing other allele;
  - coordinate build follows chromosome/position columns, independently of
    the rsID column: `HmPOS_build` for harmonised coordinates, `genome_build`
    for submitted ones; mixed coordinates or missing harmonised-build metadata
    give an empty `ScoringFile.genome_build`, with mixed columns logged;
  - OR/HR weights are logged;
  - non-additive rows are dropped;
  - `keep(id, position)` skips rows a reference cannot match, while the ratio check and the weight-mass denominator still read the whole file. `harmonize_weights` counts the skipped rows as unmatched.
- **`write_pgs_scoring_file`** converts an ldpred3 table to a Catalog file with `effect_weight = WEIGHT/SD_REF`. The genome build is required, `#variants_number` is the row count, and `#sd_source` is `unspecified` when the table does not record it. `#variants_genome_wide` is not written, because the table may be one chromosome or a hits list. For plink2 `--score`, strip the `#` lines first.

## 5. Deliberate differences from ldpred3

ldpred3 0.9.9 reads and aligns through allelign, so these now hold for ldpred3 too. They are listed against ldpred3 0.9.8, the last release with its own reader.

- Case and control count columns give effective N when no explicit effective-N column exists. ldpred3 would use a total-N column.
- An effective-N column (`Effective_N`, `N_effective`, `all_inv_var_meta_effective`) wins over a total-N column such as `N` in the same header. ldpred3 read the total N.
- New aliases sit after ldpred3's in every list, so files ldpred3 maps are mapped identically, apart from the effective-N rule above. Files ldpred3 could not map (PLINK2 with REF/ALT, REGENIE, Neale, `hm_` sets) now map.
- A `TEST` column restricts input to additive association rows before duplicate
  QC; earlier readers could treat covariate coefficients as variant effects.
- `write_pgs_scoring_file` writes `#sd_source=unspecified` instead of `empirical` when the source is unknown.
- `in_long_range_ld` normalises chromosome labels. APOE is off by default.

## 6. Downloading public summary statistics

`resolve_ref(ref)` names one published file; `download_ref(ref, dest)` fetches it atomically and writes a `<dest>.meta.json` sidecar (URL, provider, trait, SHA-256, filter counts). On the CLI: `allelign download REF [-o OUT] [--dry-run] [--keep-ids FILE] [--build GRCh37|GRCh38]`.

A bare reference also stands in for the file itself. `fetch_ref(ref)` downloads into the cache (`cache_dir()`: `$ALLELIGN_CACHE_DIR`, else `~/.cache/allelign`) under `<REF>/<remote filename>` and returns it without network access thereafter; published files are immutable, so delete the cached file to force a fresh download. `allelign download REF` without `-o` does the same and prints the cached path. And `read_sumstats("GCST006630")`, `inspect_columns`, `detect_columns` and the `inspect`/`clean`/`align` commands call it for any reference that is not an existing local file (a local file always wins), recording the provenance under `parse_log["source"]`. A `PGS…` reference is refused there: it is a scoring file, not sumstats.

**Table 3. Source references.**

| Reference | File | Build |
|---|---|---|
| `GCST91000728` | GWAS Catalog sumstats, harmonised when published so (`harmonised: true`), else the one submitted file (`harmonised: false`; several submitted files are refused and named); trait and sample from the Catalog API | declared in the file / as submitted |
| `FINNGEN:R11:I9_AF` | one FinnGen endpoint; `rsids` may be comma-joined | GRCh38 |
| `COVIDHGI:R7:C2` | COVID-19 HGI analysis `A2`/`B1`/`B2`/`C2`, ALL population | GRCh37 |
| `GBMI:ASTHMA:EUR` | GBMI `052021` endpoint in one ancestry stratum | GRCh38 |
| `EQTL:QTS000001:QTD000001:ENSG00000008128` | one eQTL Catalogue molecular trait, filtered at download | GRCh38 |
| `PGS000337` | PGS Catalog score; harmonised preferred (GRCh37 unless `--build`) | in the file |

- Resolution HEAD-checks the URL, so `--dry-run` announces a download that
  will not fail halfway on a 404. The GBMI stratum and the eQTL trait are
  required, not defaulted: the multi-ancestry meta and the all-traits bulk
  file are not what a single-trait fit wants. `--keep-ids FILE` (an rsID
  list, `.bim` or `.pvar`) streams only matching rows; a comma-joined
  `rsids` cell matches on any part. The file is then read with the same
  rules as any other input; the provider columns are in the alias table.
- A submitted GWAS Catalog file is *not* harmonised: its build and allele
  conventions are the submitter's (for `GCST006630`, GRCh37 `CHR:POS` ids;
  when a file has no chrom/pos columns, `read_sumstats` unpacks such packed
  ids into coordinates and logs `coords_from_id`).
  The plan, the sidecar, `parse_log["source"]` and a `RuntimeWarning` say
  so; `--sumstats-build`/`--reanchor-on-identifier` remain yours to set.
  The substituted file is found in the study's FTP directory (from its
  Catalog record, else its 1000-accession batch); a filtered download
  (`--keep-ids`, `--no-meta`) still requires an explicit `-o`, so the
  cache only ever holds whole published files.

## 7. References

- Chen W, Wu Y, Zheng Z, et al. [Improved analyses of GWAS summary statistics
  by reducing data heterogeneity and
  errors](https://doi.org/10.1038/s41467-021-27438-7). *Nature
  Communications* (2021). The DENTIST statistic behind the
  LD-consistency screen.
- Privé F, Arbel J, Aschard H, et al. [Identifying and correcting for
  misspecifications in GWAS summary statistics and polygenic
  scores](https://doi.org/10.1016/j.xhgg.2022.100136). *HGG Advances* (2022).
  The SD-consistency and frequency checks, and the per-variant N imputation.
- Lambert SA, Gil L, Jupp S, et al. [The Polygenic Score Catalog as an open
  database for reproducibility and systematic
  evaluation](https://doi.org/10.1038/s41588-021-00783-5). *Nature Genetics*
  (2021). The scoring-file format `read_scoring_file` parses.
- Sollis E, Mosaku A, Abid A, et al. [The NHGRI-EBI GWAS Catalog:
  knowledgebase and deposition
  resource](https://doi.org/10.1093/nar/gkac1010). *Nucleic Acids Research*
  (2023). The harmonised `hm_` deposit layout.
- Yengo L, Vedantam S, Marouli E, et al. [A saturated map of common genetic
  variants associated with human
  height](https://doi.org/10.1038/s41586-022-05275-y). *Nature* (2022). The
  LD-screen example above.
