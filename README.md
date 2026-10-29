# allelign

[![CI](https://github.com/bvilhjal/allelign/actions/workflows/ci.yml/badge.svg)](https://github.com/bvilhjal/allelign/actions/workflows/ci.yml)

Read, align and quality-control GWAS summary statistics and PGS effect sizes.

Every method in this family starts the same way: parse a summary-statistics file in one of many dialects, get the effect alleles right against a reference, and catch bad input before it turns into a confident wrong number. allelign does those steps once, for all of them. It is extracted from ldpred3, plus the PGS Catalog reader from multipgs and the long-range LD table from bipred. On the inputs ldpred3 already handles it reproduces ldpred3 bit for bit; `tests/test_parity_ldpred3.py` checks that.

NumPy is the only dependency. Numba speeds up the parser about threefold, and SciPy provides float32 eigensolves for the LD screen; both are optional (`[fast]`).

## Install

From PyPI:

```bash
pip install "allelign[fast]"
```

From GitHub, or a sibling checkout as with the rest of the family:

```bash
pip install "allelign[fast] @ git+https://github.com/bvilhjal/allelign.git"
pip install -e "../allelign[fast]"
```

## Quick start

```python
import allelign as al

ss = al.read_sumstats("gwas.tsv.gz")                  # any common dialect
keep, qc_log = al.qc_sumstats(ss)                     # validity, duplicates, N, MAF, INFO
ref = al.read_bim("reference.bim").freeze()           # or read_pvar, or al.Variants(...)
h = al.harmonize(ss.subset(keep), ref)                # beta now counts ref.a1
print(al.coverage_message(al.reference_coverage(ss.subset(keep), ref, h)) or "coverage ok")

# Checks against the reference panel (af_ref: frequency of ref.a1).
sd_keep, sd_log, _ = al.sd_consistency_mask(h.beta, h.se, h.n_eff, af_ref[h.var_index])

# LD consistency: z-scores their LD neighbours contradict. LD comes from the
# reference's genotypes, one block at a time; or pass your own [(R, idx), ...]
# blocks to al.ld_consistency_screen.
bed = al.Bed("reference.bed", variants=ref, samples=eur_iids)
ld_keep = al.ld_consistency_screen_genotypes(bed, h.var_index, h.beta / h.se)

# PGS effect sizes
score = al.read_scoring_file("PGS000001_hmPOS_GRCh37.txt.gz")
w = al.harmonize_weights(score, ref)                  # also aligns AF, reports matched weight mass
```

Every filter can stamp a per-row status array (`al.new_status(len(ss))`, passed as `status=`), so you can see which stage dropped each variant and write the table with `al.write_status_table`.

## Command line

```bash
allelign inspect gwas.tsv.gz
allelign clean gwas.tsv.gz -o clean.tsv.gz --status-out status.tsv
allelign align gwas.tsv.gz --reference ref.bim -o aligned.tsv.gz --sumstats-build GRCh38 --reference-build GRCh37 --reanchor-on-identifier
allelign align gwas.tsv.gz --reference ref.bim -o aligned.tsv.gz --ld-screen --ld-keep eur_samples.txt --status-out status.tsv
allelign pgs-export prs.weights.tsv -o PGS_my.txt --genome-build GRCh37
allelign download GCST91000728 -o gwas.h.tsv.gz          # GWAS Catalog; other providers below
allelign download GCST91000728 -o gwas.h.tsv.gz --keep-ids ref.bim
allelign download PGS000337 -o score.txt.gz --dry-run
allelign download GCST006630                             # into the cache; prints its path
allelign align GCST006630 --reference ref.bim -o aligned.tsv.gz   # refs work anywhere a file does
```

Each command writes `<out>.log.json` with the parse, QC, harmonisation and coverage records; `download` writes a `<out>.meta.json` provenance sidecar (URL, SHA-256, filter counts) instead.

## What it covers

- **Reading.** ldpred3's alias table and dialect rules (SAIGE, BOLT, METAL, GWAS-SSF, PGC), OR to log-OR, SE recovered from p without underflow. allelign adds:
  - PLINK2 `A1` against `REF`/`ALT` or `OMITTED`;
  - REGENIE `GENPOS` and `LOG10P`;
  - GWAS Catalog `hm_` columns as one consistent set;
  - PGC `FRQ_U_*`;
  - Neale `chr:pos:ref:alt`;
  - FinnGen;
  - effective N from case and control counts.
- **Downloading.** `allelign download` fetches a published study by reference (GWAS Catalog, PGS Catalog, FinnGen, COVID-HGI, GBMI, eQTL Catalogue), optionally restricted to the rows a reference can match, with a provenance sidecar. A GWAS Catalog study is harmonised when the Catalog publishes it so, else its one submitted file (marked `harmonised: false`). The same bare reference works anywhere a sumstats file does (`allelign align GCST006630 ...`, `read_sumstats("GCST006630")`): it is downloaded once into `~/.cache/allelign` (`$ALLELIGN_CACHE_DIR`) and reused offline.
- **Alignment.** ldpred3's harmoniser:
  - identifier matching with coordinate checks, so a build mix-up is unmatched rather than silently accepted;
  - strand flips;
  - ambiguous pairs dropped;
  - multi-allelic sites;
  - coverage diagnosis that names a build mismatch;
  - identifier re-anchoring;
  - a declared-build check.
- **QC.**
  - File-level filters.
  - SD consistency against reference frequencies.
  - Allele-frequency concordance, with a safety valve for ancestry mismatch.
  - Imputed per-variant N, the genomic-control undo, implied N, λGC and long-range LD regions.
  - The split-half LD-consistency screen (DENTIST-style) with full-window confirmation, on dense, int8, low-rank or lazily provided LD, or on LD formed block by block from a PLINK `.bed` reference.
- **Effect sizes.** ldpred3 weight tables, PGS Catalog reading and writing, and alignment of either to a reference.

[docs/guide.md](docs/guide.md) has the rules, defaults and the deliberate differences from ldpred3. A hands-on pass over download → inspect → clean → align is [docs/walkthrough.md](docs/walkthrough.md), with the same flow as a notebook in [docs/walkthrough.ipynb](docs/walkthrough.ipynb).

## Consumers

- **ldpred3** reads, aligns, QCs and writes weights through allelign. Its modules re-export allelign under their old names, so sibling imports keep working. It keeps the pipeline-specific parts: whole-block DENTIST with low-rank Woodbury updates, the BLAS gate for a pooled screen, its float32 eigensolver (passed to the screen), and the qc_log report blocks. allelign's parity tests are the contract.
- **ppb** takes chromosome labels, reverse complements and strand ambiguity from `allelign.alleles` and keeps its own position-only join.
- **bipred** takes the region table, long-range mask, strict implied-N diagnostic and weight-file I/O from allelign. Its wrapper includes APOE by default; its LDpred2-style SD diagnostic stays in bipred. Trait preparation still goes through ldpred3.
- **gwfm, multipgs and SMARTpred** use allelign for reading, alignment, QC and weight-file I/O. multipgs aligns both scoring files and fitted weights with `harmonize_weights`. SMARTpred and the LDpred3 harvester share the header-only `resolve_gwas_catalog_columns` rules while retaining streamed filtering. LD caches, prepared traits, the fit and pipeline reports stay in ldpred3.

## References

- Chen W, Wu Y, Zheng Z, et al. [Improved analyses of GWAS summary statistics by reducing data heterogeneity and errors](https://doi.org/10.1038/s41467-021-27438-7). *Nature Communications* (2021). The LD-consistency screen is DENTIST's statistic.
- Privé F, Arbel J, Aschard H, et al. [Identifying and correcting for misspecifications in GWAS summary statistics and polygenic scores](https://doi.org/10.1016/j.xhgg.2022.100136). *HGG Advances* (2022). Source of the SD and frequency checks and the per-variant N imputation.
- Lambert SA, Gil L, Jupp S, et al. [The Polygenic Score Catalog as an open database for reproducibility and systematic evaluation](https://doi.org/10.1038/s41588-021-00783-5). *Nature Genetics* (2021).
- Sollis E, Mosaku A, Abid A, et al. [The NHGRI-EBI GWAS Catalog: knowledgebase and deposition resource](https://doi.org/10.1093/nar/gkac1010). *Nucleic Acids Research* (2023).
- Yengo L, Vedantam S, Marouli E, et al. [A saturated map of common genetic variants associated with human height](https://doi.org/10.1038/s41586-022-05275-y). *Nature* (2022). The screen example in the guide uses its chr22 statistics.

## AI assistance

Parts of the code and documentation were generated with AI assistance and reviewed by the maintainer. The licence terms are unchanged.

## Status

Version 0.2.0. MIT licence. If you use allelign, please cite it — see
[CITATION.cff](CITATION.cff).
