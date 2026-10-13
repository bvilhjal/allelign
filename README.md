# allelign

Read, align and quality-control GWAS summary statistics and PGS effect sizes.

Every method in this family starts the same way: parse a summary-statistics file in one of many dialects, get the effect alleles right against a reference, and catch bad input before it turns into a confident wrong number. allelign does those steps once, for all of them. It is extracted from ldpred3, plus the PGS Catalog reader from multipgs and the long-range LD table from bipred. On the inputs ldpred3 already handles it reproduces ldpred3 bit for bit; `tests/test_parity_ldpred3.py` checks that.

NumPy is the only dependency. Numba speeds up the parser about threefold, and SciPy provides float32 eigensolves for the LD screen; both are optional (`[fast]`).

## Install

From a sibling checkout, as with the rest of the family:

```bash
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
```

Each command writes `<out>.log.json` with the parse, QC, harmonisation and coverage records.

## What it covers

- **Reading.** ldpred3's alias table and dialect rules (SAIGE, BOLT, METAL, GWAS-SSF, PGC), OR to log-OR, SE recovered from p without underflow. allelign adds:
  - PLINK2 `A1` against `REF`/`ALT` or `OMITTED`;
  - REGENIE `GENPOS` and `LOG10P`;
  - GWAS Catalog `hm_` columns as one consistent set;
  - PGC `FRQ_U_*`;
  - Neale `chr:pos:ref:alt`;
  - FinnGen;
  - effective N from case and control counts.
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

[docs/guide.md](docs/guide.md) has the rules, defaults and the deliberate differences from ldpred3.

## Consumers

- **ldpred3** (0.9.9) reads, aligns, QCs and writes weights through allelign. Its modules re-export allelign under their old names, so sibling imports keep working. It keeps the pipeline-specific parts: whole-block DENTIST with low-rank Woodbury updates, the BLAS gate for a pooled screen, its float32 eigensolver (passed to the screen), and the qc_log report blocks. allelign's parity tests are the contract.
- **ppb** takes chromosome labels, reverse complements and strand ambiguity from `allelign.alleles` and keeps its own position-only join.
- **bipred, gwfm, multipgs and SMARTpred** import reading, alignment, QC and weight-file I/O from allelign directly. multipgs reads PGS Catalog files with `read_scoring_file`. They still take LD caches, prepared traits, the fit, ldpred3's screen and its n_eff report helpers from ldpred3.

## Status

Version 0.1.0.dev5. Private, not on PyPI. MIT licence.
