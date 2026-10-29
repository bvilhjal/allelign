# allelign walkthrough

The workflow of an analysis start to finish: fetch a published file, see
what is in it, clean it, align it against a reference. The same flow in
Python is [walkthrough.ipynb](walkthrough.ipynb); the rules and defaults
behind every step are in [guide.md](guide.md).

Studies run to hundreds of megabytes and more, so resolve before
downloading, and restrict a download to your reference's variants when the
whole file is more than you need.

## 1. Fetch

```bash
allelign download GCST91000728 -o gwas.h.tsv.gz --dry-run   # plan, no download
allelign download GCST91000728 -o gwas.h.tsv.gz
allelign download GCST91000728 -o gwas.h.tsv.gz --keep-ids reference.bim
allelign download FINNGEN:R11:I9_AF -o finngen.tsv.gz --keep-ids reference.bim
allelign download EQTL:QTS000001:QTD000001:ENSG00000008128 -o eqtl.tsv
```

The reference grammar covers the GWAS Catalog (`GCST…`), the PGS Catalog
(`PGS…`), FinnGen, COVID-HGI, GBMI and the eQTL Catalogue (guide.md §6).
`--keep-ids` takes an rsID list, a `.bim` or a `.pvar`, and streams only
matching rows; a comma-joined FinnGen `rsids` cell matches on any part.
The eQTL reference names one molecular trait, and only that trait's rows
are written, because the bulk file holds every trait the dataset tested.

Each download leaves `<out>.meta.json` beside the file: URL, provider,
trait, SHA-256 and filter counts. Keep it with the data.

You can also skip naming a file at all: `allelign download GCST006630`
fetches into the cache (`~/.cache/allelign`) and prints its path, and the
later steps accept the bare reference in place of a file —
`allelign inspect GCST006630`, `allelign align GCST006630 --reference ...`
— downloading once and reusing the cached file. Where the Catalog has no
harmonised file (as for `GCST006630`), the submitted file is used and
flagged as such; its build and alleles are the submitter's.

## 2. Inspect

```bash
allelign inspect gwas.h.tsv.gz
```

This prints the detected column mapping without reading data rows, and
exits non-zero when a field is missing. Providers that publish no sample
size (FinnGen, the eQTL Catalogue) need the constant on the later steps:

```bash
allelign clean finngen.tsv.gz --n-cases 41000 --n-controls 320000 -o clean.tsv.gz
```

## 3. Clean

```bash
allelign clean gwas.h.tsv.gz -o clean.tsv.gz --status-out status.tsv
```

File-level QC drops invalid rows, duplicates, low-N, rare and
poorly-imputed variants (thresholds in guide.md §3). `--status-out` writes
one row per input variant recording the stage that kept or dropped it, and
`clean.tsv.gz.log.json` holds the parse and QC records.

## 4. Align

```bash
allelign align clean.tsv.gz --reference reference.bim -o aligned.tsv.gz \
    --status-out status.tsv
```

Effects come out on the reference's counted alleles. When coverage is thin
the closing message names the cause. Unmatched identifiers plus a build
mismatch means declare the builds and reanchor on rsIDs:

```bash
allelign align gwas.h.tsv.gz --reference reference.bim -o aligned.tsv.gz \
    --sumstats-build GRCh38 --reference-build GRCh37 --reanchor-on-identifier
```

Absent identifiers simply mean little overlap. With a `.bed` beside the
`.bim`, `--ld-screen --ld-keep eur_samples.txt` adds the LD-consistency
screen and marks its drops `dentist_out`; a drop rate of several percent
usually indicts the reference, not the statistics (guide.md §3, Limits).

## 5. Scores

```bash
allelign download PGS000337 -o score.txt.gz
allelign pgs-export prs.weights.tsv -o PGS_my.txt --genome-build GRCh37
```

A PGS Catalog score downloads like a study and is aligned to a reference
with `harmonize_weights`; the notebook shows this on PGS000001.
`pgs-export` converts fitted ldpred3 weights the other way.
