# allelign walkthrough: download, process, align

A hands-on pass over the workflow every analysis starts with: obtain a
published summary-statistics or scoring file, see what is in it, clean it,
align it against a reference, and read the records each step leaves behind.
The same flow in Python is [walkthrough.ipynb](walkthrough.ipynb); the rules
and defaults behind every step are [guide.md](guide.md).

Every input here is public. Full studies are large (hundreds of MB to GB),
so the pattern throughout is: resolve first with `--dry-run`, download a
slice restricted to your reference's variants when the whole file is more
than you need, and keep the sidecars — they are the record of what you
analysed.

## 1. Choose and fetch the input

Name the study by its source reference. `--dry-run` resolves the reference,
checks the URL with a HEAD request, and prints the plan without
downloading:

```console
$ allelign download GCST91000728 -o gwas.h.tsv.gz --dry-run
{
  "genome_build": null,
  "label": "GCST91000728 (Allergic rhinitis)",
  "note": "harmonised set; the declared build is in the file header",
  "provider": "gwas",
  "ref": "GCST91000728",
  "remote_bytes": 430478461,
  "sample": "10,582 European ancestry cases, 28,256 European ancestry controls",
  "trait": "Allergic rhinitis",
  "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/..."
}
```

The six providers (guide.md, Table 3): `GCST…` (GWAS Catalog harmonised),
`FINNGEN:R11:I9_AF`, `COVIDHGI:R7:C2`, `GBMI:ASTHMA:EUR`,
`EQTL:QTS000001:QTD000001:ENSG00000008128` (one molecular trait, filtered
at download) and `PGS000337` (a PGS Catalog score).

The download:

```bash
# A whole study, streamed and atomic:
allelign download GCST91000728 -o gwas.h.tsv.gz

# Only the variants your reference panel can match — the file on disk is
# then as small as your reference, not as large as the study:
allelign download GCST91000728 -o gwas.h.tsv.gz --keep-ids reference.bim

# A FinnGen endpoint; a comma-joined rsids cell matches on any part:
allelign download FINNGEN:R11:I9_AF -o finngen.tsv.gz --keep-ids reference.bim

# One eQTL Catalogue gene's cis window (the trait filter is mandatory):
allelign download EQTL:QTS000001:QTD000001:ENSG00000008128 -o eqtl.tsv
```

Every download writes `<out>.meta.json` beside the file: the URL, provider,
trait, genome build, SHA-256, byte count and — when rows were filtered —
`n_input`, `n_kept`, `n_dropped`. Keep it with the data; it is how the run
stays reproducible.

```console
$ cat finngen.tsv.gz.meta.json
{
  "allelign": "0.1.0",
  "genome_build": "GRCh38",
  "label": "FinnGen R11 I9_AF",
  "n_bytes_written": 5218901,
  "n_dropped": 981376,
  "n_input": 1059488,
  "n_kept": 78112,
  "provider": "finngen",
  "ref": "FINNGEN:R11:I9_AF",
  "sha256": "…",
  "url": "https://storage.googleapis.com/finngen-public-data-r11/…",
  "fetched_at": "2026-10-29T10:41:02.938522+00:00"
}
```

## 2. See what you got

`allelign inspect` reports the detected column mapping without reading data
rows — run it before any long step:

```console
$ allelign inspect finngen.tsv.gz
{
  "layout": "finngen",
  "missing": [],
  "mapping": {"id": "rsids", "chrom": "#chrom", "pos": "pos", "ea": "alt", ...},
  "n_eff_source": "none"
}
```

`inspect` exits non-zero when a field is missing. FinnGen publishes no
sample-size column; give the constant on later steps
(`--n-eff 100000`, or case/control counts which allelign turns into an
effective N):

```bash
allelign clean finngen.tsv.gz --n-cases 41000 --n-controls 320000 -o clean.tsv.gz
```

## 3. Clean

File-level QC drops invalid rows, duplicates, low-N, rare and
poorly-imputed variants (guide.md §3 has the thresholds):

```bash
allelign clean gwas.h.tsv.gz -o clean.tsv.gz --status-out status.tsv
```

- `clean.tsv.gz` — the canonical table (`id chrom pos ea oa beta se n_eff eaf info`)
- `status.tsv` — one row per input variant, with the QC stage that kept or dropped it
- `clean.tsv.gz.log.json` — parse, QC and status-count records

```console
$ allelign: kept 412,006 of 430,781 rows -> clean.tsv.gz
```

## 4. Align to a reference

`allelign align` runs the same read and QC, then matches to a `.bim`/`.pvar`
reference, orients effect alleles to the reference's counted allele, and
diagnoses coverage:

```bash
allelign align clean.tsv.gz --reference reference.bim -o aligned.tsv.gz \
    --status-out status.tsv
```

What the coverage message is telling you (guide.md §2):

- `unmatched identifiers + build mismatch` — a GRCh38 study against a GRCh37
  reference. Declare the builds and reanchor on rsIDs, or lift properly:

  ```bash
  allelign align gwas.h.tsv.gz --reference reference.bim -o aligned.tsv.gz \
      --sumstats-build GRCh38 --reference-build GRCh37 --reanchor-on-identifier
  ```

- `absent identifiers` — the study and reference simply overlap little.
- a sparse hits-only deposit aligns few rows; that is expected.

Add the LD-consistency screen when the reference has genotypes beside the
`.bim` (a `.bed`/`.fam`), restricting LD to the ancestry that matches the
GWAS:

```bash
allelign align clean.tsv.gz --reference reference.bim -o aligned.tsv.gz \
    --ld-screen --ld-keep eur_samples.txt --status-out status.tsv
```

Dropped rows carry the `dentist_outlier` status. A drop rate of several
percent usually indicts the reference, not the statistics: with 500-sample
reference LD and a very large GWAS, treat the mask as a diagnostic
(guide.md §3, "Limits").

## 5. Scores

A PGS Catalog score downloads like a study and reads through the same
rules; `harmonize_weights` (walkthrough.ipynb §3) aligns it to a reference
and reports the matched share of squared weight mass. In the other
direction, fitted weights become a publishable scoring file:

```bash
allelign download PGS000337 -o PGS000337.txt.gz
allelign pgs-export prs.weights.tsv -o PGS_my.txt --genome-build GRCh37
```

## What to keep

| File | From | What it is |
|---|---|---|
| `<out>.meta.json` | `download` | provenance: URL, provider, trait, SHA-256, filter counts |
| `status.tsv` | `clean`, `align` | per-input-variant record of the stage that kept or dropped it |
| `<out>.log.json` | `clean`, `align` | parse, QC, harmonisation, coverage records |
| `aligned.tsv.gz` | `align` | effects on the reference's counted alleles, ready for fitting |
