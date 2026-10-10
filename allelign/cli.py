"""Command-line interface: ``allelign inspect | clean | align | pgs-export``."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import __version__
from . import status as _st


def _columns(pairs):
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"allelign: --column expects FIELD=COLUMN, got {item!r}")
        field, col = item.split("=", 1)
        out[field.strip()] = col.strip()
    return out


def _read(args):
    from .sumstats import read_sumstats
    kwargs = _columns(args.column)
    if args.n_eff is not None:
        kwargs["n_eff"] = args.n_eff
    if args.n_cases is not None or args.n_controls is not None:
        kwargs["n_cases"] = args.n_cases
        kwargs["n_controls"] = args.n_controls
    return read_sumstats(args.sumstats, **kwargs)


def _qc(args, ss, status):
    from .qc import qc_sumstats
    if args.no_qc:
        return np.ones(len(ss), dtype=bool), {"skipped": True}
    return qc_sumstats(ss, min_n_ratio=args.min_n_ratio, min_maf=args.min_maf,
                       min_info=args.min_info, max_chisq=args.max_chisq,
                       drop_duplicates=not args.keep_duplicates, status=status)


def _write_log(out, log):
    with open(f"{out}.log.json", "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")


def _write_status(path, ss, status):
    _st.write_status_table(path, id=ss.id, chrom=ss.chrom, pos=ss.pos,
                           ea=ss.ea, oa=ss.oa, codes=status)


def cmd_inspect(args):
    from .sumstats import inspect_columns
    info = inspect_columns(args.sumstats, **_columns(args.column))
    print(json.dumps(info, indent=2))
    return 1 if info["missing"] else 0


def cmd_clean(args):
    ss = _read(args)
    status = _st.new_status(len(ss))
    keep, qc_log = _qc(args, ss, status)
    out = ss.subset(keep)
    out.write(args.out)
    if args.status_out:
        _write_status(args.status_out, ss, status)
    _write_log(args.out, {"allelign": __version__, "parse": ss.parse_log,
                          "qc": qc_log, "status": _st.status_counts(status)})
    print(f"allelign: kept {len(out):,} of {len(ss):,} rows -> {args.out}",
          file=sys.stderr)
    return 0


def cmd_align(args):
    from .build import check_build
    from .align import (coverage_message, harmonize, reanchor_on_identifier,
                            reference_coverage)
    from .qc import gwas_af_on_counted_allele
    from .sumstats import Sumstats
    from .variants import read_variants

    build = check_build(args.sumstats_build, args.reference_build)
    ref = read_variants(args.reference).freeze()
    ss = _read(args)
    status = _st.new_status(len(ss))
    keep, qc_log = _qc(args, ss, status)
    rows = np.flatnonzero(keep)
    sub = ss.subset(keep)
    log = {"allelign": __version__, "parse": ss.parse_log, "qc": qc_log,
           "build": build}
    if args.reanchor_on_identifier:
        sub, log["reanchor"], rkeep = reanchor_on_identifier(
            sub, ref, return_keep=True)
        status[rows[~rkeep]] = _st.REANCHOR_DROPPED
        rows = rows[rkeep]
    sub_status = _st.new_status(len(sub)) if args.status_out else None
    h = harmonize(sub, ref, drop_ambiguous=not args.keep_ambiguous,
                  status=sub_status)
    if sub_status is not None:
        # Per-row harmonisation reasons (the row loop) only when a status
        # table was asked for; otherwise the harmonize log has the counts.
        status[rows] = np.where(status[rows] == 0, sub_status, status[rows])
    log["harmonize"] = h.log
    coverage = reference_coverage(sub, ref, h)
    log["coverage"] = coverage
    message = coverage_message(coverage, label=str(args.sumstats))
    if message:
        print(f"allelign: {message}", file=sys.stderr)
    aligned = Sumstats(
        id=np.asarray(ref.id, dtype=object)[h.var_index],
        chrom=np.asarray(ref.chrom, dtype=object)[h.var_index],
        pos=np.asarray(ref.pos, dtype=np.int64)[h.var_index],
        ea=np.asarray(ref.a1, dtype=object)[h.var_index],
        oa=np.asarray(ref.a2, dtype=object)[h.var_index],
        beta=h.beta, se=h.se, n_eff=h.n_eff,
        eaf=gwas_af_on_counted_allele(sub.eaf, h.src_index, h.flipped),
        info=np.asarray(sub.info, dtype=float)[h.src_index])
    aligned.write(args.out)
    if args.status_out:
        _write_status(args.status_out, ss, status)
        log["status"] = _st.status_counts(status)
    else:
        log["status_before_harmonize"] = _st.status_counts(status)
    _write_log(args.out, log)
    print(f"allelign: aligned {len(h):,} of {len(ss):,} rows to "
          f"{len(ref):,} reference variants -> {args.out}", file=sys.stderr)
    return 0


def cmd_pgs_export(args):
    from .weights import read_weights, write_pgs_scoring_file
    wt = read_weights(args.weights, sd_source=args.sd_source)
    write_pgs_scoring_file(args.out, wt, genome_build=args.genome_build,
                           pgs_id=args.pgs_id, trait_reported=args.trait)
    print(f"allelign: wrote {len(wt):,} variants -> {args.out}", file=sys.stderr)
    return 0


def _add_read_options(p):
    p.add_argument("sumstats", help="summary-statistics file (.gz accepted)")
    p.add_argument("--column", action="append", metavar="FIELD=COLUMN",
                   help="force a field to a column (name or 0-based index)")
    p.add_argument("--n-eff", type=float, default=None,
                   help="constant effective N for every row")
    p.add_argument("--n-cases", type=float, default=None)
    p.add_argument("--n-controls", type=float, default=None)


def _add_qc_options(p):
    p.add_argument("--no-qc", action="store_true", help="skip file-level QC")
    p.add_argument("--min-n-ratio", type=float, default=0.7)
    p.add_argument("--min-maf", type=float, default=0.01)
    p.add_argument("--min-info", type=float, default=0.7)
    p.add_argument("--max-chisq", type=float, default=None)
    p.add_argument("--keep-duplicates", action="store_true")
    p.add_argument("--status-out", default=None,
                   help="per-input-row status table (TSV, .gz accepted)")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="allelign",
        description="Read, align and QC GWAS summary statistics and PGS weights.")
    ap.add_argument("--version", action="version", version=f"allelign {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="show the detected column mapping")
    p.add_argument("sumstats")
    p.add_argument("--column", action="append", metavar="FIELD=COLUMN")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("clean", help="parse, QC and write the canonical table")
    _add_read_options(p)
    _add_qc_options(p)
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("align", help="parse, QC and align to a .bim/.pvar")
    _add_read_options(p)
    _add_qc_options(p)
    p.add_argument("--reference", required=True, help=".bim or .pvar file")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--keep-ambiguous", action="store_true",
                   help="keep strand-ambiguous (A/T, C/G) variants")
    p.add_argument("--reanchor-on-identifier", action="store_true",
                   help="take coordinates from the reference's rsIDs")
    p.add_argument("--sumstats-build", default=None)
    p.add_argument("--reference-build", default=None)
    p.set_defaults(func=cmd_align)

    p = sub.add_parser("pgs-export",
                       help="ldpred3 weights -> PGS Catalog scoring file")
    p.add_argument("weights")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--genome-build", required=True, choices=("GRCh37", "GRCh38"))
    p.add_argument("--pgs-id", default=None)
    p.add_argument("--trait", default=None)
    p.add_argument("--sd-source", choices=("empirical", "hwe"), default=None)
    p.set_defaults(func=cmd_pgs_export)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as error:
        print(f"allelign: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
