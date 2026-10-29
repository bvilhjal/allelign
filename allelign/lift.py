"""Lift summary statistics between GRCh37 and GRCh38, and detect a build mismatch.

A GRCh38 file aligned to a GRCh37 reference matches about 1% of variants by
coordinate and still yields numbers. :func:`detect_build` names both builds
by asking which liftover makes the file's coordinates agree with the
reference's, and :func:`liftover` moves the coordinates with the UCSC chain
file. :func:`~allelign.align.harmonize` does both automatically when an
alignment shows the build-mismatch signature (``liftover="auto"``).

The UCSC chain files are downloaded once into the allelign cache
(:func:`~allelign.download.cache_dir`) and checked against their SHA-256.
Each position is lifted as a single base, as UCSC liftOver does for one-base
regions; a position in a reversed block has its alleles reverse-complemented.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import download as _dl
from .alleles import normalize_chrom, reverse_complement
from .build import normalize_build

__all__ = ["CHAIN_FILES", "Chain", "fetch_chain", "read_chain", "load_chain",
           "lift_positions", "liftover", "detect_build"]

#: UCSC chain files by ``(source, target)`` build, with their SHA-256.
CHAIN_FILES = {
    ("GRCh37", "GRCh38"): (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz",
        "5c0598e500ceb5a78c73086929e8ef993aec309bcafb595139b53d440b125a1d"),
    ("GRCh38", "GRCh37"): (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz",
        "14a712e8e147d9fc8e9d87d51977b46f6f8ddb93efbe5d0843d86b6205f587b1"),
}

#: Lifted positions are kept only on these chromosomes (not alt or random contigs).
PRIMARY_CHROMOSOMES = frozenset([str(c) for c in range(1, 23)] + ["X", "Y", "MT"])

#: :func:`detect_build` names another build only when its agreement with the
#: reference is at least this many times that of the next-best hypothesis.
DETECTION_MARGIN = 2.0


@dataclass(frozen=True)
class Chain:
    """Parsed chain: per source chromosome, sorted ungapped blocks.

    ``blocks[chrom]`` is ``(start, end, target_chrom, target_start, reverse,
    target_size)``, with 0-based half-open source intervals and the target
    start on the target's own strand.
    """

    source: str | None
    target: str | None
    blocks: dict


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_chain(source, target, *, cache=None, opener=None):
    """Path of the UCSC chain file from ``source`` to ``target``, downloading it once.

    The file lands in ``<cache>/liftover/`` (``cache`` overrides
    :func:`~allelign.download.cache_dir`) and must match its recorded SHA-256.
    """
    key = (normalize_build(source), normalize_build(target))
    if key not in CHAIN_FILES:
        raise ValueError(f"no chain file from {key[0]} to {key[1]}; "
                         f"available: {sorted(CHAIN_FILES)}")
    url, sha256 = CHAIN_FILES[key]
    directory = (Path(cache) if cache is not None else _dl.cache_dir()) / "liftover"
    path = directory / url.rsplit("/", 1)[-1]
    if path.is_file() and _sha256(path) == sha256:
        return path
    directory.mkdir(parents=True, exist_ok=True)
    opener = opener if opener is not None else _dl._open
    try:
        with opener(url) as response, _dl._atomic_binary(path) as handle:
            shutil.copyfileobj(response, handle)
    except OSError as exc:
        raise ValueError(f"could not download the {key[0]} to {key[1]} chain "
                         f"file from {url} ({exc})") from exc
    if _sha256(path) != sha256:
        path.unlink()
        raise ValueError(f"{url}: the downloaded chain file does not match its "
                         "recorded SHA-256")
    return path


def read_chain(path, *, source=None, target=None):
    """Parse a UCSC chain file (plain or gzip) into a :class:`Chain`.

    Chromosome names are normalised (:func:`~allelign.alleles.normalize_chrom`).
    Source blocks that overlap are refused: a position must lift to one place.
    """
    path = Path(path)
    lists = defaultdict(lambda: ([], [], [], [], [], []))
    current = None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "chain":
                current = lists[normalize_chrom(fields[2])]
                t_pos = int(fields[5])
                q_name, q_size = normalize_chrom(fields[7]), int(fields[8])
                reverse, q_pos = fields[9] == "-", int(fields[10])
                continue
            if current is None:
                raise ValueError(f"{path}: block line before the first chain header")
            size = int(fields[0])
            for column, value in zip(current, (t_pos, t_pos + size, q_name, q_pos,
                                               reverse, q_size)):
                column.append(value)
            if len(fields) == 3:
                t_pos += size + int(fields[1])
                q_pos += size + int(fields[2])
    blocks = {}
    for chrom, (start, end, q_name, q_start, reverse, q_size) in lists.items():
        order = np.argsort(np.asarray(start, dtype=np.int64), kind="stable")
        start = np.asarray(start, dtype=np.int64)[order]
        end = np.asarray(end, dtype=np.int64)[order]
        if np.any(start[1:] < end[:-1]):
            raise ValueError(f"{path}: overlapping source blocks on chromosome "
                             f"{chrom}; such chain files are not supported")
        blocks[chrom] = (start, end, np.asarray(q_name, dtype=object)[order],
                         np.asarray(q_start, dtype=np.int64)[order],
                         np.asarray(reverse, dtype=bool)[order],
                         np.asarray(q_size, dtype=np.int64)[order])
    return Chain(normalize_build(source), normalize_build(target), blocks)


@lru_cache(maxsize=4)
def _cached_chain(path, source, target):
    return read_chain(path, source=source, target=target)


def load_chain(source, target, *, cache=None, opener=None):
    """The parsed UCSC chain from ``source`` to ``target`` (downloaded once, parsed once)."""
    source, target = normalize_build(source), normalize_build(target)
    path = fetch_chain(source, target, cache=cache, opener=opener)
    return _cached_chain(str(path), source, target)


def _normalized(chrom):
    """Normalised labels for a chromosome column, one call per distinct label."""
    values, inverse = np.unique(np.asarray(chrom).astype(str), return_inverse=True)
    labels = np.array([normalize_chrom(v) if v.strip() else "" for v in values],
                      dtype=object)
    return labels[inverse]


def lift_positions(chain, chrom, pos):
    """Lift 1-based positions; returns ``(chrom, pos, reverse, mapped)`` arrays.

    Unmapped positions, missing ones (``pos <= 0``) and positions landing
    off the primary chromosomes come back as ``''`` and ``0`` with
    ``mapped`` false.
    """
    norm = _normalized(chrom)
    pos = np.asarray(pos, dtype=np.int64)
    n = pos.size
    out_chrom = np.full(n, "", dtype=object)
    out_pos = np.zeros(n, dtype=np.int64)
    reverse = np.zeros(n, dtype=bool)
    for label in np.unique(norm):
        if label not in chain.blocks:
            continue
        rows = np.flatnonzero((norm == label) & (pos > 0))
        start, end, q_name, q_start, q_reverse, q_size = chain.blocks[label]
        p0 = pos[rows] - 1
        block = np.searchsorted(start, p0, side="right") - 1
        inside = block >= 0
        inside[inside] = p0[inside] < end[block[inside]]
        rows, p0, block = rows[inside], p0[inside], block[inside]
        q = q_start[block] + (p0 - start[block])
        q = np.where(q_reverse[block], q_size[block] - q - 1, q)
        out_chrom[rows] = q_name[block]
        out_pos[rows] = q + 1
        reverse[rows] = q_reverse[block]
    mapped = (out_pos > 0) & np.isin(out_chrom, list(PRIMARY_CHROMOSOMES))
    out_chrom[~mapped] = ""
    out_pos[~mapped] = 0
    reverse[~mapped] = False
    return out_chrom, out_pos, reverse, mapped


def _reverse_complemented(alleles, reverse):
    """``alleles`` with the ``reverse`` rows reverse-complemented (non-ACGT kept)."""
    alleles = np.asarray(alleles).astype(str)
    out = alleles.copy()
    if reverse.any():
        flipped = alleles[reverse]
        values, inverse = np.unique(flipped, return_inverse=True)
        mapped = np.array([reverse_complement(v) or v for v in values])
        out[reverse] = mapped[inverse]
    return out


def liftover(sumstats, source, target, *, chain=None, cache=None, opener=None):
    """Move ``sumstats`` from build ``source`` to ``target``; returns ``(lifted, log)``.

    The rows stay in place, so indices into the input remain valid. A row that
    cannot be lifted loses its identifier and coordinates and so matches
    nothing; rows without coordinates are left as they are. In reversed
    blocks the alleles are reverse-complemented, so ``beta`` keeps counting
    the same allele.
    """
    source, target = normalize_build(source), normalize_build(target)
    if chain is None:
        chain = load_chain(source, target, cache=cache, opener=opener)
    pos = np.asarray(sumstats.pos, dtype=np.int64)
    has_coordinates = (pos > 0) & (_normalized(sumstats.chrom) != "")
    new_chrom, new_pos, reverse, mapped = lift_positions(chain, sumstats.chrom, pos)
    unmapped = has_coordinates & ~mapped
    chrom = np.where(has_coordinates, new_chrom, "").astype(str)
    ids = np.asarray(sumstats.id).astype(str).copy()
    ids[unmapped] = ""
    lifted = replace(sumstats, id=ids, chrom=chrom,
                     pos=np.where(has_coordinates, new_pos, 0),
                     ea=_reverse_complemented(sumstats.ea, reverse),
                     oa=_reverse_complemented(sumstats.oa, reverse))
    log = {"source": source, "target": target, "n_rows": int(pos.size),
           "n_lifted": int(mapped.sum()), "n_reverse_strand": int(reverse.sum()),
           "n_unmapped": int(unmapped.sum()),
           "n_without_coordinates": int((~has_coordinates).sum()),
           "chain": CHAIN_FILES.get((source, target), ("custom",))[0]}
    return lifted, log


_CHROM_CODES = {**{str(c): c for c in range(1, 23)}, "X": 23, "Y": 24, "MT": 25}


def _locus_keys(norm_chrom, pos):
    """``chrom * 10**10 + pos`` for primary chromosomes and positive positions, else -1."""
    codes = np.array([_CHROM_CODES.get(c, 0) for c in norm_chrom], dtype=np.int64)
    pos = np.asarray(pos, dtype=np.int64)
    return np.where((codes > 0) & (pos > 0), codes * 10**10 + pos, -1)


def _agreement(keys, reference_keys):
    """Fraction of ``keys`` (ignoring -1) found among the sorted ``reference_keys``."""
    valid = keys >= 0
    if not valid.any():
        return 0.0
    keys = keys[valid]
    found = np.searchsorted(reference_keys, keys)
    found = np.minimum(found, reference_keys.size - 1)
    return float(np.mean(reference_keys[found] == keys) * valid.mean())


def detect_build(sumstats, variants, *, sample=50_000, cache=None, opener=None):
    """Which genome builds make ``sumstats`` agree with the ``variants`` reference.

    Up to ``sample`` rows with coordinates are compared with the reference's
    positions as they are and after each liftover. ``relation`` is ``same``
    when they agree as they are, ``differ`` when a liftover agrees at least
    :data:`DETECTION_MARGIN` times better (then ``sumstats_build`` and
    ``reference_build`` name the two builds), else ``unclear``. The two
    chain files are downloaded on first use. Identifiers are not needed.
    """
    pos = np.asarray(sumstats.pos, dtype=np.int64)
    norm = _normalized(sumstats.chrom)
    rows = np.flatnonzero((pos > 0) & (norm != ""))
    report = {"n_tested": 0, "agreement": {}, "relation": "unclear",
              "sumstats_build": None, "reference_build": None}
    if rows.size == 0:
        return report
    if rows.size > sample:
        rows = rows[np.linspace(0, rows.size - 1, sample).astype(np.int64)]
    report["n_tested"] = int(rows.size)
    reference_keys = np.unique(_locus_keys(_normalized(variants.chrom),
                                           np.asarray(variants.pos, dtype=np.int64)))
    reference_keys = reference_keys[reference_keys >= 0]
    if reference_keys.size == 0:
        return report
    chrom = np.asarray(sumstats.chrom)[rows]
    agreement = {"as_is": _agreement(_locus_keys(norm[rows], pos[rows]), reference_keys)}
    for source, target in CHAIN_FILES:
        chain = load_chain(source, target, cache=cache, opener=opener)
        new_chrom, new_pos, _, _ = lift_positions(chain, chrom, pos[rows])
        agreement[f"{source}->{target}"] = _agreement(_locus_keys(new_chrom, new_pos),
                                                      reference_keys)
    report["agreement"] = agreement
    ranked = sorted(agreement, key=agreement.get, reverse=True)
    best, runner_up = agreement[ranked[0]], agreement[ranked[1]]
    if best == 0 or best < DETECTION_MARGIN * runner_up:
        return report
    if ranked[0] == "as_is":
        report["relation"] = "same"
        return report
    source, target = ranked[0].split("->")
    report.update(relation="differ", sumstats_build=source, reference_build=target)
    return report
