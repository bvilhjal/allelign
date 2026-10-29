"""Resolve and download public GWAS summary statistics and PGS scores.

A *source reference* names one published file. Resolving it answers "which
URL, which genome build, and what is known about the study", and
:func:`download_ref` fetches it into a local file with a provenance sidecar.
The dialect rules that parse the result are :mod:`allelign.sumstats`' (the
providers here publish columns it already maps), and PGS scores are read by
:func:`allelign.read_scoring_file`.

Supported references (case-insensitive; ``GCST``/``PGS`` accessions bare):

==============  ==============================================================
reference        file
==============  ==============================================================
``GCST91000728`` GWAS Catalog summary statistics, harmonised when the
Catalog publishes a harmonised file, else the one submitted file (the
plan and sidecar then say ``harmonised: false``; several submitted files
are refused rather than guessed)
``FINNGEN:R11:I9_AF``  one FinnGen release endpoint (GRCh38)
``COVIDHGI:R7:C2``     one COVID-19 HGI release-7 analysis (GRCh37)
``GBMI:ASTHMA:EUR``    one GBMI endpoint in one ancestry stratum (GRCh38)
``EQTL:QTS000001:QTD000001:ENSG00000008128``  one eQTL Catalogue molecular
trait, row-filtered at download (GRCh38)
``PGS000337``    one PGS Catalog score, harmonised if published (build
preference GRCh37, or ``build=``)
==============  ==============================================================

GBMI's stratum is required rather than defaulted to the multi-ancestry
meta: a single-ancestry LD reference cannot represent it. The same rule
keeps the eQTL Catalogue trait mandatory: a dataset's bulk file holds every
trait it tested.

:func:`fetch_ref` returns a local path for a reference, downloading once
into :func:`cache_dir` (``$ALLELIGN_CACHE_DIR``, else ``~/.cache/allelign``)
and reusing it afterwards without touching the network. It is what lets
:func:`allelign.read_sumstats` accept a bare ``GCST006630``.

Network access uses :data:`USER_AGENT` over ``urllib`` only; the opener is
injectable so the suite runs offline.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .sumstats import COLUMN_ALIASES

__all__ = ["cache_dir", "download_ref", "fetch_ref", "parse_ref",
           "read_keep_ids", "resolve_ref"]

GWAS_FTP = "https://ftp.ebi.ac.uk/pub/databases/gwas"
SUMSTATS_FTP = f"{GWAS_FTP}/summary_statistics"
HARMONISED_LIST = f"{SUMSTATS_FTP}/harmonised_list.txt"
GWAS_STUDY_REST = "https://www.ebi.ac.uk/gwas/rest/api/v2/studies/{accession}"
GWAS_PUBLICATION_REST = ("https://www.ebi.ac.uk/gwas/rest/api/v2/"
                         "publications/{pmid}")
PGS_SCORE_REST = "https://www.pgscatalog.org/rest/score/{pgs_id}"

FINNGEN_HOST = "https://storage.googleapis.com"
FINNGEN_BUCKET = "finngen-public-data-{release}"
FINNGEN_PATH = "summary_stats/finngen_{release}_{endpoint}.gz"
# The bucket name lowercases the release; the object keeps it as parsed.
FINNGEN_REF = re.compile(r"^FINNGEN:(R\d{1,2}):([A-Z0-9][A-Z0-9_]{0,39})$")

COVIDHGI_HOST = "https://storage.googleapis.com"
COVIDHGI_PATH = ("covid19-hg-public/freeze_7/results/20220403/main/"
                 "sumstats/COVID19_HGI_{analysis}_ALL_leave_23andme_"
                 "20220403_GRCh37.tsv.gz")
COVIDHGI_REF = re.compile(r"^COVIDHGI:(R7):(A2|B1|B2|C2)$")

GBMI_HOST = "https://gbmi-sumstats.s3.amazonaws.com"
GBMI_RELEASE = "052021"
# Endpoint code as it appears in the object name, keyed by the uppercased
# ref form because the path is mixed case.
GBMI_ENDPOINTS = {
    "AAA": "AAA", "ACAPP": "AcApp", "ASTHMA": "Asthma", "COPD": "COPD",
    "GOUT": "Gout", "HCM": "HCM", "HF": "HF", "IPF": "IPF",
    "POAG": "POAG", "STROKE": "Stroke", "THC": "ThC", "VTE": "VTE",
}
# Ancestry stratum -> path token ("" for the multi-ancestry meta).
GBMI_STRATA = {"ALL": "", "AFR": "afr_", "AMR": "amr_", "EAS": "eas_",
               "EUR": "eur_", "SAS": "sas_"}
GBMI_REF = re.compile(
    r"^GBMI:(" + "|".join(sorted(GBMI_ENDPOINTS)) + r"):("
    + "|".join(sorted(GBMI_STRATA)) + r")$")

EQTL_HOST = "https://ftp.ebi.ac.uk"
EQTL_PATH = ("pub/databases/spot/eQTL/sumstats/{study}/{dataset}/"
             "{dataset}.all.tsv.gz")
EQTL_REF = re.compile(
    r"^EQTL:(QTS\d{6}):(QTD\d{6}):([A-Z0-9][A-Z0-9._-]{0,63})$")
EQTL_ROW_FILTER_COLUMN = "molecular_trait_id"

GCST_REF = re.compile(r"^GCST\d{3,}$")
PGS_REF = re.compile(r"^PGS\d{6}$")

#: Reported to every HTTP service, so hosts can tell a tool from a browser.
USER_AGENT = f"allelign/{__version__} (+https://github.com/bvilhjal/allelign)"

#: Some providers carry several comma-joined identifiers per row (FinnGen's
#: ``rsids``); a joined cell matches a keep set when any part does.
ID_SEPARATOR = ","
_ID_COLUMNS = frozenset(COLUMN_ALIASES["id"])

TIMEOUT = 60          # index, HEAD and metadata requests
STREAM_TIMEOUT = 300  # data transfers
CHUNK = 1 << 20


def _open(url, method="GET", timeout=TIMEOUT):
    """Open ``url`` with the allelign user agent.

    The suite replaces this module-level hook with a fake response source,
    which keeps every test offline.
    """
    request = urllib.request.Request(url, method=method,
                                     headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def _request_failed(exc, label):
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
        return f"{label}: file not found (404)"
    return f"{label}: request failed ({exc})"


def _head(url, label, opener):
    try:
        with opener(url, method="HEAD") as response:
            return {"remote_bytes": int(
                response.headers.get("Content-Length", 0) or 0)}
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        raise ValueError(_request_failed(exc, label)) from exc


def _fetch_bytes(url, label, opener, timeout=TIMEOUT):
    try:
        with opener(url, timeout=timeout) as response:
            return response.read()
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        raise ValueError(_request_failed(exc, label)) from exc


def _fetch_json(url, label, opener):
    try:
        return json.loads(_fetch_bytes(url, label, opener).decode("utf-8"))
    except ValueError:
        raise ValueError(f"{label}: invalid JSON from {url}") from None


def parse_ref(ref):
    """``{provider: ..., ...}`` for one source reference, or ``None``.

    The reference is uppercased first; ``provider`` is one of ``gwas``,
    ``finngen``, ``covidhgi``, ``gbmi``, ``eqtl`` and ``pgs``.
    """
    text = (ref or "").strip().upper()
    match = GCST_REF.match(text)
    if match:
        return {"provider": "gwas", "accession": text}
    match = FINNGEN_REF.match(text)
    if match:
        return {"provider": "finngen", "release": match.group(1),
                "endpoint": match.group(2)}
    match = COVIDHGI_REF.match(text)
    if match:
        return {"provider": "covidhgi", "analysis": match.group(2)}
    match = GBMI_REF.match(text)
    if match:
        return {"provider": "gbmi", "endpoint": match.group(1),
                "stratum": match.group(2)}
    match = EQTL_REF.match(text)
    if match:
        return {"provider": "eqtl", "study": match.group(1),
                "dataset": match.group(2), "trait": match.group(3)}
    match = PGS_REF.match(text)
    if match:
        return {"provider": "pgs", "pgs_id": text}
    return None


def finngen_url(release, endpoint):
    """Download URL for one FinnGen release endpoint (bucket lowercased)."""
    return (f"{FINNGEN_HOST}/"
            f"{FINNGEN_BUCKET.format(release=release.lower())}/"
            f"{FINNGEN_PATH.format(release=release, endpoint=endpoint)}")


def covidhgi_url(analysis):
    """Download URL for one COVID-19 HGI release-7 analysis (GRCh37)."""
    return f"{COVIDHGI_HOST}/{COVIDHGI_PATH.format(analysis=analysis)}"


def gbmi_url(endpoint, stratum):
    """Download URL for one GBMI endpoint in one ancestry stratum.

    Both components are looked up rather than interpolated: the endpoint is
    mixed case in the path and the stratum carries its own trailing
    underscore (empty for the multi-ancestry meta).
    """
    return (f"{GBMI_HOST}/{GBMI_ENDPOINTS[endpoint]}_Bothsex_"
            f"{GBMI_STRATA[stratum]}inv_var_meta_GBMI_{GBMI_RELEASE}"
            "_nbbkgt1.txt.gz")


def eqtl_url(study, dataset):
    """Download URL for one eQTL Catalogue dataset's bulk file (GRCh38)."""
    return f"{EQTL_HOST}/{EQTL_PATH.format(study=study, dataset=dataset)}"


def _harmonised_index(opener):
    """``{accession: ftp path}`` from the Catalog's harmonised index."""
    try:
        text = _fetch_bytes(HARMONISED_LIST, "the harmonised index", opener,
                            STREAM_TIMEOUT).decode("utf-8", "replace")
    except ValueError as exc:
        raise ValueError(f"GCST: catalog index lookup failed ({exc})") from exc
    out = {}
    for line in text.splitlines():
        found = re.search(r"/(GCST\d+)/harmonised/(\S+\.h\.tsv\.gz)",
                          line.strip())
        if found:
            out[found.group(1)] = line.strip().lstrip("./")
    return out


def _gwas_study(accession, opener):
    """The Catalog's validated study record for one accession."""
    payload = _fetch_json(GWAS_STUDY_REST.format(accession=accession),
                          accession, opener)
    if (not isinstance(payload, dict)
            or payload.get("accession_id") != accession):
        raise ValueError(f"{accession}: invalid study metadata from the "
                         "GWAS Catalog")
    return payload


def _gwas_study_metadata(accession, opener, payload=None):
    """Trait, sample and publication fields from the Catalog's study API."""
    if payload is None:
        payload = _gwas_study(accession, opener)
    sample = payload.get("initial_sample_size") or ""
    trait = payload.get("disease_trait") or accession
    if not isinstance(sample, str) or not isinstance(trait, str):
        raise ValueError(f"{accession}: invalid study metadata from the "
                         "GWAS Catalog")
    meta = {"accession": accession, "trait": trait.strip(),
            "sample": sample.strip(),
            "pmid": str(payload.get("pubmed_id") or "")}
    # The publication title is decoration; an outage there must not block a
    # download whose study metadata is already in hand.
    if meta["pmid"]:
        try:
            pub = _fetch_json(
                GWAS_PUBLICATION_REST.format(pmid=meta["pmid"]),
                accession, opener)
            if isinstance(pub, dict) and isinstance(pub.get("title"), str):
                meta["title"] = pub["title"].strip()
        except ValueError:
            pass
    return meta


_SUBMITTED_SUFFIXES = (".tsv.gz", ".txt.gz", ".csv.gz", ".tsv", ".txt",
                       ".csv", ".gz")


def _gwas_ftp_dir(accession, payload=None):
    """The study's FTP directory: the study record's, else its 1000-batch."""
    recorded = (payload or {}).get("full_summary_stats")
    if isinstance(recorded, str) and accession in recorded:
        return ("https://" + recorded.split("://", 1)[-1]).rstrip("/")
    digits, number = accession[4:], int(accession[4:])
    low = (number - 1) // 1000 * 1000 + 1
    batch = f"GCST{low:0{len(digits)}d}-GCST{low + 999:0{len(digits)}d}"
    return f"{SUMSTATS_FTP}/{batch}/{accession}"


def _submitted_gwas_url(accession, payload, opener):
    """URL of the one submitted sumstats file in the study's FTP directory.

    Zero candidates, and more than one, are both errors: with several files
    (one per trait or ancestry) no rule here can know which the caller
    meant, so they are named instead of guessed.
    """
    if payload is not None and \
            payload.get("full_summary_stats_available") is False:
        raise ValueError(f"{accession}: no harmonised summary-statistics "
                         "file in the GWAS Catalog and the study record "
                         "reports no full summary statistics")
    directory = _gwas_ftp_dir(accession, payload)
    try:
        listing = _fetch_bytes(directory + "/", accession, opener).decode(
            "utf-8", "replace")
    except ValueError as exc:
        raise ValueError(f"{accession}: no harmonised summary-statistics "
                         f"file in the GWAS Catalog and its submitted-file "
                         f"directory could not be listed ({exc})") from exc
    candidates = {}
    for href in re.findall(r'href="([^"]+)"', listing):
        name = urllib.parse.unquote(href)
        if (href.startswith(("/", "?", "http")) or href.endswith("/")
                or not name.lower().endswith(_SUBMITTED_SUFFIXES)
                or "readme" in name.lower()):
            continue
        candidates[name] = f"{directory}/{href}"
    candidates = list(candidates.items())
    if len(candidates) == 1:
        return candidates[0][1]
    if not candidates:
        raise ValueError(f"{accession}: no harmonised summary-statistics "
                         f"file in the GWAS Catalog and no submitted "
                         f"summary-statistics file in {directory}")
    names = ", ".join(name for name, _url in candidates)
    raise ValueError(f"{accession}: no harmonised summary-statistics file "
                     f"in the GWAS Catalog and {len(candidates)} submitted "
                     f"files ({names}) in {directory}; refusing to guess "
                     "which one is meant")


def _resolve_gwas(accession, opener):
    relative = _harmonised_index(opener).get(accession)
    payload = None
    if relative is not None:
        plan = {"url": f"{SUMSTATS_FTP}/{relative}", "harmonised": True,
                "note": "harmonised set; the declared build is in the file "
                        "header"}
    else:
        # No harmonised file: fall back to the submitted one, and say so
        # everywhere, because its build and allele conventions are the
        # submitter's, not the Catalog's.
        try:
            payload = _gwas_study(accession, opener)
        except ValueError:
            payload = None
        plan = {"url": _submitted_gwas_url(accession, payload, opener),
                "harmonised": False,
                "note": "submitted file (not harmonised by the GWAS "
                        "Catalog); genome build and allele conventions "
                        "are as submitted"}
    plan.update({"ref": accession, "provider": "gwas", "label": accession,
                 "genome_build": None, "trait": accession})
    plan.update(_head(plan["url"], accession, opener))
    # The study record decorates the sidecar; the indexed (or listed) file
    # is the ground truth for whether the download exists.
    try:
        plan.update(_gwas_study_metadata(accession, opener, payload))
        if plan.get("trait") and plan["trait"] != accession:
            plan["label"] = f"{accession} ({plan['trait']})"
    except ValueError:
        pass
    return plan


def _resolve_pgs(pgs_id, build, opener):
    payload = _fetch_json(PGS_SCORE_REST.format(pgs_id=pgs_id), pgs_id, opener)
    if not isinstance(payload, dict):
        raise ValueError(f"{pgs_id}: invalid score metadata from the "
                         "PGS Catalog")
    plain = payload.get("ftp_scoring_file")
    urls = {}
    for name, entry in (payload.get("ftp_harmonized_scoring_files")
                        or {}).items():
        if isinstance(entry, dict) and isinstance(entry.get("positions"), str):
            urls[name] = entry["positions"]
    choice = build or "GRCh37"
    order = [choice, "GRCh37" if choice != "GRCh37" else "GRCh38"]
    genome_build = next((b for b in order if urls.get(b)), None)
    url = urls[genome_build] if genome_build else plain
    if not isinstance(url, str) or not url:
        raise ValueError(f"{pgs_id}: no scoring file published")
    plan = {"ref": pgs_id, "provider": "pgs", "label": pgs_id, "url": url,
            "urls": urls or None, "genome_build": genome_build,
            "trait": payload.get("trait_reported") or "",
            "note": ("harmonised scoring file" if genome_build else
                     "submitted scoring file (no harmonised positions)")}
    if plan["trait"]:
        plan["label"] = f"{pgs_id} ({plan['trait']})"
    plan.update(_head(url, pgs_id, opener))
    return plan


def resolve_ref(ref, *, build=None, opener=None):
    """Everything a download needs about one source reference.

    Returns a plan dict with ``ref, provider, label, url, genome_build,
    trait, remote_bytes`` and provider-specific extras. ``build`` only
    selects among a PGS Catalog score's harmonised builds. Raises
    ``ValueError`` with a user-readable message for a malformed reference,
    an unknown study, or a dead URL (checked with a HEAD request, so a
    download announced by ``--dry-run`` does not fail half way through).
    """
    if opener is None:
        opener = _open
    parsed = parse_ref(ref)
    if parsed is None:
        raise ValueError(f"{(ref or '').strip()!r}: expected a reference like "
                         "GCST91000728, FINNGEN:R11:I9_AF, COVIDHGI:R7:C2, "
                         "GBMI:ASTHMA:EUR, "
                         "EQTL:QTS000001:QTD000001:ENSG00000008128 or "
                         "PGS000337")
    provider = parsed["provider"]
    if provider == "gwas":
        return _resolve_gwas(parsed["accession"], opener)
    if provider == "pgs":
        return _resolve_pgs(parsed["pgs_id"], build, opener)
    if provider == "finngen":
        label = f"FinnGen {parsed['release']} {parsed['endpoint']}"
        extra = {"url": finngen_url(parsed["release"],
                                    parsed["endpoint"]),
                 "genome_build": "GRCh38",
                 "note": "beta is per ALT; af_alt is the effect-allele "
                         "frequency; rsids may be comma-joined"}
    elif provider == "covidhgi":
        label = f"COVID-HGI R7 {parsed['analysis']}"
        extra = {"url": covidhgi_url(parsed["analysis"]),
                 "genome_build": "GRCh37",
                 "note": "all_inv_var_meta_effective is the per-variant "
                         "effective N; the ALL-population file"}
    elif provider == "gbmi":
        label = (f"GBMI {parsed['endpoint']} {parsed['stratum']} "
                 f"({GBMI_RELEASE})")
        extra = {"url": gbmi_url(parsed["endpoint"], parsed["stratum"]),
                 "genome_build": "GRCh38",
                 "note": "beta is per ALT; per-variant N_case and N_ctrl "
                         "columns, no single N"}
    else:
        label = f"eQTL Catalogue {parsed['dataset']} {parsed['trait']}"
        extra = {"url": eqtl_url(parsed["study"], parsed["dataset"]),
                 "genome_build": "GRCh38",
                 "row_filter": (EQTL_ROW_FILTER_COLUMN, parsed["trait"]),
                 "note": "rows filtered to the molecular trait at "
                         "download; no sample-size column (supply N)"}
    plan = {"ref": ref.strip().upper(), "provider": provider,
            "label": label, "trait": label, **extra}
    plan.update(_head(plan["url"], label, opener))
    return plan


@contextmanager
def _atomic_binary(path):
    """Stream bytes so only a complete file is ever published at ``path``."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            yield fh
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _text_lines(response, url):
    """Iterate decoded lines of a plain or gzip remote file."""
    source = gzip.GzipFile(fileobj=response) if url.endswith(".gz") \
        else response
    for raw in source:
        yield raw.decode("utf-8", "replace").rstrip("\r\n")


def _sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download_ref(ref, dest, *, build=None, keep_ids=None, meta=True,
                 opener=None):
    """Download one source reference into ``dest`` (atomic, streamed).

    ``keep_ids`` restricts the output to rows whose id column matches (a
    joined multi-rsID cell matches when any part does); the eQTL
    Catalogue's trait filter always applies. Filtering needs a text pass,
    so a gzip remote is decompressed and the output follows ``dest``'s own
    suffix. With ``meta`` (the default) a provenance sidecar
    ``<dest>.meta.json`` records the URL, provider, trait, SHA-256 and
    filter counts. Returns the metadata dict.
    """
    if opener is None:
        opener = _open
    plan = resolve_ref(ref, build=build, opener=opener)
    return _download_plan(plan, dest, keep_ids=keep_ids, meta=meta,
                          opener=opener)


def _download_plan(plan, dest, *, keep_ids=None, meta=True, opener):
    """Stream an already-resolved plan into ``dest``; see download_ref."""
    dest = Path(dest)
    url = plan["url"]
    label = plan["label"]
    filtering = keep_ids is not None or plan.get("row_filter") is not None
    counts = {}
    try:
        response = opener(url, timeout=STREAM_TIMEOUT)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        raise ValueError(_request_failed(exc, label)) from exc
    try:
        with response, _atomic_binary(dest) as out:
            if not filtering:
                while chunk := response.read(CHUNK):
                    out.write(chunk)
            else:
                lines = _text_lines(response, url)
                try:
                    header = next(lines)
                except StopIteration:
                    raise ValueError(f"{label}: the remote file is "
                                     "empty") from None
                writer = None
                if dest.name.endswith(".gz"):
                    writer = gzip.GzipFile(filename="", mode="wb", fileobj=out,
                                           compresslevel=1, mtime=0)
                try:
                    counts = _write_filtered(lines, header, writer or out,
                                             keep_ids, plan.get("row_filter"))
                finally:
                    if writer is not None:
                        writer.close()
    except (urllib.error.HTTPError, urllib.error.URLError, OSError,
            EOFError) as exc:
        raise ValueError(f"{label}: download failed ({exc})") from exc
    record = {"allelign": __version__, "ref": plan["ref"],
              "provider": plan["provider"], "label": label, "url": url,
              "genome_build": plan.get("genome_build"),
              "trait": plan.get("trait") or None,
              "remote_bytes": plan.get("remote_bytes"),
              "sha256": _sha256_of(dest),
              "n_bytes_written": dest.stat().st_size,
              "fetched_at": datetime.now(timezone.utc).isoformat()}
    record.update(counts)
    for field in ("harmonised", "note"):
        if plan.get(field) is not None:
            record[field] = plan[field]
    if keep_ids is not None:
        record["keep_ids"] = {"n_ids": len(keep_ids)}
    if meta:
        sidecar = dest.with_name(dest.name + ".meta.json")
        with _atomic_binary(sidecar) as fh:
            payload = json.dumps(record, indent=2, sort_keys=True,
                                 default=str)
            fh.write(payload.encode("utf-8"))
            fh.write(b"\n")
    return record


#: Environment variable naming the download cache directory.
CACHE_DIR_ENV = "ALLELIGN_CACHE_DIR"


def cache_dir():
    """The download cache: ``$ALLELIGN_CACHE_DIR``, else ``~/.cache/allelign``.

    ``$XDG_CACHE_HOME`` replaces ``~/.cache`` when set.
    """
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "allelign"


def _sidecar(path):
    return path.with_name(path.name + ".meta.json")


def _cached_file(directory):
    """The one completely downloaded file in a cache directory, or None.

    Complete means its provenance sidecar is beside it: downloads are
    atomic and the sidecar is written last, so a file without one is
    never trusted.
    """
    if not directory.is_dir():
        return None
    files = [p for p in sorted(directory.iterdir())
             if p.is_file() and not p.name.startswith(".")
             and not p.name.endswith(".meta.json")]
    if len(files) == 1 and _sidecar(files[0]).is_file():
        return files[0]
    return None


def fetch_ref(ref, *, build=None, cache=None, opener=None):
    """A local file for one source reference, downloading into the cache once.

    The file lands at ``<cache>/<REF>/<remote filename>`` with its
    provenance sidecar; later calls return it without any network access.
    Published summary statistics are immutable, so a cached file is never
    refreshed -- delete it to force a new download. ``cache`` overrides
    :func:`cache_dir`. Returns a :class:`pathlib.Path`.
    """
    if opener is None:
        opener = _open
    if parse_ref(ref) is None:
        # resolve_ref raises the user-readable form of this error.
        resolve_ref(ref, build=build, opener=opener)
    key = re.sub(r"[^A-Za-z0-9._-]", "_", ref.strip().upper())
    directory = (Path(cache) if cache is not None else cache_dir()) / key
    cached = _cached_file(directory)
    if cached is not None:
        return cached
    plan = resolve_ref(ref, build=build, opener=opener)
    name = urllib.parse.unquote(
        urllib.parse.urlparse(plan["url"]).path.rsplit("/", 1)[-1])
    dest = directory / name
    if dest.is_file() and _sidecar(dest).is_file():
        return dest
    directory.mkdir(parents=True, exist_ok=True)
    _download_plan(plan, dest, opener=opener)
    return dest


def _write_filtered(lines, header, out, keep_ids, row_filter):
    """Write ``header`` plus the rows passing both filters; return counts."""
    columns = [c.strip().lower() for c in header.split("\t")]
    id_index = next((i for i, c in enumerate(columns)
                     if c in _ID_COLUMNS), None)
    if keep_ids is not None and id_index is None:
        raise ValueError("keep-ids: the remote file has no id column "
                         f"among {columns}")
    filter_index = filter_value = None
    if row_filter is not None:
        name, value = row_filter
        try:
            filter_index = columns.index(name)
        except ValueError:
            raise ValueError(f"no {name} column in the remote file; cannot "
                             "select the molecular trait") from None
        filter_value = value.lower()
    out.write(header.encode("utf-8"))
    out.write(b"\n")
    n_input = n_kept = 0
    for line in lines:
        if not line:
            continue
        n_input += 1
        cells = line.split("\t")
        if keep_ids is not None:
            parts = cells[id_index].split(ID_SEPARATOR)
            if not any(part.strip() in keep_ids for part in parts):
                continue
        if filter_index is not None and \
                cells[filter_index].strip().lower() != filter_value:
            continue
        n_kept += 1
        out.write(line.encode("utf-8"))
        out.write(b"\n")
    return {"n_input": n_input, "n_kept": n_kept,
            "n_dropped": n_input - n_kept}


def read_keep_ids(path):
    """A keep set from a one-id-per-line file, a ``.bim`` or a ``.pvar``."""
    from .variants import read_variants
    path = str(path)
    if path.endswith((".bim", ".pvar")):
        return {str(i) for i in read_variants(path).id}
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            token = line.split("#", 1)[0].strip()
            if token:
                out.add(token)
    return out
