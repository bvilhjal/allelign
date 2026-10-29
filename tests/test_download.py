"""Download resolution and streaming, offline.

Every request is served by a fake opener through the ``opener`` hook and a
monkeypatched ``allelign.download._open``; no test touches the network. The
provider URLs and API field names were verified against the live services
when this module was written.
"""

import gzip
import hashlib
import json
import urllib.error

import numpy as np
import pytest

from allelign import download as dl
from allelign import read_scoring_file, read_sumstats
from allelign.cli import main

HARMONISED_INDEX = (
    "./GCST91000001-GCST91001000/GCST91000728/harmonised/"
    "GCST91000728.h.tsv.gz\n"
    "./GCST91000001-GCST91001000/GCST91000729/harmonised/"
    "GCST91000729.h.tsv.gz\n"
)
GWAS_URL = ("https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
            "GCST91000001-GCST91001000/GCST91000728/harmonised/"
            "GCST91000728.h.tsv.gz")
STUDY = {"accession_id": "GCST91000728",
         "initial_sample_size": "10,582 European ancestry cases, "
                                "28,256 European ancestry controls",
         "disease_trait": "Allergic rhinitis", "pubmed_id": 30013184}
GBMI_URL = ("https://gbmi-sumstats.s3.amazonaws.com/"
            "Asthma_Bothsex_eur_inv_var_meta_GBMI_052021_nbbkgt1.txt.gz")


class FakeResponse:
    def __init__(self, body=b"", headers=None):
        self._body = body.encode() if isinstance(body, str) else body
        self.headers = {k: str(v) for k, v in (headers or {}).items()}
        self._pos = 0

    def read(self, n=-1):
        if n is None or n < 0:
            chunk, self._pos = self._body[self._pos:], len(self._body)
        else:
            chunk = self._body[self._pos:self._pos + n]
            self._pos += len(chunk)
        return chunk

    def __iter__(self):
        yield from self._body.splitlines(keepends=True)

    def __enter__(self):
        self._pos = 0
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Callable standing in for ``urllib.request.urlopen``."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, url, method="GET", timeout=None):
        self.calls.append((method, url))
        target = self.routes[url]
        if isinstance(target, Exception):
            raise target
        return target() if callable(target) else target


def http_error_404():
    return urllib.error.HTTPError("x", 404, "Not Found", None, None)


def gwas_routes(**overrides):
    routes = {
        dl.HARMONISED_LIST: FakeResponse(HARMONISED_INDEX),
        GWAS_URL: FakeResponse(b"", headers={"Content-Length": 12345}),
        dl.GWAS_STUDY_REST.format(accession="GCST91000728"):
            FakeResponse(json.dumps(STUDY)),
        dl.GWAS_PUBLICATION_REST.format(pmid="30013184"):
            FakeResponse(json.dumps({"title": "Genome-wide association "
                                              "study of allergic rhinitis"})),
    }
    routes.update(overrides)
    return FakeOpener(routes)


def parse_only(ref):
    return dl.parse_ref(ref)


def test_parse_ref_providers_and_case():
    assert parse_only("GCST91000728") == {"provider": "gwas",
                                          "accession": "GCST91000728"}
    assert parse_only("gcst91000728")["accession"] == "GCST91000728"
    assert parse_only("finngen:r11:i9_af") == {
        "provider": "finngen", "release": "R11", "endpoint": "I9_AF"}
    assert parse_only("COVIDHGI:R7:C2") == {"provider": "covidhgi",
                                            "analysis": "C2"}
    assert parse_only("GBMI:ACAPP:ALL") == {"provider": "gbmi",
                                            "endpoint": "ACAPP",
                                            "stratum": "ALL"}
    eqtl = parse_only("EQTL:QTS000001:QTD000001:ENSG00000008128")
    assert eqtl == {"provider": "eqtl", "study": "QTS000001",
                    "dataset": "QTD000001", "trait": "ENSG00000008128"}
    assert parse_only("PGS000337") == {"provider": "pgs",
                                       "pgs_id": "PGS000337"}
    assert parse_only("FINNGEN:R11") is None
    assert parse_only("GBMI:ASTHMA:MARTIAN") is None
    assert parse_only("EQTL:QTS000001:QTD000001") is None
    assert parse_only("") is None


def test_provider_urls_exact():
    assert dl.finngen_url("R11", "I9_AF") == (
        "https://storage.googleapis.com/finngen-public-data-r11/"
        "summary_stats/finngen_R11_I9_AF.gz")
    assert dl.covidhgi_url("C2") == (
        "https://storage.googleapis.com/covid19-hg-public/freeze_7/"
        "results/20220403/main/sumstats/"
        "COVID19_HGI_C2_ALL_leave_23andme_20220403_GRCh37.tsv.gz")
    # Mixed-case endpoint token and the stratum's trailing underscore.
    assert dl.gbmi_url("ACAPP", "ALL") == (
        "https://gbmi-sumstats.s3.amazonaws.com/"
        "AcApp_Bothsex_inv_var_meta_GBMI_052021_nbbkgt1.txt.gz")
    assert dl.gbmi_url("ASTHMA", "EUR") == GBMI_URL
    assert dl.eqtl_url("QTS000001", "QTD000001") == (
        "https://ftp.ebi.ac.uk/pub/databases/spot/eQTL/sumstats/"
        "QTS000001/QTD000001/QTD000001.all.tsv.gz")


def test_resolve_gwas_uses_index_rest_and_head():
    opener = gwas_routes()
    plan = dl.resolve_ref("GCST91000728", opener=opener)
    assert plan["url"] == GWAS_URL
    assert plan["remote_bytes"] == 12345
    assert plan["trait"] == "Allergic rhinitis"
    assert "Allergic rhinitis" in plan["label"]
    assert plan["sample"].startswith("10,582")
    assert plan["title"].startswith("Genome-wide")
    assert ("HEAD", GWAS_URL) in opener.calls


def test_resolve_gwas_unknown_accession():
    # Not in the harmonised index, no such study, no submitted-file
    # directory at its computed 1000-batch either.
    opener = gwas_routes(**{
        dl.GWAS_STUDY_REST.format(accession="GCST99999999"):
            http_error_404(),
        "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
        "GCST99999001-GCST100000000/GCST99999999/": http_error_404()})
    with pytest.raises(ValueError, match="no harmonised summary-statistics"):
        dl.resolve_ref("GCST99999999", opener=opener)


def test_resolve_head_404_names_the_file():
    opener = FakeOpener({GBMI_URL: http_error_404()})
    with pytest.raises(ValueError, match="file not found"):
        dl.resolve_ref("GBMI:ASTHMA:EUR", opener=opener)


def test_resolve_pgs_prefers_requested_build():
    score = {"trait_reported": "Coronary artery disease",
             "ftp_scoring_file": "https://host/PGS000337.txt.gz",
             "ftp_harmonized_scoring_files": {
                 "GRCh37": {"positions": "https://host/hm37.txt.gz"},
                 "GRCh38": {"positions": "https://host/hm38.txt.gz"}}}
    routes = {dl.PGS_SCORE_REST.format(pgs_id="PGS000337"):
              FakeResponse(json.dumps(score)),
              "https://host/hm37.txt.gz": FakeResponse(),
              "https://host/hm38.txt.gz": FakeResponse()}
    opener = FakeOpener(routes)
    plan = dl.resolve_ref("PGS000337", opener=opener)
    assert (plan["url"], plan["genome_build"]) == ("https://host/hm37.txt.gz",
                                                   "GRCh37")
    plan = dl.resolve_ref("PGS000337", build="GRCh38", opener=opener)
    assert (plan["url"], plan["genome_build"]) == ("https://host/hm38.txt.gz",
                                                   "GRCh38")
    assert "Coronary artery disease" in plan["label"]


def test_resolve_pgs_without_harmonised_uses_submitted():
    score = {"trait_reported": "X",
             "ftp_scoring_file": "https://host/PGS000001.txt.gz"}
    opener = FakeOpener({
        dl.PGS_SCORE_REST.format(pgs_id="PGS000001"):
            FakeResponse(json.dumps(score)),
        "https://host/PGS000001.txt.gz": FakeResponse()})
    plan = dl.resolve_ref("PGS000001", opener=opener)
    assert plan["url"] == "https://host/PGS000001.txt.gz"
    assert plan["genome_build"] is None


def test_download_verbatim_writes_sidecar(tmp_path):
    body = b"#chrom\tpos\n1\t100\n1\t200\n"
    opener = FakeOpener({GBMI_URL: FakeResponse(
        body, headers={"Content-Length": len(body)})})
    out = tmp_path / "gbmi.tsv.gz"
    record = dl.download_ref("GBMI:ASTHMA:EUR", out, opener=opener)
    assert out.read_bytes() == body
    sidecar = json.loads((tmp_path / "gbmi.tsv.gz.meta.json").read_text())
    assert sidecar["sha256"] == hashlib.sha256(body).hexdigest()
    assert sidecar["provider"] == "gbmi"
    assert sidecar["n_bytes_written"] == len(body)
    assert record["genome_build"] == "GRCh38"
    assert sidecar["ref"] == "GBMI:ASTHMA:EUR"


FINNGEN_HEADER = ("rsids\t#chrom\tpos\tref\talt\tbeta\tsebeta\t"
                  "af_alt\tpval\tmlogp")
FINNGEN_ROWS = ["rs1\t1\t100\tA\tG\t0.01\t0.001\t0.4\t1e-2\t2.0",
                "rs2,rs3\t1\t200\tC\tT\t-0.02\t0.002\t0.6\t2e-3\t2.7",
                "rs4\t1\t300\tA\tT\t0.03\t0.003\t0.1\t3e-4\t3.5"]
FINNGEN_URL = dl.finngen_url("R11", "I9_AF")


def _gz(text_lines):
    return gzip.compress(("\n".join(text_lines) + "\n").encode())


def test_download_keep_ids_streams_and_roundtrips(tmp_path):
    body = _gz([FINNGEN_HEADER] + FINNGEN_ROWS)
    opener = FakeOpener({FINNGEN_URL: FakeResponse(
        body, headers={"Content-Length": len(body)})})
    out = tmp_path / "finngen.tsv.gz"
    record = dl.download_ref("finngen:r11:i9_af", out,
                             keep_ids={"rs3"}, opener=opener)
    # the comma-joined row matches on rs3; rs1 and rs4 do not
    text = gzip.decompress(out.read_bytes()).decode()
    lines = [l for l in text.splitlines() if l]
    assert lines[0] == FINNGEN_HEADER
    assert lines[1:] == [FINNGEN_ROWS[1]]
    assert record["n_input"] == 3 and record["n_kept"] == 1
    assert record["keep_ids"] == {"n_ids": 1}
    # the dialect rules of read_sumstats apply to the filtered file
    ss = read_sumstats(out, n_eff=100000)
    assert list(np.asarray(ss.id, dtype=object)) == ["rs2,rs3"]
    assert abs(float(ss.eaf[0]) - 0.6) < 1e-12


def test_download_eqtl_row_filter(tmp_path):
    header = ("chromosome\tpos\tref\talt\tmolecular_trait_id\tbeta\tse\t"
              "pvalue\tgene_id")
    rows = ["1\t10\tA\tG\tENSG1\t0.1\t0.01\t0.5\tENSG1",
            "1\t20\tC\tT\tENSG2\t-0.1\t0.01\t0.4\tENSG2",
            "1\t30\tT\tA\tENSG1\t0.2\t0.02\t0.1\tENSG1"]
    eqtl = dl.eqtl_url("QTS000001", "QTD000001")
    body = _gz([header] + rows)
    opener = FakeOpener({eqtl: FakeResponse(
        body, headers={"Content-Length": len(body)})})
    out = tmp_path / "eqtl.tsv"
    record = dl.download_ref("EQTL:QTS000001:QTD000001:ENSG1", out,
                             opener=opener)
    text = out.read_text()
    assert text.splitlines()[0] == header
    assert "ENSG2" not in text
    assert record["n_input"] == 3 and record["n_kept"] == 2


def test_download_gbmi_reads_through_alias_table(tmp_path):
    header = ("#chr\tpos\trsid\tref\talt\tall_meta_AF\t"
              "inv_var_meta_beta\tinv_var_meta_sebeta\tinv_var_meta_p\t"
              "all_inv_var_meta_effective\tN_case\tN_ctrl")
    rows = ["1\t100\trs1\tA\tG\t0.3\t0.05\t0.01\t1e-3\t50000\t25000\t25000",
            "1\t200\trs2\tC\tT\t0.7\t-0.04\t0.01\t2e-3\t48000\t24000\t24000"]
    body = _gz([header] + rows)
    opener = FakeOpener({GBMI_URL: FakeResponse(
        body, headers={"Content-Length": len(body)})})
    out = tmp_path / "gbmi.tsv.gz"
    dl.download_ref("GBMI:ASTHMA:EUR", out, opener=opener)
    ss = read_sumstats(out)
    assert list(np.asarray(ss.id, dtype=object)) == ["rs1", "rs2"]
    assert ss.n_eff[0] == 50000            # effective beats case/control
    assert abs(float(ss.eaf[0]) - 0.3) < 1e-12


def test_download_pgs_scoring_file_roundtrip(tmp_path):
    score = {"trait_reported": "CAD",
             "ftp_scoring_file": "https://host/PGS000337.txt.gz",
             "ftp_harmonized_scoring_files": {
                 "GRCh38": {"positions": "https://host/hm38.txt.gz"}}}
    rows = ["#HmPOS_build=GRCh38",
            "rsID\thm_chr\thm_pos\teffect_allele\tother_allele\t"
            "effect_weight",
            "rs1\t1\t100\tA\tG\t0.01",
            "rs2\t1\t200\tC\tT\t-0.02"]
    body = _gz(rows)
    opener = FakeOpener({
        dl.PGS_SCORE_REST.format(pgs_id="PGS000337"):
            FakeResponse(json.dumps(score)),
        # default build preference falls back to GRCh38 when 37 is absent
        "https://host/hm38.txt.gz": FakeResponse(
            body, headers={"Content-Length": len(body)})})
    out = tmp_path / "pgs.txt.gz"
    record = dl.download_ref("PGS000337", out, opener=opener)
    assert record["genome_build"] == "GRCh38"
    sf = read_scoring_file(out)
    assert len(sf) == 2 and sf.genome_build == "GRCh38"


def test_read_keep_ids_from_list_and_bim(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("# comment\nrs1\nrs2\n\nrs3\n")
    assert dl.read_keep_ids(ids) == {"rs1", "rs2", "rs3"}
    bim = tmp_path / "ref.bim"
    bim.write_text("1\trsB\t0\t200\tA\tC\n")
    assert dl.read_keep_ids(bim) == {"rsB"}


def test_cli_download_and_dry_run(tmp_path, monkeypatch, capsys):
    body = b"rsid\tchr\tpos\nrs1\t1\t100\n"
    opener = FakeOpener({GBMI_URL: FakeResponse(
        body, headers={"Content-Length": len(body)})})
    monkeypatch.setattr(dl, "_open", opener)
    out = tmp_path / "out.tsv"
    assert main(["download", "GBMI:ASTHMA:EUR", "-o", str(out)]) == 0
    assert out.read_bytes() == body
    assert (tmp_path / "out.tsv.meta.json").exists()
    # dry run resolves and prints without writing
    plan_opener = gwas_routes()
    monkeypatch.setattr(dl, "_open", plan_opener)
    out2 = tmp_path / "none.tsv"
    assert main(["download", "GCST91000728", "-o", str(out2),
                 "--dry-run"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["url"] == GWAS_URL
    assert not out2.exists()
    # a malformed reference exits 2 with a readable message
    assert main(["download", "NONSENSE", "-o", str(out2)]) == 2
    assert "expected a reference" in capsys.readouterr().err


DBP_DIR = ("https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
           "GCST006001-GCST007000/GCST006630")
DBP_URL = DBP_DIR + "/Evangelou_30224653_DBP.txt.gz"
DBP_STUDY = {"accession_id": "GCST006630",
             "initial_sample_size": "757,601 European ancestry individuals",
             "disease_trait": "Diastolic blood pressure",
             "pubmed_id": 30224653, "full_summary_stats_available": True,
             "full_summary_stats": "http://ftp.ebi.ac.uk/pub/databases/gwas/"
                                   "summary_statistics/GCST006001-"
                                   "GCST007000/GCST006630"}
DBP_LISTING = ('<a href="?C=N;O=D">Name</a>'
               '<a href="/pub/databases/gwas/summary_statistics/'
               'GCST006001-GCST007000/">Parent Directory</a>'
               '<a href="Evangelou_30224653_DBP.txt.gz">x</a>'
               '<a href="README_Evangelou_30224653.txt">x</a>')
SUMSTATS_BODY = _gz([
    "rsid\tchrom\tpos\teffect_allele\tother_allele\tbeta\tse\t"
    "effect_allele_frequency\tn",
    "rs1\t1\t100\tA\tG\t0.01\t0.001\t0.4\t100000",
    "rs2\t1\t200\tC\tT\t-0.02\t0.002\t0.6\t99000"])


def submitted_routes(listing=DBP_LISTING, body=SUMSTATS_BODY):
    return FakeOpener({
        dl.HARMONISED_LIST: FakeResponse(HARMONISED_INDEX),
        dl.GWAS_STUDY_REST.format(accession="GCST006630"):
            FakeResponse(json.dumps(DBP_STUDY)),
        DBP_DIR + "/": FakeResponse(listing),
        DBP_URL: FakeResponse(body, headers={"Content-Length": len(body)}),
        dl.GWAS_PUBLICATION_REST.format(pmid="30224653"):
            FakeResponse(json.dumps({"title": "Genetic analysis of over "
                                              "1 million people"}))})


def test_resolve_gwas_falls_back_to_submitted_file():
    plan = dl.resolve_ref("GCST006630", opener=submitted_routes())
    assert plan["url"] == DBP_URL
    assert plan["harmonised"] is False
    assert "not harmonised" in plan["note"]
    assert plan["trait"] == "Diastolic blood pressure"
    assert plan["genome_build"] is None


def test_resolve_gwas_refuses_to_guess_among_submitted_files():
    listing = ('<a href="TraitA.txt.gz">x</a>'
               '<a href="TraitB.txt.gz">x</a>')
    with pytest.raises(ValueError, match="refusing to guess") as err:
        dl.resolve_ref("GCST006630", opener=submitted_routes(listing))
    assert "TraitA.txt.gz" in str(err.value)
    assert "TraitB.txt.gz" in str(err.value)


def test_fetch_ref_caches_and_reuses(tmp_path):
    cache = tmp_path / "cache"
    opener = submitted_routes()
    first = dl.fetch_ref("GCST006630", cache=cache, opener=opener)
    assert first == cache / "GCST006630" / "Evangelou_30224653_DBP.txt.gz"
    assert first.read_bytes() == SUMSTATS_BODY
    sidecar = json.loads((first.parent /
                          (first.name + ".meta.json")).read_text())
    assert sidecar["harmonised"] is False
    calls = len(opener.calls)
    # a cached ref reads again with no further network traffic
    assert dl.fetch_ref("GCST006630", cache=cache, opener=opener) == first
    assert len(opener.calls) == calls


def test_read_sumstats_fetches_bare_gcst(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLELIGN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(dl, "_open", submitted_routes())
    with pytest.warns(RuntimeWarning, match="not.*harmonised"):
        ss = read_sumstats("GCST006630")
    assert list(np.asarray(ss.id, dtype=object)) == ["rs1", "rs2"]
    source = ss.parse_log["source"]
    assert source["ref"] == "GCST006630"
    assert source["harmonised"] is False
    assert source["url"] == DBP_URL


def test_read_sumstats_local_file_named_like_ref_wins(tmp_path, monkeypatch):
    local = tmp_path / "GCST006630"
    local.write_bytes(gzip.decompress(SUMSTATS_BODY))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dl, "_open", FakeOpener({}))
    ss = read_sumstats("GCST006630")
    assert list(np.asarray(ss.id, dtype=object)) == ["rs1", "rs2"]
    assert "source" not in ss.parse_log


def test_read_sumstats_rejects_pgs_ref(monkeypatch):
    monkeypatch.setattr(dl, "_open", FakeOpener({}))
    with pytest.raises(ValueError, match="PGS Catalog score"):
        read_sumstats("PGS000337")


def test_cli_download_without_out_uses_cache(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache"
    monkeypatch.setenv("ALLELIGN_CACHE_DIR", str(cache))
    monkeypatch.setattr(dl, "_open", submitted_routes())
    assert main(["download", "GCST006630"]) == 0
    captured = capsys.readouterr()
    dest = cache / "GCST006630" / "Evangelou_30224653_DBP.txt.gz"
    assert captured.out.strip() == str(dest)
    assert dest.read_bytes() == SUMSTATS_BODY
    assert "not harmonised" in captured.err
    # a filtered download must not enter (and poison) the cache
    assert main(["download", "GCST006630", "--keep-ids",
                 str(tmp_path / "ids.txt")]) == 2
    assert "--keep-ids needs -o" in capsys.readouterr().err
