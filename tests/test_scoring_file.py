"""PGS Catalog scoring-file reader rules (moved from multipgs/tests/test_catalog.py).

``test_weights_cli.py`` pins the hmPOS layout, the ratio check and the
reference prefilter; these cover the remaining options and edge rows.
"""

import gzip

import numpy as np
import pytest

from allelign import read_scoring_file
from allelign.weights import scoring_file_id


HEADER = """###PGS CATALOG SCORING FILE
#format_version=2.0
#pgs_id=PGS000042
#trait_reported=Test trait
#genome_build=GRCh37
#weight_type={weight_type}
"""

COLUMNS = ["rsID", "chr_name", "chr_position", "effect_allele",
           "other_allele", "effect_weight"]


def _write(tmp_path, rows, columns, weight_type="beta", name="PGS000042.txt",
           gz=False):
    text = HEADER.format(weight_type=weight_type)
    text += "\t".join(columns) + "\n"
    for row in rows:
        text += "\t".join(str(v) for v in row) + "\n"
    path = tmp_path / name
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return str(path)


def test_reads_metadata_and_rows(tmp_path):
    path = _write(tmp_path,
                  [("rs1", 1, 100, "A", "G", 0.5),
                   ("rs2", 2, 200, "C", "T", -0.25)], COLUMNS)
    sf = read_scoring_file(path)
    assert len(sf) == 2
    assert sf.pgs_id == "PGS000042"
    assert sf.trait == "Test trait"
    assert sf.genome_build == "GRCh37"
    assert list(sf.id) == ["rs1", "rs2"]
    assert list(sf.pos) == [100, 200]
    assert np.allclose(sf.weight, [0.5, -0.25])
    assert sf.log["n_kept"] == 2
    assert "PGS000042" in str(sf)


def test_gzipped_file_reads_identically(tmp_path):
    rows = [("rs1", 1, 100, "A", "G", 0.5)]
    plain = read_scoring_file(_write(tmp_path, rows, COLUMNS, name="a.txt"))
    zipped = read_scoring_file(_write(tmp_path, rows, COLUMNS,
                                      name="b.txt.gz", gz=True))
    assert np.allclose(plain.weight, zipped.weight)


def test_streamed_rows_preserve_late_metadata_and_bad_row_counts(tmp_path):
    """Streaming must not change late ratio headers, flags or missing fields."""
    path = tmp_path / 'stream.txt'
    path.write_text('rsID\teffect_allele\tother_allele\teffect_weight\tis_dominant\n'
                    'rs1\tA\tG\t2\tfalse\n\n'
                    '#weight_type=OR\n'
                    'rs2\tC\tT\t0.5\tfalse\n'
                    'rs3\tA\tG\tinf\tfalse\n'
                    'rs4\tA\tG\tnan\tfalse\n'
                    'rs5\tC\tT\t2\ttrue\n'
                    'rs6\tA\n')
    s = read_scoring_file(path)
    np.testing.assert_array_equal(s.id, ['rs1', 'rs2'])
    np.testing.assert_array_equal(s.weight, np.log([2., .5]))
    assert s.meta['weight_type'] == 'OR'
    assert s.log['n_rows'] == 6 and s.log['n_kept'] == 2
    assert s.log['n_non_additive'] == 1
    assert s.log['n_unparsable_weight'] == 3
    assert s.log['n_unparsable_position'] == 0


def test_the_header_fixes_the_delimiter_for_the_whole_file(tmp_path):
    """A data row without tabs must not switch to whitespace splitting."""
    # A tabbed header, but one row joined by spaces: it cannot be split on
    # the file's delimiter, so it is a bad row, not six misaligned fields.
    text = HEADER.format(weight_type="beta")
    text += "\t".join(COLUMNS) + "\n"
    text += "\t".join(["rs1", "1", "100", "A", "G", "0.5"]) + "\n"
    text += " ".join(["rs2", "2", "200", "C", "T", "-0.25"]) + "\n"
    mixed = tmp_path / "mixed.txt"
    mixed.write_text(text, encoding="utf-8")
    sf = read_scoring_file(str(mixed))
    assert len(sf) == 1
    assert sf.log["n_unparsable_weight"] == 1


def test_unparsable_position_is_counted_as_missing(tmp_path):
    path = _write(tmp_path, [("rs1", 1, "100", "A", "G", 0.5),
                             ("rs2", 1, "NA", "C", "T", 0.2),
                             ("rs3", 1, "1.5", "G", "A", -0.1)],
                  COLUMNS)
    sf = read_scoring_file(path)
    assert list(sf.pos) == [100, 0, 0]
    assert sf.log["n_unparsable_position"] == 2
    assert sf.log["n_kept"] == 3


def test_odds_ratios_are_log_transformed(tmp_path):
    path = _write(tmp_path, [("rs1", 1, 100, "A", "G", 2.0),
                             ("rs2", 2, 200, "C", "T", 0.5)],
                  COLUMNS, weight_type="OR")
    sf = read_scoring_file(path)
    assert np.allclose(sf.weight, [np.log(2.0), np.log(0.5)])
    assert sf.log["log_transformed"] is True


def test_a_missing_effect_weight_column_is_named(tmp_path):
    path = _write(tmp_path, [("rs1", "A", "G")],
                  ["rsID", "effect_allele", "other_allele"])
    with pytest.raises(ValueError, match="effect_weight"):
        read_scoring_file(path)


def test_non_additive_rows_are_dropped_and_counted(tmp_path):
    path = _write(tmp_path,
                  [("rs1", 1, 100, "A", "G", 0.5, "False"),
                   ("rs2", 2, 200, "C", "T", 0.3, "True")],
                  COLUMNS + ["is_recessive"])
    sf = read_scoring_file(path)
    assert len(sf) == 1
    assert sf.log["n_non_additive"] == 1
    kept = read_scoring_file(path, drop_non_additive=False)
    assert len(kept) == 2


def test_harmonized_columns_are_preferred(tmp_path):
    path = _write(tmp_path, [("rs1", 9, 999, "rs1", 1, 100, "A", "G", 0.5)],
                  ["rsID", "chr_name", "chr_position", "hm_rsID", "hm_chr",
                   "hm_pos", "effect_allele", "other_allele", "effect_weight"])
    sf = read_scoring_file(path)
    assert sf.pos[0] == 100                    # hm_pos, not chr_position
    assert sf.log["harmonized_columns"] is True
    raw = read_scoring_file(path, prefer_harmonized=False)
    assert raw.pos[0] == 999


def test_inferred_other_allele_is_a_row_wise_fallback(tmp_path):
    path = _write(
        tmp_path,
        [("rs1", 1, 100, "A", "G", "T", 0.5),
         ("rs2", 1, 200, "C", "", "A", 0.3),
         ("rs3", 1, 300, "G", "NA", "C", -0.2)],
        ["rsID", "chr_name", "chr_position", "effect_allele",
         "other_allele", "hm_inferOtherAllele", "effect_weight"])
    sf = read_scoring_file(path)
    assert list(sf.oa) == ["G", "A", "C"]
    assert sf.log["n_inferred_other_allele"] == 2
    assert sf.log["columns_used"]["oa_fallback"] == "hm_inferotherallele"


def test_scoring_file_id_from_filename():
    assert scoring_file_id("/x/PGS000123_hmPOS_GRCh37.txt.gz") == "PGS000123"
    assert scoring_file_id("my_scores.txt") == "my_scores"
