"""Private names a sibling imports from allelign, pinned here without the sibling.

ldpred3 binds these underscore names directly (its ``harmonize``, ``prepare``,
``qc``, ``scale``, ``sumstats`` and ``weights`` modules). They are not part of
allelign's public API, but renaming or removing one breaks every fresh
ldpred3 install. ``test_parity_ldpred3.py`` would catch it only where ldpred3
is installed; this check runs everywhere. Update the list together with the
consumer, never by deleting a name the consumer still imports.
"""

import importlib

import pytest

LDPRED3_PRIVATE_IMPORTS = {
    "allelign._common": ["atomic_text"],
    "allelign.align": [
        "_CACHE_ATTR", "_INDEX_CACHE_LOCK", "_ReferenceLoci", "_factorize",
        "_query_locus", "_reference_coverage", "_rows", "_variant_indices"],
    "allelign.ldcheck": [
        "_SPLIT_HALF_CACHE_BYTES", "_SplitHalfWindows", "_precision_apply",
        "_split_half_apply", "_split_half_groups", "_split_half_tiles",
        "_window_has_neighbor", "_window_ld"],
    "allelign.qc": ["_autosome_mask"],
    "allelign.sumstats": ["_n_eff_scalar_value"],
}


@pytest.mark.parametrize("module_name", sorted(LDPRED3_PRIVATE_IMPORTS))
def test_ldpred3_private_imports_resolve(module_name):
    module = importlib.import_module(module_name)
    missing = [name for name in LDPRED3_PRIVATE_IMPORTS[module_name]
               if not hasattr(module, name)]
    assert not missing, f"{module_name}: {missing}"
