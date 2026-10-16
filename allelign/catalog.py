"""Header rules for streamed GWAS Catalog normalisation.

This deposit profile preserves the Catalog consumers' column priorities and
output names. It is deliberately narrower than the general sumstats reader:
``n`` here records reported sample size, not an inferred effective N; the
consumer must apply its study-specific sample-size policy.
"""

__all__ = ["resolve_gwas_catalog_columns"]

_ALIASES = {
    "rsid": ("hm_rsid", "rsid", "rs_id", "variant_id"),
    "chrom": ("hm_chrom", "chromosome", "chr"),
    "pos": ("hm_pos", "base_pair_location", "position"),
    "effect_allele": ("hm_effect_allele", "effect_allele"),
    "other_allele": ("hm_other_allele", "other_allele"),
    "beta": ("hm_beta", "beta"),
    "se": ("standard_error", "se", "standarderror"),
    "eaf": ("hm_effect_allele_frequency", "effect_allele_frequency", "eaf"),
    "pval": ("p_value", "pvalue", "p"),
    "n": ("n", "sample_size", "n_total"),
}
_ODDS = ("hm_odds_ratio", "odds_ratio")
# A z-score with a standard error IS an effect: beta = z * se, sign included.
_ZSCORE = ("hm_z_score", "z_score", "zscore", "z")
_REQUIRED = ("rsid", "effect_allele", "other_allele", "se")

def resolve_gwas_catalog_columns(fieldnames, extra_aliases=None):
    """Map normalised names onto whichever harmonised layout this file uses.

    ``extra_aliases`` adds provider-scoped source names per target, tried
    before the Catalog aliases. Returns ``(mapping, odds, zscore)`` using
    the original header spellings; the latter two columns supply per-row
    fallback effects. A submitted frequency is discarded beside harmonised
    alleles without a harmonised frequency. This is header-only: consumers
    keep their streaming row filters and their sample-size provenance.
    """
    have = {}
    for field in (fieldnames or []):
        have.setdefault(field.strip().lower(), field.strip())
    aliases = dict(_ALIASES)
    for target, names in (extra_aliases or {}).items():
        # Provider names FIRST (only providers pass extras), so a generic
        # column (``beta`` beside ``all_inv_var_meta_beta``) cannot override
        # the explicit mapping.
        aliases[target] = tuple(names) + tuple(aliases.get(target, ()))
    mapping = {}
    for target, names in aliases.items():
        for alias in names:
            if alias in have:
                mapping[target] = have[alias]
                break
    # allelign's rule: the submitted frequency counts the submitted effect
    # allele, which harmonisation may have swapped, so beside hm_ alleles it
    # needs its hm_ counterpart or it describes the other allele there.
    if ("hm_effect_allele" in have
            and mapping.get("effect_allele") == have["hm_effect_allele"]
            and "hm_effect_allele_frequency" not in have):
        mapping.pop("eaf", None)
    odds = next((have[o] for o in _ODDS if o in have), None)
    zscore = next((have[z] for z in _ZSCORE if z in have), None)
    missing = [c for c in _REQUIRED if c not in mapping]
    if "beta" not in mapping and odds is None and zscore is None:
        missing.append("beta/odds_ratio/z_score")
    if missing:
        raise ValueError(f"unmappable columns {missing}; header={fieldnames}")
    return mapping, odds, zscore

