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
    fallback effects. Alleles, effects (including fallbacks) and frequency
    come from one orientation: harmonised when a complete set is available,
    submitted otherwise. Provider allele/effect aliases choose their own
    orientation first. This is header-only: consumers keep their streaming
    row filters and their sample-size provenance.
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
    # A finite submitted beta must not override a harmonised OR, nor may a
    # row with missing hm_beta fall back to a submitted OR or z-score.
    directional = ("effect_allele", "other_allele", "beta", "eaf")
    provider_modes = {
        mapping[target].lower().startswith("hm_")
        for target in directional[:3]
        if target in mapping and mapping[target].lower()
        in (extra_aliases or {}).get(target, ())}
    if len(provider_modes) > 1:
        raise ValueError("provider aliases mix submitted and harmonised effects/alleles")
    modes = (next(iter(provider_modes)),) if provider_modes else (True, False)
    chosen, odds, zscore = {}, None, None
    for harmonised in modes:
        def pick(names):
            return next((have[name] for name in names if name in have
                         and name.startswith("hm_") == harmonised), None)

        candidate = {target: pick(aliases[target]) for target in directional}
        candidate_odds, candidate_z = pick(_ODDS), pick(_ZSCORE)
        if (candidate["effect_allele"] and candidate["other_allele"]
                and (candidate["beta"] or candidate_odds or candidate_z)):
            chosen = {target: name for target, name in candidate.items() if name}
            odds, zscore = candidate_odds, candidate_z
            break
    for target in directional:
        mapping.pop(target, None)
    mapping.update(chosen)
    missing = [c for c in _REQUIRED if c not in mapping]
    if "beta" not in mapping and odds is None and zscore is None:
        missing.append("beta/odds_ratio/z_score")
    if missing:
        raise ValueError(f"unmappable columns {missing}; header={fieldnames}")
    return mapping, odds, zscore

