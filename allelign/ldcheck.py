"""LD-consistency screen: z-scores that their LD neighbours contradict.

A variant whose z-score disagrees with what its neighbours predict through the
LD matrix is a sign of an allele error, a mis-reported statistic or LD that is
not the matrix the GWAS was drawn against. The screen follows DENTIST's
split-half statistic (Chen et al. 2021, Nat Commun 12:7117)::

    zhat_a = R[a,B] pinv(R[B,B]) z_B
    T_a    = (z_a - zhat_a)^2 / (1 - R[a,B] pinv(R[B,B]) R[B,a])   ~ chi2_1

on random halves of each window, and confirms every flag with the full-window
leave-one-out statistic ``T_j = (Omega z)_j^2 / Omega_jj``,
``Omega = (R + cI)^{-1}``. A confirmed variant is dropped only when it has an
LD neighbour (``|r| >= min_neighbor_r``) and its residual is not a private
effect consistent with ``R`` (``|(Omega z)_j| > private_z_ratio |z_j|``). It is
not the complete published DENTIST pipeline, so its calibration does not
transfer; validate the mask for the study at hand.

LD is supplied as blocks ``[(R, idx), ...]`` partitioning the variants. ``R``
can be a dense float array, an int8 array on ldpred3's ``round(127 r)`` scale,
a low-rank block (any object with ``U``, ``residual_diag`` and ``scale``
attributes, such as ldpred3's ``LowRankLD``), or any object with a
``window(rows)`` method returning dense correlations. Windows are solved in
float32. :func:`ld_blocks_from_genotypes` builds blocks from reference
genotypes; :func:`chromosome_blocks` splits a sorted panel into blocks.

Ported from ldpred3's ``ld_consistency_screen``; for the same blocks, z and
seed the mask is the same (serial path, SciPy present).
"""

from __future__ import annotations

import functools
import warnings

import numpy as np

from ._common import finite_control, integer_at_least, validate_boolean, validate_seed
from .warning_markers import LD_SCREEN_HIGH_DROP, LD_SCREEN_UNTESTED

__all__ = ["ld_consistency_screen", "ld_consistency_screen_selected",
           "dentist_statistic", "loo_statistic",
           "ld_blocks_from_genotypes", "dense_blocks", "chromosome_blocks",
           "DEFAULT_WINDOW", "MIN_WINDOW", "DEFAULT_THRESHOLD",
           "DEFAULT_ROUNDS"]

#: Variants per window (bounds the pseudo-inverse cost).
DEFAULT_WINDOW = 1000
#: Windows below this are skipped: too few variants per half to predict from.
MIN_WINDOW = 50
#: Eigenvalues below this fraction of the largest are dropped, not inverted.
DEFAULT_SPLIT_HALF_EIGENVALUE_FLOOR = 1e-3
#: chi2_1 at p = 5e-8.
DEFAULT_THRESHOLD = 29.72
DEFAULT_ROUNDS = 4
#: Confirm a flag only when a surviving neighbour has at least this |r|.
DEFAULT_MIN_NEIGHBOR_R = 0.1
#: Ridge on the full-window confirmation inverse.
DEFAULT_LOO_RIDGE = 0.01
#: Spectral floor for the confirmation window (int8 and thresholded LD are
#: not positive definite). 0.01 zeroed the null drop rate on real D8 and D32
#: HapMap3+ windows in ldpred3's calibration while keeping planted recovery.
DEFAULT_LOO_EIGENVALUE_FLOOR = 0.01
#: A flagged variant with ``|(Omega z)_j| <= ratio |z_j|`` is a private effect.
DEFAULT_PRIVATE_Z_RATIO = 2.0
DEFAULT_DROP_FRACTION_WARN = 0.05
DEFAULT_UNTESTED_FRACTION_WARN = 0.05

_WORK_DTYPE = np.float32
_Q8 = 127.0
_FLOOR_TOLERANCE = 0.99
_SHIFT_GRID = 1e-4
#: Window LD a block may keep between rounds.
_SPLIT_HALF_CACHE_BYTES = 64 * 1024 ** 2

try:                                    # optional: float32 ssyevd via SciPy
    from scipy.linalg import eigh as _scipy_eigh
    _HAVE_SCIPY = True
except ImportError:                     # pragma: no cover - environment
    _HAVE_SCIPY = False


def _ridge_for_floor(lo, ridge, floor):
    """Ridge making every eigenvalue of ``R + cI`` at least ``floor``.

    One uniform shift, rounded up onto a 1e-4 grid so the result is
    reproducible across BLAS builds; ``ridge`` is returned unchanged when it
    already suffices (any float LD block).
    """
    if floor <= 0.0:
        return ridge
    if lo + ridge >= _FLOOR_TOLERANCE * floor:
        return ridge
    return float(np.ceil((floor - lo) / _SHIFT_GRID) * _SHIFT_GRID)


def _is_lowrank(block):
    return all(hasattr(block, a) for a in ("U", "residual_diag", "scale"))


def _block_size(block):
    if hasattr(block, "window") and not isinstance(block, np.ndarray):
        size = getattr(block, "size", None)
        if size is None:
            try:
                size = len(block)
            except TypeError:
                size = None
        return None if size is None else int(size)
    if _is_lowrank(block):
        return int(np.asarray(block.U).shape[0])
    arr = np.asarray(block)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("each dense LD block must be a square matrix")
    return int(arr.shape[0])


def _window_ld(block, local):
    """Dense float32 correlations for the rows ``local`` of one block."""
    if hasattr(block, "window") and not isinstance(block, np.ndarray):
        return np.asarray(block.window(local), dtype=_WORK_DTYPE)
    if _is_lowrank(block):
        factor = np.multiply(np.asarray(block.U)[local],
                             _WORK_DTYPE(block.scale), dtype=_WORK_DTYPE)
        out = factor @ factor.T
        out[np.diag_indices(len(local))] += np.asarray(
            block.residual_diag, dtype=_WORK_DTYPE)[local]
        return out
    raw = np.asarray(block)
    window = raw[np.ix_(local, local)]
    if raw.dtype == np.int8:
        # ldpred3's dense int8 stores round(127 r).
        return np.asarray(window, dtype=_WORK_DTYPE) * _WORK_DTYPE(1.0 / _Q8)
    return np.asarray(window, dtype=_WORK_DTYPE)


def _screen_eigh(a):
    """Every eigenpair of a symmetric window, ascending (float32 ``ssyevd``)."""
    if a.dtype == _WORK_DTYPE and _HAVE_SCIPY:
        return _scipy_eigh(a, check_finite=False, driver="evd")
    return np.linalg.eigh(a)


def _split_half_factor(ld, predictors, targets, eigenvalue_floor,
                       eigh=_screen_eigh):
    """The z-free part of the split-half statistic, or ``None`` if no rank."""
    within = ld[np.ix_(predictors, predictors)]
    across = ld[np.ix_(targets, predictors)]
    values, vectors = eigh(within)
    keep = values > eigenvalue_floor * max(float(values.max()), 1e-12)
    if not keep.any():
        return None
    retained = vectors[:, keep]
    values = values[keep]
    projected = across @ retained
    leverage = ((projected * projected) / values).sum(axis=1)
    return retained, values, projected, np.clip(1.0 - leverage, 1e-6, None)


def _split_half_apply(factor, z, predictors, targets):
    if factor is None:
        return np.zeros(len(targets))
    retained, values, projected, residual = factor
    predicted = projected @ ((retained.T @ z[predictors]) / values)
    return (z[targets] - predicted) ** 2 / residual


def dentist_statistic(ld, z, predictors, targets, *,
                      eigenvalue_floor=DEFAULT_SPLIT_HALF_EIGENVALUE_FLOOR):
    """Split-half ``T`` for ``targets`` predicted from ``predictors``.

    ``ld`` is a dense correlation matrix, ``z`` the matching z-scores, and the
    two disjoint index arrays select positions in both. Under the working
    Gaussian model with exact LD each value is approximately chi2_1.
    """
    ld = np.asarray(ld, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if ld.ndim != 2 or ld.shape[0] != ld.shape[1] or ld.shape[0] == 0:
        raise ValueError("ld must be a non-empty square matrix")
    if z.shape != (ld.shape[0],) or not np.all(np.isfinite(z)):
        raise ValueError("z must be a matching finite vector")
    if not np.all(np.isfinite(ld)):
        raise ValueError("ld must contain only finite values")

    def indices(value, name):
        value = np.asarray(value)
        if (value.ndim != 1 or value.size == 0
                or not np.issubdtype(value.dtype, np.integer)):
            raise ValueError(f"{name} must be a non-empty integer vector")
        value = value.astype(np.int64, copy=False)
        if (np.any((value < 0) | (value >= z.size))
                or np.unique(value).size != value.size):
            raise ValueError(f"{name} contains invalid or repeated indices")
        return value

    predictors = indices(predictors, "predictors")
    targets = indices(targets, "targets")
    if np.intersect1d(predictors, targets).size:
        raise ValueError("predictors and targets must be disjoint")
    eigenvalue_floor = finite_control("eigenvalue_floor", eigenvalue_floor,
                                      lower=0.0)
    if eigenvalue_floor >= 1:
        raise ValueError("eigenvalue_floor must be < 1")
    return _split_half_apply(
        _split_half_factor(ld, predictors, targets, eigenvalue_floor),
        z, predictors, targets)


def _window_has_neighbor(ld, min_neighbor_r):
    off = np.abs(ld, dtype=_WORK_DTYPE)
    np.fill_diagonal(off, 0.0)
    return off.max(axis=1) >= min_neighbor_r


def _precision_factor(ld, ridge, floor=DEFAULT_LOO_EIGENVALUE_FLOOR,
                      eigh=_screen_eigh):
    """``V``, ``1/(w+c)`` and ``Omega_jj`` for the full-window LOO statistic.

    One eigendecomposition does the PSD repair and the inversion together:
    ``c`` is the ridge raised until every eigenvalue clears ``floor``,
    ``Omega = V diag(1/(w + c)) V'`` and ``Omega_jj = sum_k V_jk^2/(w_k + c)``.
    """
    w, V = eigh(np.asarray(ld, dtype=_WORK_DTYPE))
    shift = _ridge_for_floor(float(w.min()), float(ridge), float(floor))
    inv_w = _WORK_DTYPE(1.0) / (w + _WORK_DTYPE(shift))
    diag = (V * V) @ inv_w
    return V, inv_w, np.maximum(diag, _WORK_DTYPE(1e-12))


def _precision_apply(factor, z):
    V, inv_w, diag = factor
    tvec = V @ (inv_w * (V.T @ np.asarray(z, dtype=_WORK_DTYPE)))
    stat = (tvec * tvec) / diag
    return stat, tvec


def loo_statistic(ld, z, *, ridge=DEFAULT_LOO_RIDGE,
                  eigenvalue_floor=DEFAULT_LOO_EIGENVALUE_FLOOR):
    """Leave-one-out ``T_j = (Omega z)_j^2 / Omega_jj`` for every variant.

    ``(Omega z)_j = Omega_jj (z_j - zhat_j)`` with ``zhat_j`` predicted from
    the rest of the window, so ``T_j`` is the squared standardised residual of
    ``z_j``. ``Omega = (R + cI)^{-1}`` with the ridge raised until every
    eigenvalue clears ``eigenvalue_floor``. Returns ``(T, Omega z)``.
    """
    ld = np.asarray(ld)
    z = np.asarray(z, dtype=np.float64)
    if ld.ndim != 2 or ld.shape[0] != ld.shape[1] or z.shape != (ld.shape[0],):
        raise ValueError("ld must be square and z must match it")
    stat, tvec = _precision_apply(
        _precision_factor(_window_ld(ld, np.arange(ld.shape[0])),
                          ridge, eigenvalue_floor), z)
    return stat.astype(np.float64), tvec.astype(np.float64)


def _confirmed_drops(ld, z, candidates, threshold, *,
                     min_neighbor_r=DEFAULT_MIN_NEIGHBOR_R,
                     ridge=DEFAULT_LOO_RIDGE,
                     private_z_ratio=DEFAULT_PRIVATE_Z_RATIO, shared=None,
                     eigh=_screen_eigh):
    """Split-half flags that also fail the full-window LOO statistic.

    A large z whose ``|(Omega z)_j|`` is at most ``private_z_ratio |z_j|`` is a
    private effect consistent with ``R``; if it is the window's largest
    eligible residual, nothing in the window is dropped. Isolated variants
    (no neighbour above ``min_neighbor_r``) are kept. ``shared`` memoises the
    window's neighbour mask and precision factor across traits.
    """
    if candidates.size == 0:
        return candidates
    if shared is None:
        shared = {}
    if not shared:
        shared["has_nbr"] = _window_has_neighbor(ld, min_neighbor_r)
        shared["precision"] = _precision_factor(ld, ridge, eigh=eigh)
    has_nbr = shared["has_nbr"]
    stat, tvec = _precision_apply(shared["precision"], z)
    private = np.abs(tvec) <= private_z_ratio * np.abs(z)
    eligible = np.where(has_nbr, stat, -np.inf)
    lead = int(np.argmax(eligible))
    if eligible[lead] > threshold and private[lead]:
        return candidates[:0]
    take = ((stat[candidates] > threshold)
            & has_nbr[candidates]
            & ~private[candidates])
    return candidates[take]


def _split_half_groups(keep):
    groups = {}
    for trait in range(keep.shape[0]):
        groups.setdefault(keep[trait].tobytes(), []).append(trait)
    return groups


def _split_half_tiles(live, window):
    """One round's windows over ``live``; the last is slid to the end."""
    starts = list(range(0, max(live.size - window + 1, 1), window))
    tail = max(0, live.size - window)
    if starts[-1] != tail:
        starts.append(tail)
    tiles = (live[start:start + window] for start in starts)
    return [local for local in tiles if local.size >= MIN_WINDOW]


class _SplitHalfWindows:
    """One block's window LD and confirmation memo, kept across rounds.

    Entries are keyed by source rows; :meth:`retain` keeps only the windows
    the next round tiles, within :data:`_SPLIT_HALF_CACHE_BYTES`.
    """

    def __init__(self, block, source_rows):
        self._block, self._source_rows = block, source_rows
        self._entries = {}
        self._charged = 0

    def _rows(self, local):
        return local if self._source_rows is None else self._source_rows[local]

    def get(self, local):
        rows = self._rows(local)
        key = rows.tobytes()
        hit = self._entries.get(key)
        if hit is not None:
            return hit[0]
        entry = (_window_ld(self._block, rows), {})
        cost = 2 * entry[0].nbytes + 16 * rows.size
        if self._charged + cost <= _SPLIT_HALF_CACHE_BYTES:
            self._entries[key] = (entry, cost)
            self._charged += cost
        return entry

    def retain(self, tiles):
        wanted = {self._rows(local).tobytes() for local in tiles}
        for key in [key for key in self._entries if key not in wanted]:
            self._charged -= self._entries.pop(key)[1]


def _settle_block(task, *, z, window, threshold, eigenvalue_floor, eigh):
    """Screen one block through every round; return its own masks.

    A block's schedule depends only on that block, its z-scores and its own
    random streams, so serial and pooled runs agree by construction. Traits
    whose survivors coincide share each window's LD and factors.
    """
    block, idx, source_rows, round_seeds = task
    n_traits = z.shape[0]
    keep = np.ones((n_traits, idx.size), dtype=bool)
    tested = np.zeros((n_traits, idx.size), dtype=bool)
    zb = z[:, idx]
    dropped = np.zeros((n_traits, len(round_seeds)), dtype=np.int64)
    windows = _SplitHalfWindows(block, source_rows)
    for round_no, round_seed in enumerate(round_seeds):
        groups = _split_half_groups(keep)
        n_before = keep.sum(axis=1)
        for members in groups.values():
            rng = np.random.default_rng(round_seed)
            live = np.where(keep[members[0]])[0]
            if live.size < MIN_WINDOW:
                continue
            for local in _split_half_tiles(live, window):
                ld, shared = windows.get(local)
                order = rng.permutation(local.size)
                half = local.size // 2
                first, second = order[:half], order[half:]
                splits = [
                    (targets, predictors, _split_half_factor(
                        ld, predictors, targets, eigenvalue_floor, eigh))
                    for targets, predictors in ((first, second),
                                                (second, first))]
                for trait in members:
                    tested[trait, local] = True
                    zw = zb[trait, local]
                    flagged = []
                    for targets, predictors, factor in splits:
                        stat = _split_half_apply(factor, zw, predictors,
                                                 targets)
                        bad = targets[stat > threshold]
                        if bad.size:
                            flagged.append(bad)
                    if flagged:
                        candidates = np.unique(np.concatenate(flagged))
                        drop = _confirmed_drops(ld, zw, candidates, threshold,
                                                shared=shared, eigh=eigh)
                        if drop.size:
                            keep[trait, local[drop]] = False
        dropped[:, round_no] = n_before - keep.sum(axis=1)
        if round_no + 1 < len(round_seeds):
            windows.retain([
                local for members in _split_half_groups(keep).values()
                for local in _split_half_tiles(
                    np.where(keep[members[0]])[0], window)])
    return keep, dropped, tested


def _validate_blocks(blocks, m, *, contiguous=False):
    """An exact, non-overlapping cover of ``0..m-1`` by ``(LD, idx)`` pairs."""
    if m < 1 or not blocks:
        raise ValueError("blocks must cover at least one variant")
    seen = np.zeros(m, dtype=bool)
    out = []
    for item in blocks:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("each LD block must be an (LD, index) pair")
        corr, idx = item
        idx = np.asarray(idx)
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError("each block index must be a 1-D integer array")
        idx = idx.astype(np.int64, copy=False)
        if idx.size == 0:
            raise ValueError("LD blocks must not be empty")
        if idx.min() < 0 or idx.max() >= m:
            raise ValueError(f"block indices must lie in [0, {m})")
        if seen[idx].any() or np.unique(idx).size != idx.size:
            raise ValueError("LD blocks overlap or repeat an index")
        if contiguous and idx.size > 1 and np.any(np.diff(idx) != 1):
            raise ValueError("each LD block must cover consecutive indices")
        seen[idx] = True
        size = _block_size(corr)
        if size is not None and size != idx.size:
            raise ValueError(f"an LD block of size {size} has {idx.size} indices")
        out.append((corr, idx))
    if not seen.all():
        raise ValueError("LD blocks do not cover every variant")
    return out


def _progress_event(progress, step, done, total):
    if progress is not None:
        progress({"step": str(step), "done": int(done), "total": int(total),
                  "unit": "block"})


def _report_screen(keep, tested, per_round, total, *, label, verbose,
                   stats_out, warning_stacklevel):
    """One trait's counts, drop/untested warnings and verbose printout."""
    n_tested = int(tested.sum())
    n_untested = int((~tested).sum())
    n_dropped = int((~keep).sum())
    if stats_out is not None:
        stats_out.update(n_tested=n_tested, n_untested=n_untested,
                         n_dropped=n_dropped)
    if n_tested and (n_dropped / n_tested) > DEFAULT_DROP_FRACTION_WARN:
        warnings.warn(
            f"{label}{LD_SCREEN_HIGH_DROP} {n_dropped:,} of "
            f"{n_tested:,} tested variants ({n_dropped / n_tested:.0%}). At "
            "the default chi2 threshold the expected null rate is near zero; "
            "rates of several percent usually mean the LD reference is "
            "thresholded or indefinite (the eigenvalue floor and the leverage "
            "then disagree). Inspect the reference before treating the mask "
            "as QC.", RuntimeWarning, stacklevel=warning_stacklevel)
    if n_untested and (n_tested == 0
                       or n_untested / total >= DEFAULT_UNTESTED_FRACTION_WARN):
        warnings.warn(
            f"{label}LD-consistency screen tested {n_tested:,} of {total:,} "
            f"variants ({n_untested:,} {LD_SCREEN_UNTESTED} of size "
            f"{MIN_WINDOW} and were kept unevaluated). A sparse panel can be "
            "largely unscreened while still recording that a screen ran.",
            RuntimeWarning, stacklevel=warning_stacklevel)
    if verbose:
        remaining = total
        for round_no, count in enumerate(per_round):
            remaining -= int(count)
            print(f"  {label}LD screen round {round_no + 1}: dropped "
                  f"{count:,}, {remaining:,} remain", flush=True)
        extra = (f" ({n_untested:,} untested, kept without evaluation)"
                 if n_untested else "")
        print(f"  {label}LD screen tested {n_tested:,} of {total:,} "
              f"variants{extra}", flush=True)
    return {"n_tested": n_tested, "n_untested": n_untested,
            "n_dropped": n_dropped, "per_round": [int(c) for c in per_round]}


def _run_screen(specs, z, total, *, rounds, window, threshold,
                eigenvalue_floor, seed, ncores, verbose, progress,
                progress_label, stats_out, eigh, warning_stacklevel):
    """Validate controls and screen ``(block, idx, source_rows)`` specs."""
    validate_boolean(verbose=verbose)
    if progress is not None and not callable(progress):
        raise TypeError(f"progress must be callable or None, got "
                        f"{type(progress).__name__}")
    seed = validate_seed(seed)
    rounds = integer_at_least("rounds", rounds, 1)
    window = integer_at_least("window", window, MIN_WINDOW)
    threshold = finite_control("threshold", threshold)
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    eigenvalue_floor = finite_control("eigenvalue_floor", eigenvalue_floor,
                                      lower=0.0)
    if eigenvalue_floor >= 1:
        raise ValueError("eigenvalue_floor must be < 1")
    ncores = integer_at_least("ncores", ncores, 1)
    eigh = _screen_eigh if eigh is None else eigh

    single = np.ndim(z) == 1
    zt = np.ascontiguousarray(
        np.asarray(z, dtype=_WORK_DTYPE).reshape(total, -1).T)
    # One independent stream per (block, round), derived in a fixed order.
    root = np.random.SeedSequence(seed)
    tasks = [(*spec, child.spawn(rounds))
             for spec, child in zip(specs, root.spawn(len(specs)))]
    settle = functools.partial(_settle_block, z=zt, window=window,
                               threshold=threshold,
                               eigenvalue_floor=eigenvalue_floor, eigh=eigh)

    def _settled(source):
        out = []
        for done, item in enumerate(source, 1):
            out.append(item)
            _progress_event(progress, progress_label, done, len(tasks))
        return out

    if ncores > 1 and len(tasks) > 1:
        # Blocks are independent; the caller is responsible for a BLAS that
        # tolerates concurrent LAPACK calls (pin it to one thread).
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=ncores) as executor:
            results = _settled(executor.map(settle, tasks))
    else:
        results = _settled(settle(task) for task in tasks)

    n_traits = zt.shape[0]
    keep = np.ones((n_traits, total), dtype=bool)
    tested = np.zeros((n_traits, total), dtype=bool)
    per_round = np.zeros((n_traits, rounds), dtype=np.int64)
    for (block_keep, dropped, block_tested), spec in zip(results, specs):
        idx = spec[1]
        keep[:, idx] = block_keep
        tested[:, idx] = block_tested
        per_round += dropped
    stats = []
    for trait in range(n_traits):
        stats.append(_report_screen(
            keep[trait], tested[trait], per_round[trait], total,
            label="" if single else f"z column {trait}: ", verbose=verbose,
            stats_out=stats_out, warning_stacklevel=warning_stacklevel + 1))
    return (keep[0] if single else keep.T), (stats[0] if single else stats)


def _check_z(z, total):
    z = np.asarray(z, dtype=np.float64)
    if (z.ndim not in (1, 2) or z.shape[0] != total
            or (z.ndim == 2 and z.shape[1] == 0)):
        raise ValueError(
            f"z has shape {z.shape}, but the blocks span {total} variants; "
            "pass (m,) or (m, n_traits)")
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains non-finite values; filter them first")
    return z


def ld_consistency_screen(blocks, z, *, rounds=DEFAULT_ROUNDS,
                          window=DEFAULT_WINDOW, threshold=DEFAULT_THRESHOLD,
                          eigenvalue_floor=DEFAULT_SPLIT_HALF_EIGENVALUE_FLOOR,
                          seed=0, ncores=1, verbose=False, progress=None,
                          progress_label="LD consistency screen",
                          return_stats=False, stats_out=None, _eigh=None,
                          _warning_stacklevel=2):
    """Keep-mask of variants whose z-scores are consistent with their LD.

    Parameters
    ----------
    blocks : list of (R, idx)
        LD blocks whose indices partition ``0..m-1`` (see the module docs for
        the accepted ``R`` forms). Use the LD the statistics will be fitted
        or interpreted with.
    z : array_like, shape (m,) or (m, n_traits)
        ``beta / se`` in block index order. Each column is screened on its own
        schedule (bit-identical to screening it alone); several columns share
        the costly window work while their survivors coincide.
    rounds : int
        Passes with fresh random splits; outliers are removed as found, so a
        later pass can see a variant masked by a bad neighbour.
    window, threshold, eigenvalue_floor, seed
        Window size (variants), chi-square cut, pinv truncation and the seed
        of the random splits.
    ncores : int
        Screen this many blocks concurrently. The mask equals ``ncores=1``
        whatever the pool does, but the windows' LAPACK eigensolves then run
        concurrently: use it only with a reentrant BLAS pinned to one thread.
    progress : callable, optional
        Called from this thread with ``{"step", "done", "total", "unit"}``
        after each block.
    return_stats : bool
        Also return per-trait counts: ``n_tested``, ``n_untested`` (variants
        in a block smaller than :data:`MIN_WINDOW`, kept unevaluated),
        ``n_dropped`` and ``per_round``.

    Returns
    -------
    keep : ndarray of bool shaped like ``z`` (and ``stats`` when requested)
        High drop rates and large untested fractions emit RuntimeWarning.
    """
    try:
        blocks = list(blocks)
        total = sum(len(idx) for _block, idx in blocks)
    except (TypeError, ValueError):
        raise ValueError("blocks must be a sequence of (LD, index) pairs") from None
    z = _check_z(z, total)
    blocks = _validate_blocks(blocks, total)
    validate_boolean(return_stats=return_stats)
    keep, stats = _run_screen(
        [(block, idx, None) for block, idx in blocks], z, total,
        rounds=rounds, window=window, threshold=threshold,
        eigenvalue_floor=eigenvalue_floor, seed=seed, ncores=ncores,
        verbose=verbose, progress=progress, progress_label=progress_label,
        stats_out=stats_out, eigh=_eigh,
        warning_stacklevel=_warning_stacklevel + 1)
    return (keep, stats) if return_stats else keep


def ld_consistency_screen_selected(blocks, selection, z, *,
                                   rounds=DEFAULT_ROUNDS,
                                   window=DEFAULT_WINDOW,
                                   threshold=DEFAULT_THRESHOLD,
                                   eigenvalue_floor=DEFAULT_SPLIT_HALF_EIGENVALUE_FLOOR,
                                   seed=0, ncores=1, verbose=False,
                                   progress=None,
                                   progress_label="LD consistency screen",
                                   return_stats=False, stats_out=None,
                                   _eigh=None, _warning_stacklevel=2):
    """Screen the variants ``selection`` of contiguous blocks, in place.

    ``blocks`` tile ``0..M-1`` contiguously (a whole LD cache); ``selection``
    is a strictly increasing index vector into that space and ``z`` is
    aligned to it. Only window-sized submatrices of each source block are
    formed, so the selected panel is never materialised. The result, seed
    streams and progress events equal subsetting the blocks first and then
    calling :func:`ld_consistency_screen`.
    """
    try:
        source = list(blocks)
        source_total = sum(len(idx) for _block, idx in source)
    except (TypeError, ValueError):
        raise ValueError("blocks must be a sequence of (LD, index) pairs") from None
    source = _validate_blocks(source, source_total, contiguous=True)
    expected = 0
    for _block, idx in source:
        if int(idx[0]) != expected:
            raise ValueError("LD blocks must tile contiguous indices 0..m-1")
        expected += int(idx.size)
    values = np.asarray(selection)
    if (values.ndim != 1 or not np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.bool_)):
        raise ValueError("selection must be a one-dimensional integer array")
    selection = values.astype(np.int64, copy=False)
    if selection.size == 0:
        raise ValueError("selection contains no variants")
    if selection[0] < 0 or selection[-1] >= source_total:
        raise IndexError(f"selection indices must lie in [0, {source_total})")
    if selection.size > 1 and np.any(np.diff(selection) <= 0):
        raise ValueError("selection must be strictly increasing and unique")
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (selection.size,):
        raise ValueError(f"z has shape {z.shape}, but selection spans "
                         f"{selection.size} variants")
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains non-finite values; filter them first")
    validate_boolean(return_stats=return_stats)
    specs = []
    out_start = 0
    for block, idx in source:
        start, stop = int(idx[0]), int(idx[-1]) + 1
        left = int(np.searchsorted(selection, start, side="left"))
        right = int(np.searchsorted(selection, stop, side="left"))
        if left == right:
            continue
        source_rows = selection[left:right] - start
        count = int(source_rows.size)
        specs.append((block, np.arange(out_start, out_start + count,
                                       dtype=np.int64), source_rows))
        out_start += count
    if out_start != selection.size:
        raise ValueError("LD blocks do not cover every selected variant")
    keep, stats = _run_screen(
        specs, z, int(selection.size), rounds=rounds, window=window,
        threshold=threshold, eigenvalue_floor=eigenvalue_floor, seed=seed,
        ncores=ncores, verbose=verbose, progress=progress,
        progress_label=progress_label, stats_out=stats_out, eigh=_eigh,
        warning_stacklevel=_warning_stacklevel + 1)
    return (keep, stats) if return_stats else keep


def dense_blocks(R, sizes):
    """Cut a dense correlation matrix into diagonal blocks of ``sizes``."""
    R = np.asarray(R)
    sizes = [int(s) for s in sizes]
    if R.ndim != 2 or R.shape[0] != R.shape[1] or sum(sizes) != R.shape[0]:
        raise ValueError("sizes must partition a square matrix")
    out, start = [], 0
    for s in sizes:
        out.append((R[start:start + s, start:start + s],
                    np.arange(start, start + s, dtype=np.int64)))
        start += s
    return out


def chromosome_blocks(chrom, *, max_size=2000):
    """Block sizes for a panel sorted by chromosome and position.

    Each chromosome run is split into the fewest near-equal blocks of at most
    ``max_size`` variants. Fixed-size blocks are adequate for a QC screen; use
    published LD-block boundaries when blocks must be near-independent.
    """
    from .alleles import normalize_chrom
    max_size = integer_at_least("max_size", max_size, MIN_WINDOW)
    labels = [normalize_chrom(c) for c in np.asarray(chrom, dtype=object)]
    sizes = []
    seen = set()
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        if labels[i] in seen:
            raise ValueError(f"chromosome {labels[i]} is not contiguous; sort "
                             "the panel by chromosome first")
        seen.add(labels[i])
        n = j - i
        k = -(-n // max_size)
        base, extra = divmod(n, k)
        sizes += [base + 1] * extra + [base] * (k - extra)
        i = j
    return sizes


def ld_blocks_from_genotypes(genotypes, sizes, *, dtype=np.float32):
    """Correlation blocks from a reference genotype matrix.

    ``genotypes`` is ``(n_samples, m)``: allele counts with ``-1`` or NaN for
    missing calls (mean-imputed per variant). Returns ``[(R, idx), ...]`` in
    ``dtype``; monomorphic variants get zero off-diagonal correlation.
    """
    G = np.asarray(genotypes)
    if G.ndim != 2:
        raise ValueError("genotypes must be (n_samples, n_variants)")
    sizes = [int(s) for s in sizes]
    if sum(sizes) != G.shape[1]:
        raise ValueError("sizes must sum to the number of variants")
    out, start = [], 0
    for s in sizes:
        X = np.asarray(G[:, start:start + s], dtype=np.float64)
        missing = ~np.isfinite(X) | (X < 0)
        X = np.where(missing, np.nan, X)
        mean = np.nanmean(X, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        X = np.where(missing, mean, X) - mean
        sd = np.sqrt((X * X).mean(axis=0))
        X = np.divide(X, sd, out=np.zeros_like(X), where=sd > 0)
        R = (X.T @ X) / X.shape[0]
        R[np.diag_indices(s)] = 1.0
        out.append((R.astype(dtype),
                    np.arange(start, start + s, dtype=np.int64)))
        start += s
    return out
