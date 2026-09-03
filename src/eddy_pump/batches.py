"""The design of a labelling batch: strata, allocation, the draw, the controls, the estimator.

Draws nothing itself; `production/build_batches.py` calls it. `score_deciles` cuts a pool into
strata; `isotonic_acceptance` plans the acceptance per stratum; `neyman_allocation` and `solve_n`
size the draw for a target half-width; `draw_one_per_float` draws one panel per float per stratum
at equal inclusion probability n/N; `draw_controls` adds blind controls; `hajek_rate` and
`stratified_rate` are the estimators; `calibration_report` is the gate before a session (Cohen's κ
against the frozen 42-panel reference and the base rate on target: PASS or DRIFTED).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["WMO", "CYCLE_NUMBER", "PRES_ADJUSTED"]
TARGET, POS_CTRL, NEG_CTRL = "target", "pos_ctrl", "neg_ctrl"
CONTROL_STRATA = (POS_CTRL, NEG_CTRL)
N_DECILES = 10
FLOOR_SHARE = 0.05          # at least this share of the draw in every open decile (the plan's rule)
Z = 1.96
KAPPA_PASS = 0.6            # production/LABELING_PROTOCOL.md: κ > 0.6 and base rate on target
WORKSHEET_COLS = ["LABEL", "SAMPLE_ID", "WMO", "CYCLE_NUMBER", "PRES_ADJUSTED",
                  "LATITUDE", "LONGITUDE", "TIME", "EVENT_TYPE"]
BLIND_FORBIDDEN = {"score", "stratum", "src", "inclusion_probability", "candidate_id", "REF_LABEL",
                   "decile", "design_stratum", "region", "control_arm", "previously_judged"}


# --------------------------------------------------------------------------------------------- #
# keys and strata
# --------------------------------------------------------------------------------------------- #
def key3(df: pd.DataFrame) -> pd.Series:
    """The dedup key as a tuple column: (WMO, CYCLE_NUMBER, round(PRES_ADJUSTED))."""
    return pd.Series(list(zip(df.WMO.astype(int), df.CYCLE_NUMBER.round().astype(int),
                              df.PRES_ADJUSTED.round().astype(int))), index=df.index)


def score_deciles(df: pd.DataFrame, n: int = N_DECILES) -> pd.Series:
    """Decile of `score` by RANK — equal-sized strata, ties broken by the key so the cut is a
    deterministic function of the frozen pool and never of row order."""
    order = df.sort_values(["score", "WMO", "CYCLE_NUMBER", "PRES_ADJUSTED"], kind="mergesort").index
    rank = pd.Series(np.arange(len(order)), index=order)
    return (rank.reindex(df.index) * n // len(df)).astype(int)


# --------------------------------------------------------------------------------------------- #
# planned acceptance and allocation
# --------------------------------------------------------------------------------------------- #
def isotonic_acceptance(score_labelled: np.ndarray, y: np.ndarray, score_pool: np.ndarray,
                        floor: float = 0.005) -> np.ndarray:
    """A monotone map score → P(accept), fitted on labelled rows, evaluated on the pool.

    Isotonic regression (pool-adjacent-violators) on the labelled rows; the pool's scores are
    read off by nearest labelled score. Bounded below by `floor` so no stratum is planned at
    zero variance: a decile the map calls empty still gets its floor share and a real chance."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(score_labelled, y)
    return np.clip(iso.predict(score_pool), floor, 1.0 - floor)


def neyman_allocation(W: np.ndarray, p: np.ndarray, n: int, floor_share: float = FLOOR_SHARE) -> np.ndarray:
    """Integer n_h ∝ W_h·sqrt(p_h(1−p_h)), every stratum at least `floor_share` of n.

    Strata whose Neyman share falls under the floor are pinned to the floor and the remainder is
    re-allocated over the others by Neyman, repeated until nothing new is pinned. Rounding is by
    largest remainder, so the parts sum to n exactly."""
    W, p = np.asarray(W, float), np.asarray(p, float)
    if n < 0 or len(W) == 0:
        raise ValueError("n must be non-negative and there must be strata")
    if n == 0:
        return np.zeros(len(W), int)
    s = W * np.sqrt(p * (1 - p))
    if not np.all(s > 0):
        raise ValueError("every stratum needs W > 0 and 0 < p < 1")
    share = s / s.sum()
    pinned = np.zeros(len(W), bool)
    while True:
        free = ~pinned
        rest = 1.0 - floor_share * pinned.sum()
        share = np.where(pinned, floor_share, 0.0)
        share[free] = rest * s[free] / s[free].sum()
        newly = free & (share < floor_share - 1e-12)
        if not newly.any():
            break
        pinned |= newly
        if pinned.all():
            share = np.full(len(W), 1.0 / len(W))
            break
    raw = share * n
    base = np.floor(raw).astype(int)
    short = n - base.sum()
    if short > 0:
        base[np.argsort(-(raw - base), kind="mergesort")[:short]] += 1
    return base


def stratified_variance(W: np.ndarray, p: np.ndarray, n_h: np.ndarray, deff: np.ndarray | float = 1.0) -> float:
    """Var of the stratified mean, Σ W_h² p_h(1−p_h) deff_h / n_h (finite-population correction
    ignored: n ≪ N in every stratum). A stratum with n_h = 0 contributes infinity."""
    W, p, n_h = np.asarray(W, float), np.asarray(p, float), np.asarray(n_h, float)
    deff = np.broadcast_to(np.asarray(deff, float), W.shape)
    with np.errstate(divide="ignore"):
        v = W ** 2 * p * (1 - p) * deff / n_h
    return float(np.where(n_h > 0, v, np.inf).sum())


def solve_n(W_open: np.ndarray, p_open: np.ndarray, v_target: float, v_held: float = 0.0,
            floor_share: float = FLOOR_SHARE, n_max: int = 20_000) -> tuple[int, np.ndarray]:
    """The smallest total draw n whose Neyman-with-floor allocation meets `v_target` once the
    held strata's fixed variance `v_held` is added; returns (n, n_h)."""
    if v_held >= v_target:
        raise ValueError(f"the held strata alone exceed the target variance ({v_held:.3g} ≥ {v_target:.3g})")
    lo, hi = 1, n_max
    while lo < hi:
        mid = (lo + hi) // 2
        n_h = neyman_allocation(W_open, p_open, mid, floor_share)
        if stratified_variance(W_open, p_open, n_h) + v_held <= v_target:
            hi = mid
        else:
            lo = mid + 1
    n_h = neyman_allocation(W_open, p_open, lo, floor_share)
    if stratified_variance(W_open, p_open, n_h) + v_held > v_target:
        raise ValueError(f"no n ≤ {n_max} meets the target")
    return lo, n_h


# --------------------------------------------------------------------------------------------- #
# the draw
# --------------------------------------------------------------------------------------------- #
def clusters_of(S: pd.DataFrame, n: int) -> pd.Series:
    """One cluster id per row: the float, split into chunks of at most floor(N/n) candidates
    (in cycle-then-pressure order) when a float holds more than that.

    Systematic PPS needs every cluster's selection probability n·size/N ≤ 1. A float holding more
    than N/n of a stratum would exceed it, so it is chunked; the draw then takes at most one
    panel per CHUNK, which is one per float everywhere the frame allows and two or more only on
    the few floats that dominate a stratum. The report counts them."""
    N = len(S)
    m = max(1, N // n) if n > 0 else N
    order = S.sort_values(["WMO", "CYCLE_NUMBER", "PRES_ADJUSTED"], kind="mergesort")
    pos = order.groupby("WMO").cumcount()
    chunk = (pos // m).astype(int)
    cl = order.WMO.astype(str) + "#" + chunk.astype(str)
    return cl.reindex(S.index)


def draw_one_per_float(S: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """`n` candidates of one stratum, at most one per float (per chunk of a dominant float),
    every candidate with inclusion probability EXACTLY n/N.

    Systematic PPS on clusters with size = the cluster's candidates in the stratum (a random
    cluster order, one uniform start, hits at u, u+1, …, u+n−1 along the cumulative sizes scaled
    to sum to n), then one candidate uniformly within each hit cluster.
    P(candidate) = (n·size/N)·(1/size) = n/N, the same for every candidate of the stratum."""
    if n == 0:
        out = S.iloc[0:0].copy()
        out["inclusion_probability"] = pd.Series(dtype=float)
        return out
    if len(S) == 0:
        raise ValueError("empty stratum")
    N = len(S)
    if n > N:
        raise ValueError(f"n={n} panels but only {N} candidates in the stratum")
    S = S.assign(_cluster=clusters_of(S, n))
    sizes = S.groupby("_cluster").size()
    pi_c = n * sizes / N
    if (pi_c > 1 + 1e-9).any():
        raise AssertionError("a cluster's selection probability exceeds 1 after chunking — clusters_of is wrong")
    order = rng.permutation(sizes.index.to_numpy())
    cum = np.cumsum(pi_c.loc[order].to_numpy())
    cum[-1] = float(n)  # exact, against floating error
    u = rng.uniform()
    hit = np.searchsorted(cum, u + np.arange(n), side="right")
    hit = np.minimum(hit, len(order) - 1)
    chosen = order[hit]
    if len(set(chosen)) != n:
        raise AssertionError("systematic PPS produced a repeated cluster — sizes exceed one interval")
    rows = []
    by_cluster = S.groupby("_cluster").indices
    for c in chosen:
        idx = by_cluster[c]
        rows.append(S.index[idx[rng.integers(len(idx))]])
    out = S.loc[rows].drop(columns="_cluster").copy()
    out["inclusion_probability"] = n / N
    return out


def draw_controls(source: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """`n` rows uniformly from `source`, at most one per float (a simple random sample of floats
    then one row each; controls carry no inclusion probability — they never enter a rate)."""
    if n == 0 or len(source) == 0:
        return source.iloc[0:0].copy()
    floats = source.WMO.unique()
    n = min(n, len(floats))
    pick = rng.choice(floats, n, replace=False)
    by_float = source.groupby("WMO").indices
    rows = [source.index[by_float[w][rng.integers(len(by_float[w]))]] for w in pick]
    return source.loc[rows].copy()


# --------------------------------------------------------------------------------------------- #
# the sheet and the key
# --------------------------------------------------------------------------------------------- #
@dataclass
class Batch:
    batch_id: str
    science: pd.DataFrame            # design rows: KEYS, design_stratum, score, inclusion_probability, candidate_id, coords
    controls: pd.DataFrame           # KEYS, stratum (pos_ctrl/neg_ctrl), src, REF_LABEL, score, candidate_id, coords
    event_type: str
    rng: np.random.Generator
    extra_key_cols: tuple[str, ...] = ()
    order: pd.DataFrame = field(default=None, repr=False)

    def assemble(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(worksheet, answer_key): one random interleaving, SAMPLE_ID = review order."""
        s = self.science.copy()
        s["stratum"] = TARGET
        s["src"] = s["design_stratum"]
        c = self.controls.copy()
        c["inclusion_probability"] = np.nan
        c["design_stratum"] = c["stratum"]
        cols = list(dict.fromkeys(KEYS + ["stratum", "src", "design_stratum", "score", "inclusion_probability",
                                          "candidate_id", "REF_LABEL", "previously_judged", "LATITUDE", "LONGITUDE",
                                          "TIME"] + list(self.extra_key_cols)))
        for d in (s, c):
            for col in cols:
                if col not in d.columns:
                    d[col] = np.nan
        allrows = pd.concat([s[cols], c[cols]], ignore_index=True)
        if allrows.duplicated(KEYS).any():
            dup = allrows[allrows.duplicated(KEYS, keep=False)]
            raise ValueError(f"{self.batch_id}: a candidate appears twice in one batch:\n{dup[KEYS].head()}")
        perm = self.rng.permutation(len(allrows))
        allrows = allrows.iloc[perm].reset_index(drop=True)
        allrows.insert(0, "SAMPLE_ID", np.arange(len(allrows)))
        key = allrows.copy()
        ws = allrows[["SAMPLE_ID"] + KEYS + ["LATITUDE", "LONGITUDE", "TIME"]].copy()
        ws.insert(0, "LABEL", pd.NA)
        ws["EVENT_TYPE"] = self.event_type
        ws = ws[WORKSHEET_COLS]
        assert not (set(ws.columns) & BLIND_FORBIDDEN), "the worksheet must be blind"
        self.order = key
        return ws, key


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_sheets(out_dir: Path, ws: pd.DataFrame, key: pd.DataFrame, batch_id: str) -> dict:
    """The worksheet `<batch_id>.csv` and `ANSWER_KEY_do_not_open.csv` beside it (the name
    argopod's session reader looks for). Refuses to overwrite a worksheet that carries verdicts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wp, kp = out_dir / f"{batch_id}.csv", out_dir / "ANSWER_KEY_do_not_open.csv"
    if wp.exists():
        old = pd.read_csv(wp)
        if "LABEL" in old.columns and old.LABEL.notna().any():
            raise FileExistsError(f"{wp} already carries {int(old.LABEL.notna().sum())} verdicts — a labelled "
                                  f"worksheet is never overwritten; move it or pick a new batch id")
    ws.to_csv(wp, index=False)
    key.to_csv(kp, index=False)
    return {"worksheet": {"path": str(wp), "sha256": sha256_of(wp), "rows": int(len(ws))},
            "answer_key": {"path": str(kp), "sha256": sha256_of(kp), "rows": int(len(key))}}


# --------------------------------------------------------------------------------------------- #
# the estimator the design serves
# --------------------------------------------------------------------------------------------- #
def hajek_rate(y: np.ndarray, pi: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """The weighted (1/π) mean of verdicts, its float-bootstrap SE and 95 % half-width.

    Rows with a missing π are refused: a control, a calibration row or a score-selected label has
    no inclusion probability and no place in a rate."""
    y, pi, groups = np.asarray(y, float), np.asarray(pi, float), np.asarray(groups)
    if len(y) == 0:
        raise ValueError("no rows")
    if np.isnan(pi).any() or (pi <= 0).any() or (pi > 1).any():
        raise ValueError("every row of a rate needs an inclusion probability in (0, 1]")
    w = 1.0 / pi
    point = float((w * y).sum() / w.sum())
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        boots[b] = (w[sel] * y[sel]).sum() / w[sel].sum()
    se = float(boots.std(ddof=1))
    return {"rate": point, "se_float_bootstrap": se, "half_width_95": Z * se,
            "half_width_95_rel": (Z * se / point) if point > 0 else float("inf"),
            "n": int(len(y)), "floats": int(len(uniq)), "sum_weights": float(w.sum())}


def stratified_rate(y: np.ndarray, pi: np.ndarray, stratum: np.ndarray, groups: np.ndarray, N_h: dict,
                    n_boot: int = 2000, seed: int = 0) -> dict:
    """The design-based estimate for THIS design, and its variance — the headline a rate reports.

    Within a stratum every panel is one PSU drawn with the same probability, so the estimator is
    the stratified mean  p = Σ_h W_h p̂_h  with W_h = N_h / N  (the Hájek with 1/π weights equals it
    when nothing is dropped; when an uncertain verdict is dropped the stratum weight is held at its
    frame share, which is the within-stratum ratio adjustment). The variance is the closed form
    Σ_h W_h² · p̃_h(1−p̃_h)/(n_h−1) · (1 − n_h/N_h), with p̃_h = p̂_h except that a stratum with zero
    (or all) accepts is floored at the Jeffreys mean (a+½)/(n+1) for the variance term only — a
    zero count at a planned 1.6 % acceptance is the expected outcome, not zero variance.

    Two resampling checks come with it: a stratified bootstrap (panels resampled WITHIN stratum,
    the design's own resampling) and the naive float bootstrap (floats resampled with replacement
    ignoring strata). The naive one lets the fixed n_h float and so overstates the variance of a
    stratified draw — measured by Monte Carlo on the real frame: the closed form is
    unbiased within 5 %, the naive float bootstrap 1.1–1.6× too high. It is kept as the labelled
    conservative sensitivity, never the headline."""
    y, pi, stratum, groups = np.asarray(y, float), np.asarray(pi, float), np.asarray(stratum), np.asarray(groups)
    if len(y) == 0:
        raise ValueError("no rows")
    if np.isnan(pi).any() or (pi <= 0).any() or (pi > 1).any():
        raise ValueError("every row of a rate needs an inclusion probability in (0, 1]")
    N = float(sum(N_h.values()))
    strata = sorted(set(stratum))
    if set(strata) - set(N_h):
        raise ValueError(f"strata without a frame size: {sorted(set(strata) - set(N_h))}")
    rng = np.random.default_rng(seed)
    point, var, zero_strata, per = 0.0, 0.0, 0, {}
    for h in strata:
        m = stratum == h
        n_h, a_h, Nh = int(m.sum()), float(y[m].sum()), float(N_h[h])
        p_h = a_h / n_h
        W = Nh / N
        point += W * p_h
        if a_h == 0 or a_h == n_h:
            zero_strata += 1
            p_var = (a_h + 0.5) / (n_h + 1)
        else:
            p_var = p_h
        v_h = W ** 2 * p_var * (1 - p_var) / max(n_h - 1, 1) * (1 - n_h / Nh)
        var += v_h
        per[h] = {"n": n_h, "accepted": int(a_h), "rate": p_h, "W": W, "variance_share": v_h}
    for h in per:
        per[h]["variance_share"] = per[h]["variance_share"] / var if var > 0 else float("nan")
    se_design = float(np.sqrt(var))
    # stratified bootstrap: resample panels within each stratum
    idx_h = {h: np.flatnonzero(stratum == h) for h in strata}
    W_h = {h: N_h[h] / N for h in strata}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = sum(W_h[h] * y[rng.choice(idx_h[h], len(idx_h[h]), replace=True)].mean() for h in strata)
    se_strat_boot = float(boots.std(ddof=1))
    # the naive float bootstrap of the Hájek, the conservative sensitivity
    naive = hajek_rate(y, pi, groups, n_boot=n_boot, seed=seed)
    # the weighted iid comparator, same weights as the estimator, for the design effect
    w = 1.0 / pi
    p_w = float((w * y).sum() / w.sum())
    se_iid_w = float(np.sqrt((w ** 2 * (y - p_w) ** 2).sum()) / w.sum())
    return {"rate": float(point), "se_design": se_design, "half_width_95": Z * se_design,
            "half_width_95_rel": (Z * se_design / point) if point > 0 else float("inf"),
            "se_stratified_bootstrap": se_strat_boot, "se_naive_float_bootstrap": naive["se_float_bootstrap"],
            "hajek_rate_declared_pi": naive["rate"], "se_iid_weighted": se_iid_w,
            "design_effect_vs_weighted_iid": (se_design / se_iid_w) ** 2 if se_iid_w > 0 else float("nan"),
            "zero_or_full_count_strata": zero_strata, "n": int(len(y)), "floats": int(len(np.unique(groups))),
            "strata": per, "N": N}


# --------------------------------------------------------------------------------------------- #
# the calibration gate
# --------------------------------------------------------------------------------------------- #
def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, int), np.asarray(b, int)
    if len(a) == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = float(sum(((a == k).mean()) * ((b == k).mean()) for k in np.union1d(a, b)))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def calibration_report(sheet: pd.DataFrame, reference: pd.DataFrame, kappa_pass: float = KAPPA_PASS,
                       base_rate_alpha: float = 0.05) -> dict:
    """The protocol's gate: re-label the anchor blind, then κ > 0.6 against the frozen
    reference and the base rate on target (two-sided binomial test at `base_rate_alpha`).

    `sheet` carries LABEL (0/1/2/blank), `reference` carries REF_LABEL; both carry the key.
    Undecided rows (blank or 2) are excluded from κ and counted."""
    from scipy.stats import binomtest

    s = sheet.copy()
    r = reference.copy()
    for d in (s, r):
        d["_k"] = key3(d)
    m = s.merge(r[["_k", "REF_LABEL"]], on="_k", how="inner", validate="one_to_one")
    missing = len(r) - len(m)
    m["LABEL"] = pd.to_numeric(m.LABEL, errors="coerce")
    decided = m[m.LABEL.isin([0, 1])]
    k = cohen_kappa(decided.LABEL, decided.REF_LABEL) if len(decided) else float("nan")
    ref_rate = float(r.REF_LABEL.mean())
    acc = int((decided.LABEL == 1).sum())
    p_base = float(binomtest(acc, len(decided), ref_rate).pvalue) if len(decided) else float("nan")
    on_target = bool(p_base >= base_rate_alpha) if len(decided) else False
    passed = bool(len(decided) == len(r) and k > kappa_pass and on_target)
    disagree = decided[decided.LABEL != decided.REF_LABEL]
    # what the gate can and cannot see at this n: a bootstrap interval on κ and the range of
    # accepted counts the base-rate test would still pass — PASS means "not detectably drifted".
    rng = np.random.default_rng(0)
    kb = []
    a, b = decided.LABEL.to_numpy(int), decided.REF_LABEL.to_numpy(int)
    for _ in range(2000):
        i = rng.integers(len(a), size=len(a))
        kb.append(cohen_kappa(a[i], b[i]))
    k_ci = [float(np.nanpercentile(kb, 2.5)), float(np.nanpercentile(kb, 97.5))] if len(decided) else [float("nan")] * 2
    pass_region = [x for x in range(len(r) + 1) if binomtest(x, len(r), ref_rate).pvalue >= base_rate_alpha]
    tier_col = "tier" if "tier" in r.columns else None
    by_tier = None
    if tier_col:
        mt = m.merge(r[["_k", tier_col]], on="_k")
        mt = mt[mt.LABEL.isin([0, 1])]
        by_tier = {t: {"n": int(len(g)), "agree": int((g.LABEL == g.REF_LABEL).sum())} for t, g in mt.groupby(tier_col)}
    return {"n_reference": int(len(r)), "n_decided": int(len(decided)), "n_missing_from_sheet": int(missing),
            "n_undecided": int(len(m) - len(decided)), "kappa": float(k), "kappa_ci95_bootstrap": k_ci, "kappa_pass": kappa_pass,
            "base_rate_reference": ref_rate, "base_rate_observed": (acc / len(decided)) if len(decided) else float("nan"),
            "base_rate_p": p_base, "base_rate_on_target": on_target,
            "base_rate_pass_region_accepted": [min(pass_region), max(pass_region)] if pass_region else None,
            "agreement_by_tier": by_tier,
            "verdict": "PASS" if passed else "DRIFTED",
            "verdict_means": "PASS = not detectably drifted at this n; it is not evidence of anchoring",
            "disagreements": disagree[KEYS + ["LABEL", "REF_LABEL"]].to_dict("records")}


def stamp() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def rel_half_width(v: float, p: float) -> float:
    return Z * math.sqrt(v) / p
