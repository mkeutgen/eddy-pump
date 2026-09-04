#!/usr/bin/env python
"""Draw the study's labelling batches: a blind worksheet, a sealed answer key and a draw record each.

reads  data/candidates/net_carbon_v1/ (the saved lists, verified), results/net_carbon_v1/scores/,
       data/external/{calibration_reference_b6.csv, manually_verified_physical_subd_events.csv,
       letter_pool_features.parquet}, config/
writes results/net_carbon_v1/labeling/<batch>/{<batch>.csv, ANSWER_KEY_do_not_open.csv, panels/} (not in git)
       data/labels/draws/{<batch>.yaml, DRAWS_SHA256, BATCHES.md}
usage  draw_batch.py BATCH [BATCH ...] [--seed N] [--target 0.15] [--render] [--dry-run]
       draw_batch.py --list                 the batch names this script knows and their state
       draw_batch.py --report SHEET.csv     check a re-labelled 42-panel calibration sheet
       draw_batch.py --repass BATCH_ID      a fresh blind copy of a calibration batch

A batch must be named. There is no default list, and the script refuses to write a draw record for
a batch that already has one, or that is already in the label table: the record of a measured rate
says which levels the rate stands on, and a second draw under the same name would replace it with a
different frame and a different number. To see what a draw would do, add --dry-run; to draw the same
design again, give it a new name.

The design is `eddy_pump.batches`: score deciles, Neyman allocation with a 5 % floor, one panel per
float per stratum at equal inclusion probability, blind controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "pipeline"))

from eddy_pump import batches as B  # noqa: E402
from eddy_pump import candidates as C  # noqa: E402
from eddy_pump import labels as L  # noqa: E402
from eddy_pump.criteria import active_criterion, require_ruled  # noqa: E402
from eddy_pump.manifest import load_manifest  # noqa: E402
import scores as SSP  # noqa: E402

DRAWS = REPO / "data/labels/draws"
SCORES = REPO / "results/net_carbon_v1/scores"
SCORES_MANIFEST = REPO / "data/features/net_carbon_v1/SCORES_SHA256"
CANDIDATES = REPO / "data/candidates/net_carbon_v1"   # the saved lists, full tables
B6_REFERENCE = REPO / "data/external/calibration_reference_b6.csv"   # the first 42 upward panels; sha256 8b3af4d3..., the same file the draw records name
COMPANION_EVENTS = REPO / "data/external/manually_verified_physical_subd_events.csv"   # the companion's reviewed subduction events
COMPANION_CRIT = "the companion's 2024 rule"
N_CONTROLS = 20            # positive and negative each, per rate batch — the blind carry-over
CALIB_TIERS = {"clear_TP": 12, "clear_FP": 12, "borderline": 18}   # the first upward calibration set's own shape
SECONDS_PER_PANEL = 28.7
RHO_FLOAT = 0.235          # (1.87 - 1) / (4.7 - 1): the study's pooled design effect at its panels per float
RHO_SOURCE = ("intra-float correlation implied by the study's pooled float-bootstrap design effect 1.87 "
              "at 4.7 panels per float")

# --- the region the first upward draw held back ------------------------------------------------ #
# The first upward rate batch sampled only part of the pool: the levels that the earlier study had
# already sampled directly were held back and credited to that sample. The study no longer credits
# them, so they are drawn afresh here (rate_obduction_02). The region is the levels of this pool
# that the earlier study's own detection table scores at 1.96 σ or more on both physical channels —
# a fixed list of keys, not a cut on this study's residuals, so it is reconstructed from that table.
HELD_REFERENCE = REPO / "data/external/letter_pool_features.parquet"
#: The 17,243 detections of the earlier study with the two residual columns this region is cut on.
#: Copied here on 2026-09-04 from the archive checkout's data/legacy/letter_pool_features.parquet
#: (mkeutgen/eddy-pump-archive, tag archive-2026-09-03), which took them from that study's
#: pool_features.parquet. Pinned by hash: if the file moves, the region moves, and the number a rate
#: divides by moves with it.
HELD_REFERENCE_SHA256 = "67b129d89d235dd377261f955fc823f86ec588d0f2334afe43795b2fb2256a44"
HELD_GATE = 1.96
HELD_BAND_EDGES = [0, 200, 260, 400, 600, 1000]
HELD_BAND_LABELS = ["<=200", "200-260", "260-400", "400-600", "600-1000"]
HELD_LEVELS = 14697        # what rate_obduction_01's frozen record says the region holds
HELD_PANELS = 90           # the plan's budget for it: about 90 panels, one per float
HELD_PREFIX = "former_held"   # the record of rate_obduction_01 calls the same strata letter_pool_1p96|<band>

_VERIFIED_ROWS: dict[str, int] = {}   # event_type -> rows of the saved list, as verified when it was read


def saved_rows(event_type: str) -> int:
    """The pool's candidate levels, taken from the list verified when its frame was loaded."""
    if event_type not in _VERIFIED_ROWS:
        raise SystemExit(f"{event_type}: its saved candidate list has not been read and verified yet")
    return _VERIFIED_ROWS[event_type]


# --------------------------------------------------------------------------------------------- #
# the pools with their scores and identities
# --------------------------------------------------------------------------------------------- #
def load_pool_frame(study, pool) -> pd.DataFrame:
    """One row per candidate level of a pool: its key, its score, its decile and its coordinates.

    The saved list is read with `verify=True`, so the key hash, the spec and the cache block are
    checked before a single row of it reaches a draw or a denominator.
    """
    et = pool.event_type
    s = pd.read_parquet(SCORES / f"{et}.parquet")
    if not ((s.pool_id == pool.pool_id).all() and (s.spec_id == pool.spec_id).all()):
        raise SystemExit(f"{et}: the score table is not this pool's ({pool.pool_id}, {pool.spec_id})")
    s["key"] = B.key3(s)
    a, side = C.read_saved(study, pool, verify=True)
    _VERIFIED_ROWS[et] = len(a)
    a["key"] = B.key3(a)
    s = s.merge(a[["key", "candidate_id"]], on="key", how="left", validate="one_to_one")
    if s.candidate_id.isna().any():
        raise SystemExit(f"{et}: {int(s.candidate_id.isna().sum())} scored rows are not in the saved candidate list")
    c = a[["key", "LATITUDE", "LONGITUDE", "TIME"]].drop_duplicates("key")
    s = s.merge(c, on="key", how="left", validate="one_to_one")
    if len(s) != len(a):
        raise SystemExit(f"{et}: {len(s)} rows against {len(a)} saved levels")
    s["WMO"] = s.WMO.astype(int)
    s["CYCLE_NUMBER"] = s.CYCLE_NUMBER.round().astype(int)
    s["decile"] = B.score_deciles(s)
    s["design_stratum"] = "open|d" + s.decile.astype(str)
    return s


FEATURES_MANIFEST = REPO / "data/features/net_carbon_v1/FEATURES_SHA256"


def scores_manifest() -> dict:
    """The score manifest and, through the feature manifest it was built on, the cache identity."""
    scores = json.loads(SCORES_MANIFEST.read_text().split("\n", 1)[1])
    feats = json.loads(FEATURES_MANIFEST.read_text().split("\n", 1)[1])
    return {"sha256": B.sha256_of(SCORES_MANIFEST), "content": scores,
            "features_sha256": B.sha256_of(FEATURES_MANIFEST), "cache_sha256": feats["cache"]["fine_grids_sha256"]}


# --------------------------------------------------------------------------------------------- #
# labels that steer the allocation (never a number)
# --------------------------------------------------------------------------------------------- #
def upward_steering_labels(s: pd.DataFrame) -> pd.DataFrame:
    """The study's own obduction reviews joined to the pool, each with the score a model gave it
    without seeing its own label: the calibration the upward allocation is planned on (mirrors
    downward_steering_labels)."""
    u = SSP.upward_labels(set(s.key))   # columns: key (the tuple), decision, source
    m = u[["key", "decision"]].drop_duplicates("key").merge(s[["key", "score", "score_is_oof"]], on="key", how="inner")
    if not m.score_is_oof.all():
        raise SystemExit("an upward steering label carries a score fitted on itself; the scorer must flag every "
                         "labelled row as scored without seeing its own label")
    return m.rename(columns={"decision": "y"})[["key", "score", "y"]]


def downward_steering_labels(s: pd.DataFrame) -> pd.DataFrame:
    """The companion's reviewed detections joined to the pool, each with the score a model gave it
    without seeing its own label."""
    d = SSP.downward_labels(set(s.key))   # columns: key (the tuple), decision, source
    m = d[["key", "decision"]].drop_duplicates("key").merge(s[["key", "score", "score_is_oof"]], on="key", how="inner")
    if not m.score_is_oof.all():
        raise SystemExit("a downward steering label carries a score fitted on itself")
    return m.rename(columns={"decision": "y"})[["key", "score", "y"]]


def _stratum_table(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.groupby("design_stratum", sort=True)
    T = pd.DataFrame({"N": g.size(), "floats": g.WMO.nunique(), "score_min": g.score.min(), "score_max": g.score.max(),
                      "p_raw": g._p.mean(), "max_float_share": g.apply(lambda d: d.groupby("WMO").size().max() / len(d))})
    T["W"] = T.N / len(frame)
    return T


def planned_acceptance(s: pd.DataFrame, lab: pd.DataFrame, p_plan: float,
                       rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per design stratum: N, W (share of the frame drawn from), floats, score range, planned p.

    The monotone score → acceptance map is fitted on the steering labels and rescaled so the whole
    pool's mean planned acceptance is `p_plan`; the rescale factor belongs to the pool, not to a
    sub-frame. `rows` draws the table over a subset of the pool (the former held region) while
    keeping that pool-wide factor.
    """
    p_row = B.isotonic_acceptance(lab.score.to_numpy(float), lab.y.to_numpy(float), s.score.to_numpy(float))
    s = s.assign(_p=p_row)
    T_pool = _stratum_table(s)
    scale = p_plan / float((T_pool.W * T_pool.p_raw).sum())
    T = T_pool if rows is None else _stratum_table(s.loc[rows.index].assign(design_stratum=rows.design_stratum))
    T["p_planned"] = np.clip(T.p_raw * scale, 0.005, 0.995)
    T["rescale_factor"] = scale
    return T.reset_index()


# --------------------------------------------------------------------------------------------- #
# the planning base rate
# --------------------------------------------------------------------------------------------- #
def planning_base_rate(lab: pd.DataFrame) -> tuple[float, str]:
    # study-derived: the planning base rate is the steering labels' own acceptance rate
    return float(lab.y.mean()), f"the obduction steering labels' own base rate, {int(lab.y.sum())}/{len(lab)} (study-derived)"


# --------------------------------------------------------------------------------------------- #
# the rate arm
# --------------------------------------------------------------------------------------------- #
def rate_arm(study, pool, s: pd.DataFrame, lab: pd.DataFrame, p_plan: float, p_source: str, target: float,
             rng: np.random.Generator, controls: tuple[pd.DataFrame, pd.DataFrame, dict, dict],
             previously_judged: set, batch_id: str) -> tuple[B.Batch, dict]:
    T = planned_acceptance(s, lab, p_plan)
    open_T = T.reset_index(drop=True)   # the whole pool
    v_target = (target * p_plan / B.Z) ** 2
    # The one-per-float rule holds within a stratum; across the ten strata a float can recur, and
    # verdicts on one float are correlated. The study sample's pooled design effect (1.87 at 4.7
    # panels per float) implies an intra-float correlation of ~0.235; the draw is solved with the
    # design effect its own panels-per-float implies, iterated until the two agree.
    deff, iterations = 1.0, []
    for it in range(6):
        n_total, n_h = B.solve_n(open_T.W.to_numpy(), open_T.p_planned.to_numpy(), v_target / deff, 0.0)
        open_T["n"] = n_h
        rng_it = rng.spawn(1)[0]
        science, chunked = [], 0
        for r in open_T.itertuples():
            S = s[s.design_stratum == r.design_stratum]
            chunked += int((S.groupby("WMO").size() > max(1, len(S) // int(r.n))).sum()) if r.n else 0
            science.append(B.draw_one_per_float(S, int(r.n), rng_it))
        science = pd.concat(science, ignore_index=True)
        m_bar = len(science) / science.WMO.nunique()
        deff_new = 1.0 + (m_bar - 1.0) * RHO_FLOAT
        iterations.append({"n": int(n_total), "panels_per_float": float(m_bar), "design_effect": float(deff_new)})
        if abs(deff_new - deff) < 0.005:
            break
        deff = deff_new
    deff = deff_new
    open_T["inclusion_probability"] = open_T.n / open_T.N
    v_open = B.stratified_variance(open_T.W, open_T.p_planned, open_T.n)
    # this allocation against proportional allocation at the same n, and against simple random sampling
    n_prop = np.maximum(1, np.round(open_T.W / open_T.W.sum() * n_total)).astype(int)
    v_prop = B.stratified_variance(open_T.W, open_T.p_planned, n_prop)
    W_open = float(open_T.W.sum())
    p_open = float((open_T.W * open_T.p_planned).sum() / W_open)
    v_srs = W_open ** 2 * p_open * (1 - p_open) / n_total
    science["previously_judged"] = [k in previously_judged for k in science.key]
    ctrl, pos, neg, pos_meta, neg_meta = blind_controls(controls, science, rng)
    batch = B.Batch(batch_id=batch_id, science=science, controls=ctrl, event_type=pool.event_type, rng=rng)
    per_float = science.groupby("WMO").size()
    design = {
        "role": "analysis", "decides": True,
        "sampling": {"mode": "probability", "draw": "stratified_pps_one_per_float", "design": "probability",
                     "frame": f"the active pool {pool.pool_id} ({saved_rows(pool.event_type):,} candidate levels); the draw covers "
                              f"all {int(open_T.N.sum()):,} levels",
                     "strata": "rank deciles of the classifier score over the whole pool",
                     "inclusion_probability": "n_h / N_h within each stratum, exact (one panel per float per stratum by systematic PPS)",
                     "allocation": f"Neyman on planned acceptance with a {B.FLOOR_SHARE:.0%} floor per decile; n solved for the target"},
        "target": {"rel_half_width": target, "planning_base_rate": p_plan, "planning_base_rate_source": p_source,
                   "precision_of": "the pool rate",
                   "variance_target": v_target, "variance_held": 0.0, "variance_open_planned": v_open,
                   "float_design_effect": float(deff), "float_design_effect_iterations": iterations,
                   "rho_float": RHO_FLOAT, "rho_float_source": RHO_SOURCE,
                   "variance_open_with_design_effect": float(v_open * deff),
                   "expected_rel_half_width": B.rel_half_width(v_open * deff, p_plan),
                   "expected_rel_half_width_if_unclustered": B.rel_half_width(v_open, p_plan),
                   "neyman_vs_proportional_multiplier": float(v_open / v_prop) if v_prop > 0 else None,
                   "stratified_vs_srs_multiplier": float(v_open / v_srs) if v_srs > 0 else None},
        "steering_labels": {"n": int(len(lab)), "accepted": int(lab.y.sum()), "what": pos_meta.get("steering", "")},
        "n_science": int(len(science)), "n_controls": {"positive": int(len(pos)), "negative": int(len(neg))},
        "floats_in_draw": int(science.WMO.nunique()), "floats_with_more_than_one_panel": int((per_float > 1).sum()),
        "max_panels_one_float": int(per_float.max()),
        "floats_chunked_for_pps": chunked,
        "panels_per_float_in_draw": float(len(science) / science.WMO.nunique()),
        "previously_judged_in_draw": int(science.previously_judged.sum()),
        "hours_at_planning_rate": float(len(batch.science) + len(ctrl)) * SECONDS_PER_PANEL / 3600,
        "strata": [{k: (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else v))
                    for k, v in r.items()} for r in open_T.drop(columns=["rescale_factor"]).to_dict("records")],
        "rescale_factor_planned_acceptance": float(open_T.rescale_factor.iloc[0]),
        "held_strata": [],
        "controls": {"positive": pos_meta, "negative": neg_meta},
    }
    return batch, design


def blind_controls(controls, science: pd.DataFrame, rng: np.random.Generator):
    """Twenty positive and twenty negative blind controls, none of them a science row."""
    pos_src, neg_src, pos_meta, neg_meta = controls
    drawn = set(science.key)
    pos = B.draw_controls(pos_src[~pos_src.key.isin(drawn)], N_CONTROLS, rng).assign(stratum=B.POS_CTRL)
    neg = B.draw_controls(neg_src[~neg_src.key.isin(drawn) & ~neg_src.key.isin(pos.key)], N_CONTROLS, rng).assign(stratum=B.NEG_CTRL)
    pos["src"], neg["src"] = pos_meta["src"], neg_meta["src"]
    ctrl = pd.concat([pos, neg], ignore_index=True)
    ctrl["previously_judged"] = True
    return ctrl, pos, neg, pos_meta, neg_meta


# --------------------------------------------------------------------------------------------- #
# the region the first upward draw held back
# --------------------------------------------------------------------------------------------- #
def held_region(s: pd.DataFrame) -> pd.DataFrame:
    """The 14,697 upward levels the first upward draw held back, with their five pressure strata.

    They are the levels of this pool that the earlier study's own detection table (kept as data in
    `data/external/letter_pool_features.parquet`) scores at 1.96 σ or more on both physical
    channels. That is a fixed list of keys made by another detector run: this study's own residual
    columns do not reproduce it (they give 14,688 levels, 47 of them different), so the region is
    taken from that table and nowhere else. The count and the five strata are checked against the
    frozen record of `rate_obduction_01` before anything is drawn.
    """
    if not HELD_REFERENCE.exists():
        raise SystemExit(f"not ready: {HELD_REFERENCE} is absent, and it is the only description of the region "
                         f"rate_obduction_02 samples. Without it the region would be a guess; refusing to draw.")
    got = B.sha256_of(HELD_REFERENCE)
    if got != HELD_REFERENCE_SHA256:
        raise SystemExit(f"not ready: {HELD_REFERENCE.name} hashes to {got[:16]}…, not the {HELD_REFERENCE_SHA256[:16]}… "
                         f"this region was reconstructed from. The file moved; the region would move with it.")
    ff = pd.read_parquet(HELD_REFERENCE, columns=["WMO", "CYCLE_NUMBER", "PRES_ADJUSTED",
                                                  "AOU_res_abs_at_det", "ABS_SAL_res_abs_at_det"])
    ff["key"] = B.key3(ff)
    ff = ff.drop_duplicates("key")
    keys = set(ff.key[(ff.AOU_res_abs_at_det >= HELD_GATE) & (ff.ABS_SAL_res_abs_at_det >= HELD_GATE)])
    h = s[s.key.isin(keys)].copy()
    band = pd.cut(h.PRES_ADJUSTED, HELD_BAND_EDGES, labels=HELD_BAND_LABELS, include_lowest=True).astype(str)
    if not band.isin(HELD_BAND_LABELS).all():
        raise SystemExit(f"not ready: {int((~band.isin(HELD_BAND_LABELS)).sum())} levels of the region fall outside "
                         f"the five pressure bands the frozen record names; refusing to draw a region it does not describe")
    h["design_stratum"] = HELD_PREFIX + "|" + band
    check = held_region_matches_the_frozen_record(h)
    if check["mismatch"]:
        raise SystemExit("not ready: the reconstructed region is not the one rate_obduction_01 held back — "
                         + "; ".join(check["mismatch"]) + ". Refusing to draw a guessed region.")
    return h


def held_region_matches_the_frozen_record(h: pd.DataFrame) -> dict:
    """Compare a reconstruction with the frozen record of `rate_obduction_01`: total and per band."""
    import yaml

    out = {"levels": int(len(h)), "expected_levels": HELD_LEVELS, "per_stratum": {}, "expected_per_stratum": {},
           "mismatch": []}
    out["per_stratum"] = {k.split("|", 1)[1]: int(v) for k, v in h.design_stratum.value_counts().items()}
    if len(h) != HELD_LEVELS:
        out["mismatch"].append(f"{len(h):,} levels against the recorded {HELD_LEVELS:,}")
    rec = DRAWS / "rate_obduction_01.yaml"
    if rec.exists():
        frozen = yaml.safe_load(rec.read_text())
        out["expected_per_stratum"] = {r["stratum"].split("|", 1)[1]: int(r["N"]) for r in frozen["held_strata"]}
        for band, n in sorted(out["expected_per_stratum"].items()):
            got = out["per_stratum"].get(band, 0)
            if got != n:
                out["mismatch"].append(f"band {band}: {got:,} levels against the recorded {n:,}")
    return out


def held_arm(study, pool, s: pd.DataFrame, lab: pd.DataFrame, p_plan: float, p_source: str, n_panels: int,
             rng: np.random.Generator, controls: tuple[pd.DataFrame, pd.DataFrame, dict, dict],
             previously_judged: set, batch_id: str) -> tuple[B.Batch, dict]:
    """The former held region as its own probability sample: `n_panels` panels over the five
    pressure strata, one per float within a stratum, every level of a stratum at the same n_h/N_h.

    The size is a budget, not a solve: the plan gives this region about 90 panels. It is 7.9 % of
    the pool by levels, so what it adds to the pool rate's error bar is small even at that size; the
    record carries both its own expected precision and what it contributes to the pool.
    """
    h = held_region(s)
    T = planned_acceptance(s, lab, p_plan, rows=h)
    order = {f"{HELD_PREFIX}|{b}": i for i, b in enumerate(HELD_BAND_LABELS)}   # shallow to deep, not alphabetical
    T = T.sort_values("design_stratum", key=lambda c: c.map(order), kind="mergesort").reset_index(drop=True)
    T["n"] = B.neyman_allocation(T.W.to_numpy(), T.p_planned.to_numpy(), n_panels, B.FLOOR_SHARE)
    science, chunked = [], 0
    for r in T.itertuples():
        S = h[h.design_stratum == r.design_stratum]
        chunked += int((S.groupby("WMO").size() > max(1, len(S) // int(r.n))).sum()) if r.n else 0
        science.append(B.draw_one_per_float(S, int(r.n), rng))
    science = pd.concat(science, ignore_index=True)
    T["inclusion_probability"] = T.n / T.N
    m_bar = len(science) / science.WMO.nunique()
    deff = 1.0 + (m_bar - 1.0) * RHO_FLOAT
    v_region = B.stratified_variance(T.W, T.p_planned, T.n)          # of the region's own rate
    p_region = float((T.W * T.p_planned).sum())
    n_prop = np.maximum(1, np.round(T.W / T.W.sum() * n_panels)).astype(int)
    v_prop = B.stratified_variance(T.W, T.p_planned, n_prop)
    v_srs = p_region * (1 - p_region) / n_panels
    W_pool = float(len(h)) / saved_rows(pool.event_type)             # the region's share of the pool
    science["previously_judged"] = [k in previously_judged for k in science.key]
    ctrl, pos, neg, pos_meta, neg_meta = blind_controls(controls, science, rng)
    batch = B.Batch(batch_id=batch_id, science=science, controls=ctrl, event_type=pool.event_type, rng=rng)
    per_float = science.groupby("WMO").size()
    design = {
        "role": "analysis", "decides": True,
        "sampling": {"mode": "probability", "draw": "stratified_pps_one_per_float", "design": "probability",
                     "frame": f"the {len(h):,} levels of {pool.pool_id} that the first upward draw "
                              f"(rate_obduction_01) held back and credited to the earlier study's own sample; "
                              f"the study no longer credits it, so it is drawn afresh here",
                     "frame_definition": f"levels of this pool whose entry in {HELD_REFERENCE.name} scores "
                                         f"{HELD_GATE} sigma or more on both AOU and absolute salinity at the "
                                         f"detection point; the frozen record of rate_obduction_01 calls the same "
                                         f"strata letter_pool_1p96|<band>",
                     "strata": "five pressure bands, " + ", ".join(HELD_BAND_LABELS) + " dbar — the strata the "
                               "frozen record of rate_obduction_01 names for this region",
                     "inclusion_probability": "n_h / N_h within each stratum, exact (one panel per float per stratum by systematic PPS)",
                     "allocation": f"Neyman on planned acceptance with a {B.FLOOR_SHARE:.0%} floor per band, at a fixed "
                                   f"budget of {n_panels} panels (docs/PLAN.md)"},
        "target": {"rel_half_width": None, "planning_base_rate": p_plan, "planning_base_rate_source": p_source,
                   "precision_of": "this region's own rate",
                   "panel_budget": int(n_panels), "panel_budget_source": "docs/PLAN.md: about 90 panels, one per float",
                   "variance_target": None, "variance_held": 0.0, "variance_open_planned": v_region,
                   "float_design_effect": float(deff), "rho_float": RHO_FLOAT, "rho_float_source": RHO_SOURCE,
                   "variance_open_with_design_effect": float(v_region * deff),
                   "expected_rel_half_width": B.rel_half_width(v_region * deff, p_region),
                   "expected_rel_half_width_if_unclustered": B.rel_half_width(v_region, p_region),
                   "planned_rate_of_this_region": p_region,
                   "region_share_of_pool": W_pool,
                   "variance_added_to_the_pool_rate": float(W_pool ** 2 * v_region * deff),
                   "neyman_vs_proportional_multiplier": float(v_region / v_prop) if v_prop > 0 else None,
                   "stratified_vs_srs_multiplier": float(v_region / v_srs) if v_srs > 0 else None},
        "steering_labels": {"n": int(len(lab)), "accepted": int(lab.y.sum()), "what": pos_meta.get("steering", "")},
        "n_science": int(len(science)), "n_controls": {"positive": int(len(pos)), "negative": int(len(neg))},
        "floats_in_draw": int(science.WMO.nunique()), "floats_with_more_than_one_panel": int((per_float > 1).sum()),
        "max_panels_one_float": int(per_float.max()),
        "floats_chunked_for_pps": chunked,
        "panels_per_float_in_draw": float(m_bar),
        "previously_judged_in_draw": int(science.previously_judged.sum()),
        "hours_at_planning_rate": float(len(batch.science) + len(ctrl)) * SECONDS_PER_PANEL / 3600,
        "strata": [{k: (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else v))
                    for k, v in r.items()} for r in T.drop(columns=["rescale_factor"]).to_dict("records")],
        "rescale_factor_planned_acceptance": float(T.rescale_factor.iloc[0]),
        "held_strata": [],
        "reconstruction": held_region_matches_the_frozen_record(h),
        "controls": {"positive": pos_meta, "negative": neg_meta},
    }
    return batch, design


# --------------------------------------------------------------------------------------------- #
# control sources
# --------------------------------------------------------------------------------------------- #
NEG_CONTROL_SCORE_CAP = 0.5   # a rejected reference event the classifier calls near-certain is a likely detector miss, not a control


def companion_status(s: pd.DataFrame) -> pd.Series:
    """Each pool key's status in the companion's reviewed events: 'verified' (Category 1/2),
    'rejected' (Category 0), else 'none'. The same source scores.py's downward_labels uses."""
    cc = pd.read_csv(COMPANION_EVENTS)
    cc["key"] = B.key3(cc)
    cat = cc.drop_duplicates("key").set_index("key").Category
    st = s.key.map(cat)
    return st.map(lambda c: "verified" if c in (1, 2) else ("rejected" if c == 0 else "none"))


def study_status(s: pd.DataFrame) -> pd.Series:
    """Each pool key's standing verdict in the study's own reviews of this pool: 'accepted',
    'rejected' or 'none'. The upward mirror of `companion_status`."""
    u = SSP.upward_labels(set(s.key)).drop_duplicates("key").set_index("key").decision
    st = s.key.map(u)
    return st.map(lambda d: "accepted" if d == 1 else ("rejected" if d == 0 else "none"))


def upward_controls(s: pd.DataFrame, crit) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Blind controls for an obduction rate batch, from the first upward calibration set: its
    accepted events (REF_LABEL 1) are the positive controls, its rejected events (REF_LABEL 0)
    the negatives — the same 42 panels calib_obduction_b6 is built from. This mirrors
    downward_controls, which draws its controls from the companion's reviewed events."""
    ref = pd.read_csv(B6_REFERENCE)
    ref["key"] = B.key3(ref)
    lab = ref.drop_duplicates("key").set_index("key").REF_LABEL
    st = s.key.map(lab)
    pos = s[st == 1].assign(REF_LABEL=1)
    neg = s[(st == 0) & (s.score < NEG_CONTROL_SCORE_CAP)].assign(REF_LABEL=0)
    pos_meta = {"source": "the first upward calibration set's accepted events (REF_LABEL 1) with a candidate in the pool",
                "criterion": crit.id, "n_source": int(len(pos)), "reference_k": None, "reference_n": None,
                "reference_provenance": f"the 42 upward calibration panels ({B6_REFERENCE.name}); the reader reports "
                                        "the controls, the drift test pools the arms across batches",
                "src": "first upward calibration set, accepted",
                "steering": "the study's own obduction reviews joined to the pool, each with the score a model gave it "
                            "without seeing its own label"}
    neg_meta = {"source": f"the first upward calibration set's rejected events (REF_LABEL 0) with a candidate in the pool, "
                          f"classifier score < {NEG_CONTROL_SCORE_CAP}", "criterion": crit.id,
                "n_source": int(len(neg)), "score_cap": NEG_CONTROL_SCORE_CAP, "ceiling": 0.20,
                "ceiling_note": "display only; a rejected candidate the classifier calls likely is a plausible detector miss, not a control",
                "src": "first upward calibration set, rejected"}
    return pos, neg, pos_meta, neg_meta


def downward_controls(s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    st = companion_status(s)
    pos = s[st == "verified"].assign(REF_LABEL=1)
    neg = s[st == "rejected"].assign(REF_LABEL=0)
    pos_meta = {"source": "the companion's verified subduction events (Category 1/2) with a candidate in the pool", "criterion": COMPANION_CRIT,
                "n_source": int(len(pos)), "reference_k": None, "reference_n": None,
                "reference_provenance": f"judged under {COMPANION_CRIT} (clauses 1-3, cycle unit); the acceptance under clause 4 is "
                                        "unknown until calib_subduction_v1 is decided — the reader reports these, no Fisher test",
                "src": f"companion verified under {COMPANION_CRIT}",
                "steering": "the companion's reviewed detections joined to the pool (verified 1, rejected or detected-not-verified 0), "
                            "rescaled to the planning base rate"}
    neg_meta = {"source": "the companion's explicit rejections (Category 0) with a candidate in the pool", "criterion": COMPANION_CRIT,
                "n_source": int(len(neg)), "ceiling": 0.20, "src": f"companion rejected under {COMPANION_CRIT}"}
    return pos, neg, pos_meta, neg_meta


# --------------------------------------------------------------------------------------------- #
# the calibration batches
# --------------------------------------------------------------------------------------------- #
def calib_upward_b6(s: pd.DataFrame, crit, rng: np.random.Generator) -> tuple[B.Batch, dict]:
    """The first upward calibration set: the 42 events of the frozen reference kept as data."""
    reference = crit.raw["anchors"]["obduction"]     # data/external/calibration_reference_b6.csv
    ref = pd.read_csv(B6_REFERENCE)
    sha = B.sha256_of(B6_REFERENCE)
    ref["key"] = B.key3(ref)
    m = ref[["key", "REF_LABEL", "tier"]].merge(s, on="key", how="left", validate="one_to_one")
    if m.candidate_id.isna().any():
        raise SystemExit("a calibration panel is not in the active obduction pool")
    m["stratum"] = "calibration"
    m["src"] = "calibration panel, tier " + m.tier
    ctrl = m
    base_rate = f"{int(ref.REF_LABEL.sum())}/{len(ref)}"
    batch = B.Batch(batch_id="calib_obduction_b6", science=s.iloc[0:0].assign(design_stratum=pd.Series(dtype=str)),
                    controls=ctrl, event_type="physical_obduction", rng=rng, extra_key_cols=("tier",))
    design = {"role": "calibration", "decides": False,
              "sampling": {"mode": "calibration", "draw": "the frozen reference, every event", "design": None, "frame": None,
                           "inclusion_probability": None},
              "reference": {"what": "the 42 upward calibration panels (the study's first upward reference)",
                            "file": str(B6_REFERENCE.relative_to(REPO)), "sha256": sha, "n_events": int(len(ref)),
                            "base_rate": base_rate, "carried_over_as": reference},
              "check_before_labelling": f"kappa > {B.KAPPA_PASS} against REF_LABEL and the base rate on target, before every "
                                        f"session (docs/LABELING_PROTOCOL.md)",
              "panel_note": "the study panel shows the two physical channels (phys_net_carbon_v1.channels_shown); the reference was "
                            "judged on the earlier four-channel panel — the first re-labelling measures the criterion across instruments",
              "n_science": 0, "n_controls": {"calibration": int(len(ref))},
              "hours_at_planning_rate": float(len(ref)) * SECONDS_PER_PANEL / 3600,
              "reference_rows": ref[B.KEYS + ["REF_LABEL", "tier"]].to_dict("records")}
    return batch, design


def calib_upward_v1(s: pd.DataFrame, rng: np.random.Generator) -> tuple[B.Batch, dict]:
    """A fresh 42-panel upward calibration set, drawn from the study's own pool under the study's
    criterion — the upward mirror of `calib_downward`.

    The first upward set (`calib_obduction_b6`) is 42 panels carried over from the earlier study
    and judged on its four-channel panel. This one is drawn here: twelve panels the study itself
    accepted in the top score decile, twelve it rejected or never saw at the bottom, and eighteen
    in the middle where the decision is actually made. REF_LABEL is blank; it is filled once, by
    two blind passes decided by hand, exactly as the downward set was.
    """
    d = s.assign(study=study_status(s))
    used: set[int] = set()

    def take(pool_rows: pd.DataFrame, n: int, tier: str) -> pd.DataFrame:
        rows = B.draw_controls(pool_rows[~pool_rows.WMO.isin(used)], n, rng)
        if len(rows) < n:
            raise SystemExit(f"calib_obduction_v1: only {len(rows)} of {n} for tier {tier}")
        used.update(rows.WMO)
        return rows.assign(tier=tier)

    clear_tp = take(d[(d.study == "accepted") & (d.decile == 9)], CALIB_TIERS["clear_TP"], "clear_TP")
    rej_low = take(d[(d.study == "rejected") & (d.decile <= 2)], min(6, int(((d.study == "rejected") & (d.decile <= 2)).sum())), "clear_FP")
    unl_low = take(d[(d.study == "none") & (d.decile == 0)], CALIB_TIERS["clear_FP"] - len(rej_low), "clear_FP")
    mid = d[d.decile.between(5, 8)]
    bord_a = take(mid[mid.study == "accepted"], 9, "borderline")
    bord_r = take(mid[mid.study == "rejected"], min(5, int((mid.study == "rejected").sum())), "borderline")
    bord_u = take(mid[mid.study == "none"], 9 - len(bord_r), "borderline")
    m = pd.concat([clear_tp, rej_low, unl_low, bord_a, bord_r, bord_u], ignore_index=True)
    assert len(m) == sum(CALIB_TIERS.values()) and m.WMO.is_unique
    m["stratum"] = "calibration"
    m["src"] = "candidate for the upward calibration set, tier " + m.tier + ", study verdict " + m.study
    m["REF_LABEL"] = np.nan
    batch = B.Batch(batch_id="calib_obduction_v1", science=s.iloc[0:0].assign(design_stratum=pd.Series(dtype=str)),
                    controls=m, event_type="physical_obduction", rng=rng, extra_key_cols=("tier", "study"))
    design = {"role": "calibration", "decides": False,
              "sampling": {"mode": "calibration", "draw": "stratified by score decile and the study's standing verdict, one per float",
                           "design": None, "frame": None, "inclusion_probability": None},
              "composition": {"clear_TP": "12 levels the study accepted, in the top score decile",
                              "clear_FP": f"{len(rej_low)} the study rejected in deciles 0-2 + {len(unl_low)} never labelled in decile 0",
                              "borderline": f"18 in deciles 5-8: 9 the study accepted, {len(bord_r)} it rejected, {len(bord_u)} never labelled"},
              "status": "REF_LABEL is blank: to be filled once, by two blind passes decided by hand under "
                        "phys_net_carbon_v1, then frozen as the upward reference (docs/PLAN.md step 2)",
              "replaces": "calib_obduction_b6, the 42 panels carried over from the earlier study",
              "n_science": 0, "n_controls": {"calibration": int(len(m))},
              "hours_at_planning_rate": float(len(m)) * SECONDS_PER_PANEL / 3600}
    return batch, design


def calib_downward(s: pd.DataFrame, rng: np.random.Generator) -> tuple[B.Batch, dict]:
    d = s.assign(companion=companion_status(s))
    used: set[int] = set()

    def take(pool_rows: pd.DataFrame, n: int, tier: str) -> pd.DataFrame:
        rows = B.draw_controls(pool_rows[~pool_rows.WMO.isin(used)], n, rng)
        if len(rows) < n:
            raise SystemExit(f"calib_subduction_v1: only {len(rows)} of {n} for tier {tier}")
        used.update(rows.WMO)
        return rows.assign(tier=tier)

    clear_tp = take(d[(d.companion == "verified") & (d.decile == 9)], CALIB_TIERS["clear_TP"], "clear_TP")
    rej_low = take(d[(d.companion == "rejected") & (d.decile <= 2)], min(6, int(((d.companion == "rejected") & (d.decile <= 2)).sum())), "clear_FP")
    unl_low = take(d[(d.companion == "none") & (d.decile == 0)], CALIB_TIERS["clear_FP"] - len(rej_low), "clear_FP")
    mid = d[d.decile.between(5, 8)]
    bord_v = take(mid[mid.companion == "verified"], 9, "borderline")
    bord_r = take(mid[mid.companion == "rejected"], min(5, int((mid.companion == "rejected").sum())), "borderline")
    bord_u = take(mid[mid.companion == "none"], 9 - len(bord_r), "borderline")
    m = pd.concat([clear_tp, rej_low, unl_low, bord_v, bord_r, bord_u], ignore_index=True)
    assert len(m) == sum(CALIB_TIERS.values()) and m.WMO.is_unique
    m["stratum"] = "calibration"
    m["src"] = "candidate for the downward calibration set, tier " + m.tier + ", companion " + m.companion
    m["REF_LABEL"] = np.nan
    batch = B.Batch(batch_id="calib_subduction_v1", science=s.iloc[0:0].assign(design_stratum=pd.Series(dtype=str)),
                    controls=m, event_type="physical_subduction", rng=rng, extra_key_cols=("tier", "companion"))
    design = {"role": "calibration", "decides": False,
              "sampling": {"mode": "calibration", "draw": "stratified by score decile and companion status, one per float", "design": None,
                           "frame": None, "inclusion_probability": None},
              "composition": {"clear_TP": "12 companion-verified events in the top score decile",
                              "clear_FP": f"{len(rej_low)} companion-rejected in deciles 0-2 + {len(unl_low)} unlabelled in decile 0",
                              "borderline": f"18 in deciles 5-8: 9 companion-verified, {len(bord_r)} companion-rejected, {len(bord_u)} unlabelled"},
              "status": "REF_LABEL is blank: to be filled once by consensus under phys_net_carbon_v1 (who decides: open, "
                        "docs/PLAN.md), then frozen as phys_net_carbon_v1/anchors/subduction",
              "n_science": 0, "n_controls": {"calibration": int(len(m))},
              "hours_at_planning_rate": float(len(m)) * SECONDS_PER_PANEL / 3600}
    return batch, design


# --------------------------------------------------------------------------------------------- #
# the batches this script knows how to draw
# --------------------------------------------------------------------------------------------- #
#: batch name -> (pool event type, the index that seeds its own generator, what builds it).
#: The seed index is part of a batch's identity: it is what makes a draw reproducible from --seed
#: alone, so the four batches already drawn keep the indices they were drawn with.
DESIGNS: dict[str, dict] = {
    "calib_obduction_b6": {"pool": "physical_obduction", "seed_index": 0,
                           "what": "the first 42 upward calibration panels, carried over as data"},
    "calib_subduction_v1": {"pool": "physical_subduction", "seed_index": 1,
                            "what": "42 downward calibration panels, answers filled by hand"},
    "rate_obduction_01": {"pool": "physical_obduction", "seed_index": 2,
                          "what": "the upward rate: the whole pool, ten score deciles"},
    "rate_subduction_01": {"pool": "physical_subduction", "seed_index": 3,
                           "what": "the downward rate: the whole pool, ten score deciles"},
    "calib_obduction_v1": {"pool": "physical_obduction", "seed_index": 4,
                           "what": "42 fresh upward calibration panels from the study's own pool"},
    "rate_obduction_02": {"pool": "physical_obduction", "seed_index": 5,
                          "what": f"the {HELD_LEVELS:,} upward levels the first upward draw held back"},
}


class Plan:
    """What every design needs, loaded once and only when a requested batch asks for it."""

    def __init__(self, target: float, panels: int):
        self.study = load_manifest()
        self.crit = require_ruled(active_criterion())
        self.target = target
        self.panels = panels
        self.pools = {p.event_type: p for p in self.study.pools if p.tracer is None}
        self._frames: dict[str, pd.DataFrame] = {}
        self._judged: set | None = None
        self._plan_rate: tuple[float, str] | None = None

    def frame(self, event_type: str) -> pd.DataFrame:
        if event_type not in self._frames:
            self._frames[event_type] = load_pool_frame(self.study, self.pools[event_type])
        return self._frames[event_type]

    @property
    def judged(self) -> set:
        if self._judged is None:
            judged = {(int(w), int(c), int(p)) for w, c, p in L.labelled_keys()}
            judged |= set(B.key3(pd.read_csv(COMPANION_EVENTS)))
            self._judged = judged
        return self._judged

    @property
    def base_rate(self) -> tuple[float, str]:
        if self._plan_rate is None:
            self._plan_rate = planning_base_rate(upward_steering_labels(self.frame("physical_obduction")))
        return self._plan_rate

    def build(self, bid: str, rng: np.random.Generator) -> tuple[B.Batch, dict]:
        et = DESIGNS[bid]["pool"]
        s = self.frame(et)
        p_plan, p_source = self.base_rate
        if bid == "calib_obduction_b6":
            return calib_upward_b6(s, self.crit, rng)
        if bid == "calib_obduction_v1":
            return calib_upward_v1(s, rng)
        if bid == "calib_subduction_v1":
            return calib_downward(s, rng)
        if bid == "rate_obduction_01":
            return rate_arm(self.study, self.pools[et], s, upward_steering_labels(s), p_plan, p_source, self.target,
                            rng, upward_controls(s, self.crit), self.judged, bid)
        if bid == "rate_subduction_01":
            return rate_arm(self.study, self.pools[et], s, downward_steering_labels(s), p_plan,
                            p_source + " (shared with the downward arm)", self.target,
                            rng, downward_controls(s), self.judged, bid)
        if bid == "rate_obduction_02":
            return held_arm(self.study, self.pools[et], s, upward_steering_labels(s), p_plan, p_source, self.panels,
                            rng, upward_controls(s, self.crit), self.judged, bid)
        raise SystemExit(f"{bid}: no design of this name")


# --------------------------------------------------------------------------------------------- #
# what may be written
# --------------------------------------------------------------------------------------------- #
def already_drawn(bid: str) -> str | None:
    """The file that says this batch has already been drawn, or None."""
    import yaml

    rec = DRAWS / f"{bid}.yaml"
    if rec.exists():
        return B.repo_relative(rec)
    if L.STUDY_BATCHES.exists():
        raw = yaml.safe_load(L.STUDY_BATCHES.read_text(encoding="utf-8")) or {"batches": []}
        if any(b.get("batch_id") == bid for b in raw.get("batches") or []):
            return B.repo_relative(L.STUDY_BATCHES)
    return None


def refuse_a_second_draw(bid: str) -> None:
    where = already_drawn(bid)
    if where is None:
        return
    raise SystemExit(
        f"{bid} has already been drawn — {where} says so, and that file is what a rate's frame and "
        f"error bar are read from. A second draw under the same name would replace the frame the "
        f"measured rate stands on with a different one, silently. There is no flag that overrides "
        f"this. Use --dry-run to see what the draw would do, or draw the same design under a new name.")


# --------------------------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------------------------- #
def record_for(study, pool, crit, batch: B.Batch, design: dict, sheets: dict, seed: int, sm: dict) -> dict:
    return {
        "batch_id": batch.batch_id, "built": B.stamp(), "builder": "pipeline/draw_batch.py", "seed": seed,
        "study_id": study.study_id, "pool_id": pool.pool_id, "spec_id": pool.spec_id, "event_type": pool.event_type,
        "criterion_version": crit.id, "criterion_status": crit.status,
        "cache": {"path": str(study.cache.path), "fine_grids": study.cache.fine_grids, "fine_grids_sha256": study.cache.fine_grids_sha256},
        "scores_manifest": {"path": str(SCORES_MANIFEST.relative_to(REPO)), "sha256": sm["sha256"], "built": sm["content"].get("built"),
                            "features_manifest_sha256": sm["features_sha256"]},
        **design,
        "sheet_columns": B.WORKSHEET_COLS, "blind": True,
        **sheets,
    }


def render(study, pool, batch_dir: pathlib.Path, ws: pd.DataFrame) -> int:
    os.environ["ARGOPOD_RESIDUAL_CACHE"] = str(study.cache.path)
    from argopod.eventconfig import grids_fn_or_none, load_event_config
    from argopod.review.app import _event_slug  # the app's own panel-folder convention
    from argopod.review.panels import render_batch

    import shutil

    cfg = load_event_config(REPO / "config/review" / f"{pool.event_type}.yaml")
    pdir = batch_dir / "panels" / _event_slug(cfg.title)
    if pdir.exists():
        shutil.rmtree(pdir)   # a superseded draw's panels share SAMPLE_ID prefixes; never leave them beside the new ones
    out = render_batch(ws, grids_fn_or_none(cfg), cfg.panel_variables, event=cfg.title, var_labels=cfg.panel_labels, out_dir=pdir)
    return len(out)


def all_records() -> list[dict]:
    """Every draw record on disk, in name order — not only the ones this run wrote."""
    import yaml

    return [yaml.safe_load(p.read_text()) for p in sorted(DRAWS.glob("*.yaml"))
            if not (p.name.endswith(".labelled.yaml") or p.name.endswith(".reference.yaml"))]


def write_page(records: list[dict], target: float) -> None:
    lines = ["# The study's labelling batches — built " + B.stamp()[:10], "",
             "Built by `pipeline/draw_batch.py`; the design is `eddy_pump.batches`. Worksheets, keys and panels are under "
             "`results/net_carbon_v1/labeling/<batch_id>/` (not in Git); the draw records beside this page are the label "
             "table's input when a sheet comes back labelled.", "",
             "| batch | role | rows | of which science / controls | hours | expected precision | check before labelling |",
             "|---|---|---:|---|---:|---|---|"]
    for r in records:
        nc = r["n_controls"]
        ctrl = " / ".join(f"{k} {v}" for k, v in nc.items())
        t = r.get("target")
        prec = (f"±{t['expected_rel_half_width']:.1%} relative on {t.get('precision_of', 'the pool rate')}"
                + (f" (target ±{target:.0%})" if t.get("rel_half_width") else "") if t else "—")
        check = r.get("check_before_labelling") or (
            "the pool's 42 calibration panels re-labelled blind: PASS" if r["role"] == "analysis" else "—")
        lines.append(f"| `{r['batch_id']}` | {r['role']} | {r['worksheet']['rows']} | {r['n_science']} / {ctrl} | "
                     f"{r['hours_at_planning_rate']:.1f} | {prec} | {check} |")
    lines += ["", "## How to label one", "",
              "```", "make review BATCH=results/net_carbon_v1/labeling/<batch_id>/<batch_id>.csv   # the keyboard app, blind",
              "python pipeline/draw_batch.py --report results/net_carbon_v1/labeling/calib_obduction_b6/calib_obduction_b6.csv",
              "```", "",
              "The worksheet carries the key, the position and the coordinates — nothing else. The answer key beside it "
              "(`ANSWER_KEY_do_not_open.csv`) is opened by `argopod session` only once the sheet is finished.", ""]
    for r in records:
        if "target" not in r:
            continue
        t = r["target"]
        lines += [f"## `{r['batch_id']}` — the allocation", "",
                  f"Planning base rate {t['planning_base_rate']:.4f} ({t['planning_base_rate_source']}); target variance "
                  f"{t['variance_target'] if t['variance_target'] is None else format(t['variance_target'], '.3g')}, the draw "
                  f"{t['variance_open_planned']:.3g}. "
                  f"Neyman against proportional at the same n: ×{t['neyman_vs_proportional_multiplier']:.2f}.", "",
                  "| stratum | N | floats | score range | planned p | n | π = n/N |", "|---|---:|---:|---|---:|---:|---:|"]
        for s_ in r["strata"]:
            lines.append(f"| {s_['design_stratum']} | {s_['N']:,} | {s_['floats']:,} | {s_['score_min']:.4f}–{s_['score_max']:.4f} | "
                         f"{s_['p_planned']:.3f} | {s_['n']} | {s_['inclusion_probability']:.5f} |")
        lines.append("")
    (DRAWS / "BATCHES.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(seed: int) -> None:
    """`DRAWS_SHA256` over every draw record on disk plus the page, never only this run's."""
    names = sorted(p.name for p in DRAWS.glob("*.yaml")
                   if not (p.name.endswith(".labelled.yaml") or p.name.endswith(".reference.yaml")))
    (DRAWS / "DRAWS_SHA256").write_text(
        "# provenance of the study's draw records -- do not edit by hand\n"
        f"# built {B.stamp()} by pipeline/draw_batch.py, seed {seed}\n"
        + "".join(f"{B.sha256_of(DRAWS / n)}\t{n}\n" for n in names)
        + f"{B.sha256_of(DRAWS / 'BATCHES.md')}\tBATCHES.md\n")


def repass(src_id: str) -> None:
    """A blind re-labelling copy of a calibration batch: the same 42 levels in a new random order with
    new SAMPLE_IDs, LABEL blank, panels rendered afresh; its draw record points at the original
    (`reference_of`) so the load scores it against the frozen reference. The original labelled
    sheet is never opened for writing.

    The shuffle is seeded from the sha256 of the copy's name, so the same copy is the same
    permutation in any process; Python's own `hash()` is salted per process and would not be.
    """
    import yaml

    study = load_manifest()
    src = yaml.safe_load((DRAWS / f"{src_id}.yaml").read_text())
    if src["role"] != "calibration":
        raise SystemExit(f"{src_id} is not a calibration batch; a rate batch is never re-labelled, it is re-drawn")
    n = 2
    while (DRAWS / f"{src_id}_pass{n}.yaml").exists():
        n += 1
    bid = f"{src_id}_pass{n}"
    key = pd.read_csv(B.resolve_recorded_path(src["answer_key"]["path"]))
    rng = np.random.default_rng([seed_of(bid), n])
    key = key.iloc[rng.permutation(len(key))].reset_index(drop=True)
    key["SAMPLE_ID"] = np.arange(len(key))
    ws = key[["SAMPLE_ID"] + B.KEYS + ["LATITUDE", "LONGITUDE", "TIME"]].copy()
    ws.insert(0, "LABEL", pd.NA)
    ws["EVENT_TYPE"] = src["event_type"]
    ws = ws[B.WORKSHEET_COLS]
    bdir = study.output.resolve("labeling") / bid
    sheets = B.write_sheets(bdir, ws, key, bid)
    pool = next(p for p in study.pools if p.pool_id == src["pool_id"])
    n_png = render(study, pool, bdir, ws)
    rec = dict(src)
    rec.update({"batch_id": bid, "built": B.stamp(), "reference_of": src_id, "derived_from": src_id, "pass": n,
                "sampling": dict(src["sampling"], draw=f"the same levels as {src_id}, re-shuffled for a blind re-labelling (pass {n})"),
                "panels": {"dir": B.repo_relative(bdir / "panels"), "rendered": n_png, "of": int(len(ws))}, **sheets})
    (DRAWS / f"{bid}.yaml").write_text(
        f"# data/labels/draws/{bid}.yaml -- a blind re-labelling copy of {src_id} (pass {n}). Built by pipeline/draw_batch.py\n"
        f"# --repass; never edited by hand.\n" + yaml.safe_dump(rec, sort_keys=False, allow_unicode=True, width=110, default_flow_style=False),
        encoding="utf-8")
    print(f"{bid}: {len(ws)} rows, {n_png} panels -> {bdir / (bid + '.csv')}")


def seed_of(text: str) -> int:
    """A reproducible seed from a name: the first 8 bytes of its sha256.

    Python's `hash()` is salted per process, so a permutation seeded from it cannot be regenerated
    in a second process — a blind re-labelling copy has to be the same copy every time.
    """
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def print_plan(bid: str, design: dict, rows: int) -> None:
    t = design.get("target")
    print(f"{bid}: {rows} rows ({design['n_science']} science, {design['n_controls']}), "
          f"{design['hours_at_planning_rate']:.1f} h"
          + (f"; expected ±{t['expected_rel_half_width']:.1%} on {t.get('precision_of', 'the pool rate')} "
             f"(deff {t['float_design_effect']:.2f}); "
             f"vs SRS ×{t['stratified_vs_srs_multiplier']:.2f}, vs proportional ×{t['neyman_vs_proportional_multiplier']:.2f}; "
             f"{design['floats_in_draw']} floats, {design['panels_per_float_in_draw']:.2f} per float, "
             f"{design['floats_chunked_for_pps']} float-strata chunked"
             if t else ""))
    for s_ in design.get("strata", []):
        print(f"   {s_['design_stratum']:>17}  N {s_['N']:>7,}  p {s_['p_planned']:.3f}  n {s_['n']:>3}  π {s_['inclusion_probability']:.5f}")


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser(description="Draw one or more named labelling batches.")
    ap.add_argument("batches", nargs="*", metavar="BATCH", help="the batch name(s) to draw; " + ", ".join(DESIGNS))
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--target", type=float, default=0.15)
    ap.add_argument("--panels", type=int, default=HELD_PANELS,
                    help=f"the panel budget of a fixed-size batch (rate_obduction_02); default {HELD_PANELS}")
    ap.add_argument("--render", action="store_true", help="render the panel PNGs from the bound cache")
    ap.add_argument("--dry-run", action="store_true", help="print the allocation, write nothing")
    ap.add_argument("--list", action="store_true", help="print the batch names this script knows and their state")
    ap.add_argument("--report", help="a re-labelled calibration sheet: print the check and exit")
    ap.add_argument("--repass", metavar="BATCH_ID", help="a fresh blind copy of a calibration batch (rows shuffled, new SAMPLE_IDs, "
                                                        "panels re-rendered) as <BATCH_ID>_pass<N>, for a re-labelling; the labelled sheet is never touched")
    args = ap.parse_args()

    if args.list:
        for bid, d in DESIGNS.items():
            where = already_drawn(bid)
            print(f"{bid:22s} {d['pool']:20s} {'drawn: ' + where if where else 'not drawn':50s} {d['what']}")
        return

    if args.repass:
        repass(args.repass)
        return

    if args.report:
        sheet = pd.read_csv(args.report)
        key = pd.read_csv(pathlib.Path(args.report).parent / "ANSWER_KEY_do_not_open.csv")
        if key.REF_LABEL.isna().all():
            raise SystemExit("this calibration set has no REF_LABEL yet — its answers must be decided first")
        rep = B.calibration_report(sheet, key[key.REF_LABEL.notna()].assign(REF_LABEL=lambda d: d.REF_LABEL.astype(int)))
        print(json.dumps(rep, indent=1, default=str))
        return

    if not args.batches:
        raise SystemExit("name the batch to draw, e.g. `draw_batch.py rate_obduction_02`. The known names are:\n  "
                         + "\n  ".join(f"{b:22s} {d['what']}" for b, d in DESIGNS.items())
                         + "\nThere is no default: drawing everything again would overwrite the record a measured "
                           "rate stands on. `--list` shows which are already drawn, `--dry-run` prints a plan.")
    unknown = [b for b in args.batches if b not in DESIGNS]
    if unknown:
        raise SystemExit(f"no design named {unknown}; the known names are {list(DESIGNS)}")
    if not args.dry_run:
        for bid in args.batches:
            refuse_a_second_draw(bid)

    plan = Plan(args.target, args.panels)
    sm = scores_manifest()
    if sm["cache_sha256"] != plan.study.cache.fine_grids_sha256:
        raise SystemExit("the scores were made on another cache than the one the study is bound to")
    if args.render:
        C.require_bound_cache(plan.study)

    out_root = plan.study.output.resolve("labeling")
    wrote = False
    for bid in args.batches:
        rng = np.random.default_rng([args.seed, DESIGNS[bid]["seed_index"]])
        batch, design = plan.build(bid, rng)
        ws, key = batch.assemble()
        print_plan(bid, design, len(ws))
        if args.dry_run:
            continue
        pool = plan.pools[DESIGNS[bid]["pool"]]
        bdir = out_root / bid
        sheets = B.write_sheets(bdir, ws, key, bid)
        rec = record_for(plan.study, pool, plan.crit, batch, design, sheets, args.seed, sm)
        if args.render:
            n = render(plan.study, pool, bdir, ws)
            rec["panels"] = {"dir": B.repo_relative(bdir / "panels"), "rendered": n, "of": int(len(ws))}
            print(f"   panels: {n} of {len(ws)}")
        elif (bdir / "panels").exists():
            have = {p.name.split("_wmo")[0] for p in (bdir / "panels").rglob("*.png")}
            n = int(sum(str(sid) in have for sid in ws.SAMPLE_ID))
            rec["panels"] = {"dir": B.repo_relative(bdir / "panels"), "rendered": n, "of": int(len(ws)),
                             "note": "counted from disk; re-run with --render if fewer than the rows"}
        DRAWS.mkdir(parents=True, exist_ok=True)
        (DRAWS / f"{bid}.yaml").write_text(
            f"# data/labels/draws/{bid}.yaml -- the draw record of one study batch. Built by pipeline/draw_batch.py;\n"
            f"# never edited by hand. The sheet and key it hashes live under results/ (not in Git).\n"
            + yaml.safe_dump(rec, sort_keys=False, allow_unicode=True, width=110, default_flow_style=False), encoding="utf-8")
        wrote = True
    if not wrote:
        return
    write_page(all_records(), args.target)
    write_manifest(args.seed)
    print(f"-> {DRAWS}")


if __name__ == "__main__":
    main()
