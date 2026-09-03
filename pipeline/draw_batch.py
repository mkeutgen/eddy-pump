#!/usr/bin/env python
"""Draw the study's labelling batches: a blind worksheet, a sealed answer key and a draw record each.

reads  data/candidates/net_carbon_v1/ (the saved lists), results/net_carbon_v1/scores/, data/labels/audit/
       (the reuse audit), data/labels/external/calibration_reference_b6.csv, config/
writes results/net_carbon_v1/labeling/<batch>/{<batch>.csv, ANSWER_KEY_do_not_open.csv, panels/} (not in git)
       data/labels/draws/{<batch>.yaml, DRAWS_SHA256, BATCHES.md}
usage  build_batches.py [--seed N] [--target 0.15] [--render] [--dry-run]
       build_batches.py --report SHEET.csv     the calibration gate on a re-labelled 42-panel sheet
       build_batches.py --repass BATCH_ID      a fresh blind copy of a calibration batch
The design is `eddy_pump.batches`: score deciles, Neyman allocation with a 5 % floor, one panel per
float per stratum at equal inclusion probability, blind controls.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "production"))

from eddy_pump import batches as B  # noqa: E402
from eddy_pump import candidates as C  # noqa: E402
from eddy_pump import labels as L  # noqa: E402
from eddy_pump.criteria import active_criterion, load_criteria, require_ruled  # noqa: E402
from eddy_pump.manifest import load_manifest  # noqa: E402
import reuse_audit as RA  # noqa: E402

DRAWS = REPO / "data/labels/draws"
SCORES = REPO / "results/net_carbon_v1/scores"
SCORES_MANIFEST = REPO / "data/features/net_carbon_v1/SCORES_SHA256"
CANDIDATES = REPO / "data/candidates/net_carbon_v1"   # the saved lists, full tables


def saved_rows(event_type: str) -> int:
    """The row count of a pool's saved candidate list, from its sidecar."""
    return int(json.loads((CANDIDATES / f"{event_type}.json").read_text())["rows"])
COVERAGE = REPO / "data/labels/audit/coverage_by_pool_and_stratum.csv"
BUDGET = REPO / "data/labels/audit/label_budget.csv"
COMPANION_COV = REPO / "data/labels/audit/companion_coverage.csv"
REUSE_MAP = REPO / "data/labels/audit/reuse_map.parquet"
DIRECT_FRAMES = REPO / "data/labels/audit/direct_by_legacy_frame.csv"
DIRECT_FLIPS = REPO / "data/labels/audit/direct_flips.csv"
B6_REFERENCE = REPO / "data/labels/external/calibration_reference_b6.csv"   # the 42 panels; sha256 8b3af4d3..., the same file the draw records name
LEGACY_POOL = "legacy_letter_v1/physical/obduction"
B6 = "phys_obduction_letter_b6"
COMPANION_CRIT = "phys_companion_2024"
HELD_FRAME = "letter_pool_1p96"
N_CONTROLS = 20            # positive and negative each, per rate batch — the blind carry-over
CALIB_TIERS = {"clear_TP": 12, "clear_FP": 12, "borderline": 18}   # the b6 reference's own shape
SECONDS_PER_PANEL = RA.SECONDS_PER_PANEL
DESIGN = "score_stratified_spread_across_floats"
RHO_FLOAT = 0.235          # (1.87 - 1) / (4.7 - 1): the legacy pooled design effect at its panels per float
RHO_SOURCE = ("intra-float correlation implied by the direct sample's pooled float-bootstrap design effect 1.87 "
              "at 4.7 panels per float (data/labels/audit/label_budget.csv, design_effect_pooled)")


# --------------------------------------------------------------------------------------------- #
# the pools with their scores, frames and identities
# --------------------------------------------------------------------------------------------- #
def load_pool_frame(study, pool, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    et = pool.event_type
    s = pd.read_parquet(SCORES / f"{et}.parquet")
    if not ((s.pool_id == pool.pool_id).all() and (s.spec_id == pool.spec_id).all()):
        raise SystemExit(f"{et}: the score table is not this pool's ({pool.pool_id}, {pool.spec_id})")
    s["key"] = B.key3(s)
    a, side = C.read_anchor(study, pool)
    a["key"] = B.key3(a)
    s = s.merge(a[["key", "candidate_id"]], on="key", how="left", validate="one_to_one")
    if s.candidate_id.isna().any():
        raise SystemExit(f"{et}: {int(s.candidate_id.isna().sum())} scored rows are not in the anchor")
    c = pd.read_parquet(CANDIDATES / f"{et}.parquet", columns=B.KEYS + ["LATITUDE", "LONGITUDE", "TIME"])
    n_levels = saved_rows(et)
    c["key"] = B.key3(c)
    c = c.drop_duplicates("key").drop(columns=B.KEYS)
    s = s.merge(c, on="key", how="left", validate="one_to_one")
    f = frames[pool.pool_id][["key", "frame_stratum", "depth_band"]]
    s = s.merge(f, on="key", how="left", validate="one_to_one")
    if len(s) != n_levels or s.frame_stratum.isna().any():
        raise SystemExit(f"{et}: {len(s)} rows against {n_levels} saved levels, or a row without a frame")
    s["WMO"] = s.WMO.astype(int)
    s["CYCLE_NUMBER"] = s.CYCLE_NUMBER.round().astype(int)
    s["decile"] = B.score_deciles(s)
    held = (s.frame_stratum == HELD_FRAME) if pool.direction.value == "obduction" else pd.Series(False, index=s.index)
    s["region"] = np.where(held, "held", "open")
    s["design_stratum"] = s.region + "|d" + s.decile.astype(str)
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
    """The uniform b6 reviews with their OUT-OF-FOLD score: the calibration the upward
    allocation is planned on."""
    an = L.analysis_sample(LEGACY_POOL, B6)
    an = an[an.decision.isin([0, 1])]
    an = an.assign(key=list(zip(an.key_wmo.astype(int), an.key_cycle.astype(int), an.key_pres.astype(int))))
    m = an[["key", "decision"]].drop_duplicates("key").merge(s[["key", "score", "score_is_oof"]], on="key", how="inner")
    if not m.score_is_oof.all():
        raise SystemExit("an upward steering label carries an in-sample score; the scorer must flag every labelled row OOF")
    return m.rename(columns={"decision": "y"})[["key", "score", "y"]]


def downward_steering_labels(s: pd.DataFrame) -> pd.DataFrame:
    """The companion's reviewed detections joined to the pool, with their OOF score."""
    import score_study_pools as SSP

    d = SSP.downward_labels(set(s.key))   # columns: key (the tuple), decision, source
    m = d[["key", "decision"]].drop_duplicates("key").merge(s[["key", "score", "score_is_oof"]], on="key", how="inner")
    if not m.score_is_oof.all():
        raise SystemExit("a downward steering label carries an in-sample score")
    return m.rename(columns={"decision": "y"})[["key", "score", "y"]]


def planned_acceptance(s: pd.DataFrame, lab: pd.DataFrame, p_plan: float) -> pd.DataFrame:
    """Per design stratum: N, W (share of the WHOLE pool), floats, score range, planned p."""
    p_row = B.isotonic_acceptance(lab.score.to_numpy(float), lab.y.to_numpy(float), s.score.to_numpy(float))
    s = s.assign(_p=p_row)
    g = s.groupby("design_stratum", sort=True)
    T = pd.DataFrame({"N": g.size(), "floats": g.WMO.nunique(), "score_min": g.score.min(), "score_max": g.score.max(),
                      "p_raw": g._p.mean(), "max_float_share": g.apply(lambda d: d.groupby("WMO").size().max() / len(d))})
    T["W"] = T.N / len(s)
    scale = p_plan / float((T.W * T.p_raw).sum())
    T["p_planned"] = np.clip(T.p_raw * scale, 0.005, 0.995)
    T["region"] = [x.split("|")[0] for x in T.index]
    T["decile"] = [int(x.split("|d")[1]) for x in T.index]
    T["rescale_factor"] = scale
    return T.reset_index()


# --------------------------------------------------------------------------------------------- #
# the held strata (upward limb) and the target
# --------------------------------------------------------------------------------------------- #
def held_variance(pool_id: str, p_plan: float) -> tuple[float, list[dict]]:
    cov = pd.read_csv(COVERAGE)
    c = cov[(cov.pool_id == pool_id) & (cov.n_direct_candidates > 0)]
    rows, v = [], 0.0
    for r in c.itertuples():
        deff = r.direct_design_effect_within_stratum if np.isfinite(r.direct_design_effect_within_stratum) else 1.0
        vi = (r.W ** 2) * p_plan * (1 - p_plan) * deff / r.n_direct_candidates
        v += vi
        rows.append({"stratum": r.stratum, "N": int(r.N_candidates), "W": float(r.W), "n_direct": int(r.n_direct_candidates),
                     "accepted": int(r.n_direct_accepted), "design_effect_within": float(deff), "variance": float(vi),
                     "held_at": "the direct sample, phys_obduction_letter_b6, whole-Letter-pool frame"})
    return float(v), rows


def planning_base_rate() -> tuple[float, str]:
    f = pd.read_csv(DIRECT_FRAMES).set_index("legacy_frame").loc[RA.WHOLE_POOL_FRAME]
    return float(f.rate), f"the upward direct sample's own rate, {int(f.accepted)}/{int(f.n)} on the whole-Letter-pool frame"


def budget_reference(pool_id: str) -> float:
    b = pd.read_csv(BUDGET)
    return float(b[(b.pool_id == pool_id) & (b.design == DESIGN)].panels_remaining.sum())


# --------------------------------------------------------------------------------------------- #
# the rate arm
# --------------------------------------------------------------------------------------------- #
def rate_arm(study, pool, s: pd.DataFrame, lab: pd.DataFrame, p_plan: float, p_source: str, target: float,
             rng: np.random.Generator, controls: tuple[pd.DataFrame, pd.DataFrame, dict, dict],
             previously_judged: set, batch_id: str) -> tuple[B.Batch, dict]:
    T = planned_acceptance(s, lab, p_plan)
    open_T = T[T.region == "open"].reset_index(drop=True)
    v_target = (target * p_plan / B.Z) ** 2
    v_held, held_rows = held_variance(pool.pool_id, p_plan) if (T.region == "held").any() else (0.0, [])
    if v_held >= v_target:
        raise SystemExit("the held strata alone exceed the target variance")
    # The one-per-float rule holds within a stratum; across the ten strata a float can recur, and
    # verdicts on one float are correlated. The legacy sample's pooled design effect (1.87 at 4.7
    # panels per float) implies an intra-float correlation of ~0.235; the draw is solved with the
    # design effect its OWN panels-per-float implies, iterated until the two agree.
    deff, iterations = 1.0, []
    for it in range(6):
        n_total, n_h = B.solve_n(open_T.W.to_numpy(), open_T.p_planned.to_numpy(), (v_target - v_held) / deff, 0.0)
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
    pos_src, neg_src, pos_meta, neg_meta = controls
    drawn = set(science.key)
    pos = B.draw_controls(pos_src[~pos_src.key.isin(drawn)], N_CONTROLS, rng).assign(stratum=B.POS_CTRL)
    neg = B.draw_controls(neg_src[~neg_src.key.isin(drawn) & ~neg_src.key.isin(pos.key)], N_CONTROLS, rng).assign(stratum=B.NEG_CTRL)
    pos["src"], neg["src"] = pos_meta["src"], neg_meta["src"]
    ctrl = pd.concat([pos, neg], ignore_index=True)
    ctrl["previously_judged"] = True
    batch = B.Batch(batch_id=batch_id, science=science, controls=ctrl, event_type=pool.event_type, rng=rng)
    per_float = science.groupby("WMO").size()
    design = {
        "role": "analysis", "decides": True,
        "sampling": {"mode": "probability", "draw": "stratified_pps_one_per_float", "design": "probability",
                     "frame": f"the active pool {pool.pool_id} ({saved_rows(pool.event_type):,} candidate levels); the draw covers "
                              f"the OPEN region ({int(open_T.N.sum()):,} levels" +
                              (f"); the HELD region ({sum(h['N'] for h in held_rows):,} levels, {HELD_FRAME}) keeps its direct sample"
                               if held_rows else ")"),
                     "strata": "rank deciles of the classifier score over the whole pool × region",
                     "inclusion_probability": "n_h / N_h within each open stratum, exact (one panel per float per stratum by systematic PPS)",
                     "allocation": f"Neyman on planned acceptance with a {B.FLOOR_SHARE:.0%} floor per decile; n solved for the target"},
        "target": {"rel_half_width": target, "planning_base_rate": p_plan, "planning_base_rate_source": p_source,
                   "variance_target": v_target, "variance_held": v_held, "variance_open_planned": v_open,
                   "float_design_effect": float(deff), "float_design_effect_iterations": iterations,
                   "rho_float": RHO_FLOAT, "rho_float_source": RHO_SOURCE,
                   "variance_open_with_design_effect": float(v_open * deff),
                   "expected_rel_half_width": B.rel_half_width(v_open * deff + v_held, p_plan),
                   "expected_rel_half_width_if_unclustered": B.rel_half_width(v_open + v_held, p_plan),
                   "neyman_vs_proportional_multiplier": float(v_open / v_prop) if v_prop > 0 else None,
                   "stratified_vs_srs_multiplier": float(v_open / v_srs) if v_srs > 0 else None,
                   "budget_reference_panels": budget_reference(pool.pool_id), "budget_reference_design": DESIGN,
                   "budget_reference_multiplier": RA.SCORE_STRAT},
        "steering_labels": {"n": int(len(lab)), "accepted": int(lab.y.sum()), "what": pos_meta.get("steering", "")},
        "n_science": int(len(science)), "n_controls": {"positive": int(len(pos)), "negative": int(len(neg))},
        "floats_in_draw": int(science.WMO.nunique()), "floats_with_more_than_one_panel": int((per_float > 1).sum()),
        "max_panels_one_float": int(per_float.max()),
        "floats_chunked_for_pps": chunked,
        "panels_per_float_in_draw": float(len(science) / science.WMO.nunique()),
        "previously_judged_in_draw": int(science.previously_judged.sum()),
        "hours_at_planning_rate": float(len(batch.science) + len(ctrl)) * SECONDS_PER_PANEL / 3600,
        "strata": [{k: (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else v))
                    for k, v in r.items()} for r in open_T.drop(columns=["region", "decile", "rescale_factor"]).to_dict("records")],
        "rescale_factor_planned_acceptance": float(open_T.rescale_factor.iloc[0]),
        "held_strata": held_rows,
        "controls": {"positive": pos_meta, "negative": neg_meta},
    }
    return batch, design


# --------------------------------------------------------------------------------------------- #
# control sources
# --------------------------------------------------------------------------------------------- #
NEG_CONTROL_SCORE_CAP = 0.5   # a b6 reject the classifier calls near-certain is a likely b6 miss, not a control


def blind_rejudgement_history(decision_of_source: int) -> dict:
    """What a BLIND re-judgement of a direct b6 verdict returns, honestly counted: one pair per
    (candidate, re-judging sheet), the re-judging sheet a real review (no `_sessions` snapshot,
    no answer key), written after the accepting sheet, on the target arm; the source restricted
    to the direct uniform verdicts."""
    M = pd.read_parquet(REUSE_MAP)
    R = L.legacy_only(L.load_reviews())
    Bt = L.load_batches().set_index("batch_id")
    src = M[(M.reuse_status == "direct") & (M.decision == decision_of_source)][["review_id", "batch_id"]]
    src = src.merge(R[["review_id", "candidate_id"]], on="review_id")
    src["written"] = src.batch_id.map(Bt.first_written)
    later = R[(R.blind == True) & R.decision.isin([0, 1]) & ~R.role.isin(["snapshot", "answer_key"])
              & (R.control_arm.isna() | (R.control_arm == "target"))][["candidate_id", "decision", "batch_id", "sheet_sha256"]].copy()
    later["written"] = later.batch_id.map(Bt.first_written)
    j = src[["candidate_id", "batch_id", "written"]].merge(later, on="candidate_id", suffixes=("_src", ""))
    j = j[(j.batch_id != j.batch_id_src) & (j.written > j.written_src)].drop_duplicates(["candidate_id", "sheet_sha256"])
    return {"k": int(j.decision.sum()), "n": int(len(j)), "candidates": int(j.candidate_id.nunique()),
            "sheets": int(j.sheet_sha256.nunique()),
            "what": f"direct b6 {'accepts' if decision_of_source else 'rejects'} later re-judged BLIND in a later real sheet "
                    f"(target arm; one pair per candidate and sheet): re-accepted / re-judged"}


def upward_controls(s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    M = pd.read_parquet(REUSE_MAP)
    R = L.load_reviews()
    d = M[M.reuse_status == "direct"].merge(R[["review_id", "key_wmo", "key_cycle", "key_pres", "batch_id"]], on="review_id",
                                            suffixes=("", "_r"))
    d = d.assign(key=list(zip(d.key_wmo.astype(int), d.key_cycle.astype(int), d.key_pres.astype(int))))
    d = d[["key", "decision", "batch_id", "candidate_id"]].merge(s.drop(columns=["candidate_id"]), on="key", how="inner", validate="one_to_one")
    d["REF_LABEL"] = d.decision.astype(int)
    # A control carries the STANDING verdict: a direct verdict the ledger's own re-looks later
    # overturned (data/labels/audit/direct_flips.csv, 32 candidates) is not a control of anything —
    # four of the six positive controls the first batch "rejected" were such flips.
    flipped = set(pd.read_csv(DIRECT_FLIPS).candidate_id)
    n_flipped = int(d.candidate_id.isin(flipped).sum())
    d = d[~d.candidate_id.isin(flipped)]
    f = pd.read_csv(DIRECT_FRAMES).set_index("legacy_frame").loc[RA.WHOLE_POOL_FRAME]
    hist_pos, hist_neg = blind_rejudgement_history(1), blind_rejudgement_history(0)
    pos_meta = {"source": "the direct uniform verdicts (reuse_status direct), accepted, standing (not overturned by a re-look)", "criterion": B6,
                "n_source": int((d.REF_LABEL == 1).sum()), "flipped_excluded_from_source": n_flipped,
                "reference_k": hist_pos["k"], "reference_n": hist_pos["n"],
                "reference_provenance": hist_pos["what"], "reference_candidates": hist_pos["candidates"], "reference_sheets": hist_pos["sheets"],
                "relook_survival_k": int(f.accepted_after_relooks), "relook_survival_n": int(f.accepted),
                "power_note": "at n = 20 a Fisher test sees a collapse (<= 60 % re-acceptance), not a moderate drift; pool the arms across batches",
                "src": f"direct verdict under {B6}, accepted",
                "steering": "the 2,387 uniform b6 reviews with their out-of-fold score"}
    neg_src = d[(d.REF_LABEL == 0) & (d.score < NEG_CONTROL_SCORE_CAP)]
    neg_meta = {"source": f"the direct uniform verdicts (reuse_status direct), rejected, standing, classifier score < {NEG_CONTROL_SCORE_CAP}", "criterion": B6,
                "n_source": int(len(neg_src)), "score_cap": NEG_CONTROL_SCORE_CAP,
                "reference_k": hist_neg["k"], "reference_n": hist_neg["n"], "reference_provenance": hist_neg["what"],
                "ceiling": 0.20, "ceiling_note": "display only; the test is Fisher against the negatives' own blind history",
                "src": f"direct verdict under {B6}, rejected"}
    return d[d.REF_LABEL == 1], neg_src, pos_meta, neg_meta


def downward_controls(s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    cc = pd.read_csv(COMPANION_COV)
    cc["key"] = B.key3(cc)
    ver = cc[cc.reuse_status == "calibration_only"][["key", "Category"]]
    rej = cc[cc.reuse_status == "audit_only"][["key", "Category"]]
    pos = ver.merge(s, on="key", how="inner").assign(REF_LABEL=1)
    neg = rej.merge(s, on="key", how="inner").assign(REF_LABEL=0)
    pos_meta = {"source": "the companion's verified subduction events (Category 1/2) with a candidate in the pool", "criterion": COMPANION_CRIT,
                "n_source": int(len(pos)), "reference_k": None, "reference_n": None,
                "reference_provenance": f"judged under {COMPANION_CRIT} (clauses 1-3, cycle unit); the acceptance under clause 4 is "
                                        "unknown until calib_subduction_v1 is adjudicated — the reader REPORTS these, no Fisher test",
                "src": f"companion verified under {COMPANION_CRIT}",
                "steering": "the companion's reviewed detections joined to the pool (verified 1, rejected or detected-not-verified 0), "
                            "rescaled to the planning base rate"}
    neg_meta = {"source": "the companion's explicit rejections (Category 0) with a candidate in the pool", "criterion": COMPANION_CRIT,
                "n_source": int(len(neg)), "ceiling": 0.20, "src": f"companion rejected under {COMPANION_CRIT}"}
    return pos, neg, pos_meta, neg_meta


# --------------------------------------------------------------------------------------------- #
# the calibration batches
# --------------------------------------------------------------------------------------------- #
def calib_upward(s: pd.DataFrame, crit, rng: np.random.Generator) -> tuple[B.Batch, dict]:
    anchor = crit.raw["anchors"]["obduction"]
    b6 = load_criteria()[B6].raw["anchor"]
    ref = pd.read_csv(B6_REFERENCE)
    sha = B.sha256_of(B6_REFERENCE)
    if not sha.startswith(b6["sha256_16"]):
        raise SystemExit(f"the b6 reference on disk ({sha[:16]}) is not the one the criterion names ({b6['sha256_16']})")
    ref["key"] = B.key3(ref)
    m = ref[["key", "REF_LABEL", "tier"]].merge(s, on="key", how="left", validate="one_to_one")
    if m.candidate_id.isna().any():
        raise SystemExit("a b6 reference event is not in the active obduction pool")
    m["stratum"] = "calibration"
    m["src"] = "b6 reference, tier " + m.tier
    ctrl = m
    batch = B.Batch(batch_id="calib_obduction_b6", science=s.iloc[0:0].assign(design_stratum=pd.Series(dtype=str)),
                    controls=ctrl, event_type="physical_obduction", rng=rng, extra_key_cols=("tier",))
    design = {"role": "calibration", "decides": False,
              "sampling": {"mode": "calibration", "draw": "the frozen reference, every event", "design": None, "frame": None,
                           "inclusion_probability": None},
              "reference": {"what": b6["what"], "file": str(B6_REFERENCE.relative_to(REPO)), "sha256": sha, "n_events": int(len(ref)),
                            "base_rate": b6["base_rate"], "carried_over_as": anchor},
              "gate": f"κ > {B.KAPPA_PASS} against REF_LABEL and the base rate on target, before every session (production/LABELING_PROTOCOL.md)",
              "panel_note": "the study panel shows the two physical channels (phys_net_carbon_v1.channels_shown); the reference was "
                            "judged on the Letter's four-channel panel — the first re-labelling measures the criterion across instruments",
              "n_science": 0, "n_controls": {"calibration": int(len(ref))},
              "hours_at_planning_rate": float(len(ref)) * SECONDS_PER_PANEL / 3600,
              "reference_rows": ref[B.KEYS + ["REF_LABEL", "tier"]].to_dict("records")}
    return batch, design


def calib_downward(s: pd.DataFrame, rng: np.random.Generator) -> tuple[B.Batch, dict]:
    cc = pd.read_csv(COMPANION_COV)
    cc["key"] = B.key3(cc)
    st = cc.set_index("key").reuse_status
    d = s.assign(companion=s.key.map(st).fillna("none").replace({"calibration_only": "verified", "audit_only": "rejected"}))
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
    m["src"] = "candidate for the downward anchor, tier " + m.tier + ", companion " + m.companion
    m["REF_LABEL"] = np.nan
    batch = B.Batch(batch_id="calib_subduction_v1", science=s.iloc[0:0].assign(design_stratum=pd.Series(dtype=str)),
                    controls=m, event_type="physical_subduction", rng=rng, extra_key_cols=("tier", "companion"))
    design = {"role": "calibration", "decides": False,
              "sampling": {"mode": "calibration", "draw": "stratified by score decile and companion status, one per float", "design": None,
                           "frame": None, "inclusion_probability": None},
              "composition": {"clear_TP": "12 companion-verified events in the top score decile",
                              "clear_FP": f"{len(rej_low)} companion-rejected in deciles 0-2 + {len(unl_low)} unlabelled in decile 0",
                              "borderline": f"18 in deciles 5-8: 9 companion-verified, {len(bord_r)} companion-rejected, {len(bord_u)} unlabelled"},
              "status": "REF_LABEL is BLANK: to be filled ONCE by consensus adjudication under phys_net_carbon_v1 (who adjudicates: open "
                        "ruling, docs/PLAN.md), then frozen as phys_net_carbon_v1/anchors/subduction",
              "n_science": 0, "n_controls": {"calibration": int(len(m))},
              "hours_at_planning_rate": float(len(m)) * SECONDS_PER_PANEL / 3600}
    return batch, design


# --------------------------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------------------------- #
def record_for(study, pool, crit, batch: B.Batch, design: dict, sheets: dict, seed: int, sm: dict) -> dict:
    return {
        "batch_id": batch.batch_id, "built": B.stamp(), "builder": "production/build_batches.py", "seed": seed,
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


def write_page(records: list[dict], target: float) -> None:
    lines = ["# The study's labelling batches — built " + B.stamp()[:10], "",
             "Built by `production/build_batches.py`; the design is `eddy_pump.batches`. Worksheets, keys and panels are under "
             "`results/net_carbon_v1/labeling/<batch_id>/` (not in Git); the draw records beside this page are the ledger's "
             "input when a sheet comes back labelled.", "",
             "| batch | role | rows | of which science / controls | hours | expected precision | gate before labelling |",
             "|---|---|---:|---|---:|---|---|"]
    for r in records:
        nc = r["n_controls"]
        ctrl = " / ".join(f"{k} {v}" for k, v in nc.items())
        prec = (f"±{r['target']['expected_rel_half_width']:.1%} relative on the pool rate (target ±{target:.0%})"
                if "target" in r else "—")
        gate = ("calib_obduction_b6 PASS the same day" if r["batch_id"] == "rate_obduction_01" else
                "calib_subduction_v1 adjudicated and frozen, then re-labelled blind: PASS" if r["batch_id"] == "rate_subduction_01" else
                "—")
        lines.append(f"| `{r['batch_id']}` | {r['role']} | {r['worksheet']['rows']} | {r['n_science']} / {ctrl} | "
                     f"{r['hours_at_planning_rate']:.1f} | {prec} | {gate} |")
    lines += ["", "## How to label one", "",
              "```", "make review BATCH=results/net_carbon_v1/labeling/<batch_id>/<batch_id>.csv   # the keyboard app, blind",
              "python production/build_batches.py --report results/net_carbon_v1/labeling/calib_obduction_b6/calib_obduction_b6.csv",
              "```", "",
              "The worksheet carries the key, the position and the coordinates — nothing else. The answer key beside it "
              "(`ANSWER_KEY_do_not_open.csv`) is opened by `argopod session` only once the sheet is finished.", ""]
    for r in records:
        if "target" not in r:
            continue
        t = r["target"]
        lines += [f"## `{r['batch_id']}` — the allocation", "",
                  f"Planning base rate {t['planning_base_rate']:.4f} ({t['planning_base_rate_source']}); target variance "
                  f"{t['variance_target']:.3g}; held strata contribute {t['variance_held']:.3g}, the open draw {t['variance_open_planned']:.3g}. "
                  f"Neyman against proportional at the same n: ×{t['neyman_vs_proportional_multiplier']:.2f}. The coverage report's budget "
                  f"for this design was {t['budget_reference_panels']:.0f} panels.", "",
                  "| stratum | N | floats | score range | planned p | n | π = n/N |", "|---|---:|---:|---|---:|---:|---:|"]
        for s_ in r["strata"]:
            lines.append(f"| {s_['design_stratum']} | {s_['N']:,} | {s_['floats']:,} | {s_['score_min']:.4f}–{s_['score_max']:.4f} | "
                         f"{s_['p_planned']:.3f} | {s_['n']} | {s_['inclusion_probability']:.5f} |")
        lines.append("")
        if r["held_strata"]:
            lines += ["Held at their direct sample:", "", "| stratum | N | n direct | accepted | deff |", "|---|---:|---:|---:|---:|"]
            lines += [f"| {h['stratum']} | {h['N']:,} | {h['n_direct']} | {h['accepted']} | {h['design_effect_within']:.2f} |" for h in r["held_strata"]]
            lines.append("")
    (DRAWS / "BATCHES.md").write_text("\n".join(lines), encoding="utf-8")


def repass(src_id: str) -> None:
    """A blind re-labelling copy of a calibration batch: the same 42 levels in a NEW random order with
    new SAMPLE_IDs, LABEL blank, panels rendered afresh; its draw record points at the original
    (`reference_of`) so the ingestion scores it against the frozen reference. The original labelled
    sheet is never opened for writing (a labelled sheet is never opened for writing)."""
    import shutil

    import yaml

    study = load_manifest()
    src = yaml.safe_load((DRAWS / f"{src_id}.yaml").read_text())
    if src["role"] != "calibration":
        raise SystemExit(f"{src_id} is not a calibration batch; a rate batch is never re-labelled, it is re-drawn")
    n = 2
    while (DRAWS / f"{src_id}_pass{n}.yaml").exists():
        n += 1
    bid = f"{src_id}_pass{n}"
    key = pd.read_csv(src["answer_key"]["path"])
    rng = np.random.default_rng([hash(bid) % (2 ** 31), n])
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
                "panels": {"dir": str(bdir / "panels"), "rendered": n_png, "of": int(len(ws))}, **sheets})
    (DRAWS / f"{bid}.yaml").write_text(
        f"# data/labels/draws/{bid}.yaml -- a blind re-labelling copy of {src_id} (pass {n}). BUILT by production/build_batches.py\n"
        f"# --repass; never edited by hand.\n" + yaml.safe_dump(rec, sort_keys=False, allow_unicode=True, width=110, default_flow_style=False),
        encoding="utf-8")
    print(f"{bid}: {len(ws)} rows, {n_png} panels -> {bdir / (bid + '.csv')}")


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--target", type=float, default=RA.TARGET_REL)
    ap.add_argument("--render", action="store_true", help="render the panel PNGs from the bound cache")
    ap.add_argument("--dry-run", action="store_true", help="print the allocation, write nothing")
    ap.add_argument("--report", help="a re-labelled calibration sheet: print the calibration gate and exit")
    ap.add_argument("--repass", metavar="BATCH_ID", help="a fresh BLIND copy of a calibration batch (rows shuffled, new SAMPLE_IDs, "
                                                       "panels re-rendered) as <BATCH_ID>_pass<N>, for a re-labelling; the labelled sheet is never touched")
    args = ap.parse_args()

    if args.repass:
        repass(args.repass)
        return

    if args.report:
        sheet = pd.read_csv(args.report)
        key = pd.read_csv(pathlib.Path(args.report).parent / "ANSWER_KEY_do_not_open.csv")
        if key.REF_LABEL.isna().all():
            raise SystemExit("this calibration set has no REF_LABEL yet — it must be adjudicated first")
        rep = B.calibration_report(sheet, key[key.REF_LABEL.notna()].assign(REF_LABEL=lambda d: d.REF_LABEL.astype(int)))
        print(json.dumps(rep, indent=1, default=str))
        return

    study = load_manifest()
    crit = require_ruled(active_criterion())
    sm = scores_manifest()
    if sm["cache_sha256"] != study.cache.fine_grids_sha256:
        raise SystemExit("the scores were made on another cache than the one the study is bound to")
    if args.render:
        C.require_bound_cache(study)
    frames, _ = RA.load_pools()
    pools = {p.event_type: p for p in study.pools if p.tracer is None}
    obd, sub = pools["physical_obduction"], pools["physical_subduction"]
    S_obd, S_sub = load_pool_frame(study, obd, frames), load_pool_frame(study, sub, frames)
    p_plan, p_source = planning_base_rate()
    judged = L.labelled_keys(include_study=True)   # a covariate, never a filter: the study's own labelled keys count too
    judged = {(int(w), int(c), int(p)) for w, c, p in judged}
    cc = pd.read_csv(COMPANION_COV)
    judged |= set(B.key3(cc))

    out_root = study.output.resolve("labeling")
    records = []
    jobs = [
        ("calib_obduction_b6", obd, lambda rng: calib_upward(S_obd, crit, rng)),
        ("calib_subduction_v1", sub, lambda rng: calib_downward(S_sub, rng)),
        ("rate_obduction_01", obd, lambda rng: rate_arm(study, obd, S_obd, upward_steering_labels(S_obd), p_plan, p_source, args.target,
                                                       rng, upward_controls(S_obd), judged, "rate_obduction_01")),
        ("rate_subduction_01", sub, lambda rng: rate_arm(study, sub, S_sub, downward_steering_labels(S_sub), p_plan,
                                                        p_source + " (the downward pool has no direct sample of its own)", args.target,
                                                        rng, downward_controls(S_sub), judged, "rate_subduction_01")),
    ]
    for i, (bid, pool, make) in enumerate(jobs):
        rng = np.random.default_rng([args.seed, i])
        batch, design = make(rng)
        ws, key = batch.assemble()
        print(f"{bid}: {len(ws)} rows ({design['n_science']} science, {design['n_controls']}), "
              f"{design['hours_at_planning_rate']:.1f} h"
              + (f"; expected ±{design['target']['expected_rel_half_width']:.1%} (deff {design['target']['float_design_effect']:.2f}), "
                 f"budget said {design['target']['budget_reference_panels']:.0f} at ×{RA.SCORE_STRAT}; "
                 f"vs SRS ×{design['target']['stratified_vs_srs_multiplier']:.2f}, vs proportional ×{design['target']['neyman_vs_proportional_multiplier']:.2f}; "
                 f"{design['floats_in_draw']} floats, {design['panels_per_float_in_draw']:.2f} per float, "
                 f"{design['floats_chunked_for_pps']} float-strata chunked"
                 if "target" in design else ""))
        if "strata" in design:
            for s_ in design["strata"]:
                print(f"   {s_['design_stratum']:>9}  N {s_['N']:>7,}  p {s_['p_planned']:.3f}  n {s_['n']:>3}  π {s_['inclusion_probability']:.5f}")
        if args.dry_run:
            continue
        bdir = out_root / bid
        sheets = B.write_sheets(bdir, ws, key, bid)
        rec = record_for(study, pool, crit, batch, design, sheets, args.seed, sm)
        if args.render:
            n = render(study, pool, bdir, ws)
            rec["panels"] = {"dir": str(bdir / "panels"), "rendered": n, "of": int(len(ws))}
            print(f"   panels: {n} of {len(ws)}")
        elif (bdir / "panels").exists():
            have = {p.name.split("_wmo")[0] for p in (bdir / "panels").rglob("*.png")}
            n = int(sum(str(sid) in have for sid in ws.SAMPLE_ID))
            rec["panels"] = {"dir": str(bdir / "panels"), "rendered": n, "of": int(len(ws)),
                             "note": "counted from disk; re-run with --render if fewer than the rows"}
        DRAWS.mkdir(parents=True, exist_ok=True)
        (DRAWS / f"{bid}.yaml").write_text(
            f"# data/labels/draws/{bid}.yaml -- the draw record of one study batch. BUILT by production/build_batches.py;\n"
            f"# never edited by hand. The sheet and key it hashes live under results/ (not in Git).\n"
            + yaml.safe_dump(rec, sort_keys=False, allow_unicode=True, width=110, default_flow_style=False), encoding="utf-8")
        records.append(rec)
    if args.dry_run:
        return
    write_page(records, args.target)
    (DRAWS / "DRAWS_SHA256").write_text(
        "# provenance of the study's draw records -- do not edit by hand\n"
        f"# built {B.stamp()} by production/build_batches.py, seed {args.seed}\n"
        + "".join(f"{B.sha256_of(DRAWS / (r['batch_id'] + '.yaml'))}\t{r['batch_id']}.yaml\n" for r in records)
        + f"{B.sha256_of(DRAWS / 'BATCHES.md')}\tBATCHES.md\n")
    print(f"-> {DRAWS}")


if __name__ == "__main__":
    main()
