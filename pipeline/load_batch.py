#!/usr/bin/env python
"""Load a labelled study sheet into the label table's study layer.

usage  load_batch.py BATCH_ID [--allow-unfinished] [--replace] [--freeze-reference HOW] [--accept-drift]
                     [--accept-calibration-drift [--count-for-correction]]
       (`make load BATCH=<batch id>`, `make load BATCH=<batch id> FLAGS="--replace"`)
reads  data/labels/draws/<BATCH_ID>.yaml (the draw record), the worksheet(s) and the sealed key(s)
       under results/net_carbon_v1/labeling/<BATCH_ID>/
writes results/net_carbon_v1/labeling/<BATCH_ID>/<sheet>_LABELLED_<stamp>.csv (the frozen sheet, not in git)
       data/labels/{study_reviews.parquet, study_batches.yaml}, data/labels/draws/<BATCH_ID>.labelled.yaml, LABELLED_SHA256
In order, none skipped: check each key's hash and each sheet's columns; read the session (progress,
position, controls, science; the κ check for a calibration batch); check where the controls sat and
whether acceptance fell across the sitting; freeze the sheet; append the rows.

A draw longer than the 120-panel session cap is written as several sheets of one draw
(`<BATCH_ID>_s1`, `_s2`, …). They are read together here: same rows, same order, same inclusion
probabilities, so the number is the one the single sheet would have given. Position is read sheet by
sheet, because drift is drift within one sitting.

Two things stop a rate sheet or a second look here. The LAST scored copy of its pool's 42
calibration panels labelled before the sheet must have said PASS — the copy labelled before a
session is what decides whether that session counts, so an earlier copy that passed is no defence
(a refusal, unless --accept-calibration-drift; on an earlier day is a warning). And acceptance must
not fall from the first half of a sitting to the second with the stratum held fixed (p < 0.01): if
it does, the batch is loaded only once a blind re-judgement of it has been drawn, or with
--accept-drift, which the record then carries. A second look loaded over a calibration copy that
did not pass keeps its rows but corrects no rate (`counts_for_correction: false`).

A draw record written in another checkout carries that checkout's absolute paths; the sheet and the
key are found through `batches.resolve_recorded_path`, which keeps the part from `results/` onward.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eddy_pump import batches as B  # noqa: E402
from eddy_pump import labels as L  # noqa: E402
from eddy_pump.criteria import load_criteria, require_ruled  # noqa: E402

DRAWS = REPO / "data/labels/draws"
STUDY_BATCHES, STUDY_REVIEWS = L.STUDY_BATCHES, L.STUDY_REVIEWS
BASE_REVIEW_COLS = ["review_id", "row_index", "candidate_id", "pool_id", "criterion_version", "role", "decision",
                    "supersedes_review_id", "key_wmo", "key_cycle", "key_pres", "sheet_sha256", "event", "WMO",
                    "CYCLE_NUMBER", "PRES_ADJUSTED", "LABEL", "batch_id", "rank", "tier", "sampling_mode", "stratum",
                    "control_arm", "blind", "SAMPLE_ID", "src", "score"]
STUDY_EXTRA_COLS = ["inclusion_probability", "design_stratum", "study_id", "spec_id", "REF_LABEL", "previously_judged"]


def _rid(batch_id: str, sheet_sha: str, row: int) -> str:
    return hashlib.sha256(f"{batch_id}|{sheet_sha}|{row}".encode("utf-8")).hexdigest()[:16]


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def reference_for(rec: dict, key: pd.DataFrame):
    """The frozen answers of the 42 calibration panels: the key's own REF_LABEL (the first upward
    set, carried over as data) or, when the key was built blank,
    `data/labels/draws/<batch>.reference.yaml` written by `--freeze-reference`.
    Returns (frame with KEYS + REF_LABEL, provenance) or (None, {})."""
    if key.REF_LABEL.notna().any():
        ref = key[key.REF_LABEL.notna()].assign(REF_LABEL=lambda d: d.REF_LABEL.astype(int))
        return ref, {"what": rec.get("reference", {}).get("what", "the key's own REF_LABEL"), "n": int(len(ref))}
    rp = DRAWS / f"{rec.get('reference_of', rec['batch_id'])}.reference.yaml"
    if not rp.exists():
        return None, {}
    fr = yaml.safe_load(rp.read_text())
    ref = pd.DataFrame(fr["rows"])
    meta = {k: v for k, v in fr.items() if k != "rows"}
    meta["n"] = int(len(ref))
    return ref[B.KEYS + ["REF_LABEL"] + (["tier"] if "tier" in ref.columns else [])], meta


def freeze_reference(rec: dict, wp: pathlib.Path, key: pd.DataFrame, ws: pd.DataFrame, how: str) -> pathlib.Path:
    """Write the calibration set's frozen answers from a labelled pass, once. Refuses if they exist
    or if any row is undecided: a calibration set is 42 decided levels. `how` names the pass (a
    single pass, or the agreement of two)."""
    rp = DRAWS / f"{rec.get('reference_of', rec['batch_id'])}.reference.yaml"
    if rp.exists():
        raise SystemExit(f"{rp} exists — a calibration set's answers are frozen once; re-freezing them is a new "
                         f"file written by hand, not this flag")
    m = key.merge(ws[["SAMPLE_ID", "LABEL"]], on="SAMPLE_ID", validate="one_to_one")
    m["LABEL"] = pd.to_numeric(m.LABEL, errors="coerce")
    if not m.LABEL.isin([0, 1]).all():
        raise SystemExit(f"{int((~m.LABEL.isin([0, 1])).sum())} rows undecided or uncertain — every calibration "
                         f"panel needs a 0/1")
    cols = B.KEYS + [c for c in ("tier", "companion", "score", "candidate_id") if c in m.columns]
    rows = m[cols].assign(REF_LABEL=m.LABEL.astype(int)).to_dict("records")
    fr = {"batch_id": rec["batch_id"], "pool_id": rec["pool_id"], "criterion_version": rec["criterion_version"],
          "frozen": B.stamp(), "how": how, "source_sheet_sha256": _sha(wp), "n": int(len(rows)),
          "base_rate": f"{int(m.LABEL.sum())}/{len(m)}", "rows": rows}
    rp.write_text(f"# data/labels/draws/{rec['batch_id']}.reference.yaml -- the 42 calibration panels' frozen answers.\n"
                  f"# Written once by pipeline/load_batch.py --freeze-reference; never edited by hand.\n"
                  + yaml.safe_dump(fr, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    return rp


# --------------------------------------------------------------------------------------------- #
# one draw, one or more sittings
# --------------------------------------------------------------------------------------------- #
def sheet_parts(rec: dict) -> list[dict]:
    """The sheets of one draw: one for a batch written whole, one per sitting for a split batch."""
    if rec.get("sessions"):
        return [{"sheet_id": x["sheet_id"], "worksheet": x["worksheet"], "answer_key": x["answer_key"]}
                for x in rec["sessions"]]
    return [{"sheet_id": rec["batch_id"], "worksheet": rec["worksheet"], "answer_key": rec["answer_key"]}]


def read_sheets(rec: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Every sheet of a draw, read and checked, then joined in review order.

    A draw longer than the session cap is written as several sheets of the same draw; they hold the
    same rows in the same order, so joining them gives back exactly the sheet a single sitting would
    have been. Each sheet's sealed key is checked against the hash the draw record names.
    """
    parts, wss, keys = sheet_parts(rec), [], []
    meta = []
    for x in parts:
        wp = B.resolve_recorded_path(x["worksheet"]["path"])
        kp = B.resolve_recorded_path(x["answer_key"]["path"])
        if _sha(kp) != x["answer_key"]["sha256"]:
            raise SystemExit(f"{kp} does not hash to the sealed key the draw record names — refusing")
        ws, key = pd.read_csv(wp), pd.read_csv(kp)
        if list(ws.columns) != B.WORKSHEET_COLS:
            raise SystemExit(f"{wp}: columns {list(ws.columns)} are not the blind worksheet's")
        ws, key = ws.sort_values("SAMPLE_ID").reset_index(drop=True), key.sort_values("SAMPLE_ID").reset_index(drop=True)
        if not (ws[B.KEYS].round(0).to_numpy() == key[B.KEYS].round(0).to_numpy()).all():
            raise SystemExit(f"{wp}: worksheet and key disagree on a row — refusing")
        ws["_sheet"], key["_sheet"] = x["sheet_id"], x["sheet_id"]
        wss.append(ws)
        keys.append(key)
        meta.append({"sheet_id": x["sheet_id"], "path": wp, "key_path": kp, "sha256": _sha(wp), "rows": int(len(ws)),
                     "mtime": _dt.datetime.fromtimestamp(wp.stat().st_mtime).astimezone().isoformat(timespec="seconds")})
    ws = pd.concat(wss, ignore_index=True).sort_values("SAMPLE_ID").reset_index(drop=True)
    key = pd.concat(keys, ignore_index=True).sort_values("SAMPLE_ID").reset_index(drop=True)
    if not ws.SAMPLE_ID.is_unique:
        raise SystemExit(f"{rec['batch_id']}: a position appears in two sheets of the same draw — refusing")
    return ws, key, meta


def joint_sha(parts: list[dict], field: str = "sha256") -> str:
    """One hash for a draw's sheets: the sha256 of their own hashes, in order. A batch written as a
    single sheet keeps that sheet's hash, so nothing already loaded moves."""
    if len(parts) == 1:
        return parts[0][field]
    return hashlib.sha256("".join(f"{x[field]}\t{x['sheet_id']}\n" for x in parts).encode()).hexdigest()


def add_position_columns(m: pd.DataFrame) -> pd.DataFrame:
    """Where each row sat in its own sitting: the half of the sheet it was labelled in.

    The half is cut at the median position of that sheet's decided science rows — the drift the
    2026-09-04 review found is drift within one sitting, so a draw written as several sheets is read
    sheet by sheet, never across the join.
    """
    m = m.copy()
    if "_sheet" not in m.columns:
        m["_sheet"] = m.get("batch_id", "sheet")
    m["_half"] = None
    m["_half_cut_at"] = np.nan
    for sheet, g in m.groupby("_sheet"):
        t = g[(g.stratum == B.TARGET) & g.LABEL.isin([0, 1])]
        if not len(t):
            continue
        med = float(t.SAMPLE_ID.median())
        m.loc[g.index, "_half"] = np.where(g.SAMPLE_ID <= med, "first", "second")
        m.loc[g.index, "_half_cut_at"] = med
    return m


def session_block(rec: dict, parts: list[dict], ws: pd.DataFrame, key: pd.DataFrame,
                  calibration: dict | None = None) -> dict:
    """The protocol's read of the session, as data."""
    wp, kp = parts[0]["path"], parts[0]["key_path"]
    if rec["role"] == "calibration":
        ref, ref_meta = reference_for(rec, key)
        if ref is None:
            return {"kind": "calibration", "check": "no REF_LABEL yet — a calibration set whose answers are not "
                                                     "decided; loaded as evidence only"}
        rep = B.calibration_report(ws, ref)
        rep["kind"] = "calibration"
        rep["reference"] = ref_meta
        if ref_meta.get("source_sheet_sha256") == _sha(wp):
            rep["reference_is_this_pass"] = True
            rep["verdict_means"] = ("the reference is this sheet's own labels, so κ = 1 by construction; the first "
                                    "real check is the blind re-labelling before the next session")
        return rep
    from argopod.review.session import read_session

    ctrl = rec["controls"]["positive"]
    ref = (ctrl["reference_k"], ctrl["reference_n"], ctrl["reference_provenance"]) if ctrl.get("reference_k") is not None else None
    reads = [read_session(x["path"], x["key_path"], pos_ctrl_reference=ref, breakdown_cols=("src",)) for x in parts]
    rep = reads[0]
    out = {"kind": "rejudgement" if rec["role"] == "rejudgement" else "rate",
           "n_rows": sum(r.n_rows for r in reads), "n_decided": sum(r.n_decided for r in reads),
           "n_accepted": sum(r.n_accepted for r in reads), "n_rejected": sum(r.n_rejected for r in reads),
           "n_uncertain": sum(r.n_uncertain for r in reads), "n_blank": sum(r.n_blank for r in reads),
           "position": [{"bin": p.index, "sid_lo": p.sid_lo, "sid_hi": p.sid_hi, "n": p.n, "accept": p.accept} for p in rep.position],
           "trend_p_mann_whitney": rep.trend_p,
           "change_point": ({"sample_id": rep.change_point.sample_id, "accept_before": rep.change_point.accept_before,
                             "accept_after": rep.change_point.accept_after, "p_raw": rep.change_point.p_raw,
                             "p_adjusted": rep.change_point.p_adjusted, "verdict": rep.change_point.verdict}
                            if rep.change_point else None),
           "key_opened": all(r.key_opened for r in reads), "warnings": sorted({w for r in reads for w in r.warnings})}
    if len(parts) > 1:
        out["sheets"] = [{"sheet_id": x["sheet_id"], "rows": r.n_rows, "decided": r.n_decided, "accepted": r.n_accepted,
                          "trend_p_mann_whitney": r.trend_p,
                          "position": [{"bin": q.index, "n": q.n, "accept": q.accept} for q in r.position]}
                         for x, r in zip(parts, reads)]
        out["position_note"] = ("`position`, `trend_p_mann_whitney` and `change_point` above are the first sheet's; "
                                "each sitting is read on its own under `sheets`")
    m = key.merge(ws[["SAMPLE_ID", "LABEL"]], on="SAMPLE_ID")
    m["LABEL"] = pd.to_numeric(m.LABEL, errors="coerce")
    m = add_position_columns(m)
    from scipy.stats import beta as _beta, fisher_exact

    def ci(k, n):
        return [float(_beta.ppf(0.025, k + 0.5, n - k + 0.5)), float(_beta.ppf(0.975, k + 0.5, n - k + 0.5))] if n else None

    # A control is read against its standing verdict as the label table records it. There is no
    # separate re-look layer that could overturn one, so there is nothing to subtract.
    ctr = {}
    for arm in B.CONTROL_STRATA:
        c = m[(m.stratum == arm) & m.LABEL.isin([0, 1])]
        k = int((c.LABEL == 1).sum())
        ctr[arm] = {"n": int(len(c)), "accepted": k, "ci95": ci(k, len(c))}
    pos = ctr[B.POS_CTRL]
    if ref is not None and pos["n"]:
        ctr[B.POS_CTRL]["reference"] = {"k": ref[0], "n": ref[1], "what": ref[2]}
        ctr[B.POS_CTRL]["fisher_p_vs_reference"] = float(fisher_exact([[pos["accepted"], pos["n"] - pos["accepted"]], [ref[0], ref[1] - ref[0]]])[1])
    ctr[B.NEG_CTRL]["ceiling"] = rec["controls"]["negative"].get("ceiling", 0.20)
    ctr[B.NEG_CTRL]["ceiling_note"] = "display only; at equality nothing fires — the test is Fisher against the negatives' own blind history"
    hi_neg = m[(m.stratum == B.NEG_CTRL) & (m.score >= 0.5)]
    ctr[B.NEG_CTRL]["with_score_above_0p5"] = {"n": int(len(hi_neg)), "accepted": int((hi_neg.LABEL == 1).sum()),
                                               "note": "a rejected candidate the classifier calls likely is a plausible detector miss, not a control"}
    out["controls"] = ctr
    # where the controls sat: an arm bunched into one part of the sheet cannot see a reviewer who
    # changes within the sitting, which is how 17 of the 20 negative controls of the first upward
    # sheet ended up in its second half
    pos_check = {}
    for sheet, g in m.groupby("_sheet"):
        c = g[g.stratum.isin(B.CONTROL_STRATA)]
        if len(c):
            # positions inside this sheet: a later sitting of a split draw keeps the draw's own
            # numbering, and the check reads where a control sat in the sitting it was labelled in
            pos_check[str(sheet)] = B.control_position_check((c.SAMPLE_ID - g.SAMPLE_ID.min()).to_numpy(),
                                                             c.stratum.to_numpy(), int(len(g)))
    out["control_positions"] = {"per_sheet": pos_check, "placed_by_the_draw": rec.get("control_placement"),
                                "verdict": ("FAIL" if any(v["verdict"] == "FAIL" for v in pos_check.values()) else "PASS"),
                                "alpha": B.CONTROL_POSITION_ALPHA,
                                "what_a_fail_means": "an arm is bunched into part of the sheet; recorded, never a "
                                                     "reason to drop a label"}

    t = m[(m.stratum == B.TARGET) & m.LABEL.isin([0, 1])]
    out["target_by_stratum"] = {s: {"n": int(len(g)), "accepted": int(g.LABEL.sum())} for s, g in t.groupby("src")}
    out["target_raw_acceptance_unweighted"] = float(t.LABEL.mean()) if len(t) else None
    out["target_previously_judged"] = {"n": int(t.previously_judged.fillna(False).astype(bool).sum()),
                                       "accepted": int(t[t.previously_judged.fillna(False).astype(bool)].LABEL.sum())}
    # position drift on the TARGET rows alone (the reader's step 2 runs on every decided row)
    q = pd.qcut(t.SAMPLE_ID.rank(method="first"), 4, labels=False)
    out["target_position_quarters"] = [{"quarter": int(i) + 1, "n": int(len(g)), "accept": float(g.LABEL.mean())} for i, g in t.groupby(q)]
    second = (t._half == "second").to_numpy()
    out["target_halves"] = {"first": {"n": int((~second).sum()), "accept": float(t.LABEL[~second].mean())},
                            "second": {"n": int(second.sum()), "accept": float(t.LABEL[second].mean())},
                            "cut_at": {str(k): float(v) for k, v in t.groupby("_sheet")._half_cut_at.first().items()},
                            "by_stratum": {s: {"first": (float(g.LABEL[g._half == "first"].mean()) if (g._half == "first").any() else None),
                                               "second": (float(g.LABEL[g._half == "second"].mean()) if (g._half == "second").any() else None)}
                                           for s, g in t.groupby("src")}}
    # the same rank test the reader runs, but on the science rows alone: the reader's runs on every
    # decided row, controls included, and the controls are not a sample of the pool
    from scipy.stats import mannwhitneyu

    acc, rej = t.SAMPLE_ID[t.LABEL == 1], t.SAMPLE_ID[t.LABEL == 0]
    out["trend_p_mann_whitney_science_rows"] = (
        float(mannwhitneyu(acc, rej, alternative="two-sided").pvalue) if len(acc) >= 3 and len(rej) >= 3 else None)
    # the test that decides whether this batch may be loaded: acceptance in the first half of a
    # sitting against the second, holding the stratum fixed, on science rows only. A second half
    # heavier in low-score panels would show a drop with no drift at all; this cannot.
    cmh = B.cochran_mantel_haenszel(t.LABEL.to_numpy(), second, t.design_stratum.to_numpy())
    cmh["alpha"] = B.DRIFT_ALPHA
    cmh["verdict"] = "DRIFTED" if (cmh.get("p") is not None and cmh["p"] < B.DRIFT_ALPHA) else "FLAT"
    out["within_stratum_position_trend"] = cmh
    if calibration is not None:
        out["calibration"] = calibration
    if rec["role"] == "rejudgement":
        out["rejudgement"] = rejudgement_block(rec, m)
    return out


def rejudgement_block(rec: dict, m: pd.DataFrame) -> dict:
    """What the blind second look did to the first: the flip table, per half of the source sheet.

    Only rows the source sheet decided and this sheet decided count. The cells (first verdict × half
    of the source sheet) are what corrects the rate; the flip table and the Fisher tests are what a
    reader looks at.
    """
    t = m[(m.stratum == B.TARGET) & m.LABEL.isin([0, 1])].copy()
    t["original_label"] = pd.to_numeric(t.original_label, errors="coerce")
    t = t[t.original_label.isin([0, 1])]
    ft = B.flip_table(t.original_label.to_numpy(int), t.LABEL.to_numpy(int), t.original_half.to_numpy())
    ft["rejudges"] = rec["rejudges"]
    ft["n_rejudged_and_decided"] = int(len(t))
    ft["n_not_decided_this_time"] = int(((m.stratum == B.TARGET) & ~m.LABEL.isin([0, 1])).sum())
    ft["source"] = {k: rec["source"][k] for k in ("batch_id", "labelled_sheet_sha256", "decided_science_rows",
                                                  "half_cut_at_sample_id") if k in rec.get("source", {})}
    ft["never_a_rate"] = ("a re-judgement decides nothing on its own; it corrects the rate of the batch it "
                          "re-judges (pipeline/rates.py)")
    return ft


def rejudgements_of(batch_id: str) -> list[dict]:
    """Every blind re-judgement drawn for a batch, and whether it has been labelled and loaded."""
    import yaml

    out = []
    for p in sorted(DRAWS.glob("*.yaml")):
        if p.name.endswith((".labelled.yaml", ".reference.yaml")):
            continue
        r = yaml.safe_load(p.read_text())
        if r.get("role") == "rejudgement" and r.get("rejudges") == batch_id:
            out.append({"batch_id": r["batch_id"], "drawn": r["built"],
                        "loaded": (DRAWS / f"{r['batch_id']}.labelled.yaml").exists()})
    return out


def calibration_copies(pool_id: str) -> list[dict]:
    """Every scored blind copy of that pool's 42 calibration panels, oldest first.

    Two kinds of copy are left out because neither is a check. One whose set had no decided answers
    when it was loaded carries no verdict at all. And the pass whose own labels became the frozen
    answers scores κ = 1 by construction, which says nothing about whether the reading held; the
    first real check of such a set is the next blind copy.
    """
    import yaml

    out = []
    for p in sorted(DRAWS.glob("calib_*.labelled.yaml")):
        l = yaml.safe_load(p.read_text())
        rp = DRAWS / f"{l['batch_id']}.yaml"
        if not rp.exists() or yaml.safe_load(rp.read_text()).get("pool_id") != pool_id:
            continue
        s = l.get("session") or {}
        if not s.get("verdict") or s.get("reference_is_this_pass"):
            continue
        acc = (round(s["base_rate_observed"] * s["n_decided"]) if s.get("n_decided") else 0)
        ref = s.get("base_rate_reference") or 0.0
        out.append({"batch": l["batch_id"], "worksheet_mtime": l.get("worksheet_mtime", ""),
                    "verdict": s["verdict"], "kappa": s.get("kappa"),
                    "base_rate_you": f"{int(acc)}/{s.get('n_reference')}",
                    "base_rate_reference": f"{round(ref * s.get('n_reference', 0))}/{s.get('n_reference')}",
                    "base_rate_drift_relative": (float(s["base_rate_observed"] / ref - 1.0) if ref else None),
                    "undecided": int(s.get("n_undecided") or 0)})
    # by the moment each was labelled, not by the text: the records carry local times whose offsets
    # differ, and "2026-09-04T16:20:58-05:00" is later than "2026-09-04T17:00:00-04:00" as a string
    # and earlier as a moment.
    return sorted(out, key=lambda c: pd.Timestamp(c["worksheet_mtime"]))


def calibration_check_for(rec: dict, ws_first_saved: str, accept_drift: bool,
                          count_for_correction: bool) -> dict:
    """Which reading this sheet was labelled under, and whether a number may rest on it.

    The protocol's check is a fresh blind copy of the pool's 42 calibration panels, labelled before
    the session. So the copy that decides is the LAST one labelled before this sheet was saved — not
    the last one that passed. If that copy read DRIFTED, the session it checks does not count, and
    the sheet is refused: label a fresh copy on another day and load again.
    `--accept-calibration-drift` loads it anyway and the record carries the copy and its numbers; a
    re-judgement loaded that way does not correct a rate unless `--count-for-correction` says so as
    well.
    """
    copies = calibration_copies(rec["pool_id"])
    before = [c for c in copies if c["worksheet_mtime"]
              and pd.Timestamp(c["worksheet_mtime"]) < pd.Timestamp(ws_first_saved)]
    set_id = (copies[-1]["batch"].split("_pass")[0] if copies else f"calib_{rec['pool_id'].split('/')[-1]}_v1")
    what_to_do = (f"Label a fresh blind copy of the 42 panels on another day and load again:\n"
                  f"    python pipeline/draw_batch.py --repass {set_id}\n"
                  f"    make calibrate SHEET=<the labelled copy>      # it must say PASS\n"
                  f"    make load BATCH={rec['batch_id']} FLAGS=--replace\n"
                  f"To load this sheet anyway, pass --accept-calibration-drift: the rows go in, the record says which "
                  f"copy read what, and a re-judgement loaded that way does not correct a rate.")
    if not before:
        if not accept_drift:
            raise SystemExit(f"{rec['batch_id']}: no scored copy of {rec['pool_id']}'s 42 calibration panels was "
                             f"labelled before this sheet ({ws_first_saved}) — the protocol's check.\n{what_to_do}")
        check = {"batch": None, "verdict": None, "kappa": None, "base_rate_you": None,
                 "base_rate_reference": None, "sheet_first_saved": ws_first_saved,
                 "why": "no calibration copy of this pool was labelled before the sheet"}
    else:
        check = dict(before[-1])
        check["sheet_first_saved"] = ws_first_saved
        # The protocol wants the copy finished on an EARLIER DAY than the sheet, so the reading it
        # checks is yesterday's habit and not this afternoon's. Only a warning: the reviewer may
        # legitimately do both in one day.
        same_day = check["worksheet_mtime"][:10] >= ws_first_saved[:10]
        check["earlier_day"] = not same_day
        if same_day:
            check["warning"] = (f"the calibration copy was labelled on {check['worksheet_mtime'][:10]}, the same day "
                                f"as this sheet — the protocol asks for an earlier day, so that the reading it checks "
                                f"is not the one the sheet was labelled in")
            print("WARNING:", check["warning"])
        if len(before) > 1:
            check["copies_before_this_sheet"] = [{k: c[k] for k in ("batch", "worksheet_mtime", "verdict")}
                                                 for c in before]
    check["accepted_by_hand"] = False
    said = ""
    if check.get("verdict") != "PASS":
        drift = check.get("base_rate_drift_relative")
        if check.get("batch") is None:
            said = str(check.get("why"))
        else:
            said = (f"{check['batch']} read {check['verdict']}: {check['base_rate_you']} accepted against the "
                    f"reference's {check['base_rate_reference']}"
                    + (f", {abs(drift):.0%} relative {'stricter' if drift < 0 else 'looser'}" if drift is not None else "")
                    + (f", κ {check['kappa']:.2f}" if check.get("kappa") is not None else ""))
        if not accept_drift:
            raise SystemExit(f"{rec['batch_id']}: the calibration copy labelled before this sheet says the reading had "
                             f"moved — {said}. The copy labelled before a session is what decides whether that session "
                             f"counts, and this one did not pass.\n{what_to_do}")
        check["accepted_by_hand"] = True
        check["accepted_by_hand_means"] = f"loaded with --accept-calibration-drift: {said}"
        print("WARNING: loaded although the calibration copy did not pass —", said)
    if rec["role"] == "rejudgement":
        counts = bool(check.get("verdict") == "PASS" or count_for_correction)
        check["counts_for_correction"] = counts
        if not counts:
            check["does_not_count_because"] = (
                f"the calibration copy labelled before this second look did not pass ({said}); a second look read "
                f"under a reading known to have moved cannot say how another reading moved")
        elif check.get("verdict") != "PASS":
            check["counts_by_hand"] = "--count-for-correction was given although the calibration copy did not pass"
    return check


def refuse_if_the_reading_drifted(rec: dict, sess: dict, accept_drift: bool) -> None:
    """A rate batch whose acceptance fell across the sitting is not loaded on its own word.

    The test is acceptance in the first half of the sheet against the second, holding the stratum
    fixed. If it fails, the batch is loaded only once a blind re-judgement of it has been drawn —
    that is what measures how far the reading moved and corrects the rate — or once the reviewer says
    in so many words to load it anyway (`--accept-drift`), which the record then carries.
    """
    if rec["role"] != "analysis":
        return
    trend = sess.get("within_stratum_position_trend") or {}
    p_value = trend.get("p")
    sess["drift_decision"] = {"p": p_value, "alpha": B.DRIFT_ALPHA, "verdict": trend.get("verdict"),
                              "rejudgements": rejudgements_of(rec["batch_id"]), "accepted_by_hand": False}
    if p_value is None or p_value >= B.DRIFT_ALPHA:
        sess["drift_decision"]["loaded_because"] = "acceptance did not fall across the sitting"
        return
    rj = sess["drift_decision"]["rejudgements"]
    if rj:
        sess["drift_decision"]["loaded_because"] = (
            f"a blind re-judgement of this batch exists ({', '.join(x['batch_id'] for x in rj)}); the rate it feeds "
            f"is corrected once that re-judgement is labelled and loaded")
        return
    if accept_drift:
        sess["drift_decision"].update(
            accepted_by_hand=True,
            loaded_because="--accept-drift was given: the reviewer chose to load it with the drift unmeasured")
        print("WARNING: loaded with the drift unmeasured (--accept-drift); no corrected rate until a re-judgement exists")
        return
    raise SystemExit(
        f"{rec['batch_id']}: acceptance fell across the sitting — {trend['accept_first_half']:.1%} in the first half "
        f"against {trend['accept_second_half']:.1%} in the second, and not because the two halves held different "
        f"strata (p = {p_value:.4f}, below {B.DRIFT_ALPHA}). A rate measured on a reading that moved inside the "
        f"session is a number with a systematic term nobody has measured.\n"
        f"Draw the blind second look before loading this batch:\n"
        f"    make draw-batch BATCH=rejudge_{rec['batch_id'].removeprefix('rate_')}\n"
        f"It re-shows 100 of these panels, half from each half of the sitting, and its flip rate is what corrects the "
        f"rate. To load the batch anyway and leave the drift unmeasured, pass --accept-drift.")


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("batch_id")
    ap.add_argument("--allow-unfinished", action="store_true")
    ap.add_argument("--replace", action="store_true", help="load a batch already in the study layer again")
    ap.add_argument("--freeze-reference", metavar="HOW",
                    help="a calibration batch built with REF_LABEL blank: freeze this pass's labels as the set's frozen "
                         "answers (HOW names the pass, e.g. 'a single pass by the user, agreement pending')")
    ap.add_argument("--accept-drift", action="store_true",
                    help="load a rate batch whose acceptance fell within the sitting anyway; the record says so, and "
                         "the rate it feeds carries no drift correction until a blind re-judgement is loaded")
    ap.add_argument("--accept-calibration-drift", action="store_true",
                    help="load a sheet although the calibration copy labelled before it did not pass; the record "
                         "carries that copy and its numbers, and a re-judgement loaded this way corrects no rate")
    ap.add_argument("--count-for-correction", action="store_true",
                    help="with --accept-calibration-drift, let a re-judgement whose calibration copy did not pass "
                         "correct a rate all the same; the record says it was allowed by hand")
    args = ap.parse_args()
    bid = args.batch_id
    rec = yaml.safe_load((DRAWS / f"{bid}.yaml").read_text())
    crit = require_ruled(load_criteria()[rec["criterion_version"]])
    ws, key, parts = read_sheets(rec)
    wp, kp = parts[0]["path"], parts[0]["key_path"]
    lab = pd.to_numeric(ws.LABEL, errors="coerce")
    if lab.isna().any() and not args.allow_unfinished:
        raise SystemExit(f"{int(lab.isna().sum())} rows still blank — finish the sheet or pass --allow-unfinished")
    if not lab.isin([0, 1, 2]).any():
        raise SystemExit("nothing labelled")
    ws_mtime = max(x["mtime"] for x in parts)
    ws_first_saved = min(x["mtime"] for x in parts)
    if args.freeze_reference:
        if rec["role"] != "calibration":
            raise SystemExit("--freeze-reference is for a calibration batch")
        print("reference frozen:", freeze_reference(rec, wp, key, ws, args.freeze_reference))

    # The check is hard: a rate sheet or a second look is loaded only if the last scored copy of the
    # pool's 42 calibration panels labelled before it said PASS (docs/LABELING_PROTOCOL.md). Not an
    # earlier copy that passed — the copy labelled before a session is what decides whether that
    # session counts.
    calibration = None
    if rec["role"] in ("analysis", "rejudgement"):
        calibration = calibration_check_for(rec, ws_first_saved, args.accept_calibration_drift,
                                            args.count_for_correction)

    sess = session_block(rec, parts, ws, key, calibration=calibration)
    refuse_if_the_reading_drifted(rec, sess, args.accept_drift)
    sheet_sha = joint_sha([{"sheet_id": x["sheet_id"], "sha256": x["sha256"]} for x in parts])
    frozen = []
    for x in parts:
        w = x["path"]
        same = [q for q in w.parent.glob(f"{x['sheet_id']}_LABELLED_*.csv") if _sha(q) == x["sha256"]]
        if same:
            frozen.append(same[0])   # the same bytes are already frozen; never write a second copy
        else:
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            f = w.with_name(f"{x['sheet_id']}_LABELLED_{stamp}.csv")
            shutil.copyfile(w, f)
            frozen.append(f)
    frozen_path = B.repo_relative(frozen[0]) if len(frozen) == 1 else B.repo_relative(frozen[0].parent.parent)

    # --- the review rows -----------------------------------------------------------------------
    m = key.merge(ws[["SAMPLE_ID", "LABEL"]], on="SAMPLE_ID", validate="one_to_one")
    m["LABEL"] = pd.to_numeric(m.LABEL, errors="coerce")
    R = pd.DataFrame({
        "review_id": [_rid(bid, sheet_sha, i) for i in range(len(m))],
        "row_index": np.arange(len(m)),
        "candidate_id": m.candidate_id,
        "pool_id": rec["pool_id"], "criterion_version": rec["criterion_version"], "role": rec["role"],
        # 0 / 1 / 2 as the label table records them (2 = uncertain, a label but not a decision;
        # every rate filters isin([0, 1]) and labelled_keys counts it as judged)
        "decision": pd.array([int(x) if x in (0, 1, 2) else pd.NA for x in m.LABEL], dtype="Int8"),
        "supersedes_review_id": pd.array([None] * len(m), dtype="string"),
        "key_wmo": m.WMO.astype("Int64"), "key_cycle": m.CYCLE_NUMBER.round().astype("Int64"),
        "key_pres": m.PRES_ADJUSTED.round().astype("Int64"),
        "sheet_sha256": sheet_sha, "event": rec["event_type"].split("_")[0],
        "WMO": m.WMO.astype(float), "CYCLE_NUMBER": m.CYCLE_NUMBER.astype(float), "PRES_ADJUSTED": m.PRES_ADJUSTED.astype(float),
        "LABEL": m.LABEL.astype(float), "batch_id": bid,
        "rank": pd.array([pd.NA] * len(m), dtype="Int64"),
        "tier": m.tier.astype("string") if "tier" in m.columns else pd.array([None] * len(m), dtype="string"),
        "sampling_mode": rec["sampling"]["mode"], "stratum": m.design_stratum.astype("string"),
        "control_arm": m.stratum.astype("string"), "blind": True,
        "SAMPLE_ID": m.SAMPLE_ID.astype(float), "src": m.src.astype("string"), "score": m.score.astype(float),
        "inclusion_probability": m.inclusion_probability.astype(float), "design_stratum": m.design_stratum.astype("string"),
        "study_id": rec["study_id"], "spec_id": rec["spec_id"],
        "REF_LABEL": m.REF_LABEL.astype(float) if "REF_LABEL" in m.columns else np.nan,
        "previously_judged": m.previously_judged.fillna(False).astype(bool) if "previously_judged" in m.columns else False,
    })[BASE_REVIEW_COLS + STUDY_EXTRA_COLS]
    for c in ("supersedes_review_id", "tier", "src", "stratum", "control_arm", "design_stratum"):
        R[c] = R[c].astype(object).where(R[c].notna(), None)

    old = pd.read_parquet(STUDY_REVIEWS) if STUDY_REVIEWS.exists() else None
    if old is not None and (old.batch_id == bid).any():
        if not args.replace:
            raise SystemExit(f"{bid} is already in the study layer — pass --replace to load it again")
        prev_sha = set(old.loc[old.batch_id == bid, "sheet_sha256"])
        if prev_sha != {sheet_sha}:
            raise SystemExit(f"{bid}: the sheet's bytes changed since it was loaded ({prev_sha} -> {sheet_sha}); the label table is "
                             f"append-only — a re-labelled sheet is a new batch id whose rows supersede, never a replacement")
        old = old[old.batch_id != bid]   # same bytes: the session read is re-issued, the review rows are identical
    allR = pd.concat([old, R], ignore_index=True) if old is not None else R
    allR.to_parquet(STUDY_REVIEWS, index=False)

    # --- the batch record --------------------------------------------------------------------
    batch = {
        "batch_id": bid, "raw_root_id": "net_carbon_v1_labeling", "sheet_path": frozen_path,
        "sheet_sha256": sheet_sha, "worksheet_blank_sha256": rec["worksheet"]["sha256"], "answer_key_sha256": rec["answer_key"]["sha256"],
        "rows": int(len(m)), "columns_kept": B.WORKSHEET_COLS, "first_written": rec["built"], "ingested": B.stamp(),
        "study_id": rec["study_id"], "pool_id": rec["pool_id"], "spec_id": rec["spec_id"], "event": rec["event_type"].split("_")[0],
        "criterion_version": rec["criterion_version"], "criterion_evidence": f"drawn and labelled under {crit.id} ({crit.status})",
        "role": rec["role"], "decides": bool(rec["decides"]),
        "sampling": {"mode": rec["sampling"]["mode"], "draw": rec["sampling"]["draw"], "design": rec["sampling"]["design"],
                     "frame": rec["sampling"]["frame"], "inclusion_probability": "per review row (`inclusion_probability`), n_h/N_h within stratum",
                     "has_own_stratum_column": True},
        "blind": True, "answer_key_batch": B.repo_relative(kp), "derived_from": None, "invalidated": None,
        "session": sess,
    }
    raw = yaml.safe_load(STUDY_BATCHES.read_text()) if STUDY_BATCHES.exists() else {"batches": []}
    raw["batches"] = [b for b in raw["batches"] if b["batch_id"] != bid] + [batch]
    STUDY_BATCHES.write_text(
        "# data/labels/study_batches.yaml -- the study label table: one record per labelled study sheet.\n"
        "# Built by pipeline/load_batch.py from the draw records and the labelled sheets; never edited by hand.\n"
        + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    labelled = {"batch_id": bid, "ingested": batch["ingested"], "worksheet_mtime": ws_mtime,
                "worksheet_first_saved": ws_first_saved, "calibration_check": calibration,
                "labelled_sheet": {"path": frozen_path, "sha256": sheet_sha,
                                   "sheets": [{"sheet_id": x["sheet_id"], "path": B.repo_relative(f), "sha256": x["sha256"],
                                               "rows": x["rows"], "mtime": x["mtime"]}
                                              for x, f in zip(parts, frozen)] if len(parts) > 1 else None},
                "rows": int(len(m)), "decided": int(lab.isin([0, 1]).sum()), "accepted": int((lab == 1).sum()),
                "uncertain": int((lab == 2).sum()), "blank": int(lab.isna().sum()), "session": sess}
    batch["calibration_check"] = calibration
    batch["worksheet_mtime"] = ws_mtime
    (DRAWS / f"{bid}.labelled.yaml").write_text(
        f"# data/labels/draws/{bid}.labelled.yaml -- the session read and the hashes of the labelled sheet. Built by\n"
        f"# pipeline/load_batch.py; never edited by hand.\n"
        + yaml.safe_dump(labelled, sort_keys=False, allow_unicode=True, width=110, default_flow_style=False), encoding="utf-8")
    (DRAWS / "LABELLED_SHA256").write_text(
        "# provenance of the labelled study batches -- do not edit by hand\n"
        + "".join(f"{_sha(p)}\t{p.name}\n" for p in sorted(DRAWS.glob("*.labelled.yaml")))
        + f"{_sha(STUDY_BATCHES)}\tstudy_batches.yaml\n{_sha(STUDY_REVIEWS)}\tstudy_reviews.parquet\t{len(allR):,} reviews\n")
    print(json.dumps({k: v for k, v in labelled.items() if k != "session"}, indent=1))
    print(json.dumps({k: v for k, v in sess.items() if k not in ("position", "target_by_stratum")}, indent=1, default=str))


if __name__ == "__main__":
    main()
