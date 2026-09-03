#!/usr/bin/env python
"""Load a labelled study sheet into the label table's study layer.

usage  ingest_study_batch.py BATCH_ID [--allow-unfinished] [--replace] [--freeze-reference HOW]
reads  data/labels/draws/<BATCH_ID>.yaml (the draw record), the worksheet and the sealed key under
       results/net_carbon_v1/labeling/<BATCH_ID>/
writes results/net_carbon_v1/labeling/<BATCH_ID>/<BATCH_ID>_LABELLED_<stamp>.csv (the frozen sheet, not in git)
       data/labels/{study_reviews.parquet, study_batches.yaml}, data/labels/draws/<BATCH_ID>.labelled.yaml, LABELLED_SHA256
In order, none skipped: check the key's hash and the sheet's columns; read the session (progress,
position, controls, science; the κ gate for a calibration batch); freeze the sheet; append the rows.
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
sys.path.insert(0, str(REPO / "production"))

from eddy_pump import batches as B  # noqa: E402
from eddy_pump import labels as L  # noqa: E402
from eddy_pump.criteria import load_criteria, require_ruled  # noqa: E402

DRAWS = REPO / "data/labels/draws"
STUDY_BATCHES, STUDY_REVIEWS = L.STUDY_BATCHES, L.STUDY_REVIEWS
LEGACY_REVIEW_COLS = ["review_id", "row_index", "candidate_id", "pool_id", "criterion_version", "role", "decision",
                      "supersedes_review_id", "key_wmo", "key_cycle", "key_pres", "sheet_sha256", "event", "WMO",
                      "CYCLE_NUMBER", "PRES_ADJUSTED", "LABEL", "batch_id", "rank", "tier", "sampling_mode", "stratum",
                      "control_arm", "blind", "SAMPLE_ID", "src", "score"]
STUDY_EXTRA_COLS = ["inclusion_probability", "design_stratum", "study_id", "spec_id", "REF_LABEL", "previously_judged"]


def _rid(batch_id: str, sheet_sha: str, row: int) -> str:
    return hashlib.sha256(f"{batch_id}|{sheet_sha}|{row}".encode("utf-8")).hexdigest()[:16]


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def reference_for(rec: dict, key: pd.DataFrame):
    """The anchor's frozen reference: the key's own REF_LABEL (the upward anchor, carried over from
    b6) or, when the key was built blank, `data/labels/draws/<batch>.reference.yaml` written by
    `--freeze-reference`. Returns (frame with KEYS + REF_LABEL, provenance) or (None, {})."""
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
    """Write the anchor's reference from a labelled pass, ONCE. Refuses if one exists or if any row
    is undecided: an anchor is 42 decided levels. `how` names the pass (single pass / consensus)."""
    rp = DRAWS / f"{rec.get('reference_of', rec['batch_id'])}.reference.yaml"
    if rp.exists():
        raise SystemExit(f"{rp} exists — an anchor is frozen once; a re-freeze by consensus is a new file written by hand-ruled process, not this flag")
    m = key.merge(ws[["SAMPLE_ID", "LABEL"]], on="SAMPLE_ID", validate="one_to_one")
    m["LABEL"] = pd.to_numeric(m.LABEL, errors="coerce")
    if not m.LABEL.isin([0, 1]).all():
        raise SystemExit(f"{int((~m.LABEL.isin([0, 1])).sum())} rows undecided or uncertain — every anchor level needs a 0/1")
    cols = B.KEYS + [c for c in ("tier", "companion", "score", "candidate_id") if c in m.columns]
    rows = m[cols].assign(REF_LABEL=m.LABEL.astype(int)).to_dict("records")
    fr = {"batch_id": rec["batch_id"], "pool_id": rec["pool_id"], "criterion_version": rec["criterion_version"],
          "frozen": B.stamp(), "how": how, "source_sheet_sha256": _sha(wp), "n": int(len(rows)),
          "base_rate": f"{int(m.LABEL.sum())}/{len(m)}", "rows": rows}
    rp.write_text(f"# data/labels/draws/{rec['batch_id']}.reference.yaml -- the anchor's frozen reference labels. Written ONCE by\n"
                  f"# production/ingest_study_batch.py --freeze-reference; never edited by hand.\n"
                  + yaml.safe_dump(fr, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    return rp


def session_block(rec: dict, wp: pathlib.Path, kp: pathlib.Path, ws: pd.DataFrame, key: pd.DataFrame) -> dict:
    """The protocol's read of the session, as data."""
    if rec["role"] == "calibration":
        ref, ref_meta = reference_for(rec, key)
        if ref is None:
            return {"kind": "calibration", "gate": "no REF_LABEL yet — an anchor awaiting adjudication; ingested as evidence only"}
        rep = B.calibration_report(ws, ref)
        rep["kind"] = "calibration"
        rep["reference"] = ref_meta
        if ref_meta.get("source_sheet_sha256") == _sha(wp):
            rep["reference_is_this_pass"] = True
            rep["verdict_means"] = ("the reference IS this sheet's own labels, so κ = 1 by construction; the first real gate is the "
                                    "blind re-labelling before the NEXT session")
        return rep
    from argopod.review.session import read_session

    ctrl = rec["controls"]["positive"]
    ref = (ctrl["reference_k"], ctrl["reference_n"], ctrl["reference_provenance"]) if ctrl.get("reference_k") is not None else None
    rep = read_session(wp, kp, pos_ctrl_reference=ref, breakdown_cols=("src",))
    out = {"kind": "rate", "n_rows": rep.n_rows, "n_decided": rep.n_decided, "n_accepted": rep.n_accepted,
           "n_rejected": rep.n_rejected, "n_uncertain": rep.n_uncertain, "n_blank": rep.n_blank,
           "position": [{"bin": p.index, "sid_lo": p.sid_lo, "sid_hi": p.sid_hi, "n": p.n, "accept": p.accept} for p in rep.position],
           "trend_p_mann_whitney": rep.trend_p,
           "change_point": ({"sample_id": rep.change_point.sample_id, "accept_before": rep.change_point.accept_before,
                             "accept_after": rep.change_point.accept_after, "p_raw": rep.change_point.p_raw,
                             "p_adjusted": rep.change_point.p_adjusted, "verdict": rep.change_point.verdict}
                            if rep.change_point else None),
           "key_opened": rep.key_opened, "warnings": list(rep.warnings)}
    m = key.merge(ws[["SAMPLE_ID", "LABEL"]], on="SAMPLE_ID")
    m["LABEL"] = pd.to_numeric(m.LABEL, errors="coerce")
    from scipy.stats import beta as _beta, fisher_exact

    def ci(k, n):
        return [float(_beta.ppf(0.025, k + 0.5, n - k + 0.5)), float(_beta.ppf(0.975, k + 0.5, n - k + 0.5))] if n else None

    # A control is read against its STANDING verdict: a b6 verdict the ledger's own re-looks later
    # overturned (direct_flips.csv) says nothing about today's reviewer. The first batch drew its
    # controls before this was understood; the record reports both readings.
    is_b6 = ctrl.get("criterion") == "phys_obduction_letter_b6"
    flipped_keys: set = set()
    if is_b6:
        # direct_flips.csv carries LEGACY candidate ids (legacy_letter_v1/...); the key carries the study's.
        # Join through the legacy reviews to the (WMO, cycle, level) triple, which both share.
        fl = pd.read_csv(REPO / "data/labels/audit/direct_flips.csv")
        Rl = L.legacy_only(L.load_reviews())
        fk = Rl[Rl.candidate_id.isin(fl.candidate_id)][["key_wmo", "key_cycle", "key_pres"]].drop_duplicates()
        flipped_keys = set(zip(fk.key_wmo.astype(int), fk.key_cycle.astype(int), fk.key_pres.astype(int)))
    m["_k"] = list(zip(m.WMO.astype(int), m.CYCLE_NUMBER.round().astype(int), m.PRES_ADJUSTED.round().astype(int)))
    ctr = {}
    for arm in B.CONTROL_STRATA:
        c = m[(m.stratum == arm) & m.LABEL.isin([0, 1])]
        st = c[~c._k.isin(flipped_keys)]
        ctr[arm] = {"n": int(len(c)), "accepted": int((c.LABEL == 1).sum()), "ci95": ci(int((c.LABEL == 1).sum()), len(c)),
                    "overturned_in_ledger": int(c._k.isin(flipped_keys).sum()),
                    "standing": {"n": int(len(st)), "accepted": int((st.LABEL == 1).sum()), "ci95": ci(int((st.LABEL == 1).sum()), len(st))}}
    pos = ctr[B.POS_CTRL]
    if ref is not None and pos["n"]:
        ctr[B.POS_CTRL]["reference"] = {"k": ref[0], "n": ref[1], "what": ref[2]}
        ctr[B.POS_CTRL]["fisher_p_vs_reference"] = float(fisher_exact([[pos["accepted"], pos["n"] - pos["accepted"]], [ref[0], ref[1] - ref[0]]])[1])
    if is_b6:
        import build_batches as BB
        for arm, dec in ((B.POS_CTRL, 1), (B.NEG_CTRL, 0)):
            h = BB.blind_rejudgement_history(dec)
            a = ctr[arm]
            a["blind_history"] = h
            if a["n"]:
                a["fisher_p_vs_blind_history"] = float(fisher_exact([[a["accepted"], a["n"] - a["accepted"]], [h["k"], h["n"] - h["k"]]])[1])
            if a["standing"]["n"]:
                s_ = a["standing"]
                a["fisher_p_vs_blind_history_standing"] = float(fisher_exact([[s_["accepted"], s_["n"] - s_["accepted"]], [h["k"], h["n"] - h["k"]]])[1])
        ctr[B.POS_CTRL]["power_note"] = "n = 20 per arm sees a collapse, not a moderate drift; pool the arms across batches before reading strictness"
    ctr[B.NEG_CTRL]["ceiling"] = rec["controls"]["negative"].get("ceiling", 0.20)
    ctr[B.NEG_CTRL]["ceiling_note"] = "display only; at equality nothing fires — the test is Fisher against the negatives' own blind history"
    hi_neg = m[(m.stratum == B.NEG_CTRL) & (m.score >= 0.5)]
    ctr[B.NEG_CTRL]["with_score_above_0p5"] = {"n": int(len(hi_neg)), "accepted": int((hi_neg.LABEL == 1).sum()),
                                               "note": "a rejected candidate the classifier calls likely is a plausible legacy miss, not a control"}
    out["controls"] = ctr
    t = m[(m.stratum == B.TARGET) & m.LABEL.isin([0, 1])]
    out["target_by_stratum"] = {s: {"n": int(len(g)), "accepted": int(g.LABEL.sum())} for s, g in t.groupby("src")}
    out["target_raw_acceptance_unweighted"] = float(t.LABEL.mean()) if len(t) else None
    out["target_previously_judged"] = {"n": int(t.previously_judged.fillna(False).astype(bool).sum()),
                                       "accepted": int(t[t.previously_judged.fillna(False).astype(bool)].LABEL.sum())}
    # position drift on the TARGET rows alone (the reader's step 2 runs on every decided row)
    q = pd.qcut(t.SAMPLE_ID.rank(method="first"), 4, labels=False)
    out["target_position_quarters"] = [{"quarter": int(i) + 1, "n": int(len(g)), "accept": float(g.LABEL.mean())} for i, g in t.groupby(q)]
    med = t.SAMPLE_ID.median()
    out["target_halves"] = {"first": {"n": int((t.SAMPLE_ID <= med).sum()), "accept": float(t[t.SAMPLE_ID <= med].LABEL.mean())},
                            "second": {"n": int((t.SAMPLE_ID > med).sum()), "accept": float(t[t.SAMPLE_ID > med].LABEL.mean())},
                            "by_stratum": {s: {"first": (float(g[g.SAMPLE_ID <= med].LABEL.mean()) if (g.SAMPLE_ID <= med).any() else None),
                                               "second": (float(g[g.SAMPLE_ID > med].LABEL.mean()) if (g.SAMPLE_ID > med).any() else None)}
                                           for s, g in t.groupby("src")}}
    return out


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("batch_id")
    ap.add_argument("--allow-unfinished", action="store_true")
    ap.add_argument("--replace", action="store_true", help="re-ingest a batch already in the study layer")
    ap.add_argument("--freeze-reference", metavar="HOW",
                    help="a calibration batch built with REF_LABEL blank: freeze THIS pass's labels as the anchor's reference "
                         "(HOW names the pass, e.g. 'single pass by the user, consensus pending')")
    args = ap.parse_args()
    bid = args.batch_id
    rec = yaml.safe_load((DRAWS / f"{bid}.yaml").read_text())
    crit = require_ruled(load_criteria()[rec["criterion_version"]])
    wp, kp = pathlib.Path(rec["worksheet"]["path"]), pathlib.Path(rec["answer_key"]["path"])
    if _sha(kp) != rec["answer_key"]["sha256"]:
        raise SystemExit(f"{kp} does not hash to the sealed key the draw record names — refusing")
    ws, key = pd.read_csv(wp), pd.read_csv(kp)
    if list(ws.columns) != B.WORKSHEET_COLS:
        raise SystemExit(f"{wp}: columns {list(ws.columns)} are not the blind worksheet's")
    key = key.sort_values("SAMPLE_ID").reset_index(drop=True)
    ws = ws.sort_values("SAMPLE_ID").reset_index(drop=True)
    if not (ws[B.KEYS].round(0).to_numpy() == key[B.KEYS].round(0).to_numpy()).all():
        raise SystemExit("worksheet and key disagree on a row — refusing")
    lab = pd.to_numeric(ws.LABEL, errors="coerce")
    if lab.isna().any() and not args.allow_unfinished:
        raise SystemExit(f"{int(lab.isna().sum())} rows still blank — finish the sheet or pass --allow-unfinished")
    if not lab.isin([0, 1, 2]).any():
        raise SystemExit("nothing labelled")
    ws_mtime = _dt.datetime.fromtimestamp(wp.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    if args.freeze_reference:
        if rec["role"] != "calibration":
            raise SystemExit("--freeze-reference is for a calibration batch")
        print("reference frozen:", freeze_reference(rec, wp, key, ws, args.freeze_reference))

    # THE GATE IS HARD: an analysis batch is ingested only if an anchor of the same pool was
    # re-labelled PASS before its sheet was written (production/LABELING_PROTOCOL.md). The first
    # batch met it by file times alone; now nothing else is accepted.
    anchor = None
    if rec["role"] == "analysis":
        cands = []
        for p in sorted(DRAWS.glob("calib_*.labelled.yaml")):
            l = yaml.safe_load(p.read_text())
            r0 = yaml.safe_load((DRAWS / f"{l['batch_id']}.yaml").read_text())
            if r0["pool_id"] == rec["pool_id"] and l["session"].get("verdict") == "PASS" and l.get("worksheet_mtime", "") < ws_mtime:
                cands.append((l["worksheet_mtime"], l["batch_id"], l["session"]["kappa"]))
        if not cands:
            raise SystemExit(f"{bid}: no calibration anchor of {rec['pool_id']} re-labelled PASS before this sheet "
                             f"({ws_mtime}) — the protocol's gate; label the anchor first")
        anchor = {"batch_id": cands[-1][1], "worksheet_mtime": cands[-1][0], "kappa": cands[-1][2]}

    sess = session_block(rec, wp, kp, ws, key)
    sheet_sha = _sha(wp)
    existing = [p for p in wp.parent.glob(f"{bid}_LABELLED_*.csv") if _sha(p) == sheet_sha]
    if existing:
        frozen = existing[0]   # the same bytes are already frozen; never write a second copy
    else:
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        frozen = wp.with_name(f"{bid}_LABELLED_{stamp}.csv")
        shutil.copyfile(wp, frozen)

    # --- the review rows -----------------------------------------------------------------------
    m = key.merge(ws[["SAMPLE_ID", "LABEL"]], on="SAMPLE_ID", validate="one_to_one")
    m["LABEL"] = pd.to_numeric(m.LABEL, errors="coerce")
    R = pd.DataFrame({
        "review_id": [_rid(bid, sheet_sha, i) for i in range(len(m))],
        "row_index": np.arange(len(m)),
        "candidate_id": m.candidate_id,
        "pool_id": rec["pool_id"], "criterion_version": rec["criterion_version"], "role": rec["role"],
        # 0 / 1 / 2 as the legacy layer records them (2 = uncertain, a label but not a decision;
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
    })[LEGACY_REVIEW_COLS + STUDY_EXTRA_COLS]
    for c in ("supersedes_review_id", "tier", "src", "stratum", "control_arm", "design_stratum"):
        R[c] = R[c].astype(object).where(R[c].notna(), None)

    old = pd.read_parquet(STUDY_REVIEWS) if STUDY_REVIEWS.exists() else None
    if old is not None and (old.batch_id == bid).any():
        if not args.replace:
            raise SystemExit(f"{bid} is already in the study layer — pass --replace to re-ingest")
        prev_sha = set(old.loc[old.batch_id == bid, "sheet_sha256"])
        if prev_sha != {sheet_sha}:
            raise SystemExit(f"{bid}: the sheet's bytes changed since it was ingested ({prev_sha} -> {sheet_sha}); the ledger is "
                             f"append-only — a re-labelled sheet is a NEW batch id whose rows supersede, never a replacement")
        old = old[old.batch_id != bid]   # same bytes: the session read is re-issued, the review rows are identical
    allR = pd.concat([old, R], ignore_index=True) if old is not None else R
    allR.to_parquet(STUDY_REVIEWS, index=False)

    # --- the batch record --------------------------------------------------------------------
    batch = {
        "batch_id": bid, "raw_root_id": "net_carbon_v1_labeling", "sheet_path": str(frozen.relative_to(REPO)),
        "sheet_sha256": sheet_sha, "worksheet_blank_sha256": rec["worksheet"]["sha256"], "answer_key_sha256": rec["answer_key"]["sha256"],
        "rows": int(len(m)), "columns_kept": B.WORKSHEET_COLS, "first_written": rec["built"], "ingested": B.stamp(),
        "study_id": rec["study_id"], "pool_id": rec["pool_id"], "spec_id": rec["spec_id"], "event": rec["event_type"].split("_")[0],
        "criterion_version": rec["criterion_version"], "criterion_evidence": f"drawn and labelled under {crit.id} ({crit.status})",
        "role": rec["role"], "decides": bool(rec["decides"]),
        "sampling": {"mode": rec["sampling"]["mode"], "draw": rec["sampling"]["draw"], "design": rec["sampling"]["design"],
                     "frame": rec["sampling"]["frame"], "inclusion_probability": "per review row (`inclusion_probability`), n_h/N_h within stratum",
                     "has_own_stratum_column": True},
        "legacy": {"tier": None, "precedence_rank": None, "nitrate_rank": None, "in_drifted_list": False,
                   "random_session_index": None, "priority_index": None, "is_adjudication": False},
        "blind": True, "answer_key_batch": str(kp.relative_to(REPO)), "derived_from": None, "invalidated": None,
        "session": sess,
    }
    raw = yaml.safe_load(STUDY_BATCHES.read_text()) if STUDY_BATCHES.exists() else {"batches": []}
    raw["batches"] = [b for b in raw["batches"] if b["batch_id"] != bid] + [batch]
    STUDY_BATCHES.write_text(
        "# data/labels/study_batches.yaml -- the study layer of the ledger: one record per labelled study sheet.\n"
        "# BUILT by production/ingest_study_batch.py from the draw records and the labelled sheets; never edited by hand.\n"
        + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    labelled = {"batch_id": bid, "ingested": batch["ingested"], "worksheet_mtime": ws_mtime, "anchor": anchor,
                "labelled_sheet": {"path": str(frozen), "sha256": sheet_sha},
                "rows": int(len(m)), "decided": int(lab.isin([0, 1]).sum()), "accepted": int((lab == 1).sum()),
                "uncertain": int((lab == 2).sum()), "blank": int(lab.isna().sum()), "session": sess}
    batch["anchor"] = anchor
    batch["worksheet_mtime"] = ws_mtime
    (DRAWS / f"{bid}.labelled.yaml").write_text(
        f"# data/labels/draws/{bid}.labelled.yaml -- the session read and the hashes of the labelled sheet. BUILT by\n"
        f"# production/ingest_study_batch.py; never edited by hand.\n"
        + yaml.safe_dump(labelled, sort_keys=False, allow_unicode=True, width=110, default_flow_style=False), encoding="utf-8")
    (DRAWS / "LABELLED_SHA256").write_text(
        "# provenance of the labelled study batches -- do not edit by hand\n"
        + "".join(f"{_sha(p)}\t{p.name}\n" for p in sorted(DRAWS.glob("*.labelled.yaml")))
        + f"{_sha(STUDY_BATCHES)}\tstudy_batches.yaml\n{_sha(STUDY_REVIEWS)}\tstudy_reviews.parquet\t{len(allR):,} reviews\n")
    print(json.dumps({k: v for k, v in labelled.items() if k != "session"}, indent=1))
    print(json.dumps({k: v for k, v in sess.items() if k not in ("position", "target_by_stratum")}, indent=1, default=str))


if __name__ == "__main__":
    main()
