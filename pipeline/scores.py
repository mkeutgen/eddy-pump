"""Score every candidate of the two physical pools; the score only orders what a human sees.

reads  results/net_carbon_v1/features/physical_{obduction,subduction}.parquet, the label table
       (eddy_pump.labels), data/external/manually_verified_physical_subd_events.csv,
       $GLOBARGO_DATA/detected_physical_subd_events.csv
writes results/net_carbon_v1/scores/physical_{obduction,subduction}.parquet (key, score, out-of-fold flag)
       results/net_carbon_v1/models/physical_{obduction,subduction}.joblib (+ .json), the fitted models
       data/features/net_carbon_v1/SCORES_SHA256 (what was trained on, the honest AUC, hashes)
Upward: trained on the study's own obduction labels, metrics out-of-fold with folds by float.
Downward: trained on the earlier study's (the companion's) reviewed detections joined to our features;
the sign-flipped upward model is kept as a comparison. Never a label, never in a rate.

Everything a reader would argue with — which labels, which folds, which model, what the manifest
has to say — is in `eddy_pump.classifier`. This file is the run: read the features, call it twice,
write the four files.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eddy_pump import classifier as K  # noqa: E402
from eddy_pump.manifest import load_manifest  # noqa: E402

KEYS = K.KEYS
ID_COLS = K.ID_COLS
OBDUCTION_POOL = K.OBDUCTION_POOL
CRITERION = K.CRITERION
COMPANION = K.COMPANION
COMPANION_DETECTIONS = K.COMPANION_DETECTIONS
MANIFEST_DIR = REPO / "data/features/net_carbon_v1"

#: Re-exported so a caller that reaches for "the study's labels" gets the one definition of them.
#: `pipeline/draw_batch.py` imports both from here.
key3 = K.key3
feature_columns = K.feature_columns
upward_labels = K.upward_labels
downward_labels = K.downward_labels
alignment_rule = K.alignment_rule
apply_alignment = K.apply_alignment


def attach(pool: pd.DataFrame, data: K.TrainingSet, oof: np.ndarray) -> None:
    """Give every row of the pool a score, and give a labelled row its out-of-fold one instead.

    A labelled row scored by the model that was trained on it would look far better than it is,
    and those are exactly the rows the allocation is planned from. `score_is_oof` says which is
    which, and `pipeline/draw_batch.py` refuses a steering label whose score is not flagged.
    """
    pool["score_is_oof"] = False
    idx = pool.index[pool.key.isin(data.frame.key)]
    by_key = dict(zip(data.frame.key, oof))
    pool.loc[idx, "score"] = [by_key[k] for k in pool.loc[idx, "key"]]
    pool.loc[idx, "score_is_oof"] = True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    study = load_manifest()
    fdir = study.output.resolve("features")
    sdir = study.output.resolve("scores")
    mdir = study.output.resolve("models")
    sdir.mkdir(parents=True, exist_ok=True)
    obd = pd.read_parquet(fdir / "physical_obduction.parquet")
    sub = pd.read_parquet(fdir / "physical_subduction.parquet")
    obd["key"], sub["key"] = K.key3(obd), K.key3(sub)
    cols = K.feature_columns(obd)
    assert cols == K.feature_columns(sub), "the two pools must carry the same feature columns"
    backend = K.default_backend()

    # --- the upward limb: the study's own labels, honest out-of-fold, then the full fit ---------
    up = K.training_set(obd, K.upward_labels(set(obd.key)), cols)
    up_oof = K.evaluate(up, seed=a.seed, backend=backend)
    up_model = K.fit(up, seed=a.seed, backend=backend)
    obd["score"] = K.score(up_model, obd, cols, backend=backend)
    attach(obd, up, up_oof.probability)
    obd[KEYS + ["pool_id", "spec_id", "score", "score_is_oof"]].to_parquet(
        sdir / "physical_obduction.parquet", index=False)
    K.save(up_model, mdir / "physical_obduction.joblib", K.manifest(
        limb="upward", pool_id=OBDUCTION_POOL, data=up, oof=up_oof, seed=a.seed, backend=backend,
        labels_from=f"the study's own obduction reviews under {CRITERION} — a probability sample"))

    # --- the downward limb: a model trained on the companion's reviewed detections ---------------
    down = K.training_set(sub, K.downward_labels(set(sub.key)), cols)
    down_oof = K.evaluate(down, seed=a.seed, backend=backend)
    down_model = K.fit(down, seed=a.seed, backend=backend)
    sub["score"] = K.score(down_model, sub, cols, backend=backend)
    attach(sub, down, down_oof.probability)

    # the upward model transferred through the data-decided alignment: a comparison column and
    # the fallback if the companion's labels were ever withdrawn
    rule = K.alignment_rule(obd, sub, cols)
    sub["score_transfer"] = K.score(up_model, K.apply_alignment(sub, rule), cols, backend=backend)
    comp = pd.read_csv(COMPANION)
    comp["key"] = K.key3(comp)
    cs = sub.merge(comp[["key", "Category"]].drop_duplicates("key"), on="key", how="left")
    pos = cs.Category.isin([1, 2]).to_numpy()
    neg0 = (cs.Category == 0).to_numpy()
    unl = cs.Category.isna().to_numpy()
    from sklearn.metrics import roc_auc_score
    auc_pos_vs_rej = float(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(neg0.sum())], np.r_[cs.score_transfer[pos], cs.score_transfer[neg0]])) if neg0.sum() and pos.sum() else float("nan")
    auc_pos_vs_pool = float(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(unl.sum())], np.r_[cs.score_transfer[pos], cs.score_transfer[unl]])) if pos.sum() else float("nan")
    sub[KEYS + ["pool_id", "spec_id", "score", "score_is_oof", "score_transfer"]].to_parquet(
        sdir / "physical_subduction.parquet", index=False)
    K.save(down_model, mdir / "physical_subduction.joblib", K.manifest(
        limb="downward", pool_id="net_carbon_v1/physical/subduction", data=down, oof=down_oof,
        seed=a.seed, backend=backend,
        labels_from="the companion's reviewed R-detections, a different criterion and a different "
                    "frame — training evidence only, never a rate"))

    # --- the record -------------------------------------------------------------------------------
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    man = {
        "study_id": study.study_id, "built": _dt.datetime.now().isoformat(timespec="seconds"), "seed": a.seed,
        "model": backend.name,
        "features": {"n": len(cols), "excluded_ids": sorted(ID_COLS)},
        "upward": {
            "pool_rows": int(len(obd)), "labelled_rows": int(len(up.frame)),
            "labels_by_source": up.counts_by_source(), "accepted": int(up.y.sum()),
            "floats": int(pd.Series(up.groups).nunique()), "cv": up_oof.folds,
            "auc_oof_on_obduction_reviews": up_oof.auc, "rho_oof_on_obduction_reviews": up_oof.rho,
            "obduction_reviews": up_oof.measured_rows,
            "decile_calibration_on_obduction_reviews": up_oof.calibration.round(4).to_dict(orient="index"),
            "model_file": str((mdir / "physical_obduction.joblib").relative_to(REPO)),
        },
        "downward": {
            "pool_rows": int(len(sub)),
            "trained_on": "the companion's reviewed R-detections joined to the active pool by key; verified Category 1/2 -> 1, "
                          "detected-but-not-verified or Category 0 -> 0; the companion's 2024 rule; training only",
            "labelled_rows": int(len(down.frame)), "labels_by_source": down.counts_by_source(), "accepted": int(down.y.sum()),
            "floats": int(pd.Series(down.groups).nunique()), "cv": down_oof.folds,
            "auc_oof_on_companion_labels": down_oof.auc, "rho_oof_on_companion_labels": down_oof.rho,
            "decile_calibration_on_companion_labels": down_oof.calibration.round(4).to_dict(orient="index"),
            "transfer_comparison": {"what": "the upward model through the sign alignment, kept as score_transfer",
                                    "auc_verified_vs_rejected": auc_pos_vs_rej, "auc_verified_vs_pool": auc_pos_vs_pool},
            "alignment": {"flipped": sorted(c for c, r in rule.items() if r == "flip"),
                                                       "swapped_and_negated": sorted(c for c, r in rule.items() if r.startswith("swap")),
                                                       "n_flipped": sum(r == "flip" for r in rule.values()),
                                                       "n_swapped": sum(r.startswith("swap") for r in rule.values())},
            "companion_in_pool": {"verified": int(pos.sum()), "rejected": int(neg0.sum()), "unlabelled_pool": int(unl.sum())},
            "note": "trained on the companion's labels (training only); no study-criterion subduction label exists yet",
            "model_file": str((mdir / "physical_subduction.joblib").relative_to(REPO)),
        },
        "files": {n: hashlib.sha256((sdir / f"{n}.parquet").read_bytes()).hexdigest() for n in ("physical_obduction", "physical_subduction")},
    }
    (MANIFEST_DIR / "SCORES_SHA256").write_text(
        "# provenance of the study's candidate scores -- built by pipeline/scores.py; do not edit\n"
        + json.dumps(man, indent=2) + "\n")
    print(json.dumps({k: v for k, v in man.items() if k in ("upward", "downward")}, indent=2, default=str)[:3000])
    print(f"provenance -> {MANIFEST_DIR / 'SCORES_SHA256'}")


if __name__ == "__main__":
    main()
