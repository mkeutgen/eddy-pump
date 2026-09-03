"""Score every candidate of the two physical pools; the score only orders what a human sees.

reads  results/net_carbon_v1/features/physical_{obduction,subduction}.parquet, the label table
       (eddy_pump.labels), data/external/manually_verified_physical_subd_events.csv,
       $GLOBARGO_DATA/detected_physical_subd_events.csv
writes results/net_carbon_v1/scores/physical_{obduction,subduction}.parquet (key, score, out-of-fold flag)
       data/features/net_carbon_v1/SCORES_SHA256 (what was trained on, the honest AUC, hashes)
Upward: trained on the study's own obduction labels, metrics out-of-fold with folds by float.
Downward: trained on the earlier study's (the companion's) reviewed detections joined to our features;
the sign-flipped upward model is kept as a comparison. Never a label, never in a rate.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eddy_pump import labels as L  # noqa: E402
from eddy_pump.manifest import GLOBARGO_DATA, load_manifest  # noqa: E402

KEYS = ["WMO", "CYCLE_NUMBER", "PRES_ADJUSTED"]
ID_COLS = set(KEYS) | {"EVENT_TYPE", "pool_id", "spec_id", "latitude", "longitude", "abs_latitude"}
OBDUCTION_POOL = "net_carbon_v1/physical/obduction"
CRITERION = "phys_net_carbon_v1"
COMPANION = REPO / "data/external/manually_verified_physical_subd_events.csv"
COMPANION_DETECTIONS = GLOBARGO_DATA / "detected_physical_subd_events.csv"
MANIFEST_DIR = REPO / "data/features/net_carbon_v1"


def key3(df: pd.DataFrame) -> pd.Series:
    return pd.Series(list(zip(df.WMO.astype(int), df.CYCLE_NUMBER.round().astype(int), df.PRES_ADJUSTED.round().astype(int))), index=df.index)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns; geography is excluded so the score cannot learn where floats are."""
    return [c for c in df.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]


def upward_labels(pool_keys: set) -> pd.DataFrame:
    """One decision per candidate key of the active obduction pool, from the study's own reviews."""
    lab = L.analysis_sample(OBDUCTION_POOL, CRITERION)[["key_wmo", "key_cycle", "key_pres", "decision"]].assign(source="obduction_reviews")
    lab["key"] = list(zip(lab.key_wmo.astype(int), lab.key_cycle.astype(int), lab.key_pres.astype(int)))
    lab = lab.drop_duplicates("key")
    lab = lab[lab.key.isin(pool_keys)]
    return lab[["key", "decision", "source"]].reset_index(drop=True)


def grouped_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int = 5):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedGroupKFold

    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def fit_full(X: np.ndarray, y: np.ndarray, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                       l2_regularization=1.0, random_state=seed)
    return m.fit(X, y)


def downward_labels(pool_keys: set) -> pd.DataFrame:
    """The companion's reviewed detections joined to the active subduction pool, by key.

    Every companion R-detection was reviewed: a key in the verified file with Category 1/2 is a
    positive; a detection absent from it, or present with Category 0, is a negative; Category 3
    (unsure) is dropped. 2024 criterion, 1.96 frame -- training only."""
    det = pd.read_csv(COMPANION_DETECTIONS)
    det["key"] = key3(det)
    ver = pd.read_csv(COMPANION)
    ver["key"] = key3(ver)
    cat = dict(zip(ver.key, ver.Category))
    lab = det[["key", "WMO"]].drop_duplicates("key").copy()
    lab["Category"] = lab.key.map(cat)
    lab = lab[lab.Category.isna() | lab.Category.isin([0, 1, 2])]
    lab["decision"] = lab.Category.isin([1, 2]).astype(int)
    lab["source"] = np.where(lab.Category.isin([1, 2]), "companion_verified",
                             np.where(lab.Category == 0, "companion_rejected", "companion_detected_not_verified"))
    lab = lab[lab.key.isin(pool_keys)]
    return lab[["key", "decision", "source"]].reset_index(drop=True)


def alignment_rule(obd: pd.DataFrame, sub: pd.DataFrame, cols: list[str]) -> dict[str, str]:
    """For every feature: 'flip' if the subduction distribution mirrors the obduction one, else 'keep'.

    A feature whose sign follows the AOU anomaly has, on the downward limb, the negated
    distribution of the upward limb. Compare the subduction quantiles with the obduction
    quantiles and with their negation; the closer wins. Features that never change sign, and
    the AOU-free ones, come out 'keep' on their own.
    """
    qs = np.linspace(0.05, 0.95, 19)

    def q(x):
        return np.quantile(x.dropna().to_numpy(), qs)

    def mirrors(target: np.ndarray, source: np.ndarray) -> bool:
        """Does `target` look like the negation of `source`, clearly more than like `source` itself?"""
        d_same = np.mean(np.abs(target - source))
        d_flip = np.mean(np.abs(target - (-source[::-1])))
        scale = np.mean(np.abs(source)) + 1e-12
        return d_flip < 0.5 * d_same and d_flip / scale < 0.25

    rule = {}
    done = set()
    # min/max pairs: on the downward limb the anomaly is a minimum, so the subduction minimum
    # mirrors the obduction MAXIMUM -- swap the pair and negate, not flip each in place
    for c in cols:
        if c.endswith("_min_res") and c[:-8] + "_max_res" in cols:
            lo, hi = c, c[:-8] + "_max_res"
            if len(obd[lo].dropna()) >= 100 and len(sub[lo].dropna()) >= 100 and \
                    mirrors(q(sub[lo]), q(obd[hi])) and mirrors(q(sub[hi]), q(obd[lo])):
                rule[lo], rule[hi] = f"swap_negate:{hi}", f"swap_negate:{lo}"
                done.update((lo, hi))
    for c in cols:
        if c in done:
            continue
        if len(obd[c].dropna()) < 100 or len(sub[c].dropna()) < 100:
            rule[c] = "keep"
            continue
        rule[c] = "flip" if mirrors(q(sub[c]), q(obd[c])) else "keep"
    return rule


def apply_alignment(df: pd.DataFrame, rule: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for c, r in rule.items():
        if r == "flip":
            out[c] = -df[c]
        elif r.startswith("swap_negate:"):
            out[c] = -df[r.split(":", 1)[1]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    study = load_manifest()
    fdir = study.output.resolve("features")
    sdir = study.output.resolve("scores")
    sdir.mkdir(parents=True, exist_ok=True)
    obd = pd.read_parquet(fdir / "physical_obduction.parquet")
    sub = pd.read_parquet(fdir / "physical_subduction.parquet")
    obd["key"], sub["key"] = key3(obd), key3(sub)
    cols = feature_columns(obd)
    assert cols == feature_columns(sub), "the two pools must carry the same feature columns"

    # --- the upward limb: labels, honest out-of-fold, then the full fit -------------------------
    lab = upward_labels(set(obd.key))
    T = obd.merge(lab, on="key", how="inner")
    X, y, g = T[cols].to_numpy(float), T.decision.to_numpy(int), T.WMO.to_numpy(int)
    oof = grouped_oof(X, y, g, a.seed)
    from sklearn.metrics import roc_auc_score
    uni = (T.source == "obduction_reviews").to_numpy()
    auc_uniform = float(roc_auc_score(y[uni], oof[uni]))
    rho_uniform = float(np.corrcoef(oof[uni], y[uni])[0, 1])
    dec = pd.qcut(pd.Series(oof[uni]), 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"d": dec, "y": y[uni], "s": oof[uni]}).groupby("d").agg(n=("y", "size"), observed=("y", "mean"), predicted=("s", "mean"))
    model = fit_full(X, y, a.seed)
    obd["score"] = model.predict_proba(obd[cols].to_numpy(float))[:, 1]
    obd["score_is_oof"] = False
    idx = obd.index[obd.key.isin(T.key)]
    oof_by_key = dict(zip(T.key, oof))
    obd.loc[idx, "score"] = [oof_by_key[k] for k in obd.loc[idx, "key"]]
    obd.loc[idx, "score_is_oof"] = True
    obd[KEYS + ["pool_id", "spec_id", "score", "score_is_oof"]].to_parquet(sdir / "physical_obduction.parquet", index=False)

    # --- the downward limb: a model trained on the companion's reviewed detections ----------------
    dlab = downward_labels(set(sub.key))
    D = sub.merge(dlab, on="key", how="inner")
    Xd, yd, gd = D[cols].to_numpy(float), D.decision.to_numpy(int), D.WMO.to_numpy(int)
    oof_d = grouped_oof(Xd, yd, gd, a.seed)
    auc_d = float(roc_auc_score(yd, oof_d))
    rho_d = float(np.corrcoef(oof_d, yd)[0, 1])
    dec_d = pd.qcut(pd.Series(oof_d), 10, labels=False, duplicates="drop")
    cal_d = pd.DataFrame({"d": dec_d, "y": yd, "s": oof_d}).groupby("d").agg(n=("y", "size"), observed=("y", "mean"), predicted=("s", "mean"))
    model_d = fit_full(Xd, yd, a.seed)
    sub["score"] = model_d.predict_proba(sub[cols].to_numpy(float))[:, 1]
    sub["score_is_oof"] = False
    idx_d = sub.index[sub.key.isin(D.key)]
    oof_by_key_d = dict(zip(D.key, oof_d))
    sub.loc[idx_d, "score"] = [oof_by_key_d[k] for k in sub.loc[idx_d, "key"]]
    sub.loc[idx_d, "score_is_oof"] = True

    # the upward model transferred through the data-decided alignment: a comparison column and
    # the fallback if the companion's labels were ever withdrawn
    rule = alignment_rule(obd, sub, cols)
    subA = apply_alignment(sub, rule)
    sub["score_transfer"] = model.predict_proba(subA[cols].to_numpy(float))[:, 1]
    comp = pd.read_csv(COMPANION)
    comp["key"] = key3(comp)
    cs = sub.merge(comp[["key", "Category"]].drop_duplicates("key"), on="key", how="left")
    pos = cs.Category.isin([1, 2]).to_numpy()
    neg0 = (cs.Category == 0).to_numpy()
    unl = cs.Category.isna().to_numpy()
    auc_pos_vs_rej = float(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(neg0.sum())], np.r_[cs.score_transfer[pos], cs.score_transfer[neg0]])) if neg0.sum() and pos.sum() else float("nan")
    auc_pos_vs_pool = float(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(unl.sum())], np.r_[cs.score_transfer[pos], cs.score_transfer[unl]])) if pos.sum() else float("nan")
    sub[KEYS + ["pool_id", "spec_id", "score", "score_is_oof", "score_transfer"]].to_parquet(sdir / "physical_subduction.parquet", index=False)

    # --- the manifest -----------------------------------------------------------------------------
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    man = {
        "study_id": study.study_id, "built": _dt.datetime.now().isoformat(timespec="seconds"), "seed": a.seed,
        "model": "sklearn HistGradientBoostingClassifier(max_iter=300, lr=0.05, leaves=31, l2=1.0)",
        "features": {"n": len(cols), "excluded_ids": sorted(ID_COLS)},
        "upward": {
            "pool_rows": int(len(obd)), "labelled_rows": int(len(T)),
            "labels_by_source": T.source.value_counts().to_dict(), "accepted": int(y.sum()),
            "floats": int(T.WMO.nunique()), "cv": "StratifiedGroupKFold(5) grouped by float",
            "auc_oof_on_obduction_reviews": auc_uniform, "rho_oof_on_obduction_reviews": rho_uniform,
            "obduction_reviews": int(uni.sum()),
            "decile_calibration_on_obduction_reviews": cal.round(4).to_dict(orient="index"),
        },
        "downward": {
            "pool_rows": int(len(sub)),
            "trained_on": "the companion's reviewed R-detections joined to the active pool by key; verified Category 1/2 -> 1, "
                          "detected-but-not-verified or Category 0 -> 0; the companion's 2024 rule; training only",
            "labelled_rows": int(len(D)), "labels_by_source": D.source.value_counts().to_dict(), "accepted": int(yd.sum()),
            "floats": int(D.WMO.nunique()), "cv": "StratifiedGroupKFold(5) grouped by float",
            "auc_oof_on_companion_labels": auc_d, "rho_oof_on_companion_labels": rho_d,
            "decile_calibration_on_companion_labels": cal_d.round(4).to_dict(orient="index"),
            "transfer_comparison": {"what": "the upward model through the sign alignment, kept as score_transfer",
                                    "auc_verified_vs_rejected": auc_pos_vs_rej, "auc_verified_vs_pool": auc_pos_vs_pool},
            "alignment": {"flipped": sorted(c for c, r in rule.items() if r == "flip"),
                                                       "swapped_and_negated": sorted(c for c, r in rule.items() if r.startswith("swap")),
                                                       "n_flipped": sum(r == "flip" for r in rule.values()),
                                                       "n_swapped": sum(r.startswith("swap") for r in rule.values())},
            "companion_in_pool": {"verified": int(pos.sum()), "rejected": int(neg0.sum()), "unlabelled_pool": int(unl.sum())},
            "note": "trained on the companion's labels (training only); no study-criterion subduction label exists yet",
        },
        "files": {n: hashlib.sha256((sdir / f"{n}.parquet").read_bytes()).hexdigest() for n in ("physical_obduction", "physical_subduction")},
    }
    (MANIFEST_DIR / "SCORES_SHA256").write_text(
        "# provenance of the study's candidate scores -- built by pipeline/scores.py; do not edit\n"
        + json.dumps(man, indent=2) + "\n")
    print(json.dumps({k: v for k, v in man.items() if k in ("upward", "downward")}, indent=2, default=str)[:3000])
    print(f"manifest -> {MANIFEST_DIR / 'SCORES_SHA256'}")


if __name__ == "__main__":
    main()
