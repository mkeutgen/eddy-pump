"""The triage classifier: what it learns from, how honestly it is measured, what it scores.

reads  the label table (`eddy_pump.labels`), data/external/manually_verified_physical_subd_events.csv,
       $GLOBARGO_DATA/detected_physical_subd_events.csv
writes nothing on its own; `save()` writes a fitted model and its provenance where the caller says

The score decides which panels a human sees first and nothing else. It never enters a rate, a
census or a flux, so nothing here needs to be unbiased — it needs to be honestly measured, which
is what the out-of-fold split by float is for. `pipeline/scores.py` is the thin script that calls
this module; every choice a reader would argue with is here.

Four things, in the order a run does them
-----------------------------------------
1. **Where the labels come from.** One function per source, each returning the same three columns
   (`key`, `decision`, `source`): :func:`upward_labels` reads the study's own obduction reviews;
   :func:`downward_labels` reads the earlier subduction study's reviewed detections. A third
   source is coming — the ocean model's own truth, once the model experiment runs — and it arrives
   as a third function of this shape. None of them may be added to `labels.analysis_sample`: that
   door is for the study's own probability samples, and a rate reads only through it.
2. **Fitting.** :func:`fit` on a :class:`TrainingSet` — the pool's feature rows joined to the
   labels, carrying the float each row belongs to.
3. **Measuring.** :func:`evaluate` refits five times, holding out whole floats, so no row is ever
   scored by a model that has seen its own float; and it reports the area under the curve, the
   correlation and the decile calibration on the probability sample ONLY. A label that was picked
   because its score was high measures the picker, not the classifier.
4. **Scoring, and keeping the model.** :func:`score` for the whole pool; :func:`save` /
   :func:`load` for the fitted model and the provenance that says what it was made from.

A second backend
----------------
:class:`GradientBoostedTrees` is today's model: a table of features per candidate. A deep model
over the profile windows themselves is one more class implementing the same three methods —
`design` turns the candidate rows into whatever the model eats (for a window model, the residual
grids read through argopod's `CachedProfileProvider`), `fit` and `predict` do the rest. Everything
above it — the labels, the fold rule, the calibration, the provenance — stays as it is.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

from .manifest import GLOBARGO_DATA, REPO_ROOT

__all__ = [
    "KEYS",
    "ID_COLS",
    "OBDUCTION_POOL",
    "CRITERION",
    "COMPANION",
    "COMPANION_DETECTIONS",
    "key3",
    "feature_columns",
    "upward_labels",
    "downward_labels",
    "TrainingSet",
    "training_set",
    "Backend",
    "GradientBoostedTrees",
    "default_backend",
    "fit",
    "score",
    "evaluate",
    "calibrate",
    "OutOfFold",
    "alignment_rule",
    "apply_alignment",
    "manifest",
    "save",
    "load",
]

#: The event key. The fourth term, the pool, is carried by the frame, not by the key.
KEYS = ["WMO", "CYCLE_NUMBER", "PRES_ADJUSTED"]

#: Columns that are identity, not evidence. Geography is in here on purpose: a score that knows
#: where the floats are learns which oceans get labelled, and would then rank panels by fleet
#: coverage while looking like it ranked them by physics.
ID_COLS = set(KEYS) | {"EVENT_TYPE", "pool_id", "spec_id", "latitude", "longitude", "abs_latitude"}

#: The upward pool whose reviews train the upward model, and the criterion those reviews were
#: labelled under.
OBDUCTION_POOL = "net_carbon_v1/physical/obduction"
CRITERION = "phys_net_carbon_v1"

#: The earlier subduction study's two tables: every detection it made, and the verdict a human
#: gave each one. A different criterion and a different frame, so training evidence only.
COMPANION = REPO_ROOT / "data/external/manually_verified_physical_subd_events.csv"
COMPANION_DETECTIONS = GLOBARGO_DATA / "detected_physical_subd_events.csv"


# --------------------------------------------------------------------------- #
# the frame
# --------------------------------------------------------------------------- #
def key3(df: pd.DataFrame) -> pd.Series:
    """The event key of every row, as a tuple, so two tables can be joined on one column."""
    return pd.Series(
        list(zip(df.WMO.astype(int), df.CYCLE_NUMBER.round().astype(int),
                 df.PRES_ADJUSTED.round().astype(int))),
        index=df.index)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The numeric columns a model may read: everything that is not identity (see :data:`ID_COLS`)."""
    return [c for c in df.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------------- #
# where the labels come from — one function per source, all the same three columns
# --------------------------------------------------------------------------- #
def upward_labels(pool_keys: set) -> pd.DataFrame:
    """One decision per candidate key of the active obduction pool, from the study's own reviews."""
    from . import labels as L  # local: importing the label table must not be the price of `import eddy_pump`

    lab = L.analysis_sample(OBDUCTION_POOL, CRITERION)[
        ["key_wmo", "key_cycle", "key_pres", "decision"]].assign(source="obduction_reviews")
    lab["key"] = list(zip(lab.key_wmo.astype(int), lab.key_cycle.astype(int), lab.key_pres.astype(int)))
    lab = lab.drop_duplicates("key")
    lab = lab[lab.key.isin(pool_keys)]
    return lab[["key", "decision", "source"]].reset_index(drop=True)


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
                             np.where(lab.Category == 0, "companion_rejected",
                                      "companion_detected_not_verified"))
    lab = lab[lab.key.isin(pool_keys)]
    return lab[["key", "decision", "source"]].reset_index(drop=True)


#: Which sources are a probability sample of the study, and so may carry a measured rate or a
#: calibration a reader is asked to believe. Everything else is training evidence.
PROBABILITY_SOURCES = frozenset({"obduction_reviews"})


# --------------------------------------------------------------------------- #
# the training set
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainingSet:
    """The labelled rows of one pool: the features, the decision, and the float each row is from.

    The float matters as much as the decision. Two candidates on one float share a sensor, a
    calibration and an ocean; a fold that splits them measures memory, not skill.
    """

    frame: pd.DataFrame
    features: tuple[str, ...]
    label_column: str = "decision"
    group_column: str = "WMO"
    source_column: str = "source"

    @property
    def y(self) -> np.ndarray:
        return self.frame[self.label_column].to_numpy(int)

    @property
    def groups(self) -> np.ndarray:
        return self.frame[self.group_column].to_numpy(int)

    @property
    def source(self) -> np.ndarray:
        return self.frame[self.source_column].to_numpy()

    @property
    def is_probability_sample(self) -> np.ndarray:
        """Row by row: is this label from a probability sample of the study?"""
        return np.isin(self.source, list(PROBABILITY_SOURCES))

    def counts_by_source(self) -> dict[str, int]:
        return self.frame[self.source_column].value_counts().to_dict()


def training_set(pool: pd.DataFrame, labels: pd.DataFrame, features: Sequence[str]) -> TrainingSet:
    """The pool's rows that carry a label, in the pool's own order, with the label attached."""
    return TrainingSet(frame=pool.merge(labels, on="key", how="inner"), features=tuple(features))


# --------------------------------------------------------------------------- #
# the backend — one class per kind of model, three methods each
# --------------------------------------------------------------------------- #
class Backend(Protocol):
    """A model family. `design` says what the rows look like to it; `fit` and `predict` do the rest."""

    name: str

    def design(self, frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
        """The rows as this model eats them."""
        ...

    def fit(self, X: np.ndarray, y: np.ndarray, *, seed: int) -> Any:
        ...

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """P(accepted) for each row, in the row order it was handed."""
        ...


@dataclass(frozen=True)
class GradientBoostedTrees:
    """Boosted trees over the per-candidate feature table. Today's model.

    The knobs are the ones the study has always used. They are fields rather than literals so a
    saved model's provenance can say what it was fitted under, and so a second setting is a value
    rather than an edit.
    """

    max_iter: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    l2_regularization: float = 1.0

    @property
    def name(self) -> str:
        return (f"sklearn HistGradientBoostingClassifier(max_iter={self.max_iter}, "
                f"lr={self.learning_rate}, leaves={self.max_leaf_nodes}, "
                f"l2={self.l2_regularization})")

    def design(self, frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
        return frame[list(features)].to_numpy(float)

    def fit(self, X: np.ndarray, y: np.ndarray, *, seed: int):
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=self.max_iter, learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes, l2_regularization=self.l2_regularization,
            random_state=seed).fit(X, y)

    def predict(self, model, X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)[:, 1]


def default_backend() -> GradientBoostedTrees:
    """The backend a caller gets when it does not name one."""
    return GradientBoostedTrees()


# --------------------------------------------------------------------------- #
# fit, measure, score
# --------------------------------------------------------------------------- #
def fit(data: TrainingSet, *, seed: int = 0, backend: Backend | None = None):
    """One model on every labelled row. What :func:`score` uses; never what a metric comes from."""
    backend = default_backend() if backend is None else backend
    return backend.fit(backend.design(data.frame, data.features), data.y, seed=seed)


def score(model, frame: pd.DataFrame, features: Sequence[str], *,
          backend: Backend | None = None) -> np.ndarray:
    """P(accepted) for every row of `frame`, in its own order."""
    backend = default_backend() if backend is None else backend
    return backend.predict(model, backend.design(frame, features))


def calibrate(y: np.ndarray, p: np.ndarray, *, bins: int = 10) -> pd.DataFrame:
    """The decile table: in each tenth of the score, how many rows, how many were accepted, and
    what the score said. A model is calibrated when `observed` and `predicted` track each other,
    and it is useful when `observed` rises down the table even if it does not."""
    dec = pd.qcut(pd.Series(p), bins, labels=False, duplicates="drop")
    return pd.DataFrame({"d": dec, "y": y, "s": p}).groupby("d").agg(
        n=("y", "size"), observed=("y", "mean"), predicted=("s", "mean"))


@dataclass(frozen=True)
class OutOfFold:
    """What five refits on held-out floats say about a model, and on which rows it was said."""

    probability: np.ndarray
    measured_on: np.ndarray
    auc: float
    rho: float
    calibration: pd.DataFrame
    folds: str
    n_splits: int
    measured_rows: int

    def as_record(self) -> dict[str, Any]:
        """The provenance shape: the numbers and the calibration table, nothing else."""
        return {"cv": self.folds, "n_splits": self.n_splits, "auc": self.auc, "rho": self.rho,
                "measured_rows": self.measured_rows,
                "calibration": self.calibration.round(4).to_dict(orient="index")}


def evaluate(data: TrainingSet, *, seed: int = 0, n_splits: int = 5,
             backend: Backend | None = None,
             metrics_on: np.ndarray | None = None) -> OutOfFold:
    """Five refits, each holding out whole floats; the metrics on the probability sample only.

    `metrics_on` is a row mask naming the rows a metric may be computed on. Left out, it defaults
    to the probability-sample rows, and to every row when no source is one — which is the
    downward limb's situation today, and is why the downward numbers are labelled by their source
    wherever they are reported.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    backend = default_backend() if backend is None else backend
    X, y, groups = backend.design(data.frame, data.features), data.y, data.groups
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                       random_state=seed).split(X, y, groups):
        oof[te] = backend.predict(backend.fit(X[tr], y[tr], seed=seed), X[te])

    if metrics_on is None:
        probability = data.is_probability_sample
        mask = probability if probability.any() else np.ones(len(y), bool)
    else:
        mask = np.asarray(metrics_on, bool)
    return OutOfFold(
        probability=oof, measured_on=mask,
        auc=float(roc_auc_score(y[mask], oof[mask])),
        rho=float(np.corrcoef(oof[mask], y[mask])[0, 1]),
        calibration=calibrate(y[mask], oof[mask]),
        folds=f"StratifiedGroupKFold({n_splits}) grouped by float",
        n_splits=n_splits, measured_rows=int(mask.sum()))


# --------------------------------------------------------------------------- #
# transferring the upward model to the downward limb
# --------------------------------------------------------------------------- #
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
    # mirrors the obduction maximum -- swap the pair and negate, not flip each in place
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


# --------------------------------------------------------------------------- #
# keeping the fitted model
# --------------------------------------------------------------------------- #
def manifest(*, limb: str, pool_id: str, data: TrainingSet, oof: OutOfFold, seed: int,
             backend: Backend | None = None, labels_from: str = "",
             extra: dict | None = None) -> dict:
    """What a saved model has to be able to say for itself.

    Which features it read, in order; how it was measured and on how many rows; what the deciles
    looked like; where every label came from and how many came from each place. Enough to tell
    two saved models apart without opening either.
    """
    backend = default_backend() if backend is None else backend
    record = {
        "limb": limb,
        "pool_id": pool_id,
        "built": _dt.datetime.now().isoformat(timespec="seconds"),
        "backend": backend.name,
        "seed": seed,
        "features": {"n": len(data.features), "used": list(data.features),
                     "excluded_ids": sorted(ID_COLS)},
        "labels": {
            "rows": int(len(data.frame)), "accepted": int(data.y.sum()),
            "floats": int(pd.Series(data.groups).nunique()),
            "by_source": data.counts_by_source(),
            "probability_sample_rows": int(data.is_probability_sample.sum()),
            "from": labels_from,
        },
        "out_of_fold": oof.as_record(),
    }
    if extra:
        record.update(extra)
    return record


def save(model, path: str | Path, record: dict) -> Path:
    """The fitted model and its provenance. The provenance is also written beside it, readable."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "manifest": record}, path)
    path.with_suffix(".json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    return path


def load(path: str | Path) -> tuple[Any, dict]:
    """The fitted model and the provenance it was saved with."""
    import joblib

    payload = joblib.load(Path(path))
    return payload["model"], payload["manifest"]
