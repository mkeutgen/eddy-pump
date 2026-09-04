"""The triage classifier, on made-up data: no cache, no labels, no network.

The score never enters a number, so what these tests hold is not accuracy — it is the four
properties that decide whether a reported score can be believed at all.

1. A fold never splits a float. Two candidates on one float share a sensor, a calibration and an
   ocean. If one trains a model that scores the other, the out-of-fold number measures memory and
   reads as skill. The test builds floats whose label is a property of the FLOAT and checks that no
   row was ever scored by a fold that held its float in training.

2. The model does not see the pool it scores. `fit` and `score` are separate calls on separate
   frames, and a score for a labelled row is replaced by its out-of-fold one before it is written.
   Here: fit on a training set, score a frame that was never fitted on, and get a probability per
   row in the frame's own order.

3. The calibration table is read top to bottom. Its whole use is "the top decile is worth
   labelling"; if `observed` did not rise with the score on data where it plainly should, the table
   would be decoration.

4. A saved model says what it is. A model file with no record of what it was trained on cannot be
   told apart from another one a month later. `save`/`load` carry the model and its provenance
   together, and the provenance names the features, the metrics and where the labels came from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eddy_pump import classifier as K


def toy(n_floats: int = 40, per_float: int = 12, seed: int = 0) -> pd.DataFrame:
    """A pool of candidates whose acceptance is a property of the float, plus noise columns.

    `signal` carries the answer; `noise` carries nothing; `latitude` is a perfect predictor that
    the feature rule must refuse to look at, because a real one would be learning fleet coverage.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_floats):
        wmo = 1900000 + i
        good = i % 2 == 0
        for c in range(per_float):
            rows.append({
                "WMO": wmo, "CYCLE_NUMBER": c, "PRES_ADJUSTED": 100 + 10 * c,
                "pool_id": "toy/physical/obduction", "spec_id": "v1:0000000000000000",
                "latitude": 40.0 if good else -40.0,
                "signal": rng.normal(2.0 if good else -2.0, 0.7),
                "noise": rng.normal(0.0, 1.0),
                "decided": int(good),
            })
    df = pd.DataFrame(rows)
    df["key"] = K.key3(df)
    return df


def toy_labels(pool: pd.DataFrame, source: str = "obduction_reviews") -> pd.DataFrame:
    return pd.DataFrame({"key": pool.key, "decision": pool.decided, "source": source})


@pytest.fixture(scope="module")
def pool():
    return toy()


@pytest.fixture(scope="module")
def data(pool):
    cols = [c for c in K.feature_columns(pool) if c != "decided"]
    return K.training_set(pool, toy_labels(pool), cols)


# --- 1. identity is not evidence ------------------------------------------------------------- #
def test_the_feature_rule_drops_the_key_the_ids_and_the_geography(pool):
    cols = K.feature_columns(pool)
    for identity in ("WMO", "CYCLE_NUMBER", "PRES_ADJUSTED", "pool_id", "spec_id", "latitude"):
        assert identity not in cols, identity
    assert {"signal", "noise"} <= set(cols)


# --- 2. fit and score ------------------------------------------------------------------------- #
def test_fit_then_score_gives_one_probability_per_row_in_the_frames_own_order(pool, data):
    model = K.fit(data, seed=0)
    p = K.score(model, pool, data.features)
    assert p.shape == (len(pool),)
    assert ((p >= 0) & (p <= 1)).all()
    # the answer is a property of the float and the signal column carries it, so a model that
    # learned anything at all separates the two halves
    assert p[pool.decided == 1].mean() > p[pool.decided == 0].mean() + 0.3
    # scoring a shuffled frame returns the shuffled order, not the fitted order
    shuffled = pool.sample(frac=1.0, random_state=7)
    q = K.score(model, shuffled, data.features)
    assert np.allclose(q, p[shuffled.index.to_numpy()])


def test_a_second_backend_slots_in_behind_the_same_three_methods(pool, data):
    """A different model family is a class with `design`, `fit`, `predict` — nothing above changes."""

    class AlwaysHalf:
        name = "a constant, for the test"

        def design(self, frame, features):
            return frame[list(features)].to_numpy(float)

        def fit(self, X, y, *, seed):
            return {"seed": seed}

        def predict(self, model, X):
            return np.full(len(X), 0.5)

    backend = AlwaysHalf()
    model = K.fit(data, seed=3, backend=backend)
    assert model == {"seed": 3}
    assert (K.score(model, pool, data.features, backend=backend) == 0.5).all()


# --- 3. the fold never splits a float ---------------------------------------------------------- #
def test_the_out_of_fold_split_holds_out_whole_floats(data):
    """Recorded from the inside: every fold's test floats are absent from its training floats."""
    from sklearn.model_selection import StratifiedGroupKFold

    backend = K.default_backend()
    X, y, g = backend.design(data.frame, data.features), data.y, data.groups
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, g))
    assert len(splits) == 5
    for train, test in splits:
        assert not (set(g[train]) & set(g[test])), "a float was on both sides of a fold"
    assert sorted(np.concatenate([t for _, t in splits])) == list(range(len(y)))

    oof = K.evaluate(data, seed=0)
    assert oof.probability.shape == (len(y),)
    assert not np.isnan(oof.probability).any(), "every row must come back from some fold"
    assert oof.n_splits == 5 and "grouped by float" in oof.folds
    assert oof.auc > 0.8   # the answer is a float-level property, so held-out floats are learnable


def test_a_label_that_is_pure_float_noise_scores_no_better_than_a_coin():
    """The guard on the guard: if the fold leaked its float, this would score well above chance."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(40):
        wmo = 1900000 + i
        y = int(rng.integers(0, 2))          # the float's answer, unrelated to any feature
        for c in range(12):
            rows.append({"WMO": wmo, "CYCLE_NUMBER": c, "PRES_ADJUSTED": 100 + c,
                         "memorable": float(i), "noise": rng.normal(), "decided": y})
    pool = pd.DataFrame(rows)
    pool["key"] = K.key3(pool)
    cols = [c for c in K.feature_columns(pool) if c != "decided"]
    data = K.training_set(pool, toy_labels(pool), cols)
    assert K.evaluate(data, seed=0).auc < 0.75


def test_the_metrics_are_measured_on_the_probability_sample_only(pool):
    """Half the labels come from another study; the reported number is the study's own half."""
    cols = [c for c in K.feature_columns(pool) if c != "decided"]
    lab = toy_labels(pool)
    borrowed = pool.index % 2 == 1
    lab.loc[borrowed, "source"] = "companion_verified"
    data = K.training_set(pool, lab, cols)
    assert data.is_probability_sample.sum() == (~borrowed).sum()
    oof = K.evaluate(data, seed=0)
    assert oof.measured_rows == int(data.is_probability_sample.sum()) < len(data.frame)
    assert oof.calibration.n.sum() == oof.measured_rows

    # and when no source is a probability sample, the metrics fall back to every row and say so
    only_borrowed = K.training_set(pool, toy_labels(pool, "companion_verified"), cols)
    assert not only_borrowed.is_probability_sample.any()
    assert K.evaluate(only_borrowed, seed=0).measured_rows == len(pool)


# --- 4. the calibration table ------------------------------------------------------------------ #
def test_calibration_is_monotone_when_the_score_is_the_truth():
    y = np.repeat([0, 1], 500)
    p = np.r_[np.linspace(0.0, 0.5, 500), np.linspace(0.5, 1.0, 500)]
    cal = K.calibrate(y, p)
    assert len(cal) == 10
    assert cal.n.sum() == len(y)
    observed = cal.observed.to_numpy()
    assert (np.diff(observed) >= 0).all(), "the accepted fraction must not fall as the score rises"
    assert observed[0] == 0.0 and observed[-1] == 1.0
    assert (np.diff(cal.predicted.to_numpy()) > 0).all()


def test_calibration_survives_a_score_with_ties():
    """`qcut` cannot cut ten equal bins out of three values; it drops bins rather than raising."""
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    p = np.r_[np.full(50, 0.1), np.full(50, 0.9)]
    cal = K.calibrate(y, p)
    assert len(cal) < 10 and cal.n.sum() == len(y)
    assert cal.observed.iloc[-1] > cal.observed.iloc[0]


# --- 5. a saved model says what it is ----------------------------------------------------------- #
def test_save_and_load_round_trip_the_model_and_its_provenance(pool, data, tmp_path):
    model = K.fit(data, seed=0)
    oof = K.evaluate(data, seed=0)
    record = K.manifest(limb="upward", pool_id="toy/physical/obduction", data=data, oof=oof,
                        seed=0, labels_from="a made-up pool, for the tests")
    path = K.save(model, tmp_path / "toy.joblib", record)
    assert path.exists() and path.with_suffix(".json").exists()

    back, got = K.load(path)
    assert got == record
    assert np.allclose(K.score(back, pool, data.features), K.score(model, pool, data.features))

    assert got["features"]["used"] == list(data.features)
    assert "latitude" in got["features"]["excluded_ids"]
    assert got["labels"]["rows"] == len(pool) and got["labels"]["floats"] == 40
    assert got["labels"]["by_source"] == {"obduction_reviews": len(pool)}
    assert got["labels"]["from"] == "a made-up pool, for the tests"
    assert got["out_of_fold"]["auc"] == oof.auc
    assert len(got["out_of_fold"]["calibration"]) == len(oof.calibration)
    assert "HistGradientBoosting" in got["backend"]


def test_the_provenance_is_json_and_readable_without_the_model(pool, data, tmp_path):
    import json

    oof = K.evaluate(data, seed=0)
    record = K.manifest(limb="upward", pool_id="toy/physical/obduction", data=data, oof=oof, seed=0)
    K.save(K.fit(data, seed=0), tmp_path / "toy.joblib", record)
    beside = json.loads((tmp_path / "toy.json").read_text())
    assert beside["limb"] == "upward" and beside["seed"] == 0
    assert beside["out_of_fold"]["n_splits"] == 5
