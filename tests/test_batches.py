"""The labelling batches are probability draws with exact, declared inclusion probabilities —
plan step 6 *(2026-08-27)*.

Two halves. The first proves the DESIGN on synthetic data: the allocation honours its floor and
sums; the one-per-float draw gives every candidate exactly n/N (Monte Carlo) and never two
panels from one cluster; the Hájek estimator built on those probabilities is unbiased over
replications and meets the half-width the solver planned for; controls and score-selected rows
are refused by the rate; the calibration gate passes the reference against itself and fails a
shuffled copy. The second pins the four batches actually built (`data/labels/draws/`) when the
records are present: identities, blindness, hashes, the 42 + 42 calibration rows, and that every
science row's probability is its stratum's n/N.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pins import PINS  # noqa: E402
import yaml

from eddy_pump import batches as B

REPO = Path(__file__).resolve().parents[1]
DRAWS = REPO / "data/labels/draws"


# --------------------------------------------------------------------------------------------- #
# the design, on synthetic data
# --------------------------------------------------------------------------------------------- #
def _synthetic_pool(rng, n_floats=300, n_rows=20_000, dominant=None):
    """A pool with unequal floats (log-normal sizes) and a score that predicts the label."""
    sizes = rng.lognormal(3.0, 1.0, n_floats).astype(int) + 1
    if dominant is not None:
        sizes[0] = dominant
    sizes = (sizes / sizes.sum() * n_rows).astype(int) + 1
    wmo = np.repeat(np.arange(1_000_000, 1_000_000 + n_floats), sizes)
    n = len(wmo)
    cyc = rng.integers(1, 300, n)
    pres = rng.choice(np.arange(200, 1000, 20), n).astype(float)
    score = rng.beta(0.5, 3.0, n)
    df = pd.DataFrame({"WMO": wmo, "CYCLE_NUMBER": cyc, "PRES_ADJUSTED": pres, "score": score})
    df = df.drop_duplicates(["WMO", "CYCLE_NUMBER", "PRES_ADJUSTED"]).reset_index(drop=True)
    # a float effect on top of the score: the clustering the float bootstrap must see
    float_eff = dict(zip(np.unique(df.WMO), rng.normal(0, 0.8, df.WMO.nunique())))
    logit = -3.0 + 6.0 * df.score + df.WMO.map(float_eff)
    df["p_true"] = 1 / (1 + np.exp(-logit))
    return df


def test_the_floor_and_the_sum_of_the_allocation():
    W = np.array([0.1] * 10)
    p = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.2, 0.3, 0.6])
    n_h = B.neyman_allocation(W, p, 400)
    assert n_h.sum() == 400 and (n_h >= 20).all()
    assert n_h[-1] > n_h[0] and n_h[-1] == n_h.max()
    # pure Neyman shares where the floor does not bind
    s = W * np.sqrt(p * (1 - p))
    assert abs(n_h[-1] / 400 - s[-1] / s.sum()) < 0.03


def test_solve_n_meets_the_target_and_not_less():
    W = np.full(10, 0.1)
    p = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.2, 0.3, 0.6])
    p_bar = float((W * p).sum())
    v_t = (0.15 * p_bar / B.Z) ** 2
    n, n_h = B.solve_n(W, p, v_t)
    assert n_h.sum() == n
    assert B.stratified_variance(W, p, n_h) <= v_t
    n_less = B.neyman_allocation(W, p, n - 1)
    assert B.stratified_variance(W, p, n_less) > v_t
    with pytest.raises(ValueError, match="held strata alone"):
        B.solve_n(W, p, v_t, v_held=v_t)


def test_score_deciles_are_equal_sized_and_deterministic():
    rng = np.random.default_rng(1)
    df = _synthetic_pool(rng)
    d1 = B.score_deciles(df)
    d2 = B.score_deciles(df.sample(frac=1, random_state=3)).sort_index()
    assert (d1 == d2).all()
    sizes = d1.value_counts()
    assert sizes.max() - sizes.min() <= 1 and set(sizes.index) == set(range(10))
    assert df.groupby(d1).score.max().is_monotonic_increasing


def test_one_panel_per_float_gives_exactly_n_over_N_and_never_two_per_cluster():
    rng = np.random.default_rng(7)
    df = _synthetic_pool(rng, n_floats=40, n_rows=2_000, dominant=400)
    S = df[B.score_deciles(df) == 9].reset_index(drop=True)   # one stratum, with a dominant float
    n = 25
    counts = pd.Series(0, index=S.index, dtype=int)
    reps = 4000
    per_float_max = 0
    for _ in range(reps):
        d = B.draw_one_per_float(S, n, rng)
        assert len(d) == n and (d.inclusion_probability == n / len(S)).all()
        counts[d.index] += 1
        per_float_max = max(per_float_max, d.groupby("WMO").size().max())
    freq = counts / reps
    # every candidate is drawn with frequency n/N, within Monte Carlo error
    expected = n / len(S)
    se = np.sqrt(expected * (1 - expected) / reps)
    assert abs(freq.mean() - expected) < 1e-9
    assert (np.abs(freq - expected) < 5 * se).mean() > 0.99
    # the dominant float is chunked, so it can carry more than one panel; every other float at most one
    big = S.groupby("WMO").size().idxmax()
    assert S.groupby("WMO").size()[big] > len(S) // n
    assert per_float_max >= 2
    small = S[S.WMO != big]
    # a float that fits one chunk never contributes two panels: check by re-drawing without the big one
    for _ in range(300):
        d = B.draw_one_per_float(small.reset_index(drop=True), n, rng)
        assert d.groupby("WMO").size().max() == 1 or (small.groupby("WMO").size() > len(small) // n).any()


def test_the_stratified_estimator_is_unbiased_and_its_design_variance_tracks_the_truth():
    """Replications of the real design on a synthetic frame with float effects: the stratified mean
    is unbiased; the closed-form design variance tracks the true sampling variance (within Monte
    Carlo error); the naive float bootstrap is never below it (it lets the fixed n_h float)."""
    rng = np.random.default_rng(11)
    df = _synthetic_pool(rng, n_floats=400, n_rows=30_000)
    df["decile"] = B.score_deciles(df)
    df["stratum"] = "open|d" + df.decile.astype(str)
    truth = float(df.p_true.mean())
    lab = df.sample(1500, random_state=2)
    y_lab = (rng.uniform(size=len(lab)) < lab.p_true).astype(int)
    p_row = B.isotonic_acceptance(lab.score.to_numpy(), y_lab.to_numpy(), df.score.to_numpy())
    T = df.assign(_p=p_row).groupby("stratum").agg(N=("score", "size"), p=("_p", "mean"))
    T["W"] = T.N / len(df)
    v_t = (0.15 * truth / B.Z) ** 2
    n, n_h = B.solve_n(T.W.to_numpy(), T.p.to_numpy(), v_t)
    N_h = T.N.to_dict()
    ests, se_design, se_naive, se_strat = [], [], [], []
    for r in range(200):
        parts = [B.draw_one_per_float(df[df.stratum == h], int(nh), rng) for h, nh in zip(T.index, n_h)]
        d = pd.concat(parts)
        y = (rng.uniform(size=len(d)) < d.p_true).astype(int)
        res = B.stratified_rate(y.to_numpy(), d.inclusion_probability.to_numpy(), d.stratum.to_numpy(), d.WMO.to_numpy(), N_h, n_boot=100, seed=r)
        ests.append(res["rate"]); se_design.append(res["se_design"]); se_naive.append(res["se_naive_float_bootstrap"]); se_strat.append(res["se_stratified_bootstrap"])
        assert abs(res["hajek_rate_declared_pi"] - res["rate"]) < 1e-9, "nothing dropped: the Hájek IS the stratified mean"
    ests = np.array(ests)
    mc_se = ests.std(ddof=1) / np.sqrt(len(ests))
    assert abs(ests.mean() - truth) < 4 * mc_se, (ests.mean(), truth, mc_se)
    true_sd = ests.std(ddof=1)
    ratio = np.mean(se_design) / true_sd
    assert 0.8 < ratio < 1.25, f"design SE / true SD = {ratio:.2f}"
    assert 0.8 < np.mean(se_strat) / true_sd < 1.25
    assert np.mean(se_naive) >= 0.95 * np.mean(se_design), "the naive float bootstrap must not undershoot the design variance"
    assert B.Z * true_sd / truth < 0.15 * 1.25


def test_dropping_an_uncertain_verdict_keeps_the_stratum_weight():
    """Three strata, one verdict dropped from the middle one: the stratified mean holds W_h at N_h/N,
    the Hájek with the declared π does not (it re-weights by the surviving rows)."""
    rng = np.random.default_rng(2)
    N_h = {"a": 1000, "b": 1000, "c": 1000}
    rows = []
    for h, n, k in (("a", 10, 0), ("b", 10, 5), ("c", 9, 9)):   # c drew 10, one uncertain dropped
        for i in range(n):
            rows.append({"stratum": h, "y": 1 if i < k else 0, "pi": 10 / 1000, "wmo": rng.integers(1e6)})
    d = pd.DataFrame(rows)
    res = B.stratified_rate(d.y.to_numpy(float), d.pi.to_numpy(), d.stratum.to_numpy(), d.wmo.to_numpy(), N_h, n_boot=50)
    assert abs(res["rate"] - (0 + 0.5 + 1.0) / 3) < 1e-12
    assert res["hajek_rate_declared_pi"] < res["rate"]
    assert res["zero_or_full_count_strata"] == 2 and res["se_design"] > 0


def test_a_rate_refuses_rows_without_an_inclusion_probability():
    y = np.array([1, 0, 1, 0])
    pi = np.array([0.01, 0.01, np.nan, 0.01])
    with pytest.raises(ValueError, match="inclusion probability"):
        B.hajek_rate(y, pi, np.array([1, 2, 3, 4]))
    with pytest.raises(ValueError, match="inclusion probability"):
        B.stratified_rate(y, pi, np.array(["a"] * 4), np.array([1, 2, 3, 4]), {"a": 100}, n_boot=10)


def test_the_calibration_gate():
    rng = np.random.default_rng(5)
    ref = pd.DataFrame({"WMO": np.arange(42) + 5_000_000, "CYCLE_NUMBER": 10.0, "PRES_ADJUSTED": 300.0,
                        "REF_LABEL": np.r_[np.ones(22, int), np.zeros(20, int)]})
    same = ref.rename(columns={"REF_LABEL": "LABEL"})
    rep = B.calibration_report(same, ref)
    assert rep["verdict"] == "PASS" and rep["kappa"] == 1.0 and rep["n_decided"] == 42
    assert rep["kappa_ci95_bootstrap"][0] <= 1.0 and rep["base_rate_pass_region_accepted"][0] <= 22 <= rep["base_rate_pass_region_accepted"][1]
    shuffled = same.assign(LABEL=rng.permutation(same.LABEL.to_numpy()))
    rep2 = B.calibration_report(shuffled, ref)
    assert rep2["verdict"] == "DRIFTED" and rep2["kappa"] < 0.6 and rep2["base_rate_on_target"]
    strict = same.assign(LABEL=np.r_[np.ones(5, int), np.zeros(37, int)])
    rep3 = B.calibration_report(strict, ref)
    assert rep3["verdict"] == "DRIFTED" and not rep3["base_rate_on_target"]
    partial = same.copy()
    partial.loc[:9, "LABEL"] = np.nan
    rep4 = B.calibration_report(partial, ref)
    assert rep4["n_undecided"] == 10 and rep4["verdict"] == "DRIFTED"


def test_the_worksheet_is_blind_and_the_key_is_not():
    rng = np.random.default_rng(3)
    df = _synthetic_pool(rng, n_floats=30, n_rows=500)
    df["design_stratum"] = "open|d9"
    df["candidate_id"] = [f"c{i}" for i in range(len(df))]
    df["LATITUDE"], df["LONGITUDE"], df["TIME"] = 0.0, 0.0, "2020-01-01"
    sci = B.draw_one_per_float(df, 10, rng)
    ctrl = df[~df.index.isin(sci.index)].head(4).assign(stratum=[B.POS_CTRL] * 2 + [B.NEG_CTRL] * 2, src="test", REF_LABEL=[1, 1, 0, 0])
    batch = B.Batch("t", sci, ctrl, "physical_obduction", rng)
    ws, key = batch.assemble()
    assert list(ws.columns) == B.WORKSHEET_COLS and ws.LABEL.isna().all()
    assert not (set(ws.columns) & B.BLIND_FORBIDDEN)
    assert len(ws) == len(key) == 14 and (ws.SAMPLE_ID == np.arange(14)).all()
    assert set(key.stratum) == {B.TARGET, B.POS_CTRL, B.NEG_CTRL}
    assert key.loc[key.stratum == B.TARGET, "inclusion_probability"].notna().all()
    assert key.loc[key.stratum != B.TARGET, "inclusion_probability"].isna().all()
    dup = pd.concat([sci, sci.head(1)])
    with pytest.raises(ValueError, match="twice"):
        B.Batch("t", dup, ctrl, "physical_obduction", rng).assemble()


# --------------------------------------------------------------------------------------------- #
# the batches actually built
# --------------------------------------------------------------------------------------------- #
built = pytest.mark.skipif(not (DRAWS / "DRAWS_SHA256").exists(), reason="run production/build_batches.py")


def _records():
    """The DRAW records only; `<id>.labelled.yaml` is the ingestion's record of a labelled sheet."""
    return {p.stem: yaml.safe_load(p.read_text()) for p in sorted(DRAWS.glob("*.yaml"))
            if not (p.name.endswith(".labelled.yaml") or p.name.endswith(".reference.yaml"))}


@built
def test_the_four_records_match_their_manifest():
    for ln in (DRAWS / "DRAWS_SHA256").read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        sha, name = ln.split("\t")
        assert hashlib.sha256((DRAWS / name).read_bytes()).hexdigest() == sha, name
    R = _records()
    base = {b for b in R if "_pass" not in b}
    assert base == {"calib_obduction_b6", "calib_subduction_v1", "rate_obduction_01", "rate_subduction_01"}
    for b in set(R) - base:   # blind re-labelling copies of an anchor (--repass): same levels, new order
        src = R[b]["reference_of"]
        assert R[b]["derived_from"] == src and R[src]["role"] == "calibration" and R[b]["role"] == "calibration"
        assert R[b]["worksheet"]["rows"] == R[src]["worksheet"]["rows"] == 42


@built
def test_every_record_names_the_study_the_ruled_criterion_and_the_bound_cache():
    from eddy_pump.manifest import load_manifest

    study = load_manifest()
    pools = {p.pool_id: p for p in study.pools}
    for bid, r in _records().items():
        assert r["study_id"] == "net_carbon_v1" and r["criterion_version"] == "phys_net_carbon_v1"
        assert r["criterion_status"] == "ruled"
        assert r["cache"]["fine_grids_sha256"] == study.cache.fine_grids_sha256
        assert r["spec_id"] == pools[r["pool_id"]].spec_id and r["event_type"] == pools[r["pool_id"]].event_type
        assert r["blind"] is True and r["sheet_columns"] == B.WORKSHEET_COLS
        assert len(r["worksheet"]["sha256"]) == 64 and len(r["answer_key"]["sha256"]) == 64
        assert r["worksheet"]["rows"] == r["answer_key"]["rows"]


@built
def test_the_rate_arms_are_probability_draws_with_exact_inclusion_probabilities():
    for bid in ("rate_obduction_01", "rate_subduction_01"):
        r = _records()[bid]
        assert r["role"] == "analysis" and r["decides"] is True and r["sampling"]["design"] == "probability"
        strata = r["strata"]
        assert len(strata) == 10 and {s["design_stratum"] for s in strata} == {f"open|d{i}" for i in range(10)}
        n = sum(s["n"] for s in strata)
        assert n == r["n_science"]
        for s in strata:
            assert s["n"] >= int(np.floor(B.FLOOR_SHARE * n)), s
            assert abs(s["inclusion_probability"] - s["n"] / s["N"]) < 1e-12
            assert 0.005 <= s["p_planned"] <= 0.995
        # the planned precision is the target WITH the float design effect the draw's own clustering
        # implies; the allocation beats proportional allocation and simple random sampling
        t = r["target"]
        assert abs(t["expected_rel_half_width"] - t["rel_half_width"]) < 0.003
        assert t["float_design_effect"] >= 1.0 and abs(t["float_design_effect"] - (1 + (r["panels_per_float_in_draw"] - 1) * t["rho_float"])) < 0.01
        assert t["neyman_vs_proportional_multiplier"] < 0.9 and t["stratified_vs_srs_multiplier"] < 0.7
        assert r["n_controls"] == {"positive": 20, "negative": 20}
        assert r["panels_per_float_in_draw"] < 1.6, "the draw must be spread across floats (the earlier study's was 4.7 per float)"
        assert r["max_panels_one_float"] <= 8, "a float recurs across the ten strata; the record counts it, the design effect prices it"


@built
def test_the_upward_draw_recorded_a_held_region_the_rate_no_longer_credits():
    """The frozen upward draw record still carries a held region (5 strata, 1,713 direct verdicts);
    the rate report now drops it (the open region alone), to be re-sampled under the study's
    criterion. This tests the frozen record's provenance, not the live rate."""
    R = _records()
    up, down = R["rate_obduction_01"], R["rate_subduction_01"]
    assert len(up["held_strata"]) == 5 and sum(h["n_direct"] for h in up["held_strata"]) == 1713
    assert up["target"]["variance_held"] > 0 and down["held_strata"] == [] and down["target"]["variance_held"] == 0
    assert up["pool_id"].endswith("physical/obduction") and down["pool_id"].endswith("physical/subduction")
    assert sum(s["N"] for s in up["strata"]) + sum(h["N"] for h in up["held_strata"]) == PINS["pool_rows"]["physical_obduction"]
    assert sum(s["N"] for s in down["strata"]) == PINS["pool_rows"]["physical_subduction"]
    assert up["controls"]["positive"]["criterion"] == "phys_obduction_letter_b6"
    # the first batch's record still names the re-look survival (223/234) as the reference — the wrong
    # population, kept as history; the ingestion reads the controls against the blind history instead
    assert up["controls"]["positive"]["reference_k"] == 223 and up["controls"]["positive"]["reference_n"] == 234
    assert down["controls"]["positive"]["criterion"] == "phys_companion_2024"
    assert down["controls"]["positive"]["reference_k"] is None, "no Fisher reference until the downward anchor is adjudicated"


@built
def test_the_two_calibration_sets():
    R = _records()
    b6, v1 = R["calib_obduction_b6"], R["calib_subduction_v1"]
    assert b6["role"] == v1["role"] == "calibration" and b6["decides"] is v1["decides"] is False
    assert b6["worksheet"]["rows"] == v1["worksheet"]["rows"] == 42
    ref = pd.DataFrame(b6["reference_rows"])
    assert len(ref) == 42 and int(ref.REF_LABEL.sum()) == 18, "12 clear TP + 6 borderline TP in the frozen reference"
    assert ref.tier.value_counts().to_dict() == {"borderline": 18, "clear_FP": 12, "clear_TP": 12}
    assert b6["reference"]["sha256"].startswith("8b3af4d3bfe130c3")
    assert "BLANK" in v1["status"] and "consensus" in v1["status"]
    assert v1["composition"]["clear_TP"].startswith("12 ")


@built
def test_the_sheets_on_disk_when_present():
    for bid, r in _records().items():
        wp, kp = Path(r["worksheet"]["path"]), Path(r["answer_key"]["path"])
        if not wp.exists():
            pytest.skip("sheets not on this machine")
        ws, key = pd.read_csv(wp), pd.read_csv(kp)
        labelled = ws.LABEL.notna().any()
        # the key is sealed: its hash never moves. The worksheet's hash is the BLANK sheet's until the
        # reviewer writes into it; a labelled sheet is hashed by the ingestion record instead.
        assert hashlib.sha256(kp.read_bytes()).hexdigest() == r["answer_key"]["sha256"], bid
        if not labelled:
            assert hashlib.sha256(wp.read_bytes()).hexdigest() == r["worksheet"]["sha256"], bid
        assert list(ws.columns) == B.WORKSHEET_COLS
        assert not ws.duplicated(B.KEYS).any()
        assert (ws[B.KEYS].round(0).to_numpy() == key.sort_values("SAMPLE_ID")[B.KEYS].round(0).to_numpy()).all()
        assert (ws.SAMPLE_ID.to_numpy() == np.arange(len(ws))).all()
        assert (key.SAMPLE_ID.to_numpy() == ws.SAMPLE_ID.to_numpy()).all()
        if bid.startswith("rate_"):
            sci = key[key.stratum == B.TARGET]
            assert len(sci) == r["n_science"]
            by = sci.groupby("src").inclusion_probability.agg(["nunique", "size", "first"])
            assert (by["nunique"] == 1).all()
            planned = {s["design_stratum"]: (s["n"], s["inclusion_probability"]) for s in r["strata"]}
            for src, row in by.iterrows():
                assert row["size"] == planned[src][0] and abs(row["first"] - planned[src][1]) < 1e-12
            assert abs((1 / sci.inclusion_probability).sum() - sum(s["N"] for s in r["strata"])) < 1e-6
            ctrl = key[key.stratum.isin(B.CONTROL_STRATA)]
            assert len(ctrl) == 40 and ctrl.inclusion_probability.isna().all() and ctrl.REF_LABEL.isin([0, 1]).all()
        elif bid == "calib_obduction_b6":
            assert key.REF_LABEL.isin([0, 1]).all() and int(key.REF_LABEL.sum()) == 18
        else:
            assert key.REF_LABEL.isna().all() and key.tier.value_counts().to_dict() == {"borderline": 18, "clear_FP": 12, "clear_TP": 12}
            assert key.WMO.is_unique


@built
def test_an_ingested_analysis_batch_records_its_anchor_and_reads_its_controls_on_standing_verdicts():
    lab = {p.name.replace(".labelled.yaml", ""): yaml.safe_load(p.read_text()) for p in DRAWS.glob("*.labelled.yaml")}
    if "rate_obduction_01" not in lab:
        pytest.skip("rate_obduction_01 not ingested on this machine")
    l = lab["rate_obduction_01"]
    assert l["anchor"]["batch_id"] == "calib_obduction_b6" and l["anchor"]["worksheet_mtime"] < l["worksheet_mtime"]
    pc = l["session"]["controls"]["pos_ctrl"]
    assert pc["n"] == 20 and pc["accepted"] == 14 and pc["overturned_in_ledger"] == 4
    assert pc["standing"] == {"n": 16, "accepted": 14, "ci95": pc["standing"]["ci95"]}
    bh = pc["blind_history"]
    assert bh["n"] < 1000, "the honest blind history is one pair per candidate and sheet, no snapshot twins"
    assert 0.6 < bh["k"] / bh["n"] < 0.95
    assert pc["fisher_p_vs_blind_history_standing"] > 0.05
    nc = l["session"]["controls"]["neg_ctrl"]
    assert nc["with_score_above_0p5"]["n"] == 2 and nc["with_score_above_0p5"]["accepted"] == 2
    assert l["session"]["target_halves"]["first"]["accept"] > l["session"]["target_halves"]["second"]["accept"]
    cal = lab["calib_obduction_b6"]["session"]
    assert cal["verdict"] == "PASS" and cal["kappa_ci95_bootstrap"][0] < 0.6 < cal["kappa_ci95_bootstrap"][1]


@built
def test_the_rate_status_uses_the_design_variance_and_names_levels():
    p = REPO / "data/labels/audit/rate_status.csv"
    if not p.exists():
        pytest.skip("no rate report on this machine")
    T = pd.read_csv(p)
    r = T[T.pool_id == "net_carbon_v1/physical/obduction"].iloc[0]
    assert r.status == "measured" and "candidate levels" in r.denominator
    # the upward rate is the open region alone (the held region is dropped, awaiting a fresh draw)
    assert abs(r.pool_rate - r.open_rate) < 1e-9 and abs(r.open_rate - 0.1287) < 0.001
    assert r.open_se_design < r.open_se_naive_float_bootstrap
    assert abs(r.open_se_design - 0.0114) < 0.001, "design-based SE with the three zero-count deciles floored"
    assert 0.17 <= r.pool_half_width_95_rel < 0.18
    assert "accepted_levels_estimated" in T.columns and "events" not in " ".join(T.columns)
    assert r.drift_band_first_half > r.pool_rate > r.drift_band_second_half


@built
def test_the_downward_anchor_is_frozen_from_one_pass_and_the_criterion_points_at_it():
    rp = DRAWS / "calib_subduction_v1.reference.yaml"
    if not rp.exists():
        pytest.skip("the downward anchor is not frozen on this machine")
    from eddy_pump.criteria import load_criteria

    fr = yaml.safe_load(rp.read_text())
    ref = pd.DataFrame(fr["rows"])
    assert len(ref) == 42 and ref.REF_LABEL.isin([0, 1]).all() and int(ref.REF_LABEL.sum()) == 17
    assert fr["base_rate"] == "17/42" and "consensus" in fr["how"] and "adjudicated" in fr["how"]
    assert [h.get("base_rate") for h in fr["history"][:2]] == ["15/42", "20/42"]
    assert len(fr["history"][2]["rulings"]) == 5 and sum(r["ruled"] for r in fr["history"][2]["rulings"]) == 2
    assert ref.WMO.is_unique
    # 8 of the 12 companion-verified, top-score panels pass clause 4 after adjudication (7 on the first pass)
    top = ref[ref.tier == "clear_TP"]
    assert len(top) == 12 and int(top.REF_LABEL.sum()) == 8
    anchors = load_criteria()["phys_net_carbon_v1"].raw["anchors"]
    assert anchors["subduction"] == "data/labels/draws/calib_subduction_v1.reference.yaml"
    for b, k in (("calib_subduction_v1", 0.85), ("calib_subduction_v1_pass2", 0.80)):
        lab = yaml.safe_load((DRAWS / f"{b}.labelled.yaml").read_text())
        assert not lab["session"].get("reference_is_this_pass") and lab["session"]["kappa"] > k, b
        assert lab["labelled_sheet"]["sha256"] in {h.get("sheet_sha256") for h in fr["history"]}
