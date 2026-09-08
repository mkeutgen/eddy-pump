#!/usr/bin/env python
"""The rate per physical pool from the label table, with its denominator and error bar.

reads  data/labels/{study_reviews.parquet, study_batches.yaml} through eddy_pump.labels,
       data/labels/draws/*.yaml, data/candidates/net_carbon_v1/<pool>.parquet (the pool size, read
       through the check that the saved list still hashes to its sidecar)
writes data/labels/audit/{rate_status.csv, RATE_STATUS.md}
The rate is the weighted mean of the drawn sample's target verdicts with 1/π weights, over the
candidate levels the sample covers. Where a pool is only partly sampled, the uncovered remainder is
reported, never extrapolated onto. Uncertain verdicts are excluded and counted. Session flags are
reported, never used as a filter.

Beside every rate as labelled, where a blind re-judgement of the sitting has been labelled and
loaded, the same rate corrected for how the reviewer's reading moved across that sitting: the rate
if every panel had been read the way the blind second look reads it, with the sampling error and the
re-judgement's own error added in quadrature (`rate_drift_corrected*`). The net of the two limbs,
downward minus upward in accepted candidate levels, is quoted only when both limbs carry that
correction — the 2026-09-04 review found the net changing sign across the drift band.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eddy_pump import batches as B  # noqa: E402
from eddy_pump import candidates as C  # noqa: E402
from eddy_pump import labels as L  # noqa: E402
from eddy_pump.criteria import active_criterion  # noqa: E402
from eddy_pump.manifest import load_manifest  # noqa: E402

OUT = REPO / "data/labels/audit"
DRAWS = REPO / "data/labels/draws"
TARGET_REL = 0.15
SECONDS_PER_PANEL = 28.7
N_BOOT_REJUDGEMENT = 3000   # resamples of a re-judgement, for the drift correction's own error bar


def loaded_rejudgements(batch_ids) -> dict:
    """The blind re-judgements that have been labelled and loaded, by the batch each re-judges."""
    import yaml

    out = {}
    for path in sorted(DRAWS.glob("*.labelled.yaml")):
        bid = path.name.replace(".labelled.yaml", "")
        rec_path = DRAWS / f"{bid}.yaml"
        if not rec_path.exists():
            continue
        rec = yaml.safe_load(rec_path.read_text())
        if rec.get("role") != "rejudgement" or rec.get("rejudges") not in set(batch_ids):
            continue
        block = (yaml.safe_load(path.read_text())["session"] or {}).get("rejudgement")
        if block:
            out[rec["rejudges"]] = {"batch_id": bid, **block}
    return out


def halves_of(A: pd.DataFrame) -> pd.Series:
    """Which half of its own sitting each science row was labelled in.

    Cut at the median position of the batch's decided science rows — the same cut the loader's
    session read and the re-judgement's draw use, so all three speak about the same halves.
    """
    half = pd.Series("first", index=A.index, dtype=object)
    for b, g in A.groupby("batch_id"):
        half.loc[g.index] = np.where(g.SAMPLE_ID <= float(g.SAMPLE_ID.median()), "first", "second")
    return half


def corrected_rate_of(A: pd.DataFrame, N_h: dict, rejudged: dict, se_design: float) -> dict | None:
    """The rate this pool would have had if every panel had been read the way the blind
    re-judgements read them. None while no re-judgement of this pool has been loaded.

    Every science row of a batch that has a re-judgement stops contributing its own 0/1 verdict and
    contributes instead the chance that re-judgement accepts a panel of that first verdict from that
    half of the sitting. A row of a batch with no re-judgement keeps its verdict — the correction is
    never extended to a sitting nobody looked at twice.
    """
    if not rejudged:
        return None
    half = halves_of(A)
    cells, cell = {}, pd.Series("", index=A.index, dtype=object)
    for b, block in rejudged.items():
        m = A.batch_id == b
        for name, c in block["cells"].items():
            cells[f"{b}|{name}"] = c
        cell.loc[m] = b + "|" + A.decision[m].astype(int).astype(str) + "|" + half[m]
    res = B.drift_corrected_rate(A.decision.to_numpy(float), cell.to_numpy(), A.design_stratum.to_numpy(),
                                 N_h, cells, se_design, n_boot=N_BOOT_REJUDGEMENT, seed=2)
    res["rejudgements"] = {b: {"batch_id": x["batch_id"], "n": x["overall"]["n"],
                               "accept_to_reject": x["overall"]["accept_to_reject"],
                               "reject_to_accept": x["overall"]["reject_to_accept"]}
                           for b, x in rejudged.items()}
    res["batches_not_rejudged"] = sorted(set(A.batch_id) - set(rejudged))
    return res


def main() -> None:
    import yaml

    OUT.mkdir(parents=True, exist_ok=True)
    study = load_manifest()
    crit = active_criterion()
    rows, md, limbs = [], [], {}
    for pool in (p for p in study.pools if p.tracer is None):
        try:
            A = L.analysis_sample(pool.pool_id, crit.id)
        except ValueError as e:
            rows.append({"pool_id": pool.pool_id, "status": f"no analysis batch yet: {e}"})
            md.append(f"## `{pool.pool_id}` — no analysis batch labelled yet\n")
            continue
        A = A[A.decision.isin([0, 1])]
        recs = {b: yaml.safe_load((DRAWS / f"{b}.yaml").read_text()) for b in A.batch_id.unique()}
        labelled = {b: yaml.safe_load((DRAWS / f"{b}.labelled.yaml").read_text()) for b in A.batch_id.unique()}
        N_h = {}
        for r0 in recs.values():
            for s in r0["strata"]:
                N_h[s["design_stratum"]] = N_h.get(s["design_stratum"], 0) + s["N"]
        N_open = int(sum(N_h.values()))
        # the design's own estimator and variance (eddy_pump.batches.stratified_rate); the naive float
        # bootstrap is kept beside it as the labelled conservative sensitivity
        res = B.stratified_rate(A.decision.to_numpy(float), A.inclusion_probability.to_numpy(float), A.design_stratum.to_numpy(),
                                A.key_wmo.to_numpy(), N_h, n_boot=3000, seed=1)
        # the denominator comes from the saved list itself, verified: the key hash, the spec and the
        # cache block are checked before the number a rate divides by is taken from it
        N_pool = len(C.read_saved(study, pool, verify=True, columns=C.KEYS)[0])
        # the rate the blind re-judgements imply, beside the rate as labelled
        rejudged = loaded_rejudgements(A.batch_id.unique())
        corr = corrected_rate_of(A, N_h, rejudged, res["se_design"])
        # The rate is over the region the sample actually covers (N_open). N_open == N_pool when the
        # whole pool is sampled (the downward limb); when only part is (the upward limb, whose former
        # held region awaits a fresh draw), the uncovered remainder is reported, never extrapolated.
        unsampled = N_pool - N_open
        p_pool, se_pool, se_pool_naive = res["rate"], res["se_design"], res["se_naive_float_bootstrap"]
        hw_rel = B.Z * se_pool / p_pool
        v_target = (TARGET_REL * p_pool / B.Z) ** 2
        v_open_now = res["se_design"] ** 2
        n_needed = int(np.ceil(res["n"] * v_open_now / max(v_target, 1e-12))) if v_target > 0 else None
        more = max(0, n_needed - res["n"]) if n_needed else None
        # the drift band: the rate if the whole session had read like the first half / the second half
        # Each batch reads its own strata; a pool sampled in two regions (the upward limb: the open
        # region, then the former held region) adds the two contributions, weighted by stratum size.
        halves = {}
        for b, l in labelled.items():
            th = l["session"].get("target_halves")
            if th:
                for side in ("first", "second"):
                    halves[side] = halves.get(side, 0.0) + sum(
                        N_h[s] / N_open * (v[side] if v[side] is not None else 0.0)
                        for s, v in th["by_stratum"].items() if s in N_h)
        sess = {b: l["session"] for b, l in labelled.items()}
        flags = []
        for b, s in sess.items():
            if s.get("trend_p_mann_whitney") is not None and s["trend_p_mann_whitney"] < 0.05:
                flags.append(f"{b}: acceptance fell with position (Mann-Whitney p = {s['trend_p_mann_whitney']:.3f}; "
                             f"{s['position'][0]['accept']:.0%} in the first quarter, {s['position'][-1]['accept']:.0%} in the last; "
                             f"the rate would read {halves.get('first', float('nan')):.3f} like the first half, {halves.get('second', float('nan')):.3f} like the second)")
            pc = s["controls"]["pos_ctrl"]
            if pc.get("blind_history"):
                bh = pc["blind_history"]
                verdict = "read strict" if pc.get("fisher_p_vs_blind_history", 1) < 0.05 else "within the instrument's own noise"
                flags.append(f"{b}: positive controls {pc['accepted']}/{pc['n']} against the blind re-judgement history "
                             f"{bh['k']}/{bh['n']} ({bh['k'] / bh['n']:.0%}): Fisher p = {pc.get('fisher_p_vs_blind_history', float('nan')):.2f} — {verdict}")
            nc = s["controls"]["neg_ctrl"]
            if nc.get("blind_history"):
                bh = nc["blind_history"]
                flags.append(f"{b}: negative controls {nc['accepted']}/{nc['n']} against their blind history {bh['k']}/{bh['n']} "
                             f"({bh['k'] / bh['n']:.0%}): Fisher p = {nc.get('fisher_p_vs_blind_history', float('nan')):.2f}"
                             + (f"; {nc['with_score_above_0p5']['accepted']} of the {nc['with_score_above_0p5']['n']} with score >= 0.5 accepted (plausible detector misses)"
                                if nc.get("with_score_above_0p5", {}).get("n") else ""))
            elif nc["n"] and nc["accepted"] / nc["n"] > nc["ceiling"]:
                flags.append(f"{b}: negative controls {nc['accepted']}/{nc['n']} above the {nc['ceiling']:.0%} ceiling — read loose")
        anchor = next((l.get("anchor") for l in labelled.values() if l.get("anchor")), None)
        pace = None
        if anchor and anchor.get("worksheet_mtime"):
            t0 = pd.Timestamp(anchor["worksheet_mtime"]); t1 = pd.Timestamp(next(iter(labelled.values()))["worksheet_mtime"])
            pace = float((t1 - t0).total_seconds() / next(iter(labelled.values()))["rows"])
        denom = (f"candidate levels of the pool ({N_pool:,})" if unsampled == 0 else
                 f"candidate levels sampled ({N_open:,}) of the pool's {N_pool:,}; {unsampled:,} not yet sampled")
        row = {"pool_id": pool.pool_id, "status": "measured", "criterion_version": crit.id, "denominator": denom,
               "batches": ",".join(sorted(recs)), "n_target_decided": res["n"], "floats": res["floats"],
               "uncertain_excluded": int(sum(l["uncertain"] for l in labelled.values())),
               "open_N": N_open, "open_rate": res["rate"], "open_se_design": res["se_design"],
               "open_se_stratified_bootstrap": res["se_stratified_bootstrap"], "open_se_naive_float_bootstrap": res["se_naive_float_bootstrap"],
               "open_design_effect_vs_weighted_iid": res["design_effect_vs_weighted_iid"], "open_zero_count_strata": res["zero_or_full_count_strata"],
               "unsampled_levels": unsampled, "pool_size": N_pool,
               "pool_rate": p_pool, "pool_se": se_pool, "pool_se_naive_float_bootstrap": se_pool_naive,
               "pool_half_width_95": B.Z * se_pool, "pool_half_width_95_rel": hw_rel,
               "pool_half_width_95_rel_naive": B.Z * se_pool_naive / p_pool,
               "accepted_levels_estimated": p_pool * N_open, "target_rel": TARGET_REL, "meets_target": bool(hw_rel <= TARGET_REL),
               "drift_band_first_half": halves.get("first"), "drift_band_second_half": halves.get("second"),
               "open_panels_needed_at_realised_variance": n_needed, "open_panels_more": more,
               "hours_more_at_planning_pace": (more * SECONDS_PER_PANEL / 3600) if more is not None else None,
               "realised_seconds_per_panel": pace, "anchor_batch": anchor["batch_id"] if anchor else None,
               "rejudgement_batches": ",".join(sorted(x["batch_id"] for x in rejudged.values())) or None,
               "rate_drift_corrected": corr["rate"] if corr else None,
               "rate_drift_corrected_se_design": corr["se_design"] if corr else None,
               "rate_drift_corrected_se_rejudgement": corr["se_rejudgement"] if corr else None,
               "rate_drift_corrected_se_total": corr["se_total"] if corr else None,
               "rate_drift_corrected_half_width_95": corr["half_width_95"] if corr else None,
               "rate_drift_corrected_half_width_95_rel": corr["half_width_95_rel"] if corr else None,
               "accepted_levels_drift_corrected": corr["rate"] * N_open if corr else None,
               "net_accepted_levels_drift_corrected": None, "net_accepted_levels_half_width_95": None,
               "flags": " | ".join(flags)}
        rows.append(row)
        limbs[pool.pool_id] = {"row": row, "N": N_open, "corr": corr, "rate": p_pool, "se": se_pool}
        md.append(f"## `{pool.pool_id}` — {p_pool:.1%} of {N_open:,} candidate levels, ±{hw_rel:.0%} relative (target ±{TARGET_REL:.0%})\n")
        md.append(f"Sampled region ({N_open:,} levels): stratified mean {res['rate']:.4f}, design-based SE {res['se_design']:.4f} "
                  f"(stratified bootstrap {res['se_stratified_bootstrap']:.4f}; the naive float bootstrap {res['se_naive_float_bootstrap']:.4f} is the "
                  f"conservative sensitivity and overstates a stratified draw) on {res['n']} target verdicts over {res['floats']} floats "
                  f"({row['uncertain_excluded']} uncertain excluded; {res['zero_or_full_count_strata']} strata with no accept, floored at the Jeffreys mean "
                  f"in the variance)."
                  + (f" {unsampled:,} of the pool's {N_pool:,} levels are not yet in a probability sample, to be drawn under the study's criterion." if unsampled else "") + "\n")
        md.append(f"Rate: **{p_pool:.4f} ± {B.Z * se_pool:.4f}** (≈ {p_pool * N_open:,.0f} accepted candidate LEVELS in the sampled region — not events; a "
                  f"cycle-level estimand needs its own denominator). Sampling precision only, at the session-average instrument. "
                  + (f"At the variance realised, the sample needs about **{n_needed} target panels** for ±{TARGET_REL:.0%}: {more} more"
                     + (f" ({more * SECONDS_PER_PANEL / 3600:.1f} h at the planning pace" + (f", {more * pace / 3600:.1f} h at the realised {pace:.0f} s/panel)" if pace else ")") if more else "") + "." if n_needed else "")
                  + (f" **Drift band**: the rate would read {halves['first']:.3f} if the whole session had read like its first half, "
                     f"{halves['second']:.3f} like its second — a systematic term the sampling interval does not contain." if halves else "") + "\n")
        if corr:
            md.append(f"**Drift-corrected**: {corr['rate']:.4f} ± {corr['half_width_95']:.4f} "
                      f"(≈ {corr['rate'] * N_open:,.0f} accepted levels) — the rate if the reviewer had read every panel "
                      f"the way the blind re-judgement reads it. Error bar: the sampling error {corr['se_design']:.4f} and "
                      f"how well {', '.join(x['batch_id'] for x in rejudged.values())} pins the second look "
                      f"{corr['se_rejudgement']:.4f}, added in quadrature. "
                      f"{corr['rows_corrected']} of {corr['rows_corrected'] + corr['rows_kept_as_labelled']} science rows "
                      f"are corrected"
                      + (f"; {', '.join(corr['batches_not_rejudged'])} has no re-judgement and keeps its verdicts as "
                         f"labelled" if corr["batches_not_rejudged"] else "") + ".\n")
            md.append("| what the reviewer first called it | half of the sitting | re-judged | called real again |\n"
                      "|---|---|---:|---:|\n" + "".join(
                          f"| {'real' if k.split('|')[1] == '1' else 'not real'} | {k.split('|')[2]} | {v['n']} | "
                          f"{v['accepted']} ({v['rate']:.0%}) |\n" for k, v in sorted(corr["cell_rates"].items())))
        else:
            md.append("**Drift-corrected**: not available — no blind re-judgement of this limb has been labelled and "
                      "loaded yet (docs/PLAN.md).\n")
        if flags:
            md.append("Session flags (recorded, never a filter):\n" + "".join(f"- {f}\n" for f in flags))
        by = A.groupby("design_stratum").decision.agg(["size", "sum", "mean"])
        md.append("| stratum | N | n | accepted | rate | share of variance |\n|---|---:|---:|---:|---:|---:|\n" + "".join(
            f"| {s} | {N_h[s]:,} | {int(r['size'])} | {int(r['sum'])} | {r['mean']:.3f} | {res['strata'][s]['variance_share']:.0%} |\n" for s, r in by.iterrows()))
    # the net: gross downward minus the return by obduction, in accepted candidate levels. It is
    # quoted only when both limbs carry a drift correction — the 2026-09-04 review showed the net
    # changing sign across the drift band, so an uncorrected net says nothing.
    down = limbs.get("net_carbon_v1/physical/subduction")
    up = limbs.get("net_carbon_v1/physical/obduction")
    net = None
    if down and up and down["corr"] and up["corr"]:
        n_levels = down["corr"]["rate"] * down["N"] - up["corr"]["rate"] * up["N"]
        var = (down["N"] * down["corr"]["se_total"]) ** 2 + (up["N"] * up["corr"]["se_total"]) ** 2
        hw = B.Z * float(np.sqrt(var))
        net = {"accepted_levels": n_levels, "half_width_95": hw,
               "down_levels": down["corr"]["rate"] * down["N"], "up_levels": up["corr"]["rate"] * up["N"],
               "as_labelled": down["rate"] * down["N"] - up["rate"] * up["N"]}
        for limb in (down, up):
            limb["row"]["net_accepted_levels_drift_corrected"] = n_levels
            limb["row"]["net_accepted_levels_half_width_95"] = hw
        md.append(f"## The net — {n_levels:,.0f} ± {hw:,.0f} accepted candidate levels\n")
        md.append(f"Downward {net['down_levels']:,.0f} minus upward {net['up_levels']:,.0f}, both drift-corrected, both "
                  f"in accepted candidate LEVELS (not events, not carbon). The error bar carries the sampling error of "
                  f"each limb and how well each blind re-judgement pins the second look. As labelled, without the "
                  f"correction, the same difference reads {net['as_labelled']:,.0f}.\n")
    elif down and up:
        md.append("## The net — not quoted\n")
        md.append("The net of the two limbs is not quoted. The 2026-09-04 review found the reviewer's reading moving "
                  "within each of the two first sittings by two to three times the sampling error, and the net changes "
                  "sign across that band. It is quoted once a blind re-judgement of "
                  + " and ".join(sorted(k.split("/")[-1] for k, v in limbs.items() if not v["corr"]))
                  + " has been labelled and loaded (docs/PLAN.md).\n")

    T = pd.DataFrame(rows)
    T.to_csv(OUT / "rate_status.csv", index=False)
    measured = T[T.status == "measured"]
    nxt = ("the downward limb (no analysis batch yet)" if len(measured) < 2 else
           f"`{measured.sort_values('pool_half_width_95_rel', ascending=False).iloc[0].pool_id}` (the wider half-width)")
    page = (f"# RATE STATUS — the physical rates from the human labels *({B.stamp()[:10]})*\n\n"
            f"Producer: `pipeline/rates.py`. Criterion `{crit.id}`. Every rate is a weighted Hájek mean of human verdicts with "
            f"declared inclusion probabilities; the denominator is the candidate levels the sample covers. The next batch goes to "
            f"**{nxt}**.\n\n" + "\n".join(md))
    (OUT / "RATE_STATUS.md").write_text(page, encoding="utf-8")
    print(T.drop(columns=["flags"]).T.to_string())
    for f in measured["flags"]:
        if f:
            print("FLAGS:", f)
    print("next:", nxt)


if __name__ == "__main__":
    main()
