"""Compute the classifier's features for every candidate of the two physical pools.

reads  config/events.yaml, data/candidates/net_carbon_v1/physical_{obduction,subduction}.parquet,
       the cache's per-float coarse and fine residual grids
writes results/net_carbon_v1/features/<pool>.parquet (one row per candidate; not in git)
       data/features/net_carbon_v1/FEATURES_SHA256 (rows, columns, file hash, cache identity)
About 0.03 s per candidate, about an hour for both pools. `--limit-floats N` for a dry run; `--pools`.

Two refusals. The cache on disk is fingerprinted before anything is read, and a fingerprint that
is not the one the saved candidate lists were built from stops the run. Any float that contributes
no features — no grids, or a grid the extractor raised on — is named on the way past and stops the
run at the end, because features missing for a float are candidates missing from every score.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from argopod.triage.features import extract_features_batch  # noqa: E402
from eddy_pump import candidates as C  # noqa: E402
from eddy_pump.manifest import load_manifest  # noqa: E402

KEYS = ["WMO", "CYCLE_NUMBER", "PRES_ADJUSTED"]
CANDIDATES = REPO / "data/candidates/net_carbon_v1"   # the saved lists, full tables
MANIFEST_DIR = REPO / "data/features/net_carbon_v1"


def build(pool, cache_dir: pathlib.Path, flavours: dict[int, str], out: pathlib.Path,
          limit_floats: int | None, failures: list[str]) -> dict:
    channel, direction = pool.pool_id.split("/")[1:]
    cand = pd.read_parquet(CANDIDATES / f"{channel}_{direction}.parquet")
    for k in KEYS:
        cand[k] = pd.to_numeric(cand[k], errors="coerce")
    cand = cand.dropna(subset=KEYS)
    wmos = cand.WMO.unique()
    if limit_floats:
        wmos = wmos[:limit_floats]
        cand = cand[cand.WMO.isin(wmos)]
    print(f"[{pool.pool_id}] {len(cand):,} candidates on {len(wmos):,} floats -> {out}", flush=True)
    t0 = time.time()
    parts, ok, miss = [], 0, 0
    for i, wmo in enumerate(wmos):
        flavour = flavours.get(int(wmo))
        if flavour is None:
            miss += 1
            msg = f"{pool.pool_id}: float {int(wmo)} has no grids in {cache_dir}"
            failures.append(msg)
            print(f"  {msg}", flush=True)
            continue
        try:
            coarse, fine = C.read_grids(cache_dir, int(wmo), flavour)
        except Exception as exc:  # one float must not sink the run; it is named and counted
            miss += 1
            msg = f"{pool.pool_id}: float {int(wmo)} ({flavour}) would not read — {type(exc).__name__}: {exc}"
            failures.append(msg)
            print(f"  {msg}", flush=True)
            continue
        sub = cand[cand.WMO == wmo]
        try:
            F = extract_features_batch(coarse, fine, sub, list(pool.spec.variables), pool.spec.params)
        except Exception as exc:  # one float must not sink the run; it is named and counted
            miss += 1
            msg = f"{pool.pool_id}: float {int(wmo)} ({flavour}) has no features — {type(exc).__name__}: {exc}"
            failures.append(msg)
            print(f"  {msg}", flush=True)
            continue
        for k in KEYS:
            F[k] = pd.to_numeric(F[k], errors="coerce").round(0)
        keep = [c for c in F.columns if c in KEYS or pd.api.types.is_numeric_dtype(F[c])]
        parts.append(F[keep])
        ok += 1
        if (i + 1) % 100 == 0:
            done = sum(len(p) for p in parts)
            rate = (time.time() - t0) / max(1, done)
            print(f"  ...{i + 1}/{len(wmos)} floats, {done:,} rows, {rate * 1000:.1f} ms/candidate,"
                  f" ~{rate * (len(cand) - done) / 60:.0f} min left", flush=True)
    P = pd.concat(parts, ignore_index=True).drop_duplicates(KEYS)
    P["pool_id"] = pool.pool_id
    P["spec_id"] = pool.spec.spec_id
    out.parent.mkdir(parents=True, exist_ok=True)
    P.to_parquet(out, index=False)
    print(f"[{pool.pool_id}] done: {len(P):,} rows from {ok} floats ({miss} without grids or failed),"
          f" {time.time() - t0:.0f} s", flush=True)
    return {"pool_id": pool.pool_id, "spec_id": pool.spec.spec_id, "candidates": int(len(cand)),
            "rows": int(len(P)), "floats_ok": ok, "floats_missing": miss, "columns": int(P.shape[1]),
            "file": str(out.relative_to(REPO)), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "seconds": round(time.time() - t0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pools", nargs="*", default=["physical/obduction", "physical/subduction"])
    ap.add_argument("--limit-floats", type=int, default=None)
    ap.add_argument("--suffix", default="", help="appended to the output file name (dry runs)")
    a = ap.parse_args()
    study = load_manifest()
    # Refuse a cache that is not the one the saved candidate lists were built from, and keep the
    # identity MEASURED here — not the copy the identity file carries. Writing the copy into the
    # provenance would make every later comparison against it compare a number with itself.
    live = C.require_bound_cache(study)
    cache_dir = pathlib.Path(study.cache.path)
    print(f"{study.study_id}: cache {cache_dir} ({live.fine_grids} fine grids, "
          f"{live.fine_grids_sha256[:16]}…) is the one the saved lists were built from", flush=True)
    flavours = C.float_flavours(cache_dir)
    failures: list[str] = []
    records = []
    for pool in study.pools:
        channel, direction = pool.pool_id.split("/")[1:]
        if f"{channel}/{direction}" not in a.pools:
            continue
        name = f"{channel}_{direction}{a.suffix}.parquet"
        out = study.output.resolve("features", name)
        records.append(build(pool, cache_dir, flavours, out, a.limit_floats, failures))
    if failures:
        for line in failures[:20]:
            print(f"  {line}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        raise SystemExit(
            f"\n{len(failures)} float(s) produced no features, so those candidates carry no score "
            f"and would never be drawn. Nothing else written.")
    if a.limit_floats:
        print("dry run; provenance not written")
        return
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    man = {
        "study_id": study.study_id,
        "built": _dt.datetime.now().isoformat(timespec="seconds"),
        "cache": {"path": str(cache_dir), "fine_grids": live.fine_grids,
                  "fine_grids_sha256": live.fine_grids_sha256},
        "pools": records,
    }
    (MANIFEST_DIR / "FEATURES_SHA256").write_text(
        "# provenance of the study's candidate feature tables -- built by pipeline/features.py; do not edit."
        " The cache block is counted off the grids this run read, not copied from CACHE_IDENTITY.json.\n"
        + json.dumps(man, indent=2) + "\n")
    print(f"provenance -> {MANIFEST_DIR / 'FEATURES_SHA256'}")


if __name__ == "__main__":
    main()
