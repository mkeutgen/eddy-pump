"""Compute the classifier's features for every candidate of the two physical pools.

reads  config/events.yaml, data/candidates/net_carbon_v1/physical_{obduction,subduction}.parquet,
       the bound cache's per-float coarse and fine residual grids
writes results/net_carbon_v1/features/<pool>.parquet (one row per candidate; not in git)
       data/features/net_carbon_v1/FEATURES_SHA256 (rows, columns, file hash, cache identity)
About 0.03 s per candidate, about an hour for both pools. `--limit-floats N` for a dry run; `--pools`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
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
from eddy_pump.manifest import load_manifest  # noqa: E402

KEYS = ["WMO", "CYCLE_NUMBER", "PRES_ADJUSTED"]
CANDIDATES = REPO / "data/candidates/net_carbon_v1"   # the saved lists, full tables
MANIFEST_DIR = REPO / "data/features/net_carbon_v1"


def grids_for(cache_dir: pathlib.Path, wmo: int):
    c = glob.glob(str(cache_dir / f"{wmo}_*_coarse.parquet"))
    f = glob.glob(str(cache_dir / f"{wmo}_*_fine.parquet"))
    if not c or not f:
        return None
    return pd.read_parquet(c[0]), pd.read_parquet(f[0])


def build(pool, cache_dir: pathlib.Path, out: pathlib.Path, limit_floats: int | None) -> dict:
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
        gg = grids_for(cache_dir, int(wmo))
        if gg is None:
            miss += 1
            continue
        coarse, fine = gg
        sub = cand[cand.WMO == wmo]
        try:
            F = extract_features_batch(coarse, fine, sub, list(pool.spec.variables), pool.spec.params)
        except Exception as exc:  # one float must not sink the run; the manifest counts it
            print(f"  float {int(wmo)}: {exc}", flush=True)
            miss += 1
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
    cache_dir = pathlib.Path(study.cache.path)
    records = []
    for pool in study.pools:
        channel, direction = pool.pool_id.split("/")[1:]
        if f"{channel}/{direction}" not in a.pools:
            continue
        name = f"{channel}_{direction}{a.suffix}.parquet"
        out = study.output.resolve("features", name)
        records.append(build(pool, cache_dir, out, a.limit_floats))
    if a.limit_floats:
        print("dry run; manifest not written")
        return
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    man = {
        "study_id": study.study_id,
        "built": _dt.datetime.now().isoformat(timespec="seconds"),
        "cache": {"path": str(study.cache.path), "fine_grids": study.cache.fine_grids,
                  "fine_grids_sha256": study.cache.fine_grids_sha256},
        "pools": records,
    }
    (MANIFEST_DIR / "FEATURES_SHA256").write_text(
        "# provenance of the study's candidate feature tables -- built by production/build_study_features.py; do not edit\n"
        + json.dumps(man, indent=2) + "\n")
    print(f"manifest -> {MANIFEST_DIR / 'FEATURES_SHA256'}")


if __name__ == "__main__":
    main()
