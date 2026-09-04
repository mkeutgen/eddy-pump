"""Detect the study's six candidate pools from the cache; check or write the saved lists.

reads  config/events.yaml, the cache's per-float residual grids
writes --write : data/candidates/net_carbon_v1/<event_type>.parquet (+ .json), CANDIDATES.json — `make freeze-candidates`
       --verify: nothing; re-detects and compares each pool's key set with its saved list, exits 1 on
                 anything but an exact match — `make verify-candidates`, about six minutes per pool
The cache on disk must carry the fingerprint the study's saved lists were built from; a child pool
is never detected without its directional parent; a float whose grids will not open, or whose
detection raises, is counted and named, and the run stops rather than write a short list.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eddy_pump import candidates as C  # noqa: E402
from eddy_pump.manifest import load_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the saved lists and CANDIDATES.json")
    mode.add_argument("--verify", dest="verify", action="store_true",
                      help="check only; write nothing")
    ap.add_argument("--pools", nargs="*", default=None, help="channel/direction, default all six")
    a = ap.parse_args()
    study = load_manifest()
    pools = list(study.pools)
    if a.pools:
        want = set(a.pools)
        pools = [p for p in pools if f"{p.channel}/{p.direction.value}" in want]
    live = C.require_bound_cache(study)
    print(f"{study.study_id}: cache {study.cache.path} ({live.fine_grids} fine grids, "
          f"{live.fine_grids_sha256[:16]}…) is the one the saved lists were built from")
    t0 = time.time()
    failures: list[C.GridFailure] = []
    tables = C.detect_study(study, pools, failures=failures)
    print(f"detected {len(tables)} pools in {time.time() - t0:.0f} s")

    bad, report = [], []
    for p in pools:
        t = tables[p.pool_id]
        line = {"pool_id": p.pool_id, "spec_id": p.spec_id, "rows": int(len(t))}
        if a.verify:
            if not C.saved_path(study, p).exists():
                raise SystemExit(f"{p.pool_id}: no saved list at {C.saved_path(study, p)}; run --write first")
            v = C.verify_saved(study, p, t)
            line.update({"exact": v["exact"], "saved_only": v["saved_only"], "new_only": v["new_only"]})
            if not v["exact"]:
                bad.append(p.pool_id)
        report.append(line)
        print("  " + json.dumps(line))
    if failures:
        by_pool = {}
        for fail in failures:
            by_pool.setdefault(fail.pool_id, []).append(fail)
        print(f"\n{len(failures)} float(s) contributed nothing:")
        for pool_id in sorted(by_pool):
            for fail in by_pool[pool_id][:10]:
                print(f"  {fail}")
            if len(by_pool[pool_id]) > 10:
                print(f"  ... and {len(by_pool[pool_id]) - 10} more in {pool_id}")
        raise SystemExit(
            "\nthose floats are absent from the pools named, so those counts are short by whatever "
            "they hold. Nothing written. Put the cache right (scripts/fetch_caches.sh) and run "
            "again.")
    if bad:
        raise SystemExit(f"\nthe saved lists do not reproduce: {sorted(set(bad))}. Nothing written.")
    if a.verify:
        print("\nall reproduce exactly; nothing written (--verify)")
        return
    for p in pools:
        C.write_saved(study, p, tables[p.pool_id])
    print(f"\nwrote {len(pools)} saved lists under {C.saved_dir(study)} and {C.write_summary(study, report).name}")


if __name__ == "__main__":
    main()
