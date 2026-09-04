#!/usr/bin/env python
"""Build the study's fleet cache — one pair of residual grids per float — or check one.

reads  config/events.yaml (the recipe), config/fleet.csv (the 2,574 floats and the grid flavour
       each is promised), the staged raw frames named by --raw
writes --out DIR : <wmo>_<flavour>_{fine,coarse}.parquet, MANIFEST.csv, PROVENANCE.json
       --verify  : HEALTH.csv beside the cache it read (elsewhere, if that cache is the bound one)
       --check   : nothing that survives the run — it builds into a temporary directory

Three things this does:
  build    every float of the list, or a slice of it (--tier, --limit, --resume)
  --verify read a built cache back and say whether it can be trusted
  --check  rebuild a few named floats and compare every grid byte for byte with the bound cache

A build never targets the cache the saved candidate lists stand on: rewriting those grids would
move the fingerprint every published number is keyed to, so the run is refused. Build somewhere new
and compare.

A float with no staged raw frame is downloaded. ARGOPOD_SKIP_ERDDAP=1 makes those downloads go
straight to the GDAC files instead of trying the ERDDAP endpoint first, which saves minutes per
float when that endpoint is slow. This script never sets it; export it if you want it.

The work itself is generic and lives in argopod (`argopod.cache`, v0.5.1). What is here is the
study's own: which floats, which recipe, and the refusal above.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import tempfile

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from argopod.cache import build_cache, verify_cache  # noqa: E402

from eddy_pump.manifest import load_manifest  # noqa: E402

#: The build list: every float of the fleet, the grid flavour it is promised, and the wave it was
#: built in. Row for row the fleet the bound cache was built from.
FLEET = REPO / "config" / "fleet.csv"


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _refuse_the_bound_cache(out: pathlib.Path, bound: pathlib.Path) -> None:
    """A build may not write into the cache the saved candidate lists stand on."""
    if out.resolve() == bound.resolve():
        raise SystemExit(
            f"refusing to build into {out}\n"
            f"That is the cache the saved candidate lists stand on. Rebuilding it in place "
            f"would move its fingerprint, and every published number is keyed to it. "
            f"Build into a new directory and compare the two with --check.")


def do_build(a, study, policy) -> int:
    """Build the fleet, or the slice of it the flags name."""
    out = pathlib.Path(a.out).expanduser()
    _refuse_the_bound_cache(out, study.cache.path)
    manifest = build_cache(
        FLEET, policy, out,
        raw_cache=(str(pathlib.Path(a.raw).expanduser()) if a.raw else None),
        workers=a.workers, resume=a.resume, limit=a.limit, tier=a.tier)
    # A declared exclusion also gets an ok=False row, carrying its ruling. That is the record
    # working, not a failure, so it is counted separately.
    ok = manifest["ok"].astype(bool)
    left_out = manifest["excluded"].fillna(False).astype(bool) if "excluded" in manifest \
        else (~ok & False)
    failed = int((~ok & ~left_out).sum())
    print(f"{int(ok.sum())} float(s) built, {failed} failed, {int(left_out.sum())} left out by the "
          f"study; the record is {out / 'MANIFEST.csv'}")
    return 0


def do_verify(a, study, policy) -> int:
    """Read a built cache back: no surviving placeholder value, no impossible level."""
    out = pathlib.Path(a.out).expanduser()
    report = out
    if out.resolve() == study.cache.path.resolve():
        report = study.output.resolve("cache")
        report.mkdir(parents=True, exist_ok=True)
        print(f"reading the bound cache, so the health report goes to {report} instead of beside it")
    ok, health = verify_cache(out, policy, floats=FLEET, report_dir=report)
    print(f"{len(health)} grids checked; HEALTH.csv in {report}")
    return 0 if ok else 1


def do_check(a, study, policy) -> int:
    """Rebuild the named floats into a temporary directory and compare with the bound cache.

    Byte for byte, by sha256, on both grids of each float. The point is not that the numbers are
    close: a cache is only the same cache if the files are identical.
    """
    wanted = [int(w) for w in str(a.check).replace(" ", "").split(",") if w]
    fleet = pd.read_csv(FLEET)
    missing = [w for w in wanted if w not in set(fleet.WMO.astype(int))]
    if missing:
        raise SystemExit(f"not on {FLEET}: {missing}")
    out_of_scope = sorted(set(wanted) & study.excluded_wmos)
    if out_of_scope:
        raise SystemExit(
            f"{out_of_scope} are floats the study leaves out (config/events.yaml "
            f"excluded_floats), so no build produces a grid for them")
    bound = study.cache.path
    if not bound.exists():
        raise SystemExit(f"the bound cache is not on this machine: {bound}")
    raw = pathlib.Path(a.raw).expanduser() if a.raw else None
    if raw is not None:
        absent = [w for w in wanted if not (raw / f"{w}.parquet").exists()]
        if absent:
            print(f"note: {absent} have no staged raw frame in {raw} and will be downloaded")

    rows = fleet[fleet.WMO.astype(int).isin(wanted)]
    with tempfile.TemporaryDirectory(prefix="eddy-pump-check-") as tmp:
        tmp = pathlib.Path(tmp)
        print(f"rebuilding {len(rows)} float(s) into {tmp}")
        # The nine floats left out are reported one paragraph each, and not one of them can be on
        # this list (refused above), so the check keeps the count and drops the paragraphs.
        build_cache(rows, policy, tmp, raw_cache=(str(raw) if raw else None),
                    workers=a.workers, checkpoint_every=0,
                    log=lambda m: None if str(m).startswith("EXCLUDED ") else print(m))
        same, diff = [], []
        for wmo in wanted:
            built = sorted(tmp.glob(f"{wmo}_*_fine.parquet")) + \
                sorted(tmp.glob(f"{wmo}_*_coarse.parquet"))
            if not built:
                diff.append(f"{wmo}: nothing built — see the reason in {tmp / 'MANIFEST.csv'}")
                print(f"DIFF  {wmo}: nothing built")
                continue
            for path in built:
                twin = bound / path.name
                if not twin.exists():
                    diff.append(f"{path.name}: not in the bound cache")
                    print(f"DIFF  {path.name}: not in {bound}")
                    continue
                if _sha256(path) == _sha256(twin):
                    same.append(path.name)
                    print(f"SAME  {path.name}")
                else:
                    diff.append(f"{path.name}: sha256 differs")
                    print(f"DIFF  {path.name}: sha256 differs")
    print(f"\n{len(same)} file(s) identical, {len(diff)} different — against {bound}")
    for line in diff:
        print(f"  {line}")
    return 1 if diff else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="the cache directory to build into, or to read with --verify")
    ap.add_argument("--raw", help="directory of staged raw frames, <wmo>.parquet")
    ap.add_argument("--workers", type=int, default=1, help="build processes (default 1)")
    ap.add_argument("--resume", action="store_true", help="skip floats that already have a grid")
    ap.add_argument("--tier", type=int, default=None, help="build one wave of the float list only")
    ap.add_argument("--limit", type=int, default=None, help="build at most this many floats")
    ap.add_argument("--verify", action="store_true",
                    help="read the cache at --out back instead of building")
    ap.add_argument("--check", default=None, metavar="WMO[,WMO...]",
                    help="rebuild these floats in a temporary directory and compare every grid "
                         "byte for byte with the bound cache")
    a = ap.parse_args()

    study = load_manifest()
    policy = study.cache_policy()

    if a.check and a.verify:
        raise SystemExit("--check and --verify do different jobs; run one at a time")
    if a.check:
        raise SystemExit(do_check(a, study, policy))
    if not a.out:
        raise SystemExit("--out is required (the cache directory to build into, or to verify)")
    if a.verify:
        raise SystemExit(do_verify(a, study, policy))
    if not a.raw:
        print("note: no --raw, so every float will be downloaded")
    raise SystemExit(do_build(a, study, policy))


if __name__ == "__main__":
    main()
