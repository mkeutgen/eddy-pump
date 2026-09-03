"""Detect the study's six candidate pools and keep their saved lists.

reads  the residual-grid cache the study is bound to
writes data/candidates/<study_id>/<event_type>.parquet, its .json sidecar, CANDIDATES.json

Entry points: `detect_study()` (every pool, each nested in its own directional parent),
`write_saved()` / `read_saved()` for one pool's list, and `verify_saved()`, which compares a
regenerated table with the saved one and calls anything but an exact key-set match a failure.
"""

from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import re
import warnings
from pathlib import Path

import pandas as pd

from argopod.detect import detect_from_grids

from .manifest import REPO_ROOT
from .spec import CandidatePool
from .study import CacheIdentity, Study

KEYS = ["WMO", "CYCLE_NUMBER", "PRES_ADJUSTED"]
MEASURES = ["LATITUDE", "LONGITUDE", "TIME",
            "AOU_SCALE_RES_ROB", "AOU_IQRN", "ABS_SAL_SCALE_RES_ROB", "ABS_SAL_IQRN",
            "NITRATE_ADJUSTED_SCALE_RES_ROB", "NITRATE_ADJUSTED_IQRN",
            "BBP700_ADJUSTED_SCALE_RES_ROB", "BBP700_ADJUSTED_IQRN"]
ID_COLS = ["study_id", "pool_id", "spec_id", "candidate_id", "EVENT_TYPE"]
#: Where the saved lists live: data/candidates/<study_id>/.
CANDIDATES_DIR = REPO_ROOT / "data" / "candidates"
#: The summary file beside the six lists.
SUMMARY = "CANDIDATES.json"


# --------------------------------------------------------------------------- #
# the cache
# --------------------------------------------------------------------------- #
def live_cache_identity(cache_dir: Path) -> CacheIdentity:
    """The identity of the cache on disk: the number of fine grids and the sha256 of their sorted
    file names. Milliseconds, and it tells two caches apart at a glance."""
    names = sorted(p.name for p in Path(cache_dir).glob("*_fine.parquet"))
    return CacheIdentity(path=Path(cache_dir), fine_grids=len(names),
                         fine_grids_sha256=hashlib.sha256("\n".join(names).encode()).hexdigest())


def require_bound_cache(study: Study) -> CacheIdentity:
    """Refuse to detect against a cache that is not the one the study's lists are bound to."""
    live = live_cache_identity(study.cache.path)
    if not study.cache.matches(live):
        raise ValueError(
            f"the cache at {study.cache.path} is not the one {study.study_id} is bound to:\n"
            f"  bound : {study.cache.fine_grids} fine grids, {study.cache.fine_grids_sha256[:16]}…\n"
            f"  live  : {live.fine_grids} fine grids, {live.fine_grids_sha256[:16]}…\n"
            f"nothing detected, nothing written")
    return live


def _grids(cache_dir: Path) -> dict[str, dict[int, str]]:
    """{flavour: {wmo: flavour}} for every cached coarse+fine pair."""
    out: dict[str, dict[int, str]] = {}
    for path in glob.glob(str(Path(cache_dir) / "*_coarse.parquet")):
        if not Path(path.replace("_coarse.parquet", "_fine.parquet")).exists():
            continue
        m = re.match(r"(\d+)_(.+)_coarse\.parquet", Path(path).name)
        if m:
            out.setdefault(m.group(2), {})[int(m.group(1))] = m.group(2)
    return out


def _flavours_for(pool: CandidatePool, cache_dir: Path, grids) -> dict[int, str]:
    """Per float, a grid flavour that carries every channel this pool's spec needs. A spec run
    against a grid missing a channel yields nothing and raises nothing, so this is checked."""
    need = [f"{v.name}_SCALE_RES_ROB" for v in pool.spec.variables]
    ok: dict[int, str] = {}
    for flavour, wmos in grids.items():
        w0 = next(iter(wmos))
        try:
            cols = set(pd.read_parquet(Path(cache_dir) / f"{w0}_{flavour}_coarse.parquet").columns)
        except Exception:
            continue
        if not all(c in cols for c in need):
            continue
        for w in wmos:
            ok.setdefault(w, flavour)
    return ok


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def _keys(df: pd.DataFrame) -> set:
    d = df.copy()
    for k in KEYS:
        d[k] = pd.to_numeric(d[k], errors="coerce").round(0)
    return set(map(tuple, d[KEYS].dropna().to_numpy()))


def stamp(pool: CandidatePool, study: Study, df: pd.DataFrame) -> pd.DataFrame:
    """Add the identity columns every output row carries."""
    out = df.copy()
    out["study_id"] = study.study_id
    out["pool_id"] = pool.pool_id
    out["spec_id"] = pool.spec_id
    out["EVENT_TYPE"] = pool.event_type
    out["candidate_id"] = [pool.key(int(w), int(round(c)), float(p)).candidate_id
                           for w, c, p in zip(out.WMO, out.CYCLE_NUMBER, out.PRES_ADJUSTED)]
    return out


def detect_pool(pool: CandidatePool, study: Study, grids=None, cache_dir: Path | None = None) -> pd.DataFrame:
    """Raw detector output for one pool, before nesting. Deduplicated on the key."""
    cache_dir = Path(study.cache.path if cache_dir is None else cache_dir)
    grids = _grids(cache_dir) if grids is None else grids
    rows = []
    for w, fl in sorted(_flavours_for(pool, cache_dir, grids).items()):
        try:
            c = pd.read_parquet(cache_dir / f"{w}_{fl}_coarse.parquet")
            f = pd.read_parquet(cache_dir / f"{w}_{fl}_fine.parquet")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                d = detect_from_grids(c, f, list(pool.spec.variables), pool.spec.params)
        except Exception:
            continue
        if len(d):
            rows.append(d.assign(WMO=int(w)))
    cols = KEYS + MEASURES
    if not rows:
        return stamp(pool, study, pd.DataFrame(columns=cols))
    out = pd.concat(rows, ignore_index=True)
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[cols].drop_duplicates(subset=KEYS).reset_index(drop=True)
    return stamp(pool, study, out)


def detect_study(study: Study, pools: list[CandidatePool] | None = None) -> dict[str, pd.DataFrame]:
    """Every requested pool, nested in its own directional parent. A child without its parent is
    refused: the un-nested table would look like a product and be a superset of one."""
    require_bound_cache(study)
    pools = list(study.pools) if pools is None else list(pools)
    ids = {p.pool_id for p in pools}
    orphans = [p.pool_id for p in pools if p.parent is not None and p.parent.pool_id not in ids]
    if orphans:
        raise ValueError(f"children requested without their directional parents: {orphans}")
    grids = _grids(Path(study.cache.path))
    out = {p.pool_id: detect_pool(p, study, grids) for p in pools}
    for p in pools:
        if p.parent is None:
            continue
        keep = _keys(out[p.parent.pool_id])
        d = out[p.pool_id]
        if len(d):
            dd = d.copy()
            for k in KEYS:
                dd[k] = pd.to_numeric(dd[k], errors="coerce").round(0)
            out[p.pool_id] = d[[tuple(r) in keep for r in dd[KEYS].to_numpy()]].reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# the saved lists
# --------------------------------------------------------------------------- #
def saved_dir(study: Study, root: Path | None = None) -> Path:
    return (CANDIDATES_DIR if root is None else Path(root)) / study.study_id


def saved_path(study: Study, pool: CandidatePool, root: Path | None = None) -> Path:
    return saved_dir(study, root) / f"{pool.event_type}.parquet"


def content_hash(table: pd.DataFrame) -> str:
    """sha256 over the sorted key triples — the identity of a candidate SET, independent of row
    order and of the measurement columns."""
    keys = sorted(_keys(table))
    return hashlib.sha256("\n".join(f"{int(w)}|{int(c)}|{int(p)}" for w, c, p in keys).encode()).hexdigest()


def write_saved(study: Study, pool: CandidatePool, table: pd.DataFrame, root: Path | None = None) -> Path:
    """Save one pool's list: the full table sorted by key, Zstd Parquet, plus its sidecar."""
    path = saved_path(study, pool, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in KEYS + MEASURES + ID_COLS if c in table.columns]
    t = table[cols].sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    t.to_parquet(path, index=False, compression="zstd")
    side = {
        "study_id": study.study_id, "pool_id": pool.pool_id, "spec_id": pool.spec_id,
        "event_type": pool.event_type, "rows": int(len(t)), "content_sha256": content_hash(t),
        "cache": {"path": str(study.cache.path), "fine_grids": study.cache.fine_grids,
                  "fine_grids_sha256": study.cache.fine_grids_sha256},
        "written": _dt.datetime.now().isoformat(timespec="seconds"),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(side, indent=2) + "\n")
    return path


def read_saved(study: Study, pool: CandidatePool, root: Path | None = None,
               columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """One pool's saved list (optionally a column subset) and its sidecar."""
    path = saved_path(study, pool, root)
    return pd.read_parquet(path, columns=columns), json.loads(path.with_suffix(".json").read_text())


def verify_saved(study: Study, pool: CandidatePool, table: pd.DataFrame, root: Path | None = None) -> dict:
    """Compare a regenerated table with the pool's saved list. Anything but EXACT is a failure."""
    a, side = read_saved(study, pool, root, columns=KEYS)
    if side["spec_id"] != pool.spec_id:
        raise ValueError(f"{pool.pool_id}: the saved list was written under spec {side['spec_id']}, the pool is {pool.spec_id}")
    if not study.cache.matches(side["cache"]):
        raise ValueError(f"{pool.pool_id}: the saved list was written from another cache ({side['cache']['fine_grids_sha256'][:16]}…)")
    s, g = _keys(a), _keys(table)
    return {"pool_id": pool.pool_id, "saved": len(s), "regenerated": len(g),
            "exact": s == g, "saved_only": len(s - g), "new_only": len(g - s)}


def write_summary(study: Study, report: list[dict], root: Path | None = None) -> Path:
    """`CANDIDATES.json`: the six lists in one page — which cache, which specs, how many rows."""
    path = saved_dir(study, root) / SUMMARY
    path.write_text(json.dumps({
        "study_id": study.study_id, "written": _dt.datetime.now().isoformat(timespec="seconds"),
        "cache": {"path": str(study.cache.path), "fine_grids": study.cache.fine_grids,
                  "fine_grids_sha256": study.cache.fine_grids_sha256},
        "pools": report}, indent=2) + "\n")
    return path
