"""Make the src/ package importable from a plain checkout, and the strict mode.

EDDY_PUMP_STRICT=1 turns a skip caused by a missing data file into a failure, so a fresh clone
cannot go green by running nothing. Skips for other reasons (an older argopod, no git history, nothing
to test today) stay skips.
"""
import os
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

_MISSING_DATA = re.compile(r"run production/|run pipeline/|not on this machine|absent|missing|not ingested|not frozen", re.I)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if os.environ.get("EDDY_PUMP_STRICT") == "1" and rep.skipped and call.excinfo is not None:
        reason = str(call.excinfo.value)
        if _MISSING_DATA.search(reason):
            rep.outcome = "failed"
            rep.longrepr = f"strict mode: skipped for a missing data file — {reason}"
