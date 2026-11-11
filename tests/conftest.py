"""Pytest configuration and test environment initialization for KRS Dam Assistant."""

import os
import sys
from pathlib import Path

# Ensure project root is at the front of sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Silence loky core count detection warning in constrained environments
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

# On Windows, use SelectorEventLoop to prevent socketpair deadlocks in constrained environments
if sys.platform == "win32":
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Patch streamlit AppTest.from_file so that repo-root relative paths resolve correctly
try:
    from streamlit.testing.v1.app_test import AppTest

    _orig_from_file = AppTest.from_file

    @classmethod
    def _patched_from_file(cls, script_path, *args, **kwargs):
        p = Path(script_path)
        if not p.is_absolute():
            candidate = PROJECT_ROOT / p
            if candidate.exists():
                script_path = str(candidate)
        return _orig_from_file(script_path, *args, **kwargs)

    AppTest.from_file = _patched_from_file
except ImportError:
    pass
