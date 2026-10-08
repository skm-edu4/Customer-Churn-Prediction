"""Customer churn prediction package.

Importing this package sets one process-wide guard before any ML library
loads (see below), so every entry point — scripts, API, dashboard, tests —
inherits it automatically.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# macOS dual-libomp workaround
#
# xgboost links ``@rpath/libomp.dylib`` while PyTorch bundles its own
# ``libomp.dylib``. When both runtimes create thread teams in the same
# process they deadlock (or segfault) inside OpenMP barriers. Pinning the
# team size to 1 makes both runtimes coexist safely; at this project's
# dataset sizes the throughput difference is negligible.
#
# Set OMP_NUM_THREADS in your shell *before* running if you want to override.
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "2.0.0"
