from __future__ import annotations

import sys
from pathlib import Path


def ensure_shared_io_path() -> Path:
    """Expose the repository-level tensorsuiteIO package on sys.path."""
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root / "tensorsuiteIO"
