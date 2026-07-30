from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

FORBIDDEN_NAME = "bian_new_cases_20260728_20260729_final"


def assert_blind_path(path: str | Path) -> Path:
    p = Path(path).resolve()
    if FORBIDDEN_NAME in p.parts:
        raise PermissionError(f"truth-access guard rejected {p}")
    return p


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def append_audit(path: str | Path, action: str, target: str) -> None:
    record = {"at": datetime.now(timezone.utc).isoformat(), "action": action, "target": target}
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

