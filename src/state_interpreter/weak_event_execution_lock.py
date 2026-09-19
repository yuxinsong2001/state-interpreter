"""Additional execution-code hashes without editing the frozen preflight config."""

from __future__ import annotations

import json
from pathlib import Path

from state_interpreter.weak_event_protocol import file_sha256


EXPECTED_PATHS = (
    "configs/xjtu_condition3_weak_event_development_v1.json",
    "src/state_interpreter/weak_event_development.py",
    "scripts/run_weak_event_condition3_development.py",
)


def verify_execution_lock(root: Path, lock_path: Path) -> dict:
    root = root.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "execution_code_locked_not_executed":
        raise ValueError("execution lock status changed")
    artifacts = lock.get("artifacts", [])
    if tuple(item.get("path") for item in artifacts) != EXPECTED_PATHS:
        raise ValueError("execution artifact list changed")
    verified = {}
    for item in artifacts:
        expected = item.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("invalid artifact digest")
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != expected:
            raise ValueError(f"execution artifact hash mismatch: {item['path']}")
        verified[item["path"]] = expected
    return verified
