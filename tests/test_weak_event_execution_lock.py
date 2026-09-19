"""Synthetic file fixtures for the separate execution hash gate."""

import hashlib
import json

import pytest

from state_interpreter.weak_event_execution_lock import EXPECTED_PATHS, verify_execution_lock


def test_execution_lock_checks_all_three_artifacts(tmp_path):
    artifacts = []
    for relative in EXPECTED_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic {relative}", encoding="utf-8")
        artifacts.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    lock_path = tmp_path / "execution_lock.json"
    lock_path.write_text(json.dumps({"status": "execution_code_locked_not_executed", "artifacts": artifacts}), encoding="utf-8")
    assert len(verify_execution_lock(tmp_path, lock_path)) == 3
    (tmp_path / EXPECTED_PATHS[-1]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_execution_lock(tmp_path, lock_path)
