"""Metadata/hash-only preflight; no NPZ parsing and no event outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_interpreter.weak_event_protocol import WeakEventDevelopmentPolicy


def main() -> None:
    policy = WeakEventDevelopmentPolicy.load(
        ROOT / "configs/xjtu_condition3_weak_event_development_v1.json", ROOT
    )
    policy.authorize(["Bearing3_1", "Bearing3_2", "Bearing3_3"])
    print(json.dumps(policy.verify_metadata_and_hashes(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
