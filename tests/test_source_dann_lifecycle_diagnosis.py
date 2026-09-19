import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_source_dann_lifecycle",
    ROOT / "scripts" / "diagnose_source_dann_lifecycle.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_lifetime_block_boundaries() -> None:
    assert [MODULE.lifecycle_block(value) for value in (0, .1999, .2, .999, 1)] == [0, 0, 1, 4, 4]
    with pytest.raises(ValueError, match="outside"):
        MODULE.lifecycle_block(1.01)


def test_paired_stage_health_rejects_mismatched_timestamps() -> None:
    rows = []
    for arm in MODULE.ARMS:
        for seed in MODULE.SEEDS:
            for bearing in MODULE.BEARINGS:
                for block in range(MODULE.BLOCKS):
                    for point in range(3):
                        changed = (
                            arm == "source_dann"
                            and seed == MODULE.SEEDS[0]
                            and bearing == MODULE.BEARINGS[0]
                            and block == 0
                            and point == 0
                        )
                        rows.append(
                            {
                                "arm": arm,
                                "seed": str(seed),
                                "bearing_id": bearing,
                                "step_id": str(block * 3 + point + (100 if changed else 0)),
                                "normalized_lifetime": str(block / 5 + point / 100),
                                "predicted_health": str(point / 100),
                            }
                        )
    with pytest.raises(ValueError, match="timestamps differ"):
        MODULE.paired_stage_health(rows)
