import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "xjtu_condition3_source_dann_development_v1.json"
SPEC = importlib.util.spec_from_file_location(
    "preflight_source_dann_development",
    ROOT / "scripts" / "preflight_source_dann_development.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_source_dann_protocol_passes_without_cache_read() -> None:
    result = MODULE.validate(load_config())
    assert result["status"] == "source_dann_preflight_passed_no_cache_read"
    assert result["protected_bearings"] == ["Bearing3_4", "Bearing3_5"]


@pytest.mark.parametrize("protected", ["Bearing3_4", "Bearing3_5"])
def test_source_dann_rejects_protected_bearing_in_development(protected: str) -> None:
    config = load_config()
    config["development_bearings"][2] = protected
    with pytest.raises(ValueError, match="development split"):
        MODULE.validate(config)


def test_source_dann_requires_true_paired_control() -> None:
    config = load_config()
    config["paired_arms"]["source_only"]["maximum_grl_coefficient"] = 0.01
    with pytest.raises(ValueError, match="source-only"):
        MODULE.validate(config)


def test_source_dann_rejects_future_context() -> None:
    config = load_config()
    config["causal_inference"]["future_context_allowed"] = True
    with pytest.raises(ValueError, match="future context"):
        MODULE.validate(config)


def test_source_dann_verifies_frozen_hashes() -> None:
    config = copy.deepcopy(load_config())
    config["frozen_artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen artifact"):
        MODULE.validate(config)

