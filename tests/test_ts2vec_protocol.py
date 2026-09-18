import json
from pathlib import Path


def test_real_ts2vec_protocol_has_protected_dual_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/xjtu_condition3_ts2vec_development_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["development_bearings"] == [
        "Bearing3_1",
        "Bearing3_2",
        "Bearing3_3",
    ]
    assert config["protected_validation"] == ["Bearing3_4"]
    assert config["protected_blind"] == ["Bearing3_5"]
    assert config["causal_inference"]["future_context_allowed"] is False
    assert config["evaluation_gate"]["identity"]["mean_accuracy_at_most"] == 0.8
    assert config["training"]["fixed_iterations"] == 200
