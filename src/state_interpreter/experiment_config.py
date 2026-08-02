"""Read-only experiment configuration and leakage guards for v2.x runs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
DevelopmentPurpose = Literal[
    "arm_a_development",
    "arm_b_train",
    "arm_b_validation",
    "arm_b_development",
]


@dataclass(frozen=True)
class BearingSplit:
    """Immutable bearing-wise split with one protected blind holdout."""

    development_or_train: tuple[str, ...]
    validation: str
    blind_holdout: str

    @property
    def development(self) -> tuple[str, ...]:
        return self.development_or_train + (self.validation,)

    def assert_excludes_holdout(
        self, bearings: Sequence[str], *, purpose: str
    ) -> None:
        if self.blind_holdout in bearings:
            raise ValueError(
                f"blind holdout {self.blind_holdout!r} is forbidden during "
                f"{purpose}"
            )


@dataclass(frozen=True)
class FrozenArtifact:
    """Repository-relative path and preregistered SHA-256 digest."""

    path: str
    sha256: str


@dataclass(frozen=True)
class CrossConditionExperimentConfig:
    """Validated immutable subset of the v2.1 two-arm configuration."""

    config_id: str
    status: str
    source_condition: str
    target_condition: str
    split: BearingSplit
    embedding_dim: int
    calibration_steps: int
    temporal_window: int
    state_fields: tuple[str, ...]
    source_checkpoint: FrozenArtifact
    source_config: FrozenArtifact
    arm_b_train_bearings: tuple[str, ...]
    arm_b_validation_bearing: str
    arm_b_normalization_bearings: tuple[str, ...]

    def bearings_for(self, purpose: DevelopmentPurpose) -> tuple[str, ...]:
        """Return only preregistered non-blind bearings for a development task."""

        selections = {
            "arm_a_development": self.split.development,
            "arm_b_train": self.arm_b_train_bearings,
            "arm_b_validation": (self.arm_b_validation_bearing,),
            "arm_b_development": self.split.development,
        }
        bearings = selections[purpose]
        self.split.assert_excludes_holdout(bearings, purpose=purpose)
        return bearings


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _nonempty_names(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty JSON array")
    names = tuple(value)
    if any(not isinstance(item, str) or not item for item in names):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError(f"{name} contains duplicate bearings")
    return names


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 string")
    digest = value.lower()
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name} must contain 64 hexadecimal characters")
    return digest


def load_cross_condition_config(
    path: str | Path,
) -> CrossConditionExperimentConfig:
    """Load and validate v2.1 without changing the configuration file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    root = _mapping(raw, "config")

    if root.get("status") != "preregistered_not_run":
        raise ValueError("configuration is not in preregistered_not_run state")

    split_raw = _mapping(root.get("data_split"), "data_split")
    train = _nonempty_names(
        split_raw.get("development_or_train_bearings"),
        "data_split.development_or_train_bearings",
    )
    validation = split_raw.get("validation_bearing")
    holdout = split_raw.get("strict_blind_holdout")
    if not isinstance(validation, str) or not validation:
        raise ValueError("data_split.validation_bearing must be a non-empty string")
    if not isinstance(holdout, str) or not holdout:
        raise ValueError(
            "data_split.strict_blind_holdout must be a non-empty string"
        )
    all_bearings = train + (validation, holdout)
    if len(all_bearings) != len(set(all_bearings)):
        raise ValueError("train, validation, and blind holdout must be disjoint")
    split = BearingSplit(train, validation, holdout)

    audit = _mapping(root.get("blind_holdout_audit"), "blind_holdout_audit")
    if audit.get("latent_or_state_outputs_inspected") is not False:
        raise ValueError("blind holdout latent/state outputs are not isolated")
    if audit.get("signal_statistics_inspected") is not False:
        raise ValueError("blind holdout signal statistics are not isolated")

    interpreter = _mapping(root.get("shared_interpreter"), "shared_interpreter")
    if interpreter.get("parameters_may_be_tuned") is not False:
        raise ValueError("State Interpreter parameters must remain frozen")
    state_fields = tuple(interpreter.get("state_fields", ()))
    if state_fields != ("level", "trend", "movement"):
        raise ValueError("state_fields must be [level, trend, movement]")

    arms = _mapping(root.get("arms"), "arms")
    arm_a = _mapping(
        arms.get("arm_a_direct_generalization"),
        "arms.arm_a_direct_generalization",
    )
    if arm_a.get("training_allowed") is not False:
        raise ValueError("training must be disabled for direct generalization")
    if arm_a.get("normalization") != "load_source_checkpoint_statistics":
        raise ValueError("arm A must use source checkpoint normalization")

    arm_b = _mapping(
        arms.get("arm_b_adapted_encoder"), "arms.arm_b_adapted_encoder"
    )
    arm_b_train = _nonempty_names(
        arm_b.get("train_bearings"), "arms.arm_b_adapted_encoder.train_bearings"
    )
    arm_b_validation = arm_b.get("validation_bearing")
    arm_b_normalization = _nonempty_names(
        arm_b.get("normalization_fit_on"),
        "arms.arm_b_adapted_encoder.normalization_fit_on",
    )
    if arm_b_train != train:
        raise ValueError("arm B train bearings must match the shared split")
    if arm_b_validation != validation:
        raise ValueError("arm B validation bearing must match the shared split")
    if arm_b_normalization != train:
        raise ValueError("arm B normalization must be fitted on train bearings only")
    split.assert_excludes_holdout(arm_b_train, purpose="arm B training")
    split.assert_excludes_holdout(
        (str(arm_b_validation),), purpose="arm B validation"
    )

    joint = _mapping(root.get("joint_blind_evaluation"), "joint_blind_evaluation")
    if joint.get("single_script_reads_holdout_once") is not True:
        raise ValueError("blind holdout must be read once by one joint script")
    if joint.get("both_arms_reported_together") is not True:
        raise ValueError("both arms must be reported together")
    if joint.get("blind_holdout_evaluated") is not False:
        raise ValueError("blind holdout has already been evaluated")
    if joint.get("retuning_after_evaluation_allowed") is not False:
        raise ValueError("retuning after blind evaluation must be forbidden")

    return CrossConditionExperimentConfig(
        config_id=str(root.get("config_id", "")),
        status=str(root["status"]),
        source_condition=str(root.get("source_condition", "")),
        target_condition=str(root.get("target_condition", "")),
        split=split,
        embedding_dim=int(interpreter.get("embedding_dim", 0)),
        calibration_steps=int(interpreter.get("calibration_steps", 0)),
        temporal_window=int(interpreter.get("temporal_window", 0)),
        state_fields=state_fields,
        source_checkpoint=FrozenArtifact(
            path=str(arm_a.get("checkpoint", "")),
            sha256=_sha256(arm_a.get("checkpoint_sha256"), "checkpoint_sha256"),
        ),
        source_config=FrozenArtifact(
            path=str(arm_a.get("source_config", "")),
            sha256=_sha256(
                arm_a.get("source_config_sha256_at_amendment"),
                "source_config_sha256_at_amendment",
            ),
        ),
        arm_b_train_bearings=arm_b_train,
        arm_b_validation_bearing=str(arm_b_validation),
        arm_b_normalization_bearings=arm_b_normalization,
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_source_artifacts(
    config: CrossConditionExperimentConfig, repository_root: str | Path
) -> dict[str, str]:
    """Verify arm A inputs and reject paths escaping the repository root."""

    root = Path(repository_root).resolve()
    verified: dict[str, str] = {}
    for name, artifact in (
        ("source_checkpoint", config.source_checkpoint),
        ("source_config", config.source_config),
    ):
        candidate = (root / artifact.path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{name} path escapes repository root") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"{name} does not exist: {candidate}")
        actual = file_sha256(candidate)
        if actual != artifact.sha256:
            raise ValueError(
                f"{name} SHA-256 mismatch: expected {artifact.sha256}, got {actual}"
            )
        verified[name] = actual
    return verified

