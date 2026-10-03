"""Data-free demonstration of the relative State Interpreter lifecycle.

The synthetic embeddings illustrate the API only. They are not experimental
results and do not represent physical damage or remaining useful life.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Allow the example to run directly from a source checkout before installation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from state_interpreter import RelativeTemporalStateInterpreter


def main() -> None:
    torch.manual_seed(7)
    interpreter = RelativeTemporalStateInterpreter(
        embedding_dim=8,
        calibration_steps=5,
        temporal_window=4,
    )

    baseline = torch.linspace(-0.2, 0.2, 8)
    direction = torch.tensor([1.0, 0.6, 0.3, 0.15, 0.0, -0.1, -0.2, -0.3])

    print("step phase         level    trend movement")
    for step in range(12):
        # Small deterministic noise plus a gradual post-calibration drift.
        noise = 0.01 * torch.randn(8)
        progress = max(0, step - 4) * 0.08
        z_t = baseline + progress * direction + noise
        output = interpreter.update(z_t)
        if output is None:
            print(f"{step:>4} {interpreter.phase.value:<13} --       --       --")
        else:
            print(
                f"{step:>4} {interpreter.phase.value:<13} "
                f"{output.level.item():>7.4f} "
                f"{output.trend.item():>7.4f} "
                f"{output.movement.item():>8.4f}"
            )

    assert interpreter.ready
    assert output is not None
    assert output.state.shape == (3,)


if __name__ == "__main__":
    main()

