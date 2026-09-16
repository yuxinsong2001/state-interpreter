import pytest
import torch

from state_interpreter import LeftRightGaussianHMMStateInterpreter


def synthetic_sequences() -> list[torch.Tensor]:
    return [
        torch.cat(
            (
                torch.linspace(0.0, 0.3, 12),
                torch.linspace(2.0, 2.3, 12),
                torch.linspace(4.0, 4.3, 12),
            )
        )[:, None]
        + offset
        for offset in (0.0, 0.05, -0.05)
    ]


def test_hmm_fits_and_returns_normalized_online_probabilities() -> None:
    model = LeftRightGaussianHMMStateInterpreter(
        embedding_dim=1, num_states=3, max_iterations=30
    ).fit(synthetic_sequences())

    outputs = model.transform(synthetic_sequences()[0])
    assert len(outputs) == 36
    for output in outputs:
        assert output.state.shape == (4,)
        torch.testing.assert_close(
            output.stage_probabilities.sum(), torch.tensor(1.0, dtype=torch.float64)
        )
        assert 0.0 <= float(output.expected_stage) <= 1.0
    assert float(outputs[-1].expected_stage) > float(outputs[0].expected_stage)


def test_transition_matrix_is_left_to_right() -> None:
    model = LeftRightGaussianHMMStateInterpreter(
        embedding_dim=1, num_states=3, max_iterations=5
    ).fit(synthetic_sequences())

    assert float(model.transition_matrix_[0, 2]) == 0.0
    assert float(model.transition_matrix_[1, 0]) == 0.0
    assert float(model.transition_matrix_[2, :2].sum()) == 0.0
    torch.testing.assert_close(
        model.transition_matrix_.sum(dim=1), torch.ones(3, dtype=torch.float64)
    )


def test_transform_is_causal_and_resettable() -> None:
    model = LeftRightGaussianHMMStateInterpreter(
        embedding_dim=1, num_states=3, max_iterations=10
    ).fit(synthetic_sequences())
    sequence = synthetic_sequences()[0]

    prefix = model.transform(sequence[:18])
    full = model.transform(sequence)
    for prefix_output, full_output in zip(prefix, full[:18]):
        torch.testing.assert_close(
            prefix_output.stage_probabilities,
            full_output.stage_probabilities,
        )


def test_log_domain_filtering_handles_extreme_observation() -> None:
    model = LeftRightGaussianHMMStateInterpreter(
        embedding_dim=1, num_states=3, max_iterations=10
    ).fit(synthetic_sequences())

    output = model.update(torch.tensor([1_000_000.0]))

    assert bool(torch.isfinite(output.stage_probabilities).all())
    torch.testing.assert_close(
        output.stage_probabilities.sum(), torch.tensor(1.0, dtype=torch.float64)
    )


def test_hmm_rejects_invalid_use_and_inputs() -> None:
    model = LeftRightGaussianHMMStateInterpreter(embedding_dim=2, num_states=3)
    with pytest.raises(RuntimeError, match="fit"):
        model.update(torch.zeros(2))
    with pytest.raises(ValueError, match="num_states"):
        model.fit([torch.zeros(2, 2)])
    with pytest.raises(ValueError, match="shape"):
        model.fit([torch.zeros(4, 3)])
    with pytest.raises(ValueError, match="finite"):
        model.fit([torch.tensor([[0.0, 0.0], [1.0, 1.0], [float("nan"), 2.0]])])
