import torch

from state_interpreter import MLPStateInterpreter


def test_mlp_state_interpreter_output_shapes() -> None:
    model = MLPStateInterpreter(embedding_dim=256, hidden_dim=32, num_stages=4)
    fake_z_health = torch.randn(8, 256)

    output = model(fake_z_health)

    assert output.health_index.shape == (8, 1)
    assert output.stage_logits.shape == (8, 4)
    assert output.stage_probabilities.shape == (8, 4)
    assert output.rl_state.shape == (8, 5)
    torch.testing.assert_close(
        output.stage_probabilities.sum(dim=-1),
        torch.ones(8),
    )


def test_mlp_state_interpreter_rejects_wrong_embedding_shape() -> None:
    model = MLPStateInterpreter(embedding_dim=256)

    try:
        model(torch.randn(8, 128))
    except ValueError as exc:
        assert "expected embedding_dim=256" in str(exc)
    else:
        raise AssertionError("wrong embedding dimension should raise ValueError")
