import pytest
import torch

from state_interpreter.encoders import (
    DomainAdversarialHealthEncoder,
    GradientReversal,
    dann_coefficient,
)


def test_gradient_reversal_changes_only_backward_sign() -> None:
    values = torch.tensor([[1.0, -2.0]], requires_grad=True)
    output = GradientReversal()(values, coefficient=0.25)
    assert torch.equal(output, values)
    output.sum().backward()
    assert torch.allclose(values.grad, torch.full_like(values, -0.25))


def test_domain_adversarial_encoder_shapes_and_backward() -> None:
    model = DomainAdversarialHealthEncoder()
    output = model(torch.randn(4, 12, 65), grl_coefficient=0.1)
    assert output.health.shape == (4, 1)
    assert output.embedding.shape == (4, 16)
    assert output.domain_logits.shape == (4, 2)
    (output.health.square().mean() + output.domain_logits.square().mean()).backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_variable_lengths_use_only_valid_prefix() -> None:
    torch.manual_seed(7)
    model = DomainAdversarialHealthEncoder().eval()
    short = torch.randn(1, 5, 65)
    padded = torch.cat((short, torch.zeros(1, 3, 65)), dim=1)
    direct = model(short, grl_coefficient=0.0).embedding
    packed = model(
        padded,
        lengths=torch.tensor([5]),
        grl_coefficient=0.0,
    ).embedding
    assert torch.allclose(direct, packed, atol=1e-6)


def test_dann_schedule_is_bounded_and_monotone() -> None:
    values = [dann_coefficient(step / 10, maximum=0.1) for step in range(11)]
    assert values[0] == pytest.approx(0.0)
    assert values == sorted(values)
    assert values[-1] < 0.1


def test_domain_adversarial_encoder_rejects_invalid_inputs() -> None:
    model = DomainAdversarialHealthEncoder()
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(2, 5, 64))
    with pytest.raises(ValueError, match="finite"):
        model(torch.full((2, 5, 65), float("nan")))
    with pytest.raises(ValueError, match="lengths"):
        model(torch.randn(2, 5, 65), lengths=torch.tensor([6, 5]))

