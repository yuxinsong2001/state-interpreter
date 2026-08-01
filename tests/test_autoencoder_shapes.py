import pytest
import torch

from state_interpreter.encoders import SmallConvAutoEncoder


@pytest.mark.parametrize("latent_dim", [8, 16])
def test_small_conv_autoencoder_shapes(latent_dim: int) -> None:
    model = SmallConvAutoEncoder(input_channels=2, latent_dim=latent_dim)
    x = torch.randn(4, 2, 32, 32)

    output = model(x)

    assert output.z.shape == (4, latent_dim)
    assert output.reconstruction.shape == x.shape


def test_encode_is_usable_without_decoder() -> None:
    model = SmallConvAutoEncoder(input_channels=2, latent_dim=8)

    z = model.encode(torch.randn(3, 2, 32, 32))

    assert z.shape == (3, 8)


def test_autoencoder_rejects_wrong_input_shape() -> None:
    model = SmallConvAutoEncoder(input_channels=2, latent_dim=8)

    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(4, 1, 32, 32))

