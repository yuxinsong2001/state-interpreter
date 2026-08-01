import pytest
import torch

from state_interpreter.encoders import SmallConvAutoEncoder
from state_interpreter.preprocessing import LogSTFTPreprocessor


def test_log_stft_output_shape_and_finite_values() -> None:
    preprocessor = LogSTFTPreprocessor()
    vibration = torch.randn(2, 32768)

    transformed = preprocessor(vibration)

    assert transformed.shape == (2, 32, 32)
    assert transformed.dtype == torch.float32
    assert bool(torch.isfinite(transformed).all())
    assert bool((transformed >= 0).all())


def test_log_stft_is_deterministic() -> None:
    preprocessor = LogSTFTPreprocessor()
    vibration = torch.randn(2, 32768)

    first = preprocessor(vibration)
    second = preprocessor(vibration)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_log_stft_preserves_amplitude_information() -> None:
    preprocessor = LogSTFTPreprocessor()
    vibration = torch.randn(2, 32768)

    original = preprocessor(vibration)
    amplified = preprocessor(2.0 * vibration)

    assert amplified.mean() > original.mean()


def test_log_stft_connects_to_autoencoder() -> None:
    preprocessor = LogSTFTPreprocessor()
    autoencoder = SmallConvAutoEncoder(input_channels=2, latent_dim=8)

    time_frequency = preprocessor(torch.randn(2, 32768))
    output = autoencoder(time_frequency.unsqueeze(0))

    assert output.z.shape == (1, 8)
    assert output.reconstruction.shape == (1, 2, 32, 32)


def test_log_stft_rejects_non_finite_input() -> None:
    preprocessor = LogSTFTPreprocessor()
    vibration = torch.zeros(2, 32768)
    vibration[0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN or infinite"):
        preprocessor(vibration)

