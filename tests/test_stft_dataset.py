import torch

from state_interpreter.data import ChannelStandardizer


def test_channel_standardizer_fits_each_channel_independently() -> None:
    tensors = torch.stack(
        [
            torch.stack((torch.full((4, 4), 1.0), torch.full((4, 4), 10.0))),
            torch.stack((torch.full((4, 4), 3.0), torch.full((4, 4), 14.0))),
        ]
    )

    standardizer = ChannelStandardizer.fit(tensors)
    transformed = standardizer.transform(tensors)

    torch.testing.assert_close(
        transformed.mean(dim=(0, 2, 3)), torch.zeros(2), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        transformed.std(dim=(0, 2, 3), correction=0),
        torch.ones(2),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(standardizer.inverse_transform(transformed), tensors)


def test_channel_standardizer_does_not_refit_on_validation_data() -> None:
    train = torch.randn(5, 2, 4, 4)
    validation = torch.randn(3, 2, 4, 4) + 20.0

    standardizer = ChannelStandardizer.fit(train)
    original_mean = standardizer.mean.clone()
    _ = standardizer.transform(validation)

    torch.testing.assert_close(standardizer.mean, original_mean)

