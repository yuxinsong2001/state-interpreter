"""Replaceable upstream encoders for State Interpreter experiments."""

from .autoencoder import AutoEncoderOutput, SmallConvAutoEncoder
from .domain_adversarial import (
    DomainAdversarialHealthEncoder,
    DomainAdversarialOutput,
    GradientReversal,
    dann_coefficient,
)
from .feature_lstm import FeatureLSTM, FeatureLSTMOutput

__all__ = [
    "AutoEncoderOutput",
    "DomainAdversarialHealthEncoder",
    "DomainAdversarialOutput",
    "FeatureLSTM",
    "FeatureLSTMOutput",
    "GradientReversal",
    "SmallConvAutoEncoder",
    "dann_coefficient",
]
