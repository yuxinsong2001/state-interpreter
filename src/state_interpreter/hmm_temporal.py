"""Left-to-right Gaussian HMM for sequential latent-state interpretation."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch


class HMMStateOutput(NamedTuple):
    """Online state estimate produced from current and past embeddings only."""

    stage: torch.Tensor
    stage_probabilities: torch.Tensor
    expected_stage: torch.Tensor
    confidence: torch.Tensor
    state: torch.Tensor


class LeftRightGaussianHMMStateInterpreter:
    """Diagonal-Gaussian HMM with self/next-state transitions.

    Model fitting uses complete training sequences, while ``update`` performs
    causal filtering. The state order is fixed by a relative-time
    initialization and a left-to-right transition mask. Consequently, the
    learned state numbers are temporal stages; health semantics must be checked
    after fitting rather than assumed from the numeric labels.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_states: int = 3,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        variance_floor: float = 1e-4,
        transition_floor: float = 1e-6,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_states < 2:
            raise ValueError("num_states must be at least 2")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if variance_floor <= 0 or transition_floor <= 0:
            raise ValueError("floors must be positive")
        self.embedding_dim = embedding_dim
        self.num_states = num_states
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.variance_floor = variance_floor
        self.transition_floor = transition_floor
        self._fitted = False
        self._filtered_log_probabilities: torch.Tensor | None = None

    @property
    def fitted(self) -> bool:
        return self._fitted

    def _validate_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 2 or sequence.shape[1] != self.embedding_dim:
            raise ValueError(
                f"expected sequence shape [time, {self.embedding_dim}], "
                f"got {tuple(sequence.shape)}"
            )
        if len(sequence) < self.num_states:
            raise ValueError("each sequence must contain at least num_states samples")
        if not torch.is_floating_point(sequence):
            raise ValueError("sequence must be floating-point")
        if not bool(torch.isfinite(sequence).all()):
            raise ValueError("sequence contains NaN or infinite values")
        return sequence.detach().to(dtype=torch.float64, device="cpu").clone()

    def _validate_observation(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 1 or z.numel() != self.embedding_dim:
            raise ValueError(
                f"expected z shape [{self.embedding_dim}], got {tuple(z.shape)}"
            )
        if not torch.is_floating_point(z) or not bool(torch.isfinite(z).all()):
            raise ValueError("z must be a finite floating-point tensor")
        return z.detach().to(dtype=torch.float64, device="cpu").clone()

    def _transition_mask(self) -> torch.Tensor:
        mask = torch.eye(self.num_states, dtype=torch.bool)
        indices = torch.arange(self.num_states - 1)
        mask[indices, indices + 1] = True
        return mask

    def _initialize(self, sequences: Sequence[torch.Tensor]) -> None:
        all_observations = torch.cat(tuple(sequences))
        self.feature_mean_ = all_observations.mean(dim=0)
        self.feature_std_ = all_observations.std(dim=0, unbiased=False).clamp_min(
            self.variance_floor**0.5
        )
        standardized = [
            (sequence - self.feature_mean_) / self.feature_std_
            for sequence in sequences
        ]
        assignments: list[list[torch.Tensor]] = [
            [] for _ in range(self.num_states)
        ]
        for sequence in standardized:
            positions = torch.arange(len(sequence), dtype=torch.float64) / len(sequence)
            labels = torch.clamp(
                (positions * self.num_states).to(torch.long),
                max=self.num_states - 1,
            )
            for state in range(self.num_states):
                assignments[state].append(sequence[labels == state])
        self.means_ = torch.stack(
            [torch.cat(parts).mean(dim=0) for parts in assignments]
        )
        self.variances_ = torch.stack(
            [
                torch.cat(parts).var(dim=0, unbiased=False).clamp_min(
                    self.variance_floor
                )
                for parts in assignments
            ]
        )
        transition = torch.zeros(
            self.num_states, self.num_states, dtype=torch.float64
        )
        for state in range(self.num_states - 1):
            transition[state, state] = 0.95
            transition[state, state + 1] = 0.05
        transition[-1, -1] = 1.0
        self.transition_matrix_ = transition
        self.start_probabilities_ = torch.zeros(
            self.num_states, dtype=torch.float64
        )
        self.start_probabilities_[0] = 1.0

    def _log_emissions(self, standardized: torch.Tensor) -> torch.Tensor:
        difference = standardized[:, None, :] - self.means_[None, :, :]
        log_density = -0.5 * (
            torch.log(2.0 * torch.pi * self.variances_)[None, :, :]
            + difference.square() / self.variances_[None, :, :]
        )
        return log_density.sum(dim=-1)

    def _log_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        log_start = torch.full(
            (self.num_states,), -torch.inf, dtype=torch.float64
        )
        log_start[0] = 0.0
        log_transition = torch.full(
            (self.num_states, self.num_states), -torch.inf, dtype=torch.float64
        )
        mask = self._transition_mask()
        log_transition[mask] = torch.log(self.transition_matrix_[mask])
        return log_start, log_transition

    def _expectation(
        self, standardized: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emission = self._log_emissions(standardized)
        log_start, log_transition = self._log_parameters()
        length = len(standardized)
        alpha = torch.empty(length, self.num_states, dtype=torch.float64)
        alpha[0] = log_start + emission[0]
        for step in range(1, length):
            alpha[step] = emission[step] + torch.logsumexp(
                alpha[step - 1][:, None] + log_transition, dim=0
            )
        log_likelihood = torch.logsumexp(alpha[-1], dim=0)
        beta = torch.zeros(length, self.num_states, dtype=torch.float64)
        for step in range(length - 2, -1, -1):
            beta[step] = torch.logsumexp(
                log_transition + emission[step + 1][None, :] + beta[step + 1][None, :],
                dim=1,
            )
        gamma = torch.exp(alpha + beta - log_likelihood)
        xi = torch.zeros(
            self.num_states, self.num_states, dtype=torch.float64
        )
        for step in range(length - 1):
            xi += torch.exp(
                alpha[step][:, None]
                + log_transition
                + emission[step + 1][None, :]
                + beta[step + 1][None, :]
                - log_likelihood
            )
        return gamma, xi, log_likelihood

    def fit(self, sequences: Sequence[torch.Tensor]) -> "LeftRightGaussianHMMStateInterpreter":
        """Fit emission and transition parameters on ordered training sequences."""

        if not sequences:
            raise ValueError("at least one training sequence is required")
        clean = tuple(self._validate_sequence(sequence) for sequence in sequences)
        self._initialize(clean)
        standardized = tuple(
            (sequence - self.feature_mean_) / self.feature_std_ for sequence in clean
        )
        self.log_likelihood_history_: list[float] = []
        for _ in range(self.max_iterations):
            state_weight = torch.zeros(self.num_states, dtype=torch.float64)
            weighted_sum = torch.zeros(
                self.num_states, self.embedding_dim, dtype=torch.float64
            )
            weighted_square_sum = torch.zeros_like(weighted_sum)
            transition_count = torch.zeros(
                self.num_states, self.num_states, dtype=torch.float64
            )
            total_likelihood = 0.0
            for sequence in standardized:
                gamma, xi, likelihood = self._expectation(sequence)
                state_weight += gamma.sum(dim=0)
                weighted_sum += gamma.T @ sequence
                weighted_square_sum += gamma.T @ sequence.square()
                transition_count += xi
                total_likelihood += float(likelihood)
            safe_weight = state_weight.clamp_min(self.transition_floor)
            self.means_ = weighted_sum / safe_weight[:, None]
            second_moment = weighted_square_sum / safe_weight[:, None]
            self.variances_ = (
                second_moment - self.means_.square()
            ).clamp_min(self.variance_floor)
            mask = self._transition_mask()
            for state in range(self.num_states):
                allowed = mask[state]
                counts = transition_count[state, allowed] + self.transition_floor
                self.transition_matrix_[state] = 0.0
                self.transition_matrix_[state, allowed] = counts / counts.sum()
            self.log_likelihood_history_.append(total_likelihood)
            if (
                len(self.log_likelihood_history_) > 1
                and abs(
                    self.log_likelihood_history_[-1]
                    - self.log_likelihood_history_[-2]
                )
                <= self.tolerance
                * (1.0 + abs(self.log_likelihood_history_[-2]))
            ):
                break
        self.n_iterations_ = len(self.log_likelihood_history_)
        self._fitted = True
        self.reset()
        return self

    def reset(self) -> None:
        """Reset causal filtering before a new bearing sequence."""

        self._filtered_log_probabilities = None

    def update(self, z: torch.Tensor) -> HMMStateOutput:
        """Consume one embedding and return a causal filtered state estimate."""

        if not self._fitted:
            raise RuntimeError("fit must be called before update")
        current = self._validate_observation(z)
        standardized = (current - self.feature_mean_) / self.feature_std_
        log_likelihood = self._log_emissions(standardized[None, :])[0]
        log_start, log_transition = self._log_parameters()
        if self._filtered_log_probabilities is None:
            log_prior = log_start
        else:
            log_prior = torch.logsumexp(
                self._filtered_log_probabilities[:, None] + log_transition,
                dim=0,
            )
        log_posterior = log_prior + log_likelihood
        normalizer = torch.logsumexp(log_posterior, dim=0)
        if not bool(torch.isfinite(normalizer)):
            raise RuntimeError("HMM filtering produced invalid log probabilities")
        log_posterior = log_posterior - normalizer
        posterior = torch.exp(log_posterior)
        self._filtered_log_probabilities = log_posterior
        indices = torch.arange(self.num_states, dtype=torch.float64)
        expected = (posterior * indices).sum() / (self.num_states - 1)
        confidence, stage = posterior.max(dim=0)
        state = torch.cat((expected[None], posterior))
        return HMMStateOutput(
            stage=stage,
            stage_probabilities=posterior.clone(),
            expected_stage=expected,
            confidence=confidence,
            state=state,
        )

    def transform(self, sequence: torch.Tensor) -> list[HMMStateOutput]:
        """Causally filter a complete sequence after resetting episode state."""

        clean = self._validate_sequence(sequence)
        self.reset()
        return [self.update(z) for z in clean]
