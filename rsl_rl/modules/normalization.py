# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2020 Preferred Networks, Inc.


from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-2,
        until: int | None = None,
        clip: float | None = 10.0,
        outlier_clip: float | None = 10.0,
        outlier_warmup: int = 1_000_000,
    ) -> None:
        """Initialize EmpiricalNormalization module.

        .. note:: The normalization parameters are computed over the whole batch, not for each environment separately.

        Args:
            shape: Shape of input values except batch axis.
            eps: Small value for stability.
            until: If this arg is specified, the module learns input values until the sum of batch sizes exceeds it.
            clip: Clamp normalized outputs to ``[-clip, clip]`` (``None`` disables).
            outlier_clip: Winsorize inputs to ``mean +- outlier_clip * std`` before they enter the running
                moments (``None`` disables). The running variance never forgets, so a single glitched sample
                (a 1e3 joint velocity from a PhysX blow-up) otherwise inflates a feature's std for the rest of
                the run and silently rescales that input for the policy.
            outlier_warmup: Winsorizing starts once ``count`` exceeds this many samples (the moments need to be
                trustworthy first).
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.clip = clip
        self.outlier_clip = outlier_clip
        self.outlier_warmup = outlier_warmup
        # Fraction of elements winsorized by the last ``update`` (diagnostic, not checkpointed).
        self.last_outlier_frac = 0.0
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self) -> torch.Tensor:
        """Return the current running mean."""
        return self._mean.squeeze(0).clone()  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the current running standard deviation."""
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize mean and variance of values based on empirical values."""
        y = (x - self._mean) / (self._std + self.eps)
        if self.clip is not None:
            y = y.clamp(-self.clip, self.clip)
        return y

    @torch.jit.unused
    def _winsorize(self, x: torch.Tensor) -> torch.Tensor:
        if self.outlier_clip is None or int(self.count) < self.outlier_warmup:
            return x
        width = self.outlier_clip * (self._std + self.eps)
        clamped = torch.maximum(torch.minimum(x, self._mean + width), self._mean - width)
        self.last_outlier_frac = (clamped != x).float().mean()  # tensor: no device sync per step
        return clamped

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """Learn input values without computing the output values of them."""
        if not self.training:
            return
        if self.until is not None and self.count >= self.until:
            return

        x = self._winsorize(x)
        count_x = torch.tensor(x.shape[0], dtype=self.count.dtype, device=self.count.device)  # type: ignore
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)

        if torch.distributed.is_initialized():
            # Compute the global mean first, then combine the local variances around that mean
            local_mean_x = mean_x
            torch.distributed.all_reduce(count_x)
            mean_sum_x = mean_x * x.shape[0]
            torch.distributed.all_reduce(mean_sum_x)
            mean_x = mean_sum_x / count_x
            var_sum_x = x.shape[0] * (var_x + (local_mean_x - mean_x).square())
            torch.distributed.all_reduce(var_sum_x)
            var_x = var_sum_x / count_x

        self.count += count_x
        rate = count_x / self.count
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """De-normalize values based on empirical values."""
        return y * (self._std + self.eps) + self._mean


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize the scale of the
    rewards so that the value function can learn quickly. We did this by dividing the rewards by a running estimate of
    the standard deviation of the sum of discounted rewards.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-2,
        gamma: float = 0.99,
        until: int | None = None,
    ) -> None:
        """Initialize discounted-reward normalization with running moments."""
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until, clip=None, outlier_clip=None)
        self.disc_avg = _DiscountedAverage(gamma)

    def forward(self, rew: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using the running std of discounted returns."""
        if self.training:
            # Update discounted rewards
            avg = self.disc_avg.update(rew)
            # Update moments from discounted rewards
            self.emp_norm.update(avg)

        # Normalize rewards with the empirical std
        if self.emp_norm._std > 0:  # type: ignore
            return rew / self.emp_norm._std  # type: ignore
        else:
            return rew


class _DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t
    """

    def __init__(self, gamma: float) -> None:
        """Initialize discounted accumulation with a fixed discount factor."""
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        """Update and return the discounted running average."""
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg
