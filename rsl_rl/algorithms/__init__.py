# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms."""

from .distillation import Distillation
from .ppo import PPO
from ._distillation_legacy import DistillationLegacy  # noqa: I001  (UW two-camera DAgger)
from .distillation_dagger import DistillationDAgger
from .distillation_dagger_weighted import DistillationDAggerWeighted

__all__ = ["PPO", "Distillation", "DistillationDAgger", "DistillationDAggerWeighted", "DistillationLegacy"]
