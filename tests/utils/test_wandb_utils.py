# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the W&B summary writer."""

from torch.utils.tensorboard import SummaryWriter
from typing import Any

import pytest

from rsl_rl.utils import wandb_utils


def test_add_scalars_batch_preserves_tensorboard_and_uses_one_wandb_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch should retain each TB scalar but issue one W&B log call."""
    tensorboard_calls: list[tuple[str, Any, int | None, float | None, bool]] = []
    wandb_calls: list[tuple[dict[str, float], int | None]] = []

    def record_tensorboard_scalar(
        self: SummaryWriter,
        tag: str,
        scalar_value: Any,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        tensorboard_calls.append((tag, scalar_value, global_step, walltime, new_style))

    monkeypatch.setattr(SummaryWriter, "add_scalar", record_tensorboard_scalar)
    monkeypatch.setattr(
        wandb_utils.wandb,
        "log",
        lambda payload, step=None: wandb_calls.append((payload, step)),
    )
    writer = object.__new__(wandb_utils.WandbSummaryWriter)
    metrics = {
        "RankDiagnostics/rank_00/passive_velocity": 12.0,
        "RankDiagnostics/rank_01/passive_velocity": 15.0,
    }

    writer.add_scalars_batch(metrics, global_step=7)

    assert [(tag, value, step) for tag, value, step, _, _ in tensorboard_calls] == [
        ("RankDiagnostics/rank_00/passive_velocity", 12.0, 7),
        ("RankDiagnostics/rank_01/passive_velocity", 15.0, 7),
    ]
    assert wandb_calls == [(metrics, 7)]
