# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the W&B summary writer."""

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Any

import pytest

from rsl_rl.utils import wandb_utils


@pytest.mark.parametrize("mode", ["allow", "auto", "must", "never"])
def test_wandb_resume_mode_honors_valid_environment_value(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """The deterministic run ID must obey the launcher's collision policy."""
    monkeypatch.setenv("WANDB_RESUME", mode)

    assert wandb_utils._wandb_resume_mode() == mode


def test_wandb_resume_mode_defaults_to_allow_and_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep requeue compatibility by default while rejecting ambiguous input."""
    monkeypatch.delenv("WANDB_RESUME", raising=False)
    assert wandb_utils._wandb_resume_mode() == "allow"

    monkeypatch.setenv("WANDB_RESUME", "sometimes")
    with pytest.raises(ValueError, match="WANDB_RESUME must be one of"):
        wandb_utils._wandb_resume_mode()


def test_wandb_writer_uses_environment_resume_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A fresh-start launcher must reach :func:`wandb.init` as ``never``."""
    init_calls: list[dict[str, Any]] = []
    monkeypatch.setenv("WANDB_RESUME", "never")
    monkeypatch.setattr(SummaryWriter, "__init__", lambda self, log_dir, flush_secs: None)
    monkeypatch.setattr(wandb_utils.wandb, "Settings", lambda **kwargs: kwargs)
    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: init_calls.append(kwargs))

    wandb_utils.WandbSummaryWriter(
        str(tmp_path / "fresh-run"),
        flush_secs=10,
        cfg={"wandb_project": "test-project", "run_id": "fresh-run-id"},
    )

    assert init_calls[0]["id"] == "fresh-run-id"
    assert init_calls[0]["resume"] == "never"


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
