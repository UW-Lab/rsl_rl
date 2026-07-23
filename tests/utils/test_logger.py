# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the training logger."""

import torch
from pathlib import Path
from typing import Any

from rsl_rl.utils.logger import Logger


class _RecordingWriter:
    def __init__(self) -> None:
        self.scalar_calls: list[tuple[str, Any, int | None]] = []
        self.scalar_batch_calls: list[tuple[dict[str, float], int | None]] = []

    def add_scalar(self, tag: str, scalar_value: Any, global_step: int | None = None) -> None:
        self.scalar_calls.append((tag, scalar_value, global_step))

    def add_scalars_batch(self, metrics: dict[str, float], global_step: int | None = None) -> None:
        self.scalar_batch_calls.append((metrics, global_step))

    def save_video(self, video: Path, global_step: int) -> None:
        pass


def _run_log(tmp_path: Path, logger_type: str) -> _RecordingWriter:
    logger = Logger(
        log_dir=str(tmp_path),
        cfg={"num_steps_per_env": 1, "algorithm": {"rnd_cfg": None}},
        env_cfg={},
        num_envs=1,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cpu",
    )
    writer = _RecordingWriter()
    logger.writer = writer
    logger.logger_type = logger_type
    metrics = {
        "RankDiagnostics/rank_00/passive_velocity": 12.0,
        "RankDiagnostics/rank_01/passive_velocity": 15.0,
    }

    logger.log(
        it=3,
        start_it=0,
        total_it=4,
        collect_time=1.0,
        learn_time=1.0,
        loss_dict={},
        learning_rate=1e-3,
        action_std=torch.ones(1),
        rnd_weight=None,
        rank_diagnostics=metrics,
        print_minimal=True,
    )
    return writer


def test_wandb_rank_diagnostics_use_batch_writer(tmp_path: Path) -> None:
    """W&B should receive the full rank payload through its batch path."""
    writer = _run_log(tmp_path, "wandb")

    assert writer.scalar_batch_calls == [
        (
            {
                "RankDiagnostics/rank_00/passive_velocity": 12.0,
                "RankDiagnostics/rank_01/passive_velocity": 15.0,
            },
            3,
        )
    ]
    assert all(not tag.startswith("RankDiagnostics/") for tag, _, _ in writer.scalar_calls)


def test_non_wandb_rank_diagnostics_keep_individual_scalar_writes(tmp_path: Path) -> None:
    """Other writers should retain their original scalar-at-a-time behavior."""
    writer = _run_log(tmp_path, "tensorboard")

    assert writer.scalar_batch_calls == []
    assert [call for call in writer.scalar_calls if call[0].startswith("RankDiagnostics/")] == [
        ("RankDiagnostics/rank_00/passive_velocity", 12.0, 3),
        ("RankDiagnostics/rank_01/passive_velocity", 15.0, 3),
    ]
