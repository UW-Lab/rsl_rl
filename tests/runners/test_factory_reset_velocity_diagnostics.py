# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for task-provided component velocity metrics in rank diagnostics."""

import torch
from types import SimpleNamespace

from rsl_rl.runners import OnPolicyRunner


class _Buffer:
    def __init__(self) -> None:
        self.data = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        self.tags = torch.tensor([0, 1])

    def __len__(self) -> int:
        return len(self.data)


def test_reset_velocity_dictionaries_are_forwarded_to_rank_metrics() -> None:
    """Only compact task velocity diagnostics should reach per-rank telemetry."""
    reset = SimpleNamespace(
        buffer=_Buffer(),
        success_rate=torch.tensor([0.25, 0.75]),
        sampled_tags=torch.tensor([0, 1]),
        names=("air", "assembled"),
        gps_mode="per_strategy",
        last_sampling_probs=torch.tensor([0.4, 0.6]),
        success_monitor=None,
        first_step_velocity_last={
            "robot_joint_velocity_absmax": torch.tensor(240.0),
            "passive_mimic_joint_velocity_absmax": torch.tensor(130.0),
            "passive_mimic_joint_velocity_limit_min": torch.tensor(130.0),
            "passive_mimic_joint_velocity_limit_max": torch.tensor(130.0),
            "passive_mimic_joint_velocity_limit_backend_min": torch.tensor(130.0),
            "passive_mimic_joint_velocity_limit_backend_max": torch.tensor(130.0),
            "passive_mimic_joint_velocity_limit_backend_requested_mismatch_fraction": torch.tensor(0.0),
            "passive_mimic_joint_velocity_limit_backend_cache_mismatch_fraction": torch.tensor(0.0),
            "held_asset_linear_speed_max": torch.tensor(4.0),
            "held_asset_angular_speed_max": torch.tensor(8.0),
            "joint_velocity_absmax/left_inner_knuckle_joint": torch.tensor(129.0),
        },
        first_step_velocity_max_ever={
            "robot_joint_velocity_absmax": torch.tensor(7200.0),
            "passive_mimic_joint_velocity_absmax": torch.tensor(6400.0),
            "passive_mimic_joint_velocity_limit_min": torch.tensor(130.0),
            "passive_mimic_joint_velocity_limit_max": torch.tensor(130.0),
            "passive_mimic_joint_velocity_limit_backend_min": torch.tensor(130.0),
            "passive_mimic_joint_velocity_limit_backend_max": torch.tensor(130.0),
            "passive_mimic_joint_velocity_limit_backend_requested_mismatch_fraction": torch.tensor(0.0),
            "passive_mimic_joint_velocity_limit_backend_cache_mismatch_fraction": torch.tensor(0.0),
            "held_asset_linear_speed_max": torch.tensor(41.0),
            "held_asset_angular_speed_max": torch.tensor(91.0),
            "joint_velocity_absmax/left_inner_knuckle_joint": torch.tensor(6300.0),
        },
    )
    event_manager = SimpleNamespace(
        get_term_cfg=lambda name: SimpleNamespace(func=reset) if name == "reset_positioning" else None
    )
    runner = object.__new__(OnPolicyRunner)
    runner.device = "cpu"
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(event_manager=event_manager))

    metrics = runner._local_reset_diagnostics()

    assert metrics["first_step_velocity/last/passive_mimic_joint_velocity_absmax"].item() == 130.0
    assert metrics["first_step_velocity/last/held_asset_angular_speed_max"].item() == 8.0
    assert metrics["first_step_velocity/max_ever/passive_mimic_joint_velocity_absmax"].item() == 6400.0
    assert metrics["first_step_velocity/max_ever/held_asset_angular_speed_max"].item() == 91.0
    assert (
        metrics[
            "first_step_velocity/last/passive_mimic_joint_velocity_limit_backend_requested_mismatch_fraction"
        ].item()
        == 0.0
    )
    assert all("joint_velocity_absmax/left_inner_knuckle_joint" not in name for name in metrics)

    # Verify the same task-provided scalar survives the rank gather and reaches
    # the fully namespaced dictionary that Logger forwards to W&B/TensorBoard.
    runner.alg = SimpleNamespace(
        actor=torch.nn.Linear(1, 1),
        critic=torch.nn.Linear(1, 1),
        storage=SimpleNamespace(actions=None),
        learning_rate=1e-3,
    )
    runner.gpu_local_rank = 0
    runner.gpu_global_rank = 0
    runner.gpu_world_size = 1
    runner.is_distributed = False
    gathered = runner._gather_rank_diagnostics({}, {})

    assert gathered is not None
    assert (
        gathered["RankDiagnostics/rank_00/first_step_velocity/max_ever/passive_mimic_joint_velocity_absmax"] == 6400.0
    )
    assert all("joint_velocity_absmax/left_inner_knuckle_joint" not in name for name in gathered)


def test_reset_velocity_schema_is_fixed_when_task_payload_is_missing() -> None:
    """A rank-local read failure must retain every collective velocity slot."""
    runner = object.__new__(OnPolicyRunner)
    runner.device = "cpu"
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace())

    metrics = runner._local_reset_diagnostics()

    velocity_metrics = {name: value for name, value in metrics.items() if name.startswith("first_step_velocity/")}
    assert len(velocity_metrics) == 20
    assert all(torch.isnan(value) for value in velocity_metrics.values())
