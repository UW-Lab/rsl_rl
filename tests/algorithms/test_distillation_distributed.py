# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distributed correctness tests for the legacy DAgger path."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from rsl_rl.algorithms import DistillationLegacy
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.runners import DistillationRunnerSplit


def _distributed_worker(rank: int, world_size: int, init_file: str, output_dir: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        normalizer = EmpiricalNormalization(1)
        normalizer.train()
        normalizer.set_distributed_sync(True)
        local_batch = torch.tensor([[4.0 * rank], [4.0 * rank + 2.0]])
        normalizer.update(local_batch)

        policy = nn.Linear(1, 1, bias=False)
        algorithm = DistillationLegacy(policy, multi_gpu_cfg={"global_rank": rank, "world_size": world_size})
        policy.weight.grad = torch.full_like(policy.weight, float(rank + 1))
        algorithm.reduce_parameters()

        torch.save(
            {
                "mean": normalizer._mean,
                "var": normalizer._var,
                "count": normalizer.count,
                "grad": policy.weight.grad,
            },
            Path(output_dir) / f"rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_distributed_normalizer_and_gradients_agree(tmp_path: Path) -> None:
    """Two ranks should merge moments and average gradients identically."""
    init_file = tmp_path / "dist_init"
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_distributed_worker, args=(rank, 2, str(init_file), str(tmp_path)))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    rank_0 = torch.load(tmp_path / "rank_0.pt", weights_only=True)
    rank_1 = torch.load(tmp_path / "rank_1.pt", weights_only=True)
    for key in ("mean", "var", "count", "grad"):
        assert torch.equal(rank_0[key], rank_1[key]), key
    assert rank_0["count"].item() == 4
    assert rank_0["mean"].item() == pytest.approx(3.0)
    assert rank_0["var"].item() == pytest.approx(5.0)
    assert rank_0["grad"].item() == pytest.approx(1.5)


def test_distillation_checkpoint_restores_update_clock() -> None:
    """Resume must preserve annealing and backbone-unfreeze update clocks."""
    source = DistillationLegacy(nn.Linear(2, 1))
    source.num_updates = 37
    checkpoint = source.save()

    restored = DistillationLegacy(nn.Linear(2, 1))
    restored.load(checkpoint, load_cfg=None, strict=True)
    assert restored.num_updates == 37


def test_split_runner_rejects_collective_unsafe_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every distributed gradient chunk must contain at least one train row."""
    monkeypatch.setenv("WORLD_SIZE", "2")

    class FakeEnv:
        num_envs = 8

    train_cfg = {
        "student_fraction": 0.5,
        "eval_fraction": 0.875,
        "teacher_eval_fraction": 0.0,
        "algorithm": {"num_mini_batches": 2},
    }
    with pytest.raises(ValueError, match="fewer train rows than minibatches"):
        DistillationRunnerSplit(FakeEnv(), train_cfg, device="cpu")
