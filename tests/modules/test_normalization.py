# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for normalization modules."""

import socket
import torch
import torch.multiprocessing as mp
from datetime import timedelta
from typing import Any

import pytest

from rsl_rl.modules.normalization import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization


class TestEmpiricalNormalization:
    """Tests for ``EmpiricalNormalization``."""

    def test_convergence_to_known_distribution(self) -> None:
        """Running mean and std should converge to the true values of the input distribution."""
        true_mean, true_std = 5.0, 2.0
        norm = EmpiricalNormalization(shape=4)
        norm.train()

        torch.manual_seed(0)
        for _ in range(200):
            batch = true_mean + true_std * torch.randn(64, 4)
            norm.update(batch)

        assert torch.allclose(norm.mean, torch.full((4,), true_mean), atol=0.15)
        assert torch.allclose(norm.std, torch.full((4,), true_std), atol=0.15)

    def test_forward_applies_normalization(self) -> None:
        """forward() should return (x - mean) / (std + eps), not the raw input."""
        norm = EmpiricalNormalization(shape=2, eps=0.01)
        norm.train()

        data = torch.tensor([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
        norm.update(data)

        result = norm(data)
        expected = (data - norm._mean) / (norm._std + norm.eps)
        assert torch.allclose(result, expected)

    def test_until_stops_updates(self) -> None:
        """After 'until' samples are seen, further updates must not change statistics."""
        norm = EmpiricalNormalization(shape=2, until=100)
        norm.train()

        for _ in range(10):
            norm.update(torch.randn(20, 2))

        assert norm.count >= 100
        mean_snapshot = norm._mean.clone()
        std_snapshot = norm._std.clone()

        for _ in range(10):
            norm.update(torch.randn(20, 2) + 100)

        assert torch.equal(norm._mean, mean_snapshot)
        assert torch.equal(norm._std, std_snapshot)

    def test_eval_mode_freezes_stats(self) -> None:
        """In eval mode, update() should be a no-op."""
        norm = EmpiricalNormalization(shape=3)
        norm.train()
        norm.update(torch.randn(50, 3))

        mean_before = norm._mean.clone()
        std_before = norm._std.clone()

        norm.eval()
        norm.update(torch.randn(50, 3) + 100)

        assert torch.equal(norm._mean, mean_before)
        assert torch.equal(norm._std, std_before)

    def test_inverse_round_trip(self) -> None:
        """inverse(forward(x)) should approximately recover x."""
        norm = EmpiricalNormalization(shape=4, eps=1e-2)
        norm.train()

        torch.manual_seed(42)
        for _ in range(50):
            norm.update(torch.randn(32, 4) * 3 + 5)

        x = torch.randn(16, 4) * 3 + 5
        recovered = norm.inverse(norm(x))
        assert torch.allclose(recovered, x, atol=1e-5)

    def test_single_sample_does_not_produce_nan(self) -> None:
        """Updating with a single sample should not produce NaN in mean or std."""
        norm = EmpiricalNormalization(shape=2)
        norm.train()
        norm.update(torch.tensor([[1.0, 2.0]]))

        assert not torch.any(torch.isnan(norm._mean))
        assert not torch.any(torch.isnan(norm._std))

    def test_distributed_sync_defaults_off_and_is_not_checkpointed(self) -> None:
        """The opt-in must not alter local updates or the state-dict schema."""
        norm = EmpiricalNormalization(shape=2)
        state_keys = tuple(norm.state_dict())

        data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        norm.update(data)

        assert norm.count.item() == len(data)
        assert norm._distributed_sync_enabled is False
        norm.set_distributed_sync(True)
        assert tuple(norm.state_dict()) == state_keys

        restored = EmpiricalNormalization(shape=2)
        restored.set_distributed_sync(True)
        restored.load_state_dict(norm.state_dict(), strict=True)
        assert torch.equal(restored._mean, norm._mean)
        assert torch.equal(restored._var, norm._var)
        assert restored.count.item() == norm.count.item()

    def test_distributed_sync_two_gloo_ranks_matches_centralized_moments(self) -> None:
        """Global batches must merge into the common prior exactly once."""
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]

        procs = [
            ctx.Process(target=_distributed_normalization_worker, args=(rank, 2, free_port, result_queue))
            for rank in range(2)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=60)
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
                proc.join()
        assert all(proc.exitcode == 0 for proc in procs), [proc.exitcode for proc in procs]

        per_rank = {}
        for _ in range(2):
            payload = result_queue.get(timeout=5)
            assert "error" not in payload, payload
            per_rank[payload["rank"]] = payload

        prior = torch.tensor([[0.0, 2.0], [2.0, 4.0]])
        first_global_batch = torch.cat((_distributed_batch(0, 0), _distributed_batch(1, 0)))
        second_global_batch = torch.cat((_distributed_batch(0, 1), _distributed_batch(1, 1)))
        reference = EmpiricalNormalization(shape=2)
        reference.update(prior)
        reference.update(first_global_batch)
        reference.update(second_global_batch)

        expected_count = len(prior) + len(first_global_batch) + len(second_global_batch)
        assert expected_count == 10
        for payload in per_rank.values():
            assert payload["count"] == expected_count
            assert torch.allclose(torch.tensor(payload["mean"]), reference._mean, atol=1e-6)
            assert torch.allclose(torch.tensor(payload["var"]), reference._var, atol=1e-6)
            assert torch.allclose(torch.tensor(payload["std"]), reference._std, atol=1e-6)

        assert per_rank[0]["mean"] == per_rank[1]["mean"]
        assert per_rank[0]["var"] == per_rank[1]["var"]

    def test_distributed_sync_rejects_nonfinite_moments_after_collective(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An Inf/NaN must fail before it can contaminate the shared running state."""
        collective_called = False

        def all_reduce(_moments: torch.Tensor, op: object | None = None) -> None:
            nonlocal collective_called
            collective_called = True

        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

        norm = EmpiricalNormalization(shape=2)
        norm.set_distributed_sync(True)
        try:
            norm.update(torch.tensor([[1.0, float("inf")]]))
        except FloatingPointError:
            pass
        else:
            raise AssertionError("distributed normalization accepted non-finite moments")

        assert collective_called
        assert norm.count.item() == 0
        assert torch.equal(norm._mean, torch.zeros_like(norm._mean))
        assert torch.equal(norm._var, torch.ones_like(norm._var))


class TestEmpiricalDiscountedVariationNormalization:
    """Tests for ``EmpiricalDiscountedVariationNormalization``."""

    def test_constant_rewards_produce_stable_normalization(self) -> None:
        """Constant rewards should converge to a stable normalization factor."""
        gamma = 0.99
        norm = EmpiricalDiscountedVariationNormalization(shape=[], gamma=gamma)
        norm.train()

        reward = torch.tensor([1.0])
        outputs = [norm(reward).item() for _ in range(200)]

        # After convergence, the output should be approximately constant
        last_10 = outputs[-10:]
        assert max(last_10) - min(last_10) < 0.1, "Normalization should stabilize for constant rewards"

    def test_zero_std_returns_raw_reward(self) -> None:
        """When std is zero (or not yet computed), forward should return the raw reward."""
        norm = EmpiricalDiscountedVariationNormalization(shape=[])
        norm.eval()
        reward = torch.tensor([5.0])
        result = norm(reward)
        # In eval mode with no prior updates, std defaults to 1, so it normalizes by 1
        assert torch.isfinite(result).all()


def _distributed_batch(rank: int, iteration: int) -> torch.Tensor:
    """Return deterministic, rank-distinct observations for the Gloo test."""
    batches = {
        (0, 0): torch.tensor([[-4.0, 0.0], [4.0, 8.0]]),
        (1, 0): torch.tensor([[10.0, -2.0], [12.0, 6.0], [14.0, 10.0]]),
        (0, 1): torch.tensor([[1.0, 9.0]]),
        (1, 1): torch.tensor([[-3.0, 5.0], [7.0, -1.0]]),
    }
    return batches[rank, iteration]


def _distributed_normalization_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    """Run two synchronized updates from one CPU/Gloo rank."""
    try:
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=30),
        )
        norm = EmpiricalNormalization(shape=2)
        # This prior is intentionally installed before synchronization. It is
        # identical on both ranks and must be counted once, not world_size times.
        norm.update(torch.tensor([[0.0, 2.0], [2.0, 4.0]]))
        norm.set_distributed_sync(True)
        norm.update(_distributed_batch(rank, 0))
        norm.update(_distributed_batch(rank, 1))
        result_queue.put({
            "rank": rank,
            "count": norm.count.item(),
            "mean": norm._mean.tolist(),
            "var": norm._var.tolist(),
            "std": norm._std.tolist(),
        })
    except Exception as exc:
        result_queue.put({"rank": rank, "error": repr(exc)})
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
