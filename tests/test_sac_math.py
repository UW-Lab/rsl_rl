# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch
from tensordict import TensorDict
from rsl_rl.models import SACActorModel, SACCriticModel

OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}


def _obs(n=8, dim=5):
    return TensorDict({"policy": torch.randn(n, dim)}, batch_size=[n])


def test_model_actor_logp_shapes_and_bounds():
    obs = _obs()
    actor = SACActorModel(obs, OBS_GROUPS, "actor", output_dim=4, hidden_dims=[32, 32])
    actor.action_bias.copy_(torch.zeros(4))
    actor.action_range.copy_(torch.full((4,), 2.0))
    actor.log_action_range.copy_(torch.log(actor.action_range).sum())
    action, logp = actor.sample_action_logp(obs)
    assert action.shape == (8, 4)
    assert logp.shape == (8, 1)
    assert torch.all(action.abs() <= 2.0 + 1e-4)


def test_model_critic_twin_and_target_softupdate():
    obs = _obs()
    critic = SACCriticModel(obs, OBS_GROUPS, "critic", output_dim=1, num_actions=4, hidden_dims=[32, 32])
    critic.init_target_networks()
    a = torch.randn(8, 4)
    q1, q2 = critic.evaluate_all_q(obs, a)
    tq1, tq2 = critic.evaluate_all_target_q(obs, a)
    assert q1.shape == (8, 1) and q2.shape == (8, 1)
    assert torch.allclose(q1, tq1) and torch.allclose(q2, tq2)
    with torch.no_grad():
        for p in critic.critic1.parameters():
            p.add_(1.0)
    critic.soft_update_target_networks(1.0)
    q1b, _ = critic.evaluate_all_q(obs, a)
    tq1b, _ = critic.evaluate_all_target_q(obs, a)
    assert torch.allclose(q1b, tq1b)


def test_actor_layer_norm_true_raises():
    import pytest
    obs = _obs()
    with pytest.raises(NotImplementedError):
        SACActorModel(obs, OBS_GROUPS, "actor", output_dim=4, hidden_dims=[32, 32], layer_norm=True)
