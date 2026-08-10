# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import SAC
from rsl_rl.models import SACActorModel, SACCriticModel
from rsl_rl.storage import ReplayBuffer

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


def test_actor_output_std_is_tensor():
    obs = _obs()
    actor = SACActorModel(obs, OBS_GROUPS, "actor", output_dim=4, hidden_dims=[32, 32])
    # populate self.distribution via a forward pass
    actor(obs, stochastic_output=True)
    std = actor.output_std
    assert isinstance(std, torch.Tensor)
    assert std.shape[-1] == 4
    # Logger does action_std.mean().item(); must not raise:
    _ = std.mean().item()


def test_actor_output_entropy_is_tensor():
    obs = _obs()
    actor = SACActorModel(obs, OBS_GROUPS, "actor", output_dim=4, hidden_dims=[32, 32])
    actor(obs, stochastic_output=True)
    ent = actor.output_entropy
    assert isinstance(ent, torch.Tensor)
    _ = ent.mean().item()


def _mk_sac(q_aggregation="min"):
    obs = _obs(n=4, dim=5)
    actor = SACActorModel(obs, OBS_GROUPS, "actor", output_dim=3, hidden_dims=[16, 16])
    critic = SACCriticModel(obs, OBS_GROUPS, "critic", output_dim=1, num_actions=3, hidden_dims=[16, 16])
    rb = ReplayBuffer(
        num_envs=4, num_transitions_per_env=1, obs=obs, actions_shape=[3],
        device="cpu", buffer_size=64, n_steps=1, gamma=0.99,
    )
    return SAC(actor, critic, rb, device="cpu", gamma=0.99, q_aggregation=q_aggregation)


def test_combine_q_min_and_avg():
    q1 = torch.tensor([[1.0], [3.0]])
    q2 = torch.tensor([[2.0], [1.0]])
    alg_min = _mk_sac("min")
    assert torch.allclose(alg_min._combine_q(q1, q2), torch.tensor([[1.0], [1.0]]))
    alg_avg = _mk_sac("avg")
    assert torch.allclose(alg_avg._combine_q(q1, q2), torch.tensor([[1.5], [2.0]]))


def test_combine_q_unknown_raises():
    import pytest
    alg = _mk_sac("min")
    alg.q_aggregation = "bogus"
    with pytest.raises(ValueError):
        alg._combine_q(torch.zeros(1, 1), torch.zeros(1, 1))


def test_bootstrap_mask_values_and_guard():
    bootstrap = torch.tensor([[0.], [1.], [0.]])
    dones = torch.tensor([[0.], [1.], [1.]])
    mask = bootstrap + 1 - dones
    assert torch.allclose(mask, torch.tensor([[1.], [1.], [0.]]))
    bad = torch.tensor([[1.]]) + 1 - torch.tensor([[0.]])
    assert torch.any(bad > 1)


def test_nstep_target_formula():
    gamma, n = 0.9, torch.tensor([[2]])
    reward = torch.tensor([[1.5]])
    q_next = torch.tensor([[10.0]])
    mask = torch.tensor([[1.0]])
    discount = torch.pow(torch.tensor(gamma), n.to(torch.float32))
    target = reward + discount * mask * q_next
    assert torch.allclose(target, torch.tensor([[1.5 + 0.81 * 10.0]]))


def test_gamma_mismatch_raises():
    import pytest
    obs = _obs(n=4, dim=5)
    actor = SACActorModel(obs, OBS_GROUPS, "actor", output_dim=3, hidden_dims=[16, 16])
    critic = SACCriticModel(obs, OBS_GROUPS, "critic", output_dim=1, num_actions=3, hidden_dims=[16, 16])
    rb = ReplayBuffer(num_envs=4, num_transitions_per_env=1, obs=obs, actions_shape=[3],
                      device="cpu", buffer_size=64, n_steps=1, gamma=0.99)
    with pytest.raises(ValueError):
        SAC(actor, critic, rb, device="cpu", gamma=0.95)  # mismatch vs buffer 0.99


def test_process_env_step_timeout_without_obs_degrades_to_terminal():
    import torch
    # When a timeout is active but time_outs_obs is absent, SAC must NOT bootstrap from the
    # post-reset observation; it degrades safely to terminal treatment (bootstrap flag zeroed).
    alg = _mk_sac("min")
    obs = _obs(n=4, dim=5)
    alg.act(obs)  # sets transition.observations/actions
    next_obs = _obs(n=4, dim=5)
    dones = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    extras = {"time_outs": torch.tensor([[0], [1], [0], [0]])}  # timeout on env 1, no time_outs_obs
    alg.process_env_step(next_obs, torch.zeros(4, 1), dones, extras)
    # The stored bootstrap flag for the just-written transition must be all-zero (terminal treatment).
    stored_bootstrap = alg.replay_buffer.bootstrap[:, alg.replay_buffer.step - 1]
    assert torch.count_nonzero(stored_bootstrap) == 0


def test_process_env_step_timeout_substitution_multidim():
    import torch
    from tensordict import TensorDict
    # obs with a 3D leaf [N, H, W] to exercise general mask broadcast
    n = 4
    groups = {"actor": ["policy"], "critic": ["policy"]}
    def mk3d():
        return TensorDict(
            {"policy": torch.zeros(n, 5), "height_scan": torch.zeros(n, 2, 3)}, batch_size=[n]
        )
    obs = mk3d()
    actor = SACActorModel(obs, groups, "actor", output_dim=3, hidden_dims=[16, 16])
    critic = SACCriticModel(obs, groups, "critic", output_dim=1, num_actions=3, hidden_dims=[16, 16])
    rb = ReplayBuffer(num_envs=n, num_transitions_per_env=1, obs=obs, actions_shape=[3],
                      device="cpu", buffer_size=64, n_steps=1, gamma=0.99)
    alg = SAC(actor, critic, rb, device="cpu", gamma=0.99)
    alg.act(obs)
    next_obs = TensorDict(
        {"policy": torch.ones(n, 5), "height_scan": torch.ones(n, 2, 3)}, batch_size=[n]
    )  # post-reset = 1s
    term_obs = TensorDict(
        {"policy": torch.full((n, 5), 7.0), "height_scan": torch.full((n, 2, 3), 7.0)}, batch_size=[n]
    )  # terminal = 7s
    dones = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    extras = {"time_outs": torch.tensor([[0], [1], [0], [0]]), "time_outs_obs": term_obs}
    alg.process_env_step(next_obs, torch.zeros(n, 1), dones, extras)
    # env 1 (timeout) must have terminal obs (7) stored as next_obs; others keep post-reset (1)
    stored = rb.next_observations["height_scan"][:, rb.step - 1]  # last written slot
    assert torch.allclose(stored[1], torch.full((2, 3), 7.0))
    assert torch.allclose(stored[0], torch.ones(2, 3))
