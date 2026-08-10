# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch
from tensordict import TensorDict
from rsl_rl.storage import ReplayBuffer


def _obs(num_envs, dim=3):
    return TensorDict({"policy": torch.zeros(num_envs, dim)}, batch_size=[num_envs])


def _mk(num_envs=2, buffer_size=4, n_steps=1, gamma=0.9, act=1):
    return ReplayBuffer(
        num_envs=num_envs,
        num_transitions_per_env=1,
        obs=_obs(num_envs),
        actions_shape=[act],
        device="cpu",
        buffer_size=buffer_size * num_envs,  # divided by num_envs internally
        n_steps=n_steps,
        gamma=gamma,
    )


def _add(buf, obs_val, action, reward, next_val, done, bootstrap):
    n = buf.num_envs
    t = ReplayBuffer.Transition()
    t.observations = TensorDict({"policy": torch.full((n, 3), float(obs_val))}, batch_size=[n])
    t.actions = torch.full((n, 1), float(action))
    t.rewards = torch.full((n, 1), float(reward))
    t.next_observations = TensorDict({"policy": torch.full((n, 3), float(next_val))}, batch_size=[n])
    t.dones = torch.full((n, 1), float(done))
    t.bootstrap = torch.full((n, 1), float(bootstrap))
    buf.add_transition(t)


def test_circular_wraparound_step_and_count():
    buf = _mk(num_envs=2, buffer_size=4)
    for i in range(6):  # per-env capacity is 4
        _add(buf, i, i, i, i + 1, 0, 0)
    assert buf.num_transitions == 4
    assert buf.step == 2  # 6 % 4


def test_n1_sample_shapes_and_values():
    buf = _mk(num_envs=2, buffer_size=4, n_steps=1)
    _add(buf, 5.0, 1.0, 2.0, 6.0, 0, 1)
    (batch,) = list(buf.mini_batch_generator(num_mini_batch=1, mini_batch_size=2))
    obs, actions, rewards, next_obs, dones, bootstrap, eff_n = batch
    assert obs["policy"].shape == (2, 3)
    assert torch.allclose(rewards, torch.full((2, 1), 2.0))
    assert torch.allclose(bootstrap, torch.full((2, 1), 1.0))
    assert torch.all(eff_n == 1)


def test_nstep_discounted_reward_stops_at_done():
    # n=3, gamma=0.5. Rewards 1,1,1 with a done on the 2nd transition.
    buf = _mk(num_envs=1, buffer_size=8, n_steps=3, gamma=0.5)
    _add(buf, 0, 0, 1.0, 0, 0, 0)
    _add(buf, 0, 0, 1.0, 0, 1, 0)  # episode ends here
    _add(buf, 0, 0, 1.0, 0, 0, 0)
    (batch,) = list(buf.mini_batch_generator(num_mini_batch=1, mini_batch_size=1))
    _, _, rewards, _, final_dones, _, eff_n = batch
    # Only start index 0 is valid (start+max_offset < num_transitions=3 -> start<1).
    # Discounted sum: r0 + gamma*r1 = 1 + 0.5*1 = 1.5 (r2 masked by done at step1).
    assert torch.allclose(rewards, torch.tensor([[1.5]]))
    assert torch.all(final_dones == 1.0)
    assert torch.all(eff_n == 2)  # first_done at offset 1 -> effective n = 2
