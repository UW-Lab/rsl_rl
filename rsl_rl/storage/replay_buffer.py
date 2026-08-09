# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import warnings
from tensordict import TensorDict

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    class Transition:
        """Storage for a single state transition"""

        def __init__(self) -> None:
            self.observations: TensorDict | None = None
            self.actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.next_observations: TensorDict | None = None
            self.dones: torch.Tensor | None = None
            self.bootstrap: torch.Tensor | None = None

        def clear(self) -> None:
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs, actions_shape, device, buffer_size, n_steps=1, gamma=0.99):
        """
        Initialize a ReplayBuffer object.
        Args:
            - dim (int or list of int): Dimension(s) of the data to be stored.
                                    If a list, is stands for the dimensions of transition elements:
                                    [obs_dim, action_dim, reward_dim, next_obs_dim, done_dim].
            - buffer_size (int): Maximum size of buffer.
            - device (torch.device): Device on which tensors are stored.
            - n_steps (int): Number of steps for n-step returns (default: 1).
            - gamma (float): Discount factor for n-step returns (default: 0.99).
        """
        self.buffer_size = buffer_size
        self.device = device
        self.n_steps = n_steps
        self.gamma = gamma

        self.replay_buf = None
        self.num_envs = num_envs
        self.step = 0
        self.num_transitions_per_env = num_transitions_per_env
        self.num_transitions = 0
        #adjust buffer size based on number of envs
        self.buffer_size = max(buffer_size // num_envs, 1)

        #store shapes to build the buffer later
        # shape of observation for each group
        self.values_shape = {key: value.shape[1:] for key, value in obs.items()}
        #shape of actions
        self.actions_shape = tuple(actions_shape)

        # list-based storage spec: [obs, action, reward, next_obs, done]
        # dims <= 0 become None buffers (we keep all > 0 here)
        self.obs = None
        self.actions = None
        self.rewards = None
        self.next_obs = None
        self.dones = None
        self.bootstrap = None

        self.observations = TensorDict(
            {key: torch.zeros(num_envs, self.buffer_size, *value.shape[1:], device=self.device) for key, value in obs.items()},
            batch_size=[num_envs, self.buffer_size],
            device=self.device,
        )
        self.actions = torch.zeros(num_envs, self.buffer_size, *actions_shape, device=self.device)
        self.rewards = torch.zeros(num_envs, self.buffer_size, 1, device=self.device)
        self.next_observations = TensorDict(
            {key: torch.zeros(num_envs, self.buffer_size, *value.shape[1:], device=self.device) for key, value in obs.items()},
            batch_size=[num_envs, self.buffer_size],
            device=self.device,
        )
        self.dones = torch.zeros(num_envs, self.buffer_size, 1, device=self.device)
        self.bootstrap = torch.zeros(num_envs, self.buffer_size, 1, device=self.device)

        self.replay_buf = [self.observations, self.actions, self.rewards, self.next_observations, self.dones, self.bootstrap]

    def clear (self) -> None:
        """Clear the replay buffer."""
        self.step = 0
        self.num_transitions = 0

    def add_transition(self, transition: Transition) -> None:
        """Add a single transition using a RolloutStorage-style API."""
        if transition is None:
            raise ValueError("Transition is None.")
        if transition.observations is None or transition.actions is None:
            raise ValueError("Transition observations/actions must be provided.")
        if transition.rewards is None or transition.next_observations is None:
            raise ValueError("Transition rewards/next_observations must be provided.")
        if transition.dones is None or transition.bootstrap is None:
            raise ValueError("Transition dones/bootstrap must be provided.")

        self._insert(
            (
                transition.observations,
                transition.actions,
                transition.rewards,
                transition.next_observations,
                transition.dones,
                transition.bootstrap,
            )
        )

    def _insert(self, input_buf):
        """Add new states to memory in a circular manner.

        input_buf: list/tuple with entries matching self.replay_buf layout:
            [observations(TensorDict), actions(tensor), rewards(tensor), next_observations(TensorDict), dones(tensor), bootstrap(tensor)]
        Each entry should have shape [num_envs, num_inputs, ...] where num_inputs is how many time steps
        (usually 1) are being inserted per env.
        """

        def _insert_into_buffer(r_buf, i_buf):
            '''Helper function to insert i_buf into r_buf circularly.'''
            num_inputs = i_buf.shape[1]
            end_idx = self.step + num_inputs
            if end_idx > self.buffer_size:
                r_buf[:, self.step:self.buffer_size] = i_buf[:, :self.buffer_size - self.step]
                r_buf[:, :end_idx - self.buffer_size] = i_buf[:, self.buffer_size - self.step:]
            else:
                r_buf[:, self.step:end_idx] = i_buf
            return num_inputs


        num_inputs = 0
        if isinstance(self.replay_buf, list):
            # iterate over each buffer entry and insert accordingly
            for r_buf, i_buf in zip(self.replay_buf, input_buf):
                if r_buf is not None and i_buf is not None:
                    if isinstance(r_buf, TensorDict):
                        if not isinstance(i_buf, TensorDict):
                            raise ValueError("Input buffer must be a TensorDict if replay buffer is a TensorDict.")
                        # Sanity check for matching keys
                        if list(r_buf.keys()) != list(i_buf.keys()):
                            raise ValueError(f"Input buffer TensorDict keys do not \
                                             match replay buffer TensorDict keys: {list(i_buf.keys())} != {list(r_buf.keys())}")
                        # TensorDict case
                        for key in r_buf.keys():
                            r_field = r_buf[key]
                            i_field = i_buf[key]
                            # unsqueeze if needed
                            i_field = i_field.unsqueeze(1) if r_field.ndim > i_field.ndim else i_field
                            ni = _insert_into_buffer(r_field, i_field)
                            if num_inputs == 0:
                                num_inputs = ni
                            else:
                                assert num_inputs == ni, f"Mismatch in number of \
                                    inputs inserted across TensorDict fields: {num_inputs} != {ni} for key {key}."
                    else:
                        # Regular tensor case
                        # if scalar, first unsqueeze -1 and then unsqueeze 1 if needed
                        if r_buf.ndim > i_buf.ndim:
                            if i_buf.ndim == 1:
                                i_buf = i_buf.unsqueeze(-1)
                            i_buf = i_buf.unsqueeze(1)
                        ni = _insert_into_buffer(r_buf, i_buf)
                        if num_inputs == 0:
                            num_inputs = ni
                        else:
                            assert num_inputs == ni, f"Mismatch in number of \
                                    inputs inserted across TensorDict fields: {num_inputs} != {ni} for key {key}."
                else:
                    raise ValueError(f"Either replay buffer or input buffer contains None entries: r_buf={r_buf}, i_buf={i_buf}")
        else:
            raise NotImplementedError("ReplayBuffer currently only supports list-based storage.")

        # update counters for circular buffer
        self.num_transitions = min(self.buffer_size, self.num_transitions + num_inputs)
        self.step = (self.step + num_inputs) % self.buffer_size

    def mini_batch_generator(self, num_mini_batch, mini_batch_size, num_epochs=1):
        """Yield transition mini-batches (no sequence axis)."""
        assert self.replay_buf is not None, "Replay buffer is not initialized."
        valid_indices = self._generate_valid_indices()

        for _ in range(num_epochs):
            for _ in range(num_mini_batch):
                yield self._generate_batch(valid_indices, mini_batch_size)

    def _generate_valid_indices(self):
        """Generate valid (env, start) transition indices."""
        if self.num_transitions == 0:
            return None

        time_len = self.num_transitions if self.num_transitions < self.buffer_size else self.buffer_size
        env_ids = torch.arange(self.num_envs, device=self.device)
        time_ids = torch.arange(time_len, device=self.device)
        env_grid, time_grid = torch.meshgrid(env_ids, time_ids, indexing="ij")
        env_indices = env_grid.reshape(-1)
        start_indices = time_grid.reshape(-1)

        if self.n_steps > 1:
            max_offset = self.n_steps - 1
            if self.num_transitions == self.buffer_size:
                if max_offset >= self.buffer_size:
                    raise ValueError("n_steps must be <= buffer_size to avoid wrap across time.")
                starts_before_step = start_indices < self.step
                safe_before = (start_indices + max_offset) < self.step
                safe_after = (start_indices + max_offset) < (self.buffer_size + self.step)
                safe_mask = torch.where(starts_before_step, safe_before, safe_after)
            else:
                safe_mask = (start_indices + max_offset) < self.num_transitions
            env_indices = env_indices[safe_mask]
            start_indices = start_indices[safe_mask]

        return env_indices, start_indices

    def _generate_batch(self, valid_indices, mini_batch_size):
        """Sample a transition mini-batch with optional n-step target aggregation."""
        if valid_indices is None:
            raise ValueError("No valid indices available to sample from.")

        env_indices, start_indices = valid_indices
        total_transitions = len(env_indices)
        if total_transitions == 0:
            raise ValueError("Replay buffer does not contain enough data to sample a batch.")

        max_batch_size = total_transitions
        if max_batch_size < mini_batch_size:
            warnings.warn(
                f"Requested mini_batch_size={mini_batch_size} exceeds available transitions ({total_transitions}). "
                f"Using batch size {max_batch_size} instead.",
                RuntimeWarning,
            )
        batch_size = max(1, min(mini_batch_size, max_batch_size))

        sampled_idxs = torch.randint(total_transitions, size=(batch_size,), device=self.device)
        sampled_envs = env_indices[sampled_idxs]
        sampled_starts = start_indices[sampled_idxs]

        obs_buf, actions_buf, rewards_buf, next_obs_buf, dones_buf, bootstrap_buf = self.replay_buf

        if self.n_steps == 1:
            obs_out = TensorDict({k: v[sampled_envs, sampled_starts] for k, v in obs_buf.items()}, batch_size=[batch_size], device=self.device)
            actions_out = actions_buf[sampled_envs, sampled_starts]
            rewards_out = rewards_buf[sampled_envs, sampled_starts]
            next_obs_out = TensorDict({k: v[sampled_envs, sampled_starts] for k, v in next_obs_buf.items()}, batch_size=[batch_size], device=self.device)
            dones_out = dones_buf[sampled_envs, sampled_starts]
            bootstrap_out = bootstrap_buf[sampled_envs, sampled_starts]
            effective_n_steps = torch.ones(batch_size, 1, device=self.device, dtype=torch.long)
            return [
                obs_out,
                actions_out,
                rewards_out,
                next_obs_out,
                dones_out,
                bootstrap_out,
                effective_n_steps,
            ]

        #Create n-step indices
        step_offsets = torch.arange(self.n_steps, device=self.device)
        all_indices = (sampled_starts.unsqueeze(-1) + step_offsets) % self.buffer_size
        env_indices_expanded = sampled_envs.unsqueeze(-1).expand(batch_size, self.n_steps)

        all_rewards = rewards_buf[env_indices_expanded, all_indices].squeeze(-1)
        all_dones = dones_buf[env_indices_expanded, all_indices].squeeze(-1)

        # Zero out rewards after done and compute discounted sum with done masks
        all_dones_shifted = torch.cat([torch.zeros_like(all_dones[..., :1]), all_dones[..., :-1]], dim=-1)
        # Cumulative product to create masks that zero out rewards after the first done is encountered
        done_masks = torch.cumprod(1.0 - all_dones_shifted, dim=-1)
        # Discount factors: gamma^0, gamma^1, ..., gamma^(n-1)
        discounts = torch.pow(self.gamma, step_offsets)
        n_step_rewards = (all_rewards * done_masks * discounts.view(1, -1)).sum(dim=-1, keepdim=True)

        # Find the first done index for each transition to determine the correct next_obs, done, and bootstrap values
        first_done = torch.argmax((all_dones > 0).float(), dim=-1)
        no_dones = (all_dones.sum(dim=-1) == 0)
        first_done = torch.where(no_dones, torch.full_like(first_done, self.n_steps - 1), first_done)
        effective_n_steps = (first_done + 1).unsqueeze(-1).to(torch.long)
        final_step_indices = all_indices.gather(1, first_done.unsqueeze(-1)).squeeze(-1)

        obs_out = TensorDict({k: v[sampled_envs, sampled_starts] for k, v in obs_buf.items()}, batch_size=[batch_size], device=self.device)
        actions_out = actions_buf[sampled_envs, sampled_starts]
        next_obs_out = TensorDict({k: v[sampled_envs, final_step_indices] for k, v in next_obs_buf.items()}, batch_size=[batch_size], device=self.device)
        final_dones = dones_buf[sampled_envs, final_step_indices]
        final_bootstraps = bootstrap_buf[sampled_envs, final_step_indices]

        return [
            obs_out,
            actions_out,
            n_step_rewards,
            next_obs_out,
            final_dones,
            final_bootstraps,
            effective_n_steps,
        ]
