# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.utils import unpad_trajectories

from .mlp_model import MLPModel


class SACActorModel(MLPModel):
    """SAC actor model with a Tanh-squashed Gaussian output distribution."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        init_noise_std: float = 1.0,
        layer_norm: bool = False,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        **kwargs,
    ) -> None:
        if layer_norm:
            raise NotImplementedError("layer_norm not supported in v1")

        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=None,
        )

        self.output_dim = output_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # The 5.2.0 MLPModel does not provide a state-dependent standard-deviation
        # head, so replace its MLP with a joint mean/log-standard-deviation head.
        self.mlp = MLP(self._get_latent_dim(), 2 * output_dim, hidden_dims, activation)

        # Initialize the actor head so initial actions remain close to zero.
        last_linear = None
        for module in reversed(self.mlp):
            if isinstance(module, nn.Linear):
                last_linear = module
                break
        if last_linear is not None:
            torch.nn.init.normal_(last_linear.weight[:output_dim], mean=0.0, std=1e-3)
            torch.nn.init.zeros_(last_linear.bias[:output_dim])
            torch.nn.init.zeros_(last_linear.weight[output_dim:])
            torch.nn.init.constant_(last_linear.bias[output_dim:], torch.log(torch.tensor(init_noise_std + 1e-7)))

        self.register_buffer("action_bias", torch.zeros(output_dim))
        self.register_buffer("action_range", torch.ones(output_dim))
        self.register_buffer("log_action_range", torch.zeros(1))

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return Tanh-squashed and scaled actions."""
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        self._update_distribution(latent)
        if stochastic_output:
            x_t = self.distribution.rsample()
        else:
            x_t = self.distribution.mean
        return self._squash_and_scale(x_t)

    def sample_action_logp(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and return its squash- and scale-corrected log-probability."""
        latent = self.get_latent(obs)
        self._update_distribution(latent)
        x_t = self.distribution.rsample()
        tanh_x = torch.tanh(x_t)
        action = self.action_range * tanh_x + self.action_bias

        log_prob = self.distribution.log_prob(x_t).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - tanh_x.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        log_prob -= self.log_action_range

        return action, log_prob

    def _update_distribution(self, latent: torch.Tensor) -> None:
        """Update the Gaussian distribution with a clamped state-dependent log standard deviation."""
        out = self.mlp(latent)
        mean, log_std = torch.unbind(out.view(*out.shape[:-1], 2, self.output_dim), dim=-2)
        std = log_std.clamp(self.log_std_min, self.log_std_max).exp()
        self.distribution = Normal(mean, std)

    def _squash_and_scale(self, x_t: torch.Tensor) -> torch.Tensor:
        return self.action_range * torch.tanh(x_t) + self.action_bias

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def as_jit(self) -> nn.Module:
        return _TorchSACActorModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxSACActorModel(self, verbose)


class SACCriticModel(MLPModel):
    """SAC critic model with twin online Q-networks and frozen target networks."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        num_actions: int = 0,
        layer_norm: bool = False,
        **kwargs,
    ) -> None:
        if layer_norm:
            raise NotImplementedError("layer_norm not supported in v1")

        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=None,
        )

        self.num_actions = num_actions
        q_input_dim = self.obs_dim + num_actions
        self.mlp = None  # type: ignore[assignment]

        self.critic1 = MLP(q_input_dim, output_dim, hidden_dims, activation)
        self.critic2 = MLP(q_input_dim, output_dim, hidden_dims, activation)

        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        q_input = torch.cat([latent, actions], dim=-1)
        return self.critic1(q_input)

    def evaluate_all_q(self, obs: TensorDict, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.get_latent(obs)
        latent = torch.cat([latent, actions], dim=-1)
        return self.critic1(latent), self.critic2(latent)

    def evaluate_all_target_q(self, obs: TensorDict, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.get_latent(obs)
        latent = torch.cat([latent, actions], dim=-1)
        return self.critic1_target(latent), self.critic2_target(latent)

    def init_target_networks(self) -> None:
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def soft_update_target_networks(self, tau: float) -> None:
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class _TorchSACActorModel(nn.Module):
    """Exportable SAC actor model for JIT."""

    def __init__(self, model: SACActorModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.action_bias = model.action_bias.clone()
        self.action_range = model.action_range.clone()
        self.output_dim = model.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        mean = out[..., : self.output_dim]
        return self.action_range * torch.tanh(mean) + self.action_bias

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxSACActorModel(nn.Module):
    """Exportable SAC actor model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: SACActorModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.register_buffer("action_bias", model.action_bias.clone())
        self.register_buffer("action_range", model.action_range.clone())
        self.input_size = model.obs_dim
        self.output_dim = model.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        mean = out[..., : self.output_dim]
        return self.action_range * torch.tanh(mean) + self.action_bias

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
