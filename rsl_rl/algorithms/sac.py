# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import RandomNetworkDistillation, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import SACActorModel, SACCriticModel
from rsl_rl.storage import ReplayBuffer
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


class SAC:
    """Soft Actor-Critic algorithm (https://arxiv.org/abs/1812.05905).

    Uses separate actor and critic models following the v4.0.0 architecture.
    """

    actor: SACActorModel
    """The actor model."""

    critic: SACCriticModel
    """The critic model."""

    def __init__(
        self,
        actor: SACActorModel,
        critic: SACCriticModel,
        replay_buffer: ReplayBuffer,
        replay_buffer_size: int = 1_000_000,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        mini_batch_size: int = 256,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        alpha_learning_rate: float = 1e-3,
        actor_optimizer: str = "adam",
        critic_optimizer: str = "adam",
        auto_alpha: bool = True,
        alpha: float = 0.05,
        tau: float = 0.005,
        gamma: float = 0.998,
        target_entropy_scale: float = 1.0,
        device: str = "cpu",
        max_grad_norm: float = 1.0,
        policy_frequency: int = 2,
        n_steps: int = 1,
        q_aggregation: str = "min",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        """Initialize the SAC algorithm.

        Args:
            actor: The SAC actor model.
            critic: The SAC critic model.
            replay_buffer: An instance of the ReplayBuffer class.
            replay_buffer_size: Max replay buffer size.
            num_learning_epochs: How many epochs to run each update.
            num_mini_batches: Into how many mini-batches to split the replay data per epoch.
            mini_batch_size: Mini-batch size for updates.
            actor_learning_rate: LR for the actor parameters.
            critic_learning_rate: LR for the critic parameters.
            alpha_learning_rate: LR for the alpha parameter, if auto_alpha=True.
            actor_optimizer: Optimizer name for the actor (e.g., "adam", "adamw").
            critic_optimizer: Optimizer name for the critic (e.g., "adam", "adamw").
            auto_alpha: Whether to learn alpha automatically.
            alpha: Initial temperature (if auto_alpha=False) or initial value for alpha learning.
            tau: Soft update coefficient for target networks.
            gamma: Discount factor.
            target_entropy_scale: Scale factor for target entropy; target_entropy = -scale * action_dim.
            device: 'cpu' or 'cuda'.
            max_grad_norm: Max norm for gradient clipping.
            policy_frequency: Frequency of actor updates relative to critic updates.
            n_steps: Number of steps for n-step returns (default: 1).
            q_aggregation: Method used to combine twin critic values ("min" or "avg").
            rnd_cfg: Optional dictionary of RND configuration parameters. If None, RND is not used.
            symmetry_cfg: Optional dictionary of symmetry configuration parameters. If None, symmetry is not used.
            multi_gpu_cfg: Optional dictionary of multi-GPU configuration parameters. If None, multi-GPU is not used.
        """
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg:
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            symmetry_cfg["data_augmentation_func"] = resolve_callable(symmetry_cfg["data_augmentation_func"])
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Symmetry configuration exists but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            if actor.is_recurrent or critic.is_recurrent:
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # Store actor and critic
        self.actor = actor.to(device)
        self.critic = critic.to(device)

        # Replay buffer
        self.replay_buffer = replay_buffer
        self.replay_buffer_size = replay_buffer_size
        self.transition = ReplayBuffer.Transition()

        # SAC hyperparams
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        if abs(float(self.replay_buffer.gamma) - float(self.gamma)) > 1e-12:
            raise ValueError(
                f"SAC gamma ({self.gamma}) and replay buffer gamma ({self.replay_buffer.gamma}) must match; "
                "they jointly define the n-step return target."
            )
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.policy_frequency = policy_frequency
        self.update_step = 0
        self.n_steps = n_steps
        self.q_aggregation = q_aggregation
        self.max_grad_norm = max_grad_norm

        self.target_entropy = -target_entropy_scale * self.actor.output_dim

        # Initialize log_alpha and its optimizer
        if self.auto_alpha:
            self.log_alpha = torch.log(torch.tensor(self.alpha, device=self.device)).detach().clone().requires_grad_(True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_learning_rate)
        else:
            self.log_alpha = (
                torch.log(torch.tensor(self.alpha, device=self.device)).detach().clone().requires_grad_(False)
            )
            self.alpha_optimizer = None

        # Collect trainable parameters (target network params have requires_grad=False, auto-excluded)
        self.actor_parameters = [p for p in self.actor.parameters() if p.requires_grad]
        self.critic_parameters = [p for p in self.critic.parameters() if p.requires_grad]

        # Create optimizers using resolve_optimizer (matching PPO pattern)
        self.actor_optimizer = resolve_optimizer(actor_optimizer)(
            self.actor_parameters, lr=self.actor_learning_rate
        )
        self.critic_optimizer = resolve_optimizer(critic_optimizer)(
            self.critic_parameters, lr=self.critic_learning_rate
        )

        # Init target networks
        self.critic.init_target_networks()

    def _combine_q(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        if self.q_aggregation == "min":
            return torch.min(q1, q2)
        if self.q_aggregation == "avg":
            return 0.5 * (q1 + q2)
        raise ValueError(f"Unknown q_aggregation: {self.q_aggregation}")

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Select an action using the actor (stochastic during training)."""
        with torch.no_grad():
            action = self.actor(obs, stochastic_output=True)
            self.transition.observations = obs
            self.transition.actions = action
            return action

    def process_env_step(
        self, next_obs: TensorDict, rew: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> None:
        """Process a single environment step and store transition in replay buffer."""
        # Determine timeout flags for this step (bootstrap through truncations, not true terminals).
        if "time_outs" in extras:
            time_outs = extras["time_outs"].int().to(self.device)
        else:
            time_outs = torch.zeros_like(dones, device=self.device)

        # Substitute the true terminal observation for timed-out envs so the critic bootstraps
        # from the pre-reset state rather than the post-reset (auto-reset) observation. Only needed
        # when a timeout is actually active this step; the env stashes ``time_outs_obs`` on reset steps.
        if bool(time_outs.any()):
            if "time_outs_obs" in extras:
                time_outs_obs = extras["time_outs_obs"]
                mask = time_outs.squeeze(-1).bool()
                true_next_obs = {}
                for key in time_outs_obs.keys():
                    leaf = next_obs[key]
                    # Broadcast the per-env mask over an arbitrary-rank leaf (e.g. flat [N, D] or
                    # height-scan [N, H, W]); mask[:, None] only works for rank-2 leaves.
                    leaf_mask = mask.reshape(mask.shape[0], *([1] * (leaf.ndim - 1)))
                    true_next_obs[key] = torch.where(leaf_mask, time_outs_obs[key], leaf)
                true_next_obs = TensorDict(true_next_obs, batch_size=next_obs.batch_size)
            else:
                # ``time_outs_obs`` is unavailable (e.g. the env-side pre-reset stash is not active in
                # this installation). Bootstrapping from the post-reset observation would be wrong, so
                # degrade safely to treating timeouts as true terminals (zero the bootstrap flag).
                # Warn once so this silent-correctness loss is visible.
                if not getattr(self, "_warned_missing_time_outs_obs", False):
                    self._warned_missing_time_outs_obs = True
                    print(
                        "[WARNING] SAC: timeouts occurred but 'time_outs_obs' is absent from extras; "
                        "treating timeouts as terminals (no bootstrap). Correct timeout bootstrapping "
                        "requires the environment to stash pre-reset observations (time_outs_obs)."
                    )
                time_outs = torch.zeros_like(dones, device=self.device)
                true_next_obs = next_obs
        else:
            true_next_obs = next_obs

        # Update normalizers
        self.actor.update_normalization(true_next_obs)
        self.critic.update_normalization(true_next_obs)
        if self.rnd:
            self.rnd.update_normalization(true_next_obs)

        # Compute intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(true_next_obs)
            rew += self.intrinsic_rewards

        # Record transition and insert into replay buffer
        self.transition.rewards = rew
        self.transition.next_observations = true_next_obs
        self.transition.dones = dones
        self.transition.bootstrap = time_outs

        self.replay_buffer.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)

    def update(self) -> dict:
        """Perform off-policy SAC updates, returning mean losses."""
        mean_critic1_loss = 0.0
        mean_critic2_loss = 0.0
        mean_actor_loss = 0.0
        mean_alpha_loss = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        for batch in self.replay_buffer.mini_batch_generator(
            num_mini_batch=self.num_mini_batches,
            mini_batch_size=self.mini_batch_size,
            num_epochs=self.num_learning_epochs,
        ):
            (
                obs_batch,
                actions_batch,
                rewards_batch,
                next_obs_batch,
                dones_batch,
                bootstrap_batch,
                effective_n_steps,
            ) = batch

            original_batch_size = (
                obs_batch.batch_size[0] if isinstance(obs_batch, TensorDict) else obs_batch.shape[0]
            )

            num_aug = 1
            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                original_batch_size = (
                    obs_batch.batch_size[0] if isinstance(obs_batch, TensorDict) else obs_batch.shape[0]
                )
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"]
                )
                next_obs_batch, _ = data_augmentation_func(
                    obs=next_obs_batch, actions=None, env=self.symmetry["_env"]
                )
                aug_batch_size = (
                    obs_batch.batch_size[0] if isinstance(obs_batch, TensorDict) else obs_batch.shape[0]
                )
                num_aug = int(aug_batch_size / original_batch_size)
                rewards_batch = rewards_batch.repeat(num_aug, *([1] * (rewards_batch.ndim - 1)))
                dones_batch = dones_batch.repeat(num_aug, *([1] * (dones_batch.ndim - 1)))
                bootstrap_batch = bootstrap_batch.repeat(num_aug, *([1] * (bootstrap_batch.ndim - 1)))
                effective_n_steps = effective_n_steps.repeat(num_aug, *([1] * (effective_n_steps.ndim - 1)))

            ###########################################################################
            # 1) Critic update
            with torch.no_grad():
                bootstrap_mask = bootstrap_batch + 1 - dones_batch
                if torch.any(bootstrap_mask > 1):
                    raise ValueError("bootstrap_mask has values greater than 1. Check bootstrapping logic.")

                new_actions, next_log_prob = self.actor.sample_action_logp(next_obs_batch)
                next_state_entropy = -self.log_alpha.exp() * next_log_prob

                q1_target, q2_target = self.critic.evaluate_all_target_q(next_obs_batch, new_actions)
                min_target_q = self._combine_q(q1_target, q2_target)
                q_target_next = min_target_q + next_state_entropy
                n_step_discount = torch.pow(self.gamma, effective_n_steps.to(dtype=q_target_next.dtype))
                target_q = rewards_batch + n_step_discount * bootstrap_mask * q_target_next

            q1_pred, q2_pred = self.critic.evaluate_all_q(obs_batch, actions_batch)

            critic1_loss = nn.functional.mse_loss(q1_pred, target_q)
            critic2_loss = nn.functional.mse_loss(q2_pred, target_q)

            total_critic_loss = 0.5 * (critic1_loss + critic2_loss)
            self.critic_optimizer.zero_grad()
            total_critic_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters(self.critic_parameters)

            torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.max_grad_norm)
            self.critic_optimizer.step()

            ###########################################################################
            # Sample new actions for actor and alpha update
            new_actions, log_prob = self.actor.sample_action_logp(obs_batch)

            ###########################################################################
            # 2) Alpha update
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()

                if self.is_multi_gpu:
                    if self.log_alpha.grad is not None:
                        torch.distributed.all_reduce(self.log_alpha.grad, op=torch.distributed.ReduceOp.SUM)
                        self.log_alpha.grad /= self.gpu_world_size

                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
            else:
                alpha_loss = torch.tensor(0.0, device=self.device)

            entropy = self.log_alpha.exp().detach() * log_prob

            ###########################################################################
            # 3) Actor update
            if self.update_step % self.policy_frequency == 0:
                # Freeze critic parameters for actor update
                for p in self.critic_parameters:
                    p.requires_grad_(False)

                q1, q2 = self.critic.evaluate_all_q(obs_batch, new_actions)
                q_new = self._combine_q(q1, q2)
                actor_loss = (entropy - q_new).mean()

                # Symmetry loss
                if self.symmetry:
                    if not self.symmetry["use_data_augmentation"]:
                        data_augmentation_func = self.symmetry["data_augmentation_func"]
                        obs_batch, _ = data_augmentation_func(
                            obs=obs_batch, actions=None, env=self.symmetry["_env"]
                        )
                        num_aug = int(
                            (obs_batch.batch_size[0] if isinstance(obs_batch, TensorDict) else obs_batch.shape[0])
                            / original_batch_size
                        )

                    # Deterministic mean actions for symmetry loss
                    mean_actions_batch = self.actor(obs_batch.detach().clone())

                    action_mean_orig = mean_actions_batch[:original_batch_size]
                    _, actions_mean_symm_batch = data_augmentation_func(
                        obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                    )

                    if num_aug > 1:
                        mse_loss = torch.nn.MSELoss()
                        symmetry_loss = mse_loss(
                            mean_actions_batch[original_batch_size:],
                            actions_mean_symm_batch.detach()[original_batch_size:],
                        )
                        if self.symmetry["use_mirror_loss"]:
                            actor_loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                        else:
                            symmetry_loss = symmetry_loss.detach()
                    else:
                        symmetry_loss = torch.tensor(0.0, device=self.device)
                else:
                    symmetry_loss = torch.tensor(0.0, device=self.device)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()

                if self.is_multi_gpu:
                    self.reduce_parameters(self.actor_parameters)

                torch.nn.utils.clip_grad_norm_(self.actor_parameters, self.max_grad_norm)
                self.actor_optimizer.step()

                # Unfreeze critic parameters after actor update
                for p in self.critic_parameters:
                    p.requires_grad_(True)
            else:
                actor_loss = torch.tensor(0.0, device=self.device)
                symmetry_loss = torch.tensor(0.0, device=self.device)

            ###########################################################################
            # 4) Soft update target networks
            with torch.no_grad():
                self.critic.soft_update_target_networks(self.tau)

            # RND loss
            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters(self.rnd.parameters())
                self.rnd_optimizer.step()

            # Accumulate losses
            mean_critic1_loss += critic1_loss.item()
            mean_critic2_loss += critic2_loss.item()
            mean_actor_loss += actor_loss.item()
            mean_alpha_loss += alpha_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            self.update_step += 1

        # Average losses
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_critic1_loss /= num_updates
        mean_critic2_loss /= num_updates
        mean_actor_loss /= max((num_updates // self.policy_frequency), 1)
        mean_alpha_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= max((num_updates // self.policy_frequency), 1)

        loss_dict = {
            "critic1": mean_critic1_loss,
            "critic2": mean_critic2_loss,
            "actor": mean_actor_loss,
            "alpha": mean_alpha_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    def train_mode(self) -> None:
        """Set actor, critic, and RND to training mode."""
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set actor, critic, and RND to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()

    def get_policy(self) -> SACActorModel:
        """Get the policy model (actor)."""
        return self.actor

    def save(self) -> dict:
        """Return a dict of all model states for saving."""
        saved_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu() if self.auto_alpha else None,
            "alpha": self.alpha if not self.auto_alpha else None,
        }
        if self.auto_alpha and self.alpha_optimizer is not None:
            saved_dict["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            if self.rnd_optimizer:
                saved_dict["rnd_optimizer_state_dict"] = self.rnd_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict.

        Args:
            loaded_dict: Dictionary of saved model states.
            load_cfg: Dictionary specifying which components to load. If None, loads all.
            strict: Whether to strictly enforce state dict key matching.

        Returns:
            Whether the iteration counter should be restored.
        """
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }

        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            if self.auto_alpha and "alpha_optimizer_state_dict" in loaded_dict:
                self.alpha_optimizer.load_state_dict(loaded_dict["alpha_optimizer_state_dict"])
            if loaded_dict.get("log_alpha") is not None:
                self.log_alpha.data.copy_(loaded_dict["log_alpha"].to(self.device))
                self.alpha = self.log_alpha.exp().item()
            elif loaded_dict.get("alpha") is not None:
                self.alpha = loaded_dict["alpha"]
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            if self.rnd_optimizer and "rnd_optimizer_state_dict" in loaded_dict:
                self.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def clear_storage(self) -> None:
        """Clear the replay buffer."""
        if self.replay_buffer is not None:
            self.replay_buffer.clear()

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SAC:
        """Construct the SAC algorithm with actor, critic, and replay buffer.

        Args:
            obs: Initial observations from the environment.
            env: The vectorized environment.
            cfg: Configuration dictionary.
            device: Device to place models on.

        Returns:
            Initialized SAC algorithm instance.
        """
        # Resolve class callables
        alg_class: type[SAC] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[SACActorModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[SACCriticModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize the actor
        actor: SACActorModel = actor_class(
            obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
        ).to(device)
        print(f"SAC Actor: {actor}")

        # Compute action scaling from robot joint limits
        upper, lower = SAC._compute_action_scaling(env, device)
        lower_neg = -lower
        actor.action_bias.copy_(0.5 * (upper + lower_neg))
        actor.action_range.copy_(0.5 * (upper - lower_neg))
        actor.log_action_range.copy_(torch.log(actor.action_range).sum())

        # Initialize the critic
        critic: SACCriticModel = critic_class(
            obs, cfg["obs_groups"], "critic", 1, num_actions=env.num_actions, **cfg["critic"]
        ).to(device)
        print(f"SAC Critic: {critic}")

        # Initialize the replay buffer
        replay_buffer = ReplayBuffer(
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            buffer_size=cfg["algorithm"].get("replay_buffer_size", 1_000_000),
            n_steps=cfg["algorithm"].get("n_steps", 1),
            gamma=cfg["algorithm"].get("gamma", 0.998),
        )

        if cfg["algorithm"].get("gradient_noise_scale_cfg") is not None:
            raise ValueError("gradient_noise_scale_cfg is not supported by SAC in v1")
        cfg["algorithm"].pop("gradient_noise_scale_cfg", None)

        # Initialize the algorithm
        alg: SAC = alg_class(
            actor, critic, replay_buffer, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg.get("multi_gpu")
        )

        return alg

    @staticmethod
    def _compute_action_scaling(env: VecEnv, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-joint action scaling factors based on robot configuration.

        Returns:
            Tuple of (upper_scaling, lower_scaling) tensors of shape (num_actions,).
        """
        unwrapped_env = getattr(env, "unwrapped", env)

        if not hasattr(unwrapped_env, "scene") or "robot" not in unwrapped_env.scene.keys():
            raise ValueError(
                "SAC: Could not find 'robot' in env.scene. Please check the environment configuration."
            )

        robot = unwrapped_env.scene["robot"]

        # In newer IsaacLab (Newton/warp backend), articulation data are ``warp`` arrays rather than
        # torch tensors. ``soft_joint_pos_limits`` is a ``wp.array`` of shape (num_envs, num_joints)
        # with dtype ``wp.vec2f``, which resolves to a torch tensor of shape (num_envs, num_joints, 2).
        # Convert to torch before indexing; tolerate either backend for forward/backward compatibility.
        soft_limits = robot.data.soft_joint_pos_limits
        default_joint_pos = robot.data.default_joint_pos
        if not isinstance(soft_limits, torch.Tensor):
            import warp as wp

            soft_limits = wp.to_torch(soft_limits)
        if not isinstance(default_joint_pos, torch.Tensor):
            import warp as wp

            default_joint_pos = wp.to_torch(default_joint_pos)

        lower_limits = soft_limits[0, :, 0].to(device)
        upper_limits = soft_limits[0, :, 1].to(device)
        default_pos = default_joint_pos[0].to(device)

        if torch.isnan(lower_limits).any() or torch.isinf(lower_limits).any():
            raise ValueError("SAC: Found NaN or Inf in lower joint position limits.")
        if torch.isnan(upper_limits).any() or torch.isinf(upper_limits).any():
            raise ValueError("SAC: Found NaN or Inf in upper joint position limits.")

        # Get global action scale from the action manager
        action_scale = 1.0
        if hasattr(unwrapped_env, "action_manager"):
            for term in unwrapped_env.action_manager._terms.values():
                if hasattr(term.cfg, "scale"):
                    if isinstance(term.cfg.scale, (float, int)):
                        action_scale = term.cfg.scale
                        break
                    else:
                        raise NotImplementedError(
                            "SAC: Action scale is not a scalar. "
                            "Please implement handling for dict/list scales if needed."
                        )

        range_to_lower = torch.abs(lower_limits - default_pos)
        range_to_upper = torch.abs(upper_limits - default_pos)

        scaling_factors_upper = range_to_upper / action_scale
        scaling_factors_lower = range_to_lower / action_scale

        print("SAC: Computed physics-based action scaling factors.")
        print(f"  Global action scale from ActionManager: {action_scale}")
        print(f"  Scaling factors lower limits: {scaling_factors_lower}")
        print(f"  Scaling factors upper limits: {scaling_factors_upper}")

        return scaling_factors_upper, scaling_factors_lower

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        model_params = [self.actor.state_dict(), self.critic.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])

    def reduce_parameters(self, params_or_model) -> None:
        """Collect gradients from the provided params/model and average them across all GPUs.

        Accepts either an nn.Module or an iterable of parameters.
        """
        if isinstance(params_or_model, torch.nn.Module):
            params = list(params_or_model.parameters())
        else:
            params = list(params_or_model)

        grads = [param.grad.view(-1) for param in params if param.grad is not None]
        if not grads:
            return

        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
