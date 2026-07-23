# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import socket
import time
import torch
import zlib
from datetime import timedelta

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger

# ``PhysicsSceneStats`` mixes counts and byte quantities.  Keep the native
# attribute mapping here without importing any Omniverse modules at module load
# time: RSL-RL is also used by non-Isaac applications where those modules do
# not exist.
_PHYSX_ROLLOUT_MAX_COUNT_FIELDS = (
    ("physx_rigid_contact_count_rollout_max", "gpu_mem_rigid_contact_count"),
    ("physx_rigid_patch_count_rollout_max", "gpu_mem_rigid_patch_count"),
    ("physx_found_lost_pairs_rollout_max", "gpu_mem_found_lost_pairs"),
    ("physx_found_lost_aggregate_pairs_rollout_max", "gpu_mem_found_lost_aggregate_pairs"),
    ("physx_total_aggregate_pairs_rollout_max", "gpu_mem_total_aggregate_pairs"),
    ("physx_active_constraints_rollout_max", "nb_active_constraints"),
    # The misspelling is part of the generated Isaac Sim Python API.
    ("physx_axis_solver_constraints_rollout_max", "nb_axis_solver_constaints"),
    ("physx_discrete_contact_pairs_total_rollout_max", "nb_discrete_contact_pairs_total"),
    ("physx_discrete_contact_pairs_cache_hits_rollout_max", "nb_discrete_contact_pairs_with_cache_hits"),
    ("physx_discrete_contact_pairs_with_contacts_rollout_max", "nb_discrete_contact_pairs_with_contacts"),
    ("physx_new_pairs_rollout_max", "nb_new_pairs"),
    ("physx_lost_pairs_rollout_max", "nb_lost_pairs"),
    ("physx_new_touches_rollout_max", "nb_new_touches"),
    ("physx_lost_touches_rollout_max", "nb_lost_touches"),
    ("physx_partitions_rollout_max", "nb_partitions"),
)

_PHYSX_ROLLOUT_MAX_MIB_FIELDS = (
    ("physx_collision_stack_mib_rollout_max", "gpu_mem_collision_stack_size"),
    ("physx_heap_mib_rollout_max", "gpu_mem_heap"),
    ("physx_heap_broadphase_mib_rollout_max", "gpu_mem_heap_broadphase"),
    ("physx_heap_narrowphase_mib_rollout_max", "gpu_mem_heap_narrowphase"),
    ("physx_heap_solver_mib_rollout_max", "gpu_mem_heap_solver"),
    ("physx_heap_articulation_mib_rollout_max", "gpu_mem_heap_articulation"),
    ("physx_heap_simulation_mib_rollout_max", "gpu_mem_heap_simulation"),
    ("physx_heap_other_mib_rollout_max", "gpu_mem_heap_other"),
    ("physx_temp_buffer_mib_rollout_max", "gpu_mem_temp_buffer_capacity"),
    ("physx_compressed_contact_mib_rollout_max", "compressed_contact_size"),
    ("physx_required_contact_constraint_mib_rollout_max", "required_contact_constraint_memory"),
    ("physx_peak_constraint_mib_rollout_max", "peak_constraint_memory"),
)

_PHYSX_ROLLOUT_MAX_CAPACITY_FIELDS = (
    (
        "physx_rigid_contact_capacity_fraction_rollout_max",
        "gpu_mem_rigid_contact_count",
        "gpu_max_rigid_contact_count",
    ),
    (
        "physx_rigid_patch_capacity_fraction_rollout_max",
        "gpu_mem_rigid_patch_count",
        "gpu_max_rigid_patch_count",
    ),
    (
        "physx_found_lost_pairs_capacity_fraction_rollout_max",
        "gpu_mem_found_lost_pairs",
        "gpu_found_lost_pairs_capacity",
    ),
    (
        "physx_found_lost_aggregate_pairs_capacity_fraction_rollout_max",
        "gpu_mem_found_lost_aggregate_pairs",
        "gpu_found_lost_aggregate_pairs_capacity",
    ),
    (
        "physx_total_aggregate_pairs_capacity_fraction_rollout_max",
        "gpu_mem_total_aggregate_pairs",
        "gpu_total_aggregate_pairs_capacity",
    ),
    (
        "physx_collision_stack_capacity_fraction_rollout_max",
        "gpu_mem_collision_stack_size",
        "gpu_collision_stack_size",
    ),
    (
        "physx_temp_buffer_capacity_fraction_rollout_max",
        "gpu_mem_temp_buffer_capacity",
        "gpu_temp_buffer_capacity",
    ),
)

_RANK_DIAGNOSTIC_ROLLOUT_MAX_KEYS = (
    *(name for name, _ in _PHYSX_ROLLOUT_MAX_COUNT_FIELDS),
    *(name for name, _ in _PHYSX_ROLLOUT_MAX_MIB_FIELDS),
    *(name for name, _, _ in _PHYSX_ROLLOUT_MAX_CAPACITY_FIELDS),
    "cuda_used_mib_rollout_max",
    "cuda_total_mib_rollout_max",
    "torch_memory_allocated_mib_rollout_max",
    "torch_memory_reserved_mib_rollout_max",
    "torch_max_memory_allocated_mib_rollout_max",
    "torch_max_memory_reserved_mib_rollout_max",
)

_RANK_DIAGNOSTIC_ROLLOUT_MIN_KEYS = (
    # A minimum of zero means at least one policy step could not be queried.
    "physx_query_ok_rollout_min",
    "cuda_query_ok_rollout_min",
    "cuda_free_mib_rollout_min",
)

_RANK_DIAGNOSTIC_ROLLOUT_EXTREMA_KEYS = (
    *_RANK_DIAGNOSTIC_ROLLOUT_MAX_KEYS,
    *_RANK_DIAGNOSTIC_ROLLOUT_MIN_KEYS,
)
_RANK_DIAGNOSTIC_TERMINATION_EVENTS = ("success", "time_out", "rod_oob", "abnormal")
_RANK_DIAGNOSTIC_TERMINATION_TAGS = ("grasp_asset_in_air", "start_assembled", "start_grasped")
# Factory tasks may retain per-joint first-step velocity channels for local
# debugging.  Gathering every joint on every rank makes the production W&B
# payload unnecessarily large, so the generic runner exports only the compact
# component and velocity-limit/readback contract below.
_RANK_DIAGNOSTIC_FIRST_STEP_VELOCITY_KEYS = (
    "robot_joint_velocity_absmax",
    "passive_mimic_joint_velocity_absmax",
    "held_asset_linear_speed_max",
    "held_asset_angular_speed_max",
    "passive_mimic_joint_velocity_limit_min",
    "passive_mimic_joint_velocity_limit_max",
    "passive_mimic_joint_velocity_limit_backend_min",
    "passive_mimic_joint_velocity_limit_backend_max",
    "passive_mimic_joint_velocity_limit_backend_requested_mismatch_fraction",
    "passive_mimic_joint_velocity_limit_backend_cache_mismatch_fraction",
)
_BYTES_PER_MIB = float(1024**2)


class OnPolicyRunner:
    """On-policy runner for reinforcement learning algorithms."""

    alg: PPO
    """The actor-critic algorithm."""

    _RANK_DIAGNOSTIC_EXTRA_KEYS = (
        "Episode_Termination/success",
        "Episode_Termination/time_out",
        "Episode_Termination/rod_oob",
        "Episode_Termination/abnormal",
        "Episode_Termination/progress_context",
        "GPS/prob/grasp_asset_in_air",
        "GPS/prob/start_assembled",
        "GPS/prob/start_grasped",
        "GPS/success_rate/grasp_asset_in_air",
        "GPS/success_rate/start_assembled",
        "GPS/success_rate/start_grasped",
        "Curriculum/gravity",
    )

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the runner, algorithm, and logging stack."""
        self.env = env
        self.cfg = train_cfg
        self.device = device

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from the environment for algorithm construction
        obs = self.env.get_observations()

        # Create the algorithm
        alg_class: type[PPO] = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)

        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations."""
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        rank_diagnostics_enabled = self.cfg.get("rank_diagnostics", False)
        for it in range(start_it, total_it):
            start = time.time()
            rank_diagnostic_sums: dict[str, torch.Tensor] = {}
            rank_diagnostic_counts: dict[str, int] = {}
            rank_diagnostic_extrema = self._new_rank_diagnostic_extrema() if rank_diagnostics_enabled else None
            termination_event_counter_start = (
                self._local_termination_event_counter_snapshot() if rank_diagnostics_enabled else None
            )
            # Resample gSDE exploration weights once per rollout so the
            # state-dependent noise direction is consistent within the rollout
            # but refreshed between iterations. No-op for non-gSDE distributions.
            actor_dist = getattr(getattr(self.alg, "actor", None), "distribution", None)
            if actor_dist is not None and hasattr(actor_dist, "sample_weights"):
                actor_dist.sample_weights()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards if RND is used (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                    if rank_diagnostics_enabled:
                        self._accumulate_rank_diagnostics(extras, rank_diagnostic_sums, rank_diagnostic_counts)
                        # PhysicsSceneStats contains current-frame values, so
                        # sample once per policy step and retain rollout extrema
                        # instead of relying on one end-of-rollout snapshot.
                        self._accumulate_rank_diagnostic_extrema(rank_diagnostic_extrema)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

                # Per-task policy metrics (action magnitude always; entropy only for
                # heteroscedastic Gaussians since homoscedastic entropy is redundant with
                # Loss/entropy). Computed before update() clears storage; graceful no-op
                # on non-IsaacLab envs — see :meth:`_compute_per_task_policy_metrics`.
                policy_metrics = self._compute_per_task_policy_metrics()
                rank_diagnostics = (
                    self._gather_rank_diagnostics(
                        rank_diagnostic_sums,
                        rank_diagnostic_counts,
                        rank_diagnostic_extrema,
                        termination_event_counter_start,
                    )
                    if rank_diagnostics_enabled
                    else None
                )

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
                policy_metrics=policy_metrics,
                rank_diagnostics=rank_diagnostics,
            )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

    def _accumulate_rank_diagnostics(
        self,
        extras: dict,
        sums: dict[str, torch.Tensor],
        counts: dict[str, int],
    ) -> None:
        """Accumulate selected environment metrics locally on every rank."""
        # Manager-based environments split diagnostics across both namespaces:
        # terminations are episodic, while GPS/reset-curriculum values are stream
        # metrics.  Merge both instead of choosing ``episode`` whenever present,
        # which previously made every per-rank GPS channel NaN.
        step_metrics = dict(extras.get("log", {}))
        step_metrics.update(extras.get("episode", {}))
        for key in self._RANK_DIAGNOSTIC_EXTRA_KEYS:
            if key not in step_metrics:
                continue
            value = torch.as_tensor(step_metrics[key], device=self.device, dtype=torch.float32).mean()
            sums[key] = sums.get(key, torch.zeros((), device=self.device)) + value
            counts[key] = counts.get(key, 0) + 1

    def _new_rank_diagnostic_extrema(self) -> torch.Tensor:
        """Create the fixed-shape rollout-extrema vector for one rank."""
        return torch.full(
            (len(_RANK_DIAGNOSTIC_ROLLOUT_EXTREMA_KEYS),),
            float("nan"),
            device=self.device,
            dtype=torch.float32,
        )

    def _accumulate_rank_diagnostic_extrema(self, extrema: torch.Tensor | None) -> None:
        """Fold one policy step's PhysX/CUDA diagnostics into rollout extrema."""
        if extrema is None:
            return
        step_values = self._local_physics_cuda_diagnostics()
        split = len(_RANK_DIAGNOSTIC_ROLLOUT_MAX_KEYS)
        # fmax/fmin preserve a finite sample when the other argument is NaN.
        # This lets a transient stats failure leave usage extrema intact while
        # the separate query-ok minimum still records the failed query.
        extrema[:split] = torch.fmax(extrema[:split], step_values[:split])
        extrema[split:] = torch.fmin(extrema[split:], step_values[split:])

    def _local_physics_cuda_diagnostics(self) -> torch.Tensor:
        """Read one fixed-shape, best-effort PhysX/CUDA diagnostic snapshot.

        Omniverse imports and scene lookup are intentionally lazy.  Successful
        lookups are cached per process/rank, while failures return NaNs plus a
        zero query flag rather than perturbing training or changing collective
        tensor shapes.
        """
        values = {key: float("nan") for key in _RANK_DIAGNOSTIC_ROLLOUT_EXTREMA_KEYS}
        values["physx_query_ok_rollout_min"] = 0.0
        values["cuda_query_ok_rollout_min"] = 0.0

        try:
            cache = getattr(self, "_physx_stats_cache", None)
            if cache is None:
                # These modules only exist after an Isaac Sim application has
                # started.  Do not move the imports to module scope.
                from isaaclab.sim.utils.stage import get_current_stage_id
                from omni.physx import get_physx_statistics_interface
                from omni.physx.bindings._physx import PhysicsSceneStats
                from pxr import PhysicsSchemaTools

                base_env = getattr(self.env, "unwrapped", self.env)
                sim = base_env.sim
                scene = getattr(base_env, "scene", None)
                scene_path = scene.physics_scene_path if scene is not None else sim.cfg.physics_prim_path
                # The encoded value is process-local and has changed between
                # otherwise identical processes in this Isaac Sim build.
                scene_path_token = PhysicsSchemaTools.sdfPathToInt(str(scene_path))
                physics_cfg = sim.cfg.physics
                capacities = {
                    cfg_field: getattr(physics_cfg, cfg_field, None)
                    for _, _, cfg_field in _PHYSX_ROLLOUT_MAX_CAPACITY_FIELDS
                }
                cache = (
                    get_physx_statistics_interface(),
                    get_current_stage_id(),
                    scene_path_token,
                    PhysicsSceneStats(),
                    capacities,
                )
                self._physx_stats_cache = cache

            interface, stage_id, scene_path_token, stats, capacities = cache
            query_ok = interface.get_physx_scene_statistics(stage_id, scene_path_token, stats)
            values["physx_query_ok_rollout_min"] = float(bool(query_ok))
            if query_ok:
                for metric, field in _PHYSX_ROLLOUT_MAX_COUNT_FIELDS:
                    values[metric] = float(getattr(stats, field))
                for metric, field in _PHYSX_ROLLOUT_MAX_MIB_FIELDS:
                    values[metric] = float(getattr(stats, field)) / _BYTES_PER_MIB
                for metric, usage_field, cfg_field in _PHYSX_ROLLOUT_MAX_CAPACITY_FIELDS:
                    capacity = capacities.get(cfg_field)
                    if capacity is not None and float(capacity) > 0.0:
                        values[metric] = float(getattr(stats, usage_field)) / float(capacity)
            else:
                # A not-yet-attached/recreated stage should be resolved again
                # on the next policy step.
                self._physx_stats_cache = None
        except Exception:
            # Diagnostics must never stop a training run.  Invalidate a stale
            # cache and retry lazily on the next policy step.
            self._physx_stats_cache = None

        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            try:
                device_index = torch.cuda.current_device()
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                values["cuda_query_ok_rollout_min"] = 1.0
                values["cuda_free_mib_rollout_min"] = float(free_bytes) / _BYTES_PER_MIB
                values["cuda_used_mib_rollout_max"] = float(total_bytes - free_bytes) / _BYTES_PER_MIB
                values["cuda_total_mib_rollout_max"] = float(total_bytes) / _BYTES_PER_MIB
                values["torch_memory_allocated_mib_rollout_max"] = (
                    float(torch.cuda.memory_allocated(device_index)) / _BYTES_PER_MIB
                )
                values["torch_memory_reserved_mib_rollout_max"] = (
                    float(torch.cuda.memory_reserved(device_index)) / _BYTES_PER_MIB
                )
                values["torch_max_memory_allocated_mib_rollout_max"] = (
                    float(torch.cuda.max_memory_allocated(device_index)) / _BYTES_PER_MIB
                )
                values["torch_max_memory_reserved_mib_rollout_max"] = (
                    float(torch.cuda.max_memory_reserved(device_index)) / _BYTES_PER_MIB
                )
            except Exception:
                pass

        # One host-to-device copy keeps the per-policy-step instrumentation
        # cheap; extrema reduction then needs only one fmax and one fmin vector
        # operation instead of a kernel per metric.
        return torch.tensor(
            [values[key] for key in _RANK_DIAGNOSTIC_ROLLOUT_EXTREMA_KEYS],
            device=self.device,
            dtype=torch.float32,
        )

    def _local_termination_event_counter_snapshot(self) -> dict[str, torch.Tensor] | None:
        """Snapshot monotonic factory reset counters in a fixed, rank-independent schema."""
        try:
            base_env = getattr(self.env, "unwrapped", self.env)
            reset = base_env.event_manager.get_term_cfg("reset_positioning").func
            event_indices = {name: index for index, name in enumerate(reset.termination_event_names)}
            tag_indices = {name: index for index, name in enumerate(reset.names)}

            snapshot = {
                "episodes": torch.as_tensor(
                    reset.termination_event_episode_count, device=self.device, dtype=torch.long
                ).clone()
            }
            for event in _RANK_DIAGNOSTIC_TERMINATION_EVENTS:
                snapshot[event] = torch.as_tensor(
                    reset.termination_event_counts[event_indices[event]], device=self.device, dtype=torch.long
                ).clone()
            for tag in _RANK_DIAGNOSTIC_TERMINATION_TAGS:
                tag_index = tag_indices.get(tag)
                if tag_index is None:
                    snapshot[f"{tag}/episodes"] = torch.zeros((), device=self.device, dtype=torch.long)
                    for event in _RANK_DIAGNOSTIC_TERMINATION_EVENTS:
                        snapshot[f"{tag}/{event}"] = torch.zeros((), device=self.device, dtype=torch.long)
                    continue
                snapshot[f"{tag}/episodes"] = torch.as_tensor(
                    reset.termination_event_episode_counts_by_tag[tag_index],
                    device=self.device,
                    dtype=torch.long,
                ).clone()
                for event in _RANK_DIAGNOSTIC_TERMINATION_EVENTS:
                    snapshot[f"{tag}/{event}"] = torch.as_tensor(
                        reset.termination_event_counts_by_tag[tag_index, event_indices[event]],
                        device=self.device,
                        dtype=torch.long,
                    ).clone()
            return snapshot
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
            return None

    def _local_termination_event_rollout_diagnostics(
        self, start: dict[str, torch.Tensor] | None
    ) -> dict[str, torch.Tensor]:
        """Convert cumulative reset counters into exact counts and rates for one rollout."""
        nan = torch.full((), float("nan"), device=self.device)
        metrics: dict[str, torch.Tensor] = {
            "TerminationEvents/rollout/episodes/count": nan,
        }
        for event in _RANK_DIAGNOSTIC_TERMINATION_EVENTS:
            metrics[f"TerminationEvents/rollout/{event}/count"] = nan
            metrics[f"TerminationEvents/rollout/{event}/rate"] = nan
        for tag in _RANK_DIAGNOSTIC_TERMINATION_TAGS:
            metrics[f"TerminationEvents/rollout/tag/{tag}/episodes/count"] = nan
            for event in _RANK_DIAGNOSTIC_TERMINATION_EVENTS:
                metrics[f"TerminationEvents/rollout/tag/{tag}/{event}/count"] = nan
                metrics[f"TerminationEvents/rollout/tag/{tag}/{event}/rate"] = nan

        end = self._local_termination_event_counter_snapshot()
        if start is None or end is None or start.keys() != end.keys():
            return metrics

        delta = {key: end[key] - start[key] for key in start}
        episodes = delta["episodes"].float()
        metrics["TerminationEvents/rollout/episodes/count"] = episodes
        for event in _RANK_DIAGNOSTIC_TERMINATION_EVENTS:
            count = delta[event].float()
            metrics[f"TerminationEvents/rollout/{event}/count"] = count
            metrics[f"TerminationEvents/rollout/{event}/rate"] = torch.where(episodes > 0, count / episodes, nan)
        for tag in _RANK_DIAGNOSTIC_TERMINATION_TAGS:
            tag_episodes = delta[f"{tag}/episodes"].float()
            metrics[f"TerminationEvents/rollout/tag/{tag}/episodes/count"] = tag_episodes
            for event in _RANK_DIAGNOSTIC_TERMINATION_EVENTS:
                count = delta[f"{tag}/{event}"].float()
                metrics[f"TerminationEvents/rollout/tag/{tag}/{event}/count"] = count
                metrics[f"TerminationEvents/rollout/tag/{tag}/{event}/rate"] = torch.where(
                    tag_episodes > 0, count / tag_episodes, nan
                )
        return metrics

    def _local_reset_diagnostics(self) -> dict[str, torch.Tensor]:
        """Read-only fingerprints of the reset bank and its live sampling state."""
        nan = torch.full((), float("nan"), device=self.device)
        metrics = {
            "reset_buffer_finite_fraction": nan,
            "reset_buffer_mean": nan,
            "reset_buffer_std": nan,
            "reset_buffer_absmax": nan,
            "reset_buffer_sum": nan,
            "reset_buffer_sq_sum": nan,
            "reset_success_rate_mean": nan,
            "reset_success_rate_min": nan,
            "reset_success_rate_max": nan,
            "reset_success_rate_zero_fraction": nan,
            "reset_success_rate_one_fraction": nan,
            "reset_success_history_size_mean": nan,
            "reset_success_history_size_min": nan,
            "reset_success_history_size_max": nan,
            "reset_success_history_fraction_full": nan,
            "gps_probability_max": nan,
            "gps_probability_entropy": nan,
            "gps_probability_effective_slots": nan,
            "reset_readback_last_max_abs": nan,
            "reset_readback_last_rms": nan,
            "reset_readback_last_mismatch_fraction": nan,
            "reset_readback_max_abs_ever": nan,
            "reset_readback_mismatch_fraction_total": nan,
            "reset_readback_check_count": nan,
            "first_step_readback_last_max_abs": nan,
            "first_step_readback_last_rms": nan,
            "first_step_readback_last_mismatch_fraction": nan,
            "first_step_readback_max_abs_ever": nan,
            "first_step_readback_mismatch_fraction_total": nan,
            "first_step_readback_check_count": nan,
            "first_step_raw_action_rms": nan,
            "first_step_raw_action_absmax": nan,
            "first_step_processed_action_rms": nan,
            "first_step_processed_action_absmax": nan,
        }
        for index in range(3):
            metrics[f"reset_buffer_tag_fraction_{index}"] = nan
            metrics[f"reset_sampled_tag_fraction_{index}"] = nan
        # Keep the collective schema fixed even if one rank cannot read the
        # task-provided dictionaries. Unequal vector lengths would otherwise
        # strand peers in all_gather instead of degrading diagnostics to NaN.
        for window in ("last", "max_ever"):
            for name in _RANK_DIAGNOSTIC_FIRST_STEP_VELOCITY_KEYS:
                metrics[f"first_step_velocity/{window}/{name}"] = nan
        try:
            reset = self.env.unwrapped.event_manager.get_term_cfg("reset_positioning").func
            size = len(reset.buffer)
            data = reset.buffer.data[:size]
            finite = torch.isfinite(data)
            metrics["reset_buffer_finite_fraction"] = finite.float().mean()
            finite_data = data[finite]
            if finite_data.numel() > 0:
                metrics["reset_buffer_mean"] = finite_data.mean()
                metrics["reset_buffer_std"] = finite_data.std(unbiased=False)
                metrics["reset_buffer_absmax"] = finite_data.abs().max()
                metrics["reset_buffer_sum"] = finite_data.sum()
                metrics["reset_buffer_sq_sum"] = finite_data.square().sum()
            success_rate = reset.success_rate.float()
            metrics["reset_success_rate_mean"] = success_rate.mean()
            metrics["reset_success_rate_min"] = success_rate.min()
            metrics["reset_success_rate_max"] = success_rate.max()
            metrics["reset_success_rate_zero_fraction"] = (success_rate == 0.0).float().mean()
            metrics["reset_success_rate_one_fraction"] = (success_rate == 1.0).float().mean()
            for name in (
                "reset_readback_last_max_abs",
                "reset_readback_last_rms",
                "reset_readback_last_mismatch_fraction",
                "reset_readback_max_abs_ever",
                "reset_readback_check_count",
                "first_step_readback_last_max_abs",
                "first_step_readback_last_rms",
                "first_step_readback_last_mismatch_fraction",
                "first_step_readback_max_abs_ever",
                "first_step_readback_check_count",
                "first_step_raw_action_rms",
                "first_step_raw_action_absmax",
                "first_step_processed_action_rms",
                "first_step_processed_action_absmax",
            ):
                value = getattr(reset, name, None)
                if value is not None:
                    metrics[name] = torch.as_tensor(value, device=self.device, dtype=torch.float32)
            checks = getattr(reset, "reset_readback_check_count", None)
            mismatches = getattr(reset, "reset_readback_mismatch_count", None)
            if checks is not None and mismatches is not None and int(checks.item()) > 0:
                metrics["reset_readback_mismatch_fraction_total"] = mismatches.float() / checks.float()
            checks = getattr(reset, "first_step_readback_check_count", None)
            mismatches = getattr(reset, "first_step_readback_mismatch_count", None)
            if checks is not None and mismatches is not None and int(checks.item()) > 0:
                metrics["first_step_readback_mismatch_fraction_total"] = mismatches.float() / checks.float()
            # factory_v2's opt-in reset diagnostics expose both compact
            # component-level values and local per-joint detail. Export a fixed
            # allowlist so the gathered schema remains rank-independent without
            # multiplying the production payload by the robot's joint count.
            for window, attribute in (
                ("last", "first_step_velocity_last"),
                ("max_ever", "first_step_velocity_max_ever"),
            ):
                velocity_metrics = getattr(reset, attribute, None)
                if isinstance(velocity_metrics, dict):
                    for name in _RANK_DIAGNOSTIC_FIRST_STEP_VELOCITY_KEYS:
                        if name not in velocity_metrics:
                            continue
                        value = velocity_metrics[name]
                        metrics[f"first_step_velocity/{window}/{name}"] = torch.as_tensor(
                            value, device=self.device, dtype=torch.float32
                        )
            tags = reset.buffer.tags[:size]
            sampled_tags = reset.sampled_tags
            for index in range(3):
                metrics[f"reset_buffer_tag_fraction_{index}"] = (tags == index).float().mean()
                metrics[f"reset_sampled_tag_fraction_{index}"] = (sampled_tags == index).float().mean()
            # Source GPS channels directly from the sampler. Manager ``extras``
            # namespaces are ephemeral and did not reliably reach every rank's
            # rollout collector.
            probabilities = getattr(reset, "last_sampling_probs", None)
            names = getattr(reset, "names", ())
            if probabilities is not None:
                probabilities = probabilities.float()
                metrics["gps_probability_max"] = probabilities.max()
                metrics["gps_probability_entropy"] = -(probabilities * probabilities.clamp_min(1e-20).log()).sum()
                metrics["gps_probability_effective_slots"] = probabilities.square().sum().reciprocal()
                for index, name in enumerate(names):
                    if reset.gps_mode == "per_state":
                        mask = tags == index
                        if mask.any():
                            metrics[f"GPS/success_rate/{name}"] = success_rate[mask].mean()
                            metrics[f"GPS/prob/{name}"] = probabilities[mask].sum()
                    elif index < success_rate.numel() and index < probabilities.numel():
                        metrics[f"GPS/success_rate/{name}"] = success_rate[index]
                        metrics[f"GPS/prob/{name}"] = probabilities[index]
            success_monitor = getattr(reset, "success_monitor", None)
            if success_monitor is not None:
                history_sizes = success_monitor.success_size.float()
                metrics["reset_success_history_size_mean"] = history_sizes.mean()
                metrics["reset_success_history_size_min"] = history_sizes.min()
                metrics["reset_success_history_size_max"] = history_sizes.max()
                metrics["reset_success_history_fraction_full"] = (
                    (history_sizes >= float(success_monitor.history_len)).float().mean()
                )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass
        return metrics

    def _gather_rank_diagnostics(
        self,
        sums: dict[str, torch.Tensor],
        counts: dict[str, int],
        rollout_extrema: torch.Tensor | None = None,
        termination_event_counter_start: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, float] | None:
        """Gather a fixed diagnostic vector from all ranks and flatten it for rank-zero logging."""
        local = {
            key: sums[key] / counts[key] if counts.get(key) else torch.full((), float("nan"), device=self.device)
            for key in self._RANK_DIAGNOSTIC_EXTRA_KEYS
        }
        local.update(self._local_reset_diagnostics())
        local.update(self._local_termination_event_rollout_diagnostics(termination_event_counter_start))
        if rollout_extrema is None:
            rollout_extrema = self._new_rank_diagnostic_extrema()
        local.update({key: rollout_extrema[index] for index, key in enumerate(_RANK_DIAGNOSTIC_ROLLOUT_EXTREMA_KEYS)})
        # Cheap agreement fingerprints.  Identical distributed ranks must have
        # identical parameter moments and optimizer LR at rollout boundaries.
        # These make a silent DDP/optimizer divergence distinguishable from a
        # rank-local simulator or GPS-state collapse without gathering models.
        with torch.no_grad():
            for name, module in (("actor", self.alg.actor), ("critic", self.alg.critic)):
                param_sum = torch.zeros((), device=self.device)
                param_sq_sum = torch.zeros((), device=self.device)
                for parameter in module.parameters():
                    values = parameter.detach().float()
                    param_sum += values.sum()
                    param_sq_sum += values.square().sum()
                local[f"{name}_parameter_sum"] = param_sum
                local[f"{name}_parameter_sq_sum"] = param_sq_sum
                buffer_sum = torch.zeros((), device=self.device)
                buffer_sq_sum = torch.zeros((), device=self.device)
                buffer_absmax = torch.zeros((), device=self.device)
                for buffer in module.buffers():
                    values = buffer.detach().float()
                    buffer_sum += values.sum()
                    buffer_sq_sum += values.square().sum()
                    if values.numel() > 0:
                        buffer_absmax = torch.maximum(buffer_absmax, values.abs().max())
                local[f"{name}_buffer_sum"] = buffer_sum
                local[f"{name}_buffer_sq_sum"] = buffer_sq_sum
                local[f"{name}_buffer_absmax"] = buffer_absmax
                normalizer = getattr(module, "obs_normalizer", None)
                for statistic in ("_mean", "_var", "_std", "count"):
                    value = getattr(normalizer, statistic, None)
                    if value is None:
                        continue
                    values = value.detach().float()
                    key = statistic.lstrip("_")
                    local[f"{name}_obs_normalizer_{key}_sum"] = values.sum()
                    local[f"{name}_obs_normalizer_{key}_sq_sum"] = values.square().sum()
        rollout_actions = self.alg.storage.actions
        if rollout_actions is not None:
            rollout_actions = rollout_actions.float()
            local["rollout_raw_action_rms"] = rollout_actions.square().mean().sqrt()
            local["rollout_raw_action_absmax"] = rollout_actions.abs().max()
        local["optimizer_learning_rate"] = torch.tensor(
            float(self.alg.learning_rate), device=self.device, dtype=torch.float32
        )
        local["host_hash"] = torch.tensor(
            zlib.crc32(socket.gethostname().encode()) % (1 << 24), device=self.device, dtype=torch.float32
        )
        local["visible_device_hash"] = torch.tensor(
            zlib.crc32(os.getenv("CUDA_VISIBLE_DEVICES", "").encode()) % (1 << 24),
            device=self.device,
            dtype=torch.float32,
        )
        local["local_rank"] = torch.tensor(self.gpu_local_rank, device=self.device, dtype=torch.float32)
        local["device_uuid_hash"] = torch.full((), float("nan"), device=self.device)
        local["device_pci_bus_hash"] = torch.full((), float("nan"), device=self.device)
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            device_properties = torch.cuda.get_device_properties(device_index)
            device_uuid = getattr(device_properties, "uuid", "")
            local["device_uuid_hash"] = torch.tensor(
                zlib.crc32(str(device_uuid).encode()) % (1 << 24), device=self.device, dtype=torch.float32
            )
            pci_bus_id = getattr(device_properties, "pci_bus_id", "")
            local["device_pci_bus_hash"] = torch.tensor(
                zlib.crc32(str(pci_bus_id).encode()) % (1 << 24), device=self.device, dtype=torch.float32
            )
        names = tuple(local)
        vector = torch.stack([local[name] for name in names])
        if self.is_distributed:
            gathered = [torch.empty_like(vector) for _ in range(self.gpu_world_size)]
            torch.distributed.all_gather(gathered, vector)
        else:
            gathered = [vector]
        if self.gpu_global_rank != 0:
            return None
        return {
            f"RankDiagnostics/rank_{rank:02d}/{name}": float(value.item())
            for rank, rank_vector in enumerate(gathered)
            for name, value in zip(names, rank_vector)
        }

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state to a given path and upload them if external logging is used."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        if self.cfg.get("save_curriculum_state", True):
            curriculum_state = self._get_curriculum_state()
            if curriculum_state:
                saved_dict["curriculum_state"] = curriculum_state
        torch.save(saved_dict, path)
        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state from a given path.

        Args:
            path (str): Path to load the model from.
            load_cfg (dict | None): Optional dictionary that defines what models and states to load. If None, all
                models and states are loaded.
            strict (bool): Whether state_dict loading should be strict.
            map_location (str | None): Device mapping for loading the model.
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        if not self.cfg.get("reset_curriculum_on_load", False) and "curriculum_state" in loaded_dict:
            self._set_curriculum_state(loaded_dict["curriculum_state"])
        return loaded_dict["infos"]

    def _compute_per_task_policy_metrics(self) -> dict[str, float] | None:
        """Per-terrain-type policy metrics over the just-collected rollout.

        Returns a flat dict mapping ``<metric>/<terrain_name>`` → float. The logger
        prefixes each key with ``Policy/``, so e.g.
        ``per_task_action_magnitude/stepping_stone`` becomes
        ``Policy/per_task_action_magnitude/stepping_stone`` in TB / W&B.

        Always-logged:

        - **Action magnitude.** ``mean(||a||_2)`` per terrain type. Depends on the action
          *mean*, so it varies across envs even when ``std`` is global.

        Conditionally-logged:

        - **Action-distribution entropy.** Only for *heteroscedastic* Gaussians
          (state-dependent ``std``). For homoscedastic Gaussians the per-task split is
          degenerate (entropy depends only on ``std``, which is constant across envs),
          and the same signal already lives in ``Loss/entropy`` — within ~0.003 nat,
          which is just the std drift across the K minibatch update steps.

        Returns ``None`` if the env is not an IsaacLab terrain env or storage hasn't
        been populated yet — graceful no-op outside the locomotion-task setup.
        """
        try:
            terrain = self.env.unwrapped.scene.terrain  # type: ignore[attr-defined]
            terrain_types = terrain.terrain_types  # [num_envs] long
            sub_terrain_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        except (AttributeError, KeyError):
            return None

        actions = self.alg.storage.actions
        if actions is None:
            return None

        metrics: dict[str, float] = {}

        # Per-task action magnitude. ``actions`` is [num_steps, num_envs, action_dim].
        action_magnitudes = actions.norm(dim=-1)  # [num_steps, num_envs]
        for terrain_index, name in enumerate(sub_terrain_names):
            mask = terrain_types == terrain_index
            if mask.any():
                metrics[f"per_task_action_magnitude/{name}"] = action_magnitudes[:, mask].mean().item()

        # Per-task entropy — only worth logging if std varies across envs (heteroscedastic).
        params = self.alg.storage.distribution_params
        if params is not None and len(params) == 2:
            mean, std = params
            if mean.ndim == 3 and std.shape == mean.shape:
                std_varies_across_envs = std.std(dim=1).max().item() > 1e-9
                if std_varies_across_envs:
                    entropy_per_step_env = (0.5 * (2.0 * torch.pi * torch.e * std.pow(2)).log()).sum(dim=-1)
                    for terrain_index, name in enumerate(sub_terrain_names):
                        mask = terrain_types == terrain_index
                        if mask.any():
                            metrics[f"per_task_entropy/{name}"] = entropy_per_step_env[:, mask].mean().item()

        return metrics if metrics else None

    def _get_curriculum_state(self) -> dict | None:
        """Return the curriculum manager's serializable state, or ``None`` if unavailable."""
        env = getattr(self.env, "unwrapped", self.env)
        manager = getattr(env, "curriculum_manager", None)
        if manager is None or not hasattr(manager, "state_dict"):
            return None
        return manager.state_dict()

    def _set_curriculum_state(self, state: dict) -> None:
        """Restore curriculum state previously produced by :meth:`_get_curriculum_state`."""
        env = getattr(self.env, "unwrapped", self.env)
        manager = getattr(env, "curriculum_manager", None)
        if manager is None or not hasattr(manager, "load_state_dict"):
            return
        manager.load_state_dict(state)

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """Return the policy on the requested device for inference."""
        self.alg.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        return self.alg.get_policy().to(device)  # type: ignore

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the model to a Torch JIT file."""
        jit_model = self.alg.get_policy().as_jit()
        jit_model.to("cpu")

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        traced_model = torch.jit.script(jit_model)
        traced_model.save(save_path)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the model into an ONNX file."""
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),  # type: ignore
            save_path,
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,  # type: ignore
            output_names=onnx_model.output_names,  # type: ignore
        )

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository path whose git status should be logged."""
        self.logger.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(
            backend="nccl",
            rank=self.gpu_global_rank,
            world_size=self.gpu_world_size,
            timeout=timedelta(minutes=10),
        )
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(
            "[RankIdentity] "
            f"global_rank={self.gpu_global_rank} local_rank={self.gpu_local_rank} "
            f"host={socket.gethostname()} uuid={getattr(properties, 'uuid', '')} "
            f"pci_bus_id={getattr(properties, 'pci_bus_id', '')}",
            flush=True,
        )
