"""CPU functional test of the two-camera weighted DAgger stack on rsl_rl 5.5.1.

A mock VecEnv emits the UW assembly DAgger observation groups (student proprio, teacher proprio + point cloud,
side/wrist RGB, auxiliary pose targets) and a success termination. Checks: construction through
OnPolicyRunner.__init__, split masks, teacher JIT loading, rollout + weighted update, finite losses, the
per-pool success metrics written to TensorBoard, checkpoint save -> load round trip, and that loading
reproduces the student's actions.

The teacher is a small TorchScript Gaussian actor built here (flat proprio + point cloud -> mean, std).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import DistillationRunnerSplit

N, A, H, W = 64, 7, 224, 224


class _Term:
    def __init__(self, env):
        self.env = env

    def get_term(self, name):
        if name == "success":
            return self.env.success
        raise KeyError(name)


class MockEnv(VecEnv):
    def __init__(self):
        self.num_envs, self.num_actions, self.device = N, A, "cpu"
        self.max_episode_length = 3
        self.episode_length_buf = torch.zeros(N, dtype=torch.long)
        self.cfg = {"mock": True}
        self.success = torch.zeros(N, dtype=torch.bool)
        self.termination_manager = _Term(self)
        self.g = torch.Generator().manual_seed(0)

    @property
    def unwrapped(self):
        return self

    def _obs(self):
        r = lambda *s: torch.randn(N, *s, generator=self.g)  # noqa: E731
        return TensorDict(
            {
                "student_proprio": r(19),
                "proprio": r(25),
                "pointcloud": r(1152),
                "side_rgb": torch.rand(N, 3, H, W, generator=self.g),
                "wrist_rgb": torch.rand(N, 3, H, W, generator=self.g),
                "aux_target": r(18),
            },
            batch_size=[N],
        )

    def get_observations(self):
        return self._obs()

    def step(self, actions):
        assert actions.shape == (N, A) and torch.isfinite(actions).all()
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        self.success = dones & (torch.rand(N, generator=self.g) < 0.5)
        self.episode_length_buf[dones] = 0
        return self._obs(), torch.zeros(N), dones.long(), {"time_outs": dones.clone(), "log": {}}


def cfg(teacher):
    return {
        "class_name": "DistillationRunnerSplit", "seed": 1, "device": "cpu", "num_steps_per_env": 1,
        "max_iterations": 6, "save_interval": 3, "experiment_name": "mock", "run_name": "", "logger": "tensorboard",
        "student_fraction": 0.5, "eval_fraction": 0.1, "teacher_eval_fraction": 0.0, "clip_actions": None,
        "obs_groups": {"policy": ["student_proprio"], "teacher": ["proprio", "pointcloud"], "side_rgb": ["side_rgb"],
                       "wrist_rgb": ["wrist_rgb"], "aux_target": ["aux_target"]},
        "algorithm": {"class_name": "DistillationDAggerWeighted", "num_learning_epochs": 1, "num_mini_batches": 7,
                      "learning_rate": 1e-4, "gradient_length": 1, "max_grad_norm": 1.0, "optimizer": "adam",
                      "loss_type": "mse", "beta_anneal_iters": 0, "distributed_obs_normalization": True, "aux_coeff": 1.0,
                      "teacher_sample": False, "gripper_loss_type": "shared", "gripper_loss_weight": 1.0},
        "vision_policy": {"class_name": "StudentTeacherVision", "teacher_jit_path": teacher,
                          "vision_groups": ("side_rgb", "wrist_rgb"), "embed_dim": 128,
                          "student_hidden_dims": (512, 512, 512, 512), "activation": "elu", "init_noise_std": 0.1,
                          "noise_std_type": "scalar", "student_obs_normalization": True, "encoder_type": "resnet18",
                          "encoder_pretrained_path": "", "encoder_freeze_iters": 0, "predict_std": True,
                          "teacher_returns_std": True, "squash_student": False, "aux_enabled": True,
                          "aux_target_group": "aux_target",
                          "aux_target_keys": ("insertive_in_wrist", "receptive_in_wrist", "insertive_in_receptive"),
                          "aux_hidden_dims": (256, 128)},
    }


class _Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(25 + 1152, 64), torch.nn.ELU(), torch.nn.Linear(64, A))
        self.log_std = torch.nn.Parameter(torch.full((A,), -1.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(x)
        return mean, self.log_std.exp().expand_as(mean)


def test_distillation_runner_split(tmp_path=None):
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        teacher = str(Path(d) / "teacher_jit.pt")
        torch.jit.script(_Teacher()).save(teacher)
        env = MockEnv()
        runner = DistillationRunnerSplit(env, cfg(teacher), log_dir=d, device="cpu")
        assert int(runner.student_mask.sum()) > 0 and int(runner.eval_mask.sum()) == round(N * 0.1)
        runner.learn(6)
        ckpts = sorted(Path(d).glob("model_*.pt"))
        assert ckpts, "no checkpoint written"
        events = list(Path(d).rglob("events.out.tfevents.*"))
        assert events, "no tensorboard events"
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        tags = set()
        for e in events:
            ea = EventAccumulator(str(e)); ea.Reload(); tags |= set(ea.Tags()["scalars"])
        need = {"Metrics/success_student_eval", "Metrics/success_teacher_train", "Metrics/bc_loss_student_eval"}
        assert need <= tags, f"missing {need - tags}; have {sorted(tags)[:40]}"
        losses = [t for t in tags if t.startswith("Loss/")]
        assert losses, f"no Loss/ scalars: {sorted(tags)}"
        for t in losses:
            for e in events:
                ea = EventAccumulator(str(e)); ea.Reload()
                if t in ea.Tags()["scalars"]:
                    assert all(torch.isfinite(torch.tensor(s.value)) for s in ea.Scalars(t)), t
        obs = env.get_observations()
        runner.alg.policy.eval()
        with torch.inference_mode():
            a0 = runner.alg.policy.act_inference(obs)
        last = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
        runner.save(str(Path(d) / "roundtrip.pt"))
        env2 = MockEnv()
        r2 = DistillationRunnerSplit(env2, cfg(teacher), log_dir=None, device="cpu")
        r2.load(str(Path(d) / "roundtrip.pt"))
        r2.alg.policy.eval()
        with torch.inference_mode():
            a1 = r2.alg.policy.act_inference(obs)
        torch.testing.assert_close(a0, a1)
        assert last.name == "model_5.pt" and r2.current_learning_iteration == 5


if __name__ == "__main__":
    test_distillation_runner_split()
    print("DAGGER_MOCK_OK")
