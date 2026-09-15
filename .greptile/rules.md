# RSL-RL review conventions

This is the UW-Lab fork of RSL-RL, a standalone PyTorch library for PPO and
student-teacher distillation, with RND and symmetry extensions. Review the code
and APIs in this fork; newer upstream layouts are not requirements for this branch.

## Sources and review style

The configuration structure and focus on actionable bugs follow
[UWLab's Greptile setup](https://github.com/UW-Lab/UWLab/tree/main/.greptile).
[Upstream contribution guidance](https://github.com/leggedrobotics/rsl_rl/blob/main/CONTRIBUTING.md)
calls for PEP 8, Google-style docstrings, and Ruff linting and formatting. These
conventions also appear in this fork's README. Upstream main was checked at
`857de6165c5fd479726ec8ac5c9303a497766f30`: there were no checked-in Greptile or
AI reviewer instruction files to import. Copilot reviews exist upstream, but
their comments are not repository policy.

Report concrete issues introduced by the change, with a failing scenario and
the affected caller or tensor operation. Avoid speculative refactors, style
preferences, and reports about unchanged code unless the PR makes it fail.

## Existing automated checks

`.pre-commit-config.yaml` configures Ruff lint/format, codespell, Python license
headers, YAML/TOML validation, merge-conflict and case-conflict checks, symlink
and shebang checks, and private-key detection. Do not duplicate these checks.
Use `ruff.toml` for exact lint coverage and exceptions; do not assume every
style convention is enabled. This fork currently has no checked-in CI workflow.

The package declares Python >=3.9 in `pyproject.toml`; many modules use postponed
annotations. Do not blindly apply UWLab's Python-version or annotation rules.

## Repository layout and integration boundaries

- `rsl_rl/algorithms/`: PPO and distillation updates, including distributed operations.
- `rsl_rl/runners/`: environment interaction, configuration, logging, checkpoints,
  and training/inference modes.
- `rsl_rl/modules/`: actor-critic and student-teacher policies, RND, and symmetry.
- `rsl_rl/networks/`: MLPs, recurrent memory, and normalization.
- `rsl_rl/storage/`: rollout buffers, return computation, and minibatch generators.
- `rsl_rl/env/vec_env.py`: the interface supplied by external environment packages.
- `rsl_rl/utils/`: observation-group/callable resolution, trajectories, and loggers.
- `config/example_config.yaml`: example public training configuration.

Simulator launch, Gym task registration, robot assets, and task rewards belong
in environment repositories. This package has no UWLab `source/` extensions,
`extension.toml`, or extension changelog requirements. Do not request those files
or a version bump for every PR.

## Training and inference context

Use `VecEnv` and its callers together to assess observation groups and shapes.
The environment returns a TensorDict plus batched rewards, dones, and extras;
`time_outs` distinguishes time limits for value bootstrapping. Rollout storage
tracks time and environment dimensions; recurrent minibatches also track
trajectory boundaries and padding. A reshape that has the right element count
can still pair the wrong action, target, or hidden state with an observation.

Keep operations batched over environments and on the appropriate device.
Time-step, epoch, minibatch, and recurrent sequence loops are expected. Host
transfers for logging, serialization, or scalar scheduling can be intentional;
flag unnecessary new synchronization in the hot path with a specific reason.

Review feed-forward and recurrent paths affected by shared changes, as well as
PPO and distillation callers of shared runner methods. Preserve the distinction
between training resume, teacher initialization, and inference. Optional RND,
symmetry, distributed training, and logging should remain optional.

## Validation expectations

For algorithm changes, suggest a focused check that would expose the reported
failure (for example a terminal versus timeout transition, recurrent state reset,
or checkpoint round trip). Do not demand long simulator training runs or a new
test framework for configuration-only changes. This fork currently has no
checked-in `tests/` suite; do not cite tests that exist only in newer upstream.

## Greptile setup

These files configure reviews; they do not install or authorize the Greptile
GitHub App. The app must have access to `UW-Lab/rsl_rl`, and the repository must
be enabled for reviews in Greptile. No GitHub Actions workflow or repository
API-key secret is added by this configuration.
