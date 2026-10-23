# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Logger."""

from __future__ import annotations

import pathlib
import torch
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from rsl_rl.utils import LogWriter
from rsl_rl.utils import utils as logger_module
from rsl_rl.utils.logger import Logger


class RecordingWriter(LogWriter):
    """LogWriter that records scalar tags and uploaded videos."""

    def __init__(self) -> None:  # noqa: D107
        self.tags: list[str] = []
        self.videos: list[tuple[pathlib.Path, int]] = []

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:  # noqa: D102
        self.tags.append(tag)

    def save_video(self, video: pathlib.Path, it: int) -> None:  # noqa: D102
        self.videos.append((video, it))


def _make_logger(log_dir: pathlib.Path) -> Logger:
    """Construct a Logger that writes to a RecordingWriter."""
    logger = Logger(
        log_dir=str(log_dir),
        cfg={"num_steps_per_env": 1, "algorithm": {}},
        env_cfg={},
        num_envs=1,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cpu",
    )
    logger.writer = RecordingWriter()
    return logger


def _log(logger: Logger, it: int) -> None:
    """Log one iteration with placeholder metrics."""
    logger.log(
        it=it,
        start_it=0,
        total_it=it + 1,
        collect_time=0.1,
        learn_time=0.1,
        loss_dict={},
        learning_rate=1e-3,
        action_std=torch.ones(1),
        rnd_weight=None,
    )


class TestVideoUpload:
    """Tests for uploading videos from the log directory."""

    def test_each_video_is_uploaded_once(self, tmp_path: pathlib.Path) -> None:
        """Each video should be uploaded once, at the first iteration after it appears."""
        first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
        first.touch()
        logger = _make_logger(tmp_path)

        _log(logger, 0)
        second.touch()
        _log(logger, 1)
        _log(logger, 2)

        assert logger.writer.videos == [(first, 0), (second, 1)]


class TestTimeSeries:
    """Tests for the series logged against wall-clock time."""

    def test_log_writer_receives_no_time_series(self, tmp_path: pathlib.Path) -> None:
        """A LogWriter should only receive series indexed by iteration."""
        logger = _make_logger(tmp_path)
        logger.rewbuffer.append(1.0)
        logger.lenbuffer.append(10.0)

        _log(logger, 0)

        assert "Train/mean_reward" in logger.writer.tags
        assert not any(tag.endswith("/time") for tag in logger.writer.tags)


def _make_cfg_logger(tmp_path: Path, logger_cfg: str | dict, monkeypatch: pytest.MonkeyPatch) -> Logger:
    monkeypatch.setattr(Logger, "_store_code_state", lambda self: [])
    cfg = {
        "algorithm": {"rnd_cfg": None},
        "logger": logger_cfg,
        "wandb_project": "test-project",
        "neptune_project": "test-project",
    }
    return Logger(str(tmp_path), cfg, {}, 2, False, 1, 0, "cpu")


def test_custom_logger_reinitialization_preserves_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated initialization preserves the writer class name and constructor options."""
    config = {"class_name": "CaseSensitiveWriter", "flush_secs": 3}
    writer = Mock()
    resolver = Mock(return_value=writer)
    monkeypatch.setattr(logger_module, "resolve_callable", resolver)
    logger = _make_cfg_logger(tmp_path, config, monkeypatch)

    logger.init_logging_writer()
    logger.init_logging_writer()

    assert config == {"class_name": "CaseSensitiveWriter", "flush_secs": 3}
    assert resolver.call_args_list == [call("CaseSensitiveWriter"), call("CaseSensitiveWriter")]
    assert writer.call_args_list == [call(log_dir=str(tmp_path), flush_secs=3)] * 2


@pytest.mark.parametrize("alias", ["tensorboard", "TensorBoard", "wandb", "WaNdB", "neptune", "NePtUnE"])
def test_logger_aliases_are_case_insensitive(alias: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Built-in string aliases select the same writer regardless of letter case."""
    writer = Mock()
    resolver = Mock(return_value=writer)
    monkeypatch.setattr(logger_module, "resolve_callable", resolver)
    monkeypatch.setattr("torch.utils.tensorboard.SummaryWriter", writer)
    logger = _make_cfg_logger(tmp_path, alias, monkeypatch)
    canonical = alias.lower()

    warning_context = nullcontext() if canonical == "tensorboard" else pytest.warns(DeprecationWarning)
    with warning_context:
        logger.init_logging_writer()

    assert logger.cfg["logger"] == alias
    if canonical == "tensorboard":
        resolver.assert_not_called()
        writer.assert_called_once_with(log_dir=str(tmp_path), flush_secs=10)
    else:
        expected = "WandbLogWriter" if canonical == "wandb" else "NeptuneLogWriter"
        resolver.assert_called_once_with(expected)
        writer.assert_called_once_with(log_dir=str(tmp_path), project_name="test-project")
