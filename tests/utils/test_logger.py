# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for logging configuration and writer initialization."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from rsl_rl.utils import utils as logger_module
from rsl_rl.utils.logger import Logger


def _make_logger(tmp_path: Path, logger_cfg: str | dict, monkeypatch: pytest.MonkeyPatch) -> Logger:
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
    logger = _make_logger(tmp_path, config, monkeypatch)

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
    logger = _make_logger(tmp_path, alias, monkeypatch)
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
