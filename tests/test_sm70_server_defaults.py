"""sm70 SGLang server-default flag selection.

Regression for the A/B bug where ``MESHY_SM70_CUDA_GRAPH=0`` was read in a
parent process whose env did not carry the flag, so every "graph off" server
still came up with ``cuda_graph_backend_decode=full``.
"""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture()
def defaults_module(monkeypatch):
    # The module caches nothing across calls; just make sure MESHY_* graph vars
    # from the outer shell can't leak into the assertions.
    monkeypatch.delenv("MESHY_SM70_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("MESHY_SM70_CUDA_GRAPH_MAX_BS", raising=False)
    sys.modules.pop("meshy.backend.sglang_sm70", None)
    return importlib.import_module("meshy.backend.sglang_sm70")


def _cli(defaults: dict) -> list[str]:
    argv: list[str] = []
    for key, value in defaults.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]
    return argv


def test_graph_off_forces_disable_flag(defaults_module, monkeypatch) -> None:
    monkeypatch.setenv("MESHY_SM70_CUDA_GRAPH", "0")
    d = defaults_module.sm70_server_defaults()
    assert d.get("disable_cuda_graph") is True
    assert d["cuda_graph_backend_decode"] == "disabled"
    assert "--disable-cuda-graph" in _cli(d)
    assert "--cuda-graph-backend-decode full" not in " ".join(_cli(d))


def test_graph_on_uses_full_no_disable_flag(defaults_module, monkeypatch) -> None:
    monkeypatch.setenv("MESHY_SM70_CUDA_GRAPH", "1")
    monkeypatch.setenv("MESHY_SM70_CUDA_GRAPH_MAX_BS", "64")
    d = defaults_module.sm70_server_defaults()
    assert d["cuda_graph_backend_decode"] == "full"
    assert d["cuda_graph_max_bs_decode"] == 64
    assert not d.get("disable_cuda_graph")
    assert "--disable-cuda-graph" not in _cli(d)


def test_default_is_graph_on(defaults_module) -> None:
    # no env set -> on (the measured good path)
    d = defaults_module.sm70_server_defaults()
    assert d["cuda_graph_backend_decode"] == "full"


def test_explicit_kwargs_override_parent_env(defaults_module, monkeypatch) -> None:
    # Simulate the A/B launcher: parent env says on, but the child is built off.
    monkeypatch.setenv("MESHY_SM70_CUDA_GRAPH", "1")
    off = defaults_module.sm70_server_defaults(cuda_graph=False)
    assert off.get("disable_cuda_graph") is True
    on = defaults_module.sm70_server_defaults(cuda_graph=True, max_bs=32)
    assert on["cuda_graph_max_bs_decode"] == 32
    assert on["cuda_graph_backend_decode"] == "full"
