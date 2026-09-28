"""DCP resume checkpoint: optimizer/LR/step/GradScaler round-trip (T5e).

CPU-only, no process group: single-process ``torch.distributed.checkpoint``
saves and loads on CPU. Proves the state set Meshy registers with torchtitan's
CheckpointManager (model params, Adam optimizer state, ``train_state`` with
step and GradScaler) survives save -> load tensor-for-tensor, and that the
trainer's init-order parking replays scaler state after DCP load.
"""

from __future__ import annotations

import torch
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint import save as dcp_save

from meshy.backend.titan.trainer import TitanTrainer


def _build_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    )


def _populate_optimizer(model: torch.nn.Module, seed: int) -> torch.optim.Adam:
    # Step on varied data so Adam's exp_avg / exp_avg_sq hold non-trivial,
    # per-parameter distinct values.
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           weight_decay=0.1, eps=1e-8)
    for _ in range(5):
        opt.zero_grad()
        loss = model(torch.randn(8, 16)).pow(2).sum()
        loss.backward()
        opt.step()
    return opt


def _optimizer_tensor_state(opt: torch.optim.Optimizer) -> dict[str, torch.Tensor]:
    # Keyed by group-param position and Adam slot name (exp_avg / exp_avg_sq /
    # step); every value is a tensor DCP must reproduce exactly.
    return {
        f"{gi}-{pi}-{k}": v.detach().clone()
        for gi, group in enumerate(opt.param_groups)
        for pi, par in enumerate(group["params"])
        for k, v in opt.state[par].items()
        if torch.is_tensor(v)
    }


class _FakeScaler:
    """Mimics GradScaler.state_dict/load_state_dict with a plain payload.

    A real cuda GradScaler stays disabled (empty state) on a GPU-less runner,
    so it cannot exercise serialization here; the trainer methods only depend
    on these two methods. Payload mirrors GradScaler keys, with a tensor to
    prove DCP round-trips nested values.
    """

    def __init__(self, scale: float) -> None:
        self._sd = {"scale": torch.tensor(scale), "_growth_tracker": torch.tensor(0)}

    def state_dict(self) -> dict:
        return dict(self._sd)

    def load_state_dict(self, sd: dict) -> None:
        self._sd = {k: v for k, v in sd.items()}


class _TrainState:
    """Stand-in exposing only what the Stateful protocol methods touch."""

    def __init__(self) -> None:
        self.step = 0
        self._pending_scaler_state = None
        self.grad_scaler = _FakeScaler(1.0)

    state_dict = TitanTrainer.state_dict
    load_state_dict = TitanTrainer.load_state_dict


def test_dcp_save_load_optimizer_state_equal_cpu(tmp_path) -> None:
    model = _build_model(1)
    opt = _populate_optimizer(model, 100)
    src_state = _TrainState()
    src_state.step = 3
    src_state.grad_scaler = _FakeScaler(512.0)

    dcp_save(
        {
            "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "optimizer": opt,
            "train_state": src_state,
        },
        checkpoint_id=str(tmp_path / "step-3"),
    )

    # Fresh, independently trained objects: their Adam moments must differ
    # before load so equality afterwards is attributable to DCP.
    model2 = _build_model(2)
    opt2 = _populate_optimizer(model2, 200)
    dst_state = _TrainState()
    dcp_load(
        {"model": model2, "optimizer": opt2, "train_state": dst_state},
        checkpoint_id=str(tmp_path / "step-3"),
    )

    before = _optimizer_tensor_state(opt)
    after = _optimizer_tensor_state(opt2)
    assert set(before) == set(after)
    for key in before:
        assert torch.equal(after[key], before[key]), f"optimizer tensor {key} differs"

    assert dst_state.step == 3
    assert torch.equal(
        dst_state.grad_scaler.state_dict()["scale"],
        src_state.grad_scaler.state_dict()["scale"],
    )
    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), model2.state_dict().items()
    ):
        assert k1 == k2 and torch.equal(v1, v2)


def test_train_state_serializes_step_and_scaler() -> None:
    src = _TrainState()
    src.step = 7
    src.grad_scaler = _FakeScaler(256.0)
    sd = src.state_dict()
    assert sd["step"] == 7
    assert torch.equal(
        sd["loss_scale"]["scale"], src.grad_scaler.state_dict()["scale"]
    )

    dst = _TrainState()
    dst.load_state_dict(sd)
    assert dst.step == 7
    assert torch.equal(
        dst.grad_scaler.state_dict()["scale"], sd["loss_scale"]["scale"]
    )
    assert dst._pending_scaler_state is None


def test_scaler_state_parked_before_construction_then_replayed() -> None:
    """DCP load runs inside ForgeEngine.__init__, before GradScaler exists.

    The load must park the scaler payload instead of crashing; constructing
    the scaler and replaying it (as trainer.__init__ does) restores it.
    """
    payload = {"step": 4, "loss_scale": _FakeScaler(1024.0).state_dict()}

    early = _TrainState()
    del early.grad_scaler  # simulate the pre-scaler window during super().__init__
    early.load_state_dict(payload)
    assert early.step == 4
    assert early._pending_scaler_state is payload["loss_scale"]

    early.grad_scaler = _FakeScaler(1.0)
    early.grad_scaler.load_state_dict(early._pending_scaler_state)
    early._pending_scaler_state = None
    assert torch.equal(
        early.grad_scaler.state_dict()["scale"], payload["loss_scale"]["scale"]
    )
