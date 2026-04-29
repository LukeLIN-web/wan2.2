"""Unit tests for the W&B DPO run analyzer."""

from __future__ import annotations

from eval import analyze_wandb_dpo_run as analyzer


class _FakeRun:
    def __init__(
        self,
        history_rows=(),
        scan_rows=(),
        keyed_scan_rows=(),
        last_history_step=None,
    ) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.history_rows = list(history_rows)
        self.scan_rows = list(scan_rows)
        self.keyed_scan_rows = list(keyed_scan_rows)
        self.lastHistoryStep = last_history_step
        self.summary = {}

    def history(self, **kwargs):
        self.calls.append(("history", kwargs))
        return self.history_rows

    def scan_history(self, **kwargs):
        self.calls.append(("scan_history", kwargs))
        if "keys" in kwargs:
            return iter(self.keyed_scan_rows)
        return iter(self.scan_rows)


def _install_fake_api(monkeypatch, fake_run):
    class FakeApi:
        def __init__(self, timeout):
            self.timeout = timeout

        def run(self, run_path):
            assert run_path == "entity/project/run"
            return fake_run

    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.setattr(analyzer.wandb, "Api", FakeApi)


def test_load_wandb_rows_quick_samples_and_scans_tail(monkeypatch):
    fake_run = _FakeRun(
        history_rows=(
            {"_step": 0, "loss": 0.75},
            {"_step": 5, "loss": 0.25},
        ),
        keyed_scan_rows=(
            {"_step": 4, "margin": -1.0},
            {"_step": 5, "acc_win50": 0.8, "ignored": "x"},
        ),
        last_history_step=5,
    )
    _install_fake_api(monkeypatch, fake_run)

    run, rows, info = analyzer.load_wandb_rows(
        run_path="entity/project/run",
        token_path=None,
        keys=("loss", "acc_win50", "margin"),
        timeout=7,
        page_size=123,
        history_mode="quick",
        samples=10,
        tail_steps=2,
    )

    assert run is fake_run
    assert rows == [
        {"_step": 0, "loss": 0.75},
        {"_step": 4, "margin": -1.0},
        {"_step": 5, "loss": 0.25, "acc_win50": 0.8},
    ]
    assert fake_run.calls == [
        (
            "history",
            {
                "samples": 10,
                "keys": ["loss", "acc_win50", "margin"],
                "pandas": False,
            },
        ),
        (
            "scan_history",
            {
                "page_size": 123,
                "keys": ["_step", "loss", "acc_win50", "margin"],
                "min_step": 4,
                "max_step": 6,
            },
        ),
    ]
    assert info["mode"] == "quick"
    assert info["complete"] is False
    assert info["tail_range"] == {"start": 4, "end_exclusive": 6}


def test_load_wandb_rows_full_falls_back_when_keyed_scan_is_empty(
    monkeypatch,
    capsys,
):
    fake_run = _FakeRun(
        scan_rows=(
            {"_step": 1, "loss": 0.25, "acc_win50": 0.5, "ignored": "x"},
            {"_step": 0, "loss": 0.75, "margin": -1.0},
        ),
    )
    _install_fake_api(monkeypatch, fake_run)

    run, rows, info = analyzer.load_wandb_rows(
        run_path="entity/project/run",
        token_path=None,
        keys=("loss", "acc_win50", "margin"),
        timeout=7,
        page_size=123,
        history_mode="full",
    )

    assert run is fake_run
    assert rows == [
        {"_step": 0, "loss": 0.75, "margin": -1.0},
        {"_step": 1, "loss": 0.25, "acc_win50": 0.5},
    ]
    assert fake_run.calls == [
        (
            "scan_history",
            {"page_size": 123, "keys": ["_step", "loss", "acc_win50", "margin"]},
        ),
        ("scan_history", {"page_size": 123}),
    ]
    assert info["mode"] == "full"
    assert info["complete"] is True
    assert info["scan"] == "scan_history()"
    assert "falling back to full history scan" in capsys.readouterr().err


def test_requested_scan_ranges_merge_quarters_and_tail():
    fake_run = _FakeRun(last_history_step=11)

    scan_ranges = analyzer._requested_scan_ranges(
        run=fake_run,
        ranges=[("0_4", 0, 5), ("10_19", 10, 20)],
        tail_steps=3,
    )

    assert scan_ranges == [(0, 5), (9, 12)]
