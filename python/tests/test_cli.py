from __future__ import annotations

import sys

import autobatcher.__main__ as cli
import autobatcher.serve as serve


def test_cli_mode_async_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_server(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(serve, "run_server", fake_run_server)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autobatcher",
            "serve",
            "--api-key",
            "sk-test",
        ],
    )

    cli.main()

    assert captured["completion_window"] == "1h"


def test_cli_mode_batch(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_server(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(serve, "run_server", fake_run_server)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autobatcher",
            "serve",
            "--api-key",
            "sk-test",
            "--mode",
            "batch",
        ],
    )

    cli.main()

    assert captured["completion_window"] == "24h"


def test_cli_passes_batch_metadata_and_shutdown_policy(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_server(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(serve, "run_server", fake_run_server)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autobatcher",
            "serve",
            "--api-key",
            "sk-test",
            "--batch-metadata",
            "benchmark_id=bench-123",
            "--batch-metadata",
            "suite=livebench",
            "--keep-active-batches-on-close",
        ],
    )

    cli.main()

    assert captured["batch_metadata"] == {
        "benchmark_id": "bench-123",
        "suite": "livebench",
    }
    assert captured["cancel_active_batches_on_close"] is False
