from __future__ import annotations

import sys

import autobatcher.__main__ as cli
import autobatcher.serve as serve


def test_cli_accepts_arbitrary_completion_window(monkeypatch) -> None:
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
            "--completion-window",
            "72h",
        ],
    )

    cli.main()

    assert captured["completion_window"] == "72h"
