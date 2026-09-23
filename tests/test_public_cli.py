from __future__ import annotations

import json
from pathlib import Path

from causal_vla.public_results import main

ROOT = Path(__file__).resolve().parents[1]


def test_public_cli_prints_human_receipt(capsys: object) -> None:
    assert (
        main(
            [
                "--archive",
                str(ROOT / "results/cross_task_decisive_v1"),
                "--repository",
                str(ROOT),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Validated aggregate for 660 rollouts" in output
    assert "Action-flow, learned trigger" in output


def test_public_cli_emits_machine_readable_receipt(capsys: object) -> None:
    assert (
        main(
            [
                "--archive",
                str(ROOT / "results/cross_task_decisive_v1"),
                "--repository",
                str(ROOT),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["total_rollouts"] == 660
    assert payload["condition_successes"]["learned_monitor"] == 45
