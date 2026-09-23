from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal_vla.public_results import format_summary, verify_protocol, verify_public_results

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/cross_task_decisive_v1"


def test_public_aggregate_matches_released_results() -> None:
    result = verify_public_results(ARCHIVE)

    assert result.total_rollouts == 660
    assert result.trials_per_condition == 60
    assert result.condition_successes == {
        "conflict": 5,
        "always_steer": 46,
        "oracle_monitor": 45,
        "learned_monitor": 45,
        "aligned_no_steer": 46,
        "aligned_monitor": 46,
        "aligned_forced_steer": 9,
        "wrong_sign": 1,
        "random": 8,
        "monitor_prompt_correction": 49,
        "oracle_prompt_correction": 49,
    }
    assert result.learned_vs_conflict == {"wins": 40, "losses": 0, "ties": 20}
    assert result.prompt_vs_steering == {"prompt_only": 7, "steering_only": 3, "ties": 50}
    assert result.conflict_monitor_triggers == 60
    assert result.aligned_monitor_triggers == 0
    assert result.primary_pooled_absolute_gain == pytest.approx(2 / 3)
    assert result.scene_cluster_bootstrap_95 == pytest.approx((0.55, 0.7833333333))
    assert all(result.preregistered_gates.values())
    assert result.evidence_level == "aggregate_only"


def test_public_protocol_matches_aggregate_design() -> None:
    verify_protocol(ROOT)


def test_receipt_reports_all_key_denominators() -> None:
    receipt = format_summary(verify_public_results(ARCHIVE))
    assert "75.0% (45/60)" in receipt
    assert "8.3% (5/60)" in receipt
    assert "40 wins, 0 losses, 20 ties" in receipt
    assert "60/60 conflict, 0/60 aligned" in receipt
    assert "7 prompt-only, 3 steering-only, 50 ties" in receipt
    assert "all 10 preregistered gates pass" in receipt


def _copy_release(source: Path, destination: Path) -> None:
    destination.mkdir()
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


def test_release_rejects_an_added_undeclared_file(tmp_path: Path) -> None:
    copied = tmp_path / "release"
    _copy_release(ARCHIVE, copied)
    (copied / "undeclared.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inventory mismatch"):
        verify_public_results(copied)


def test_release_rejects_changed_headline_count(tmp_path: Path) -> None:
    copied = tmp_path / "release"
    _copy_release(ARCHIVE, copied)
    summary = copied / "aggregate_results.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["condition_successes_out_of_60"]["learned_monitor"] = 44
    summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="released headline"):
        verify_public_results(copied)
