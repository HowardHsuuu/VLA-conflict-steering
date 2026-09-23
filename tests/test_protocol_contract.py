from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_cross_task_protocol_declares_the_paper_matrix() -> None:
    protocol = tomllib.loads(
        (ROOT / "configs/cross_task_decisive_v1.toml").read_text(encoding="utf-8")
    )

    assert protocol["campaign_id"] == "five-suite-closed-loop-v1"
    assert protocol["scenes"] == [46, 47, 48, 49]
    assert protocol["flow_noise_seeds"] == [10057, 10157, 10257]
    assert len(protocol["conditions"]) == 11
    assert len(protocol["tasks"]) == 5
    assert protocol["decision"]["unit"] == "task_x_scene_with_three_shared_noise_replicates"
    assert protocol["decision"]["all_attempted_cells_in_intention_to_treat"] is True
