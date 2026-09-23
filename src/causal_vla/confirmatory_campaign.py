"""Resumable sequential launcher for the frozen confirmatory campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_vla.confirmatory_monitor import ConfirmatoryProtocol, Device, run_confirmatory_task
from causal_vla.confirmatory_monitor_analysis import aggregate_confirmatory_reports


def run_campaign(
    protocol_path: str | Path,
    *,
    repository: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    summary: str | Path,
    device: Device,
) -> dict[str, object]:
    """Run missing task/seed reports sequentially, then verify and aggregate all reports."""

    protocol = ConfirmatoryProtocol.load(protocol_path)
    root = Path(repository).resolve()
    report_root = Path(output_dir).resolve()
    report_paths: list[Path] = []
    for task in protocol.tasks:
        for seed in protocol.noise_seeds:
            path = report_root / f"{task.name}_noise{seed}.json"
            run_confirmatory_task(
                protocol_path,
                repository=root,
                checkpoint=checkpoint,
                task_name=task.name,
                noise_seed=seed,
                output=path,
                device=device,
            )
            report_paths.append(path)
    payload = aggregate_confirmatory_reports(
        protocol_path,
        tuple(report_paths),
        repository=root,
        checkpoint=checkpoint,
    )
    summary_path = Path(summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m causal_vla.confirmatory_campaign")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_campaign(
        args.protocol,
        repository=args.repository,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        summary=args.summary,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
