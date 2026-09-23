"""Reproducible activation of the pinned Action Atlas and LIBERO checkouts."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ActionAtlasPaths:
    """Resolved paths needed by Action Atlas and its LIBERO submodule."""

    action_atlas_root: Path
    libero_source_root: Path
    libero_benchmark_root: Path
    libero_config_dir: Path

    def to_dict(self) -> dict[str, str]:
        """Return paths as JSON-serializable absolute strings."""

        return {key: str(value) for key, value in asdict(self).items()}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_action_atlas_root() -> Path:
    """Return the submodule location used by an editable source checkout."""

    return _repository_root() / "third_party" / "action-atlas"


def _validate_checkout(action_atlas_root: Path) -> tuple[Path, Path]:
    adapter = action_atlas_root / "experiments" / "model_adapters.py"
    libero_source_root = action_atlas_root / "LIBERO"
    libero_benchmark_root = libero_source_root / "libero" / "libero"
    required = (
        adapter,
        libero_benchmark_root / "bddl_files",
        libero_benchmark_root / "init_files",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Action Atlas checkout is incomplete; missing: {rendered}")
    return libero_source_root, libero_benchmark_root


def _write_libero_config(benchmark_root: Path, config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    datasets = benchmark_root.parent / "datasets"
    values = {
        "assets": benchmark_root / "assets",
        "bddl_files": benchmark_root / "bddl_files",
        "benchmark_root": benchmark_root,
        "datasets": datasets,
        "init_states": benchmark_root / "init_files",
    }
    lines = [f'{key}: "{value}"' for key, value in sorted(values.items())]
    contents = "\n".join(lines) + "\n"
    target = config_dir / "config.yaml"
    if not target.exists() or target.read_text(encoding="utf-8") != contents:
        target.write_text(contents, encoding="utf-8")


def activate_action_atlas(
    action_atlas_root: str | Path | None = None,
    libero_config_dir: str | Path | None = None,
) -> ActionAtlasPaths:
    """Activate pinned source trees without writing to ``~/.libero``.

    LIBERO's upstream editable package does not expose its nested source directory
    reliably and performs interactive home-directory setup at import time. This
    function validates the checkout, writes a deterministic repo-local config, and
    adds only the two required source roots to ``sys.path``.
    """

    root = Path(action_atlas_root or default_action_atlas_root()).resolve()
    libero_source_root, benchmark_root = _validate_checkout(root)
    config_dir = Path(libero_config_dir or (_repository_root() / "artifacts" / "libero-config"))
    config_dir = config_dir.resolve()
    _write_libero_config(benchmark_root, config_dir)

    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    for source_root in (root, libero_source_root):
        rendered = str(source_root)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)
    importlib.invalidate_caches()

    return ActionAtlasPaths(
        action_atlas_root=root,
        libero_source_root=libero_source_root,
        libero_benchmark_root=benchmark_root,
        libero_config_dir=config_dir,
    )
