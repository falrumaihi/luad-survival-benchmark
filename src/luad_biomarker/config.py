from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class StudyConfig:
    config_path: Path
    workspace: Path
    metadata: Path
    raw_counts: Path
    vst_matrix: Path
    methods_audit: Path
    datasets: Path
    artifacts: Path
    raw: dict[str, Any]


def load_config(path: str | Path) -> StudyConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    paths = raw["paths"]
    workspace = (config_path.parent / paths["workspace"]).resolve()

    def resolve_from_workspace(key: str) -> Path:
        return (workspace / paths[key]).resolve()

    config = StudyConfig(
        config_path=config_path,
        workspace=workspace,
        metadata=resolve_from_workspace("metadata"),
        raw_counts=resolve_from_workspace("raw_counts"),
        vst_matrix=resolve_from_workspace("vst_matrix"),
        methods_audit=resolve_from_workspace("methods_audit"),
        datasets=resolve_from_workspace("datasets"),
        artifacts=resolve_from_workspace("artifacts"),
        raw=raw,
    )
    validate_config(config)
    return config


def validate_config(config: StudyConfig) -> None:
    required = [
        config.metadata,
        config.raw_counts,
        config.vst_matrix,
        config.methods_audit,
        config.datasets,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing configured inputs: " + ", ".join(missing))

    workspace = config.workspace.resolve()
    artifacts = config.artifacts.resolve()
    if artifacts == workspace:
        raise ValueError("Artifacts directory cannot be the workspace root")
    if workspace not in artifacts.parents:
        raise ValueError("Artifacts directory must remain inside the workspace")

