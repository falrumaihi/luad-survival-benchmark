from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

from .io_utils import sha256_file


def file_manifest(paths: Iterable[Path], workspace: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        stat = path.stat()
        rows.append(
            {
                "path": path.relative_to(workspace).as_posix(),
                "bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": sha256_file(path),
            }
        )
    return rows


def run_metadata(command: list[str], config_path: Path) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config_path": str(config_path),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "working_directory": os.getcwd(),
    }

