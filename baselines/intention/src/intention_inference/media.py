from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def resolve_media_path(
    raw_path: Any,
    *,
    manifest_dir: Path,
    media_root: Path | None,
) -> Path | None:
    if raw_path is None:
        return None
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate

    manifest_candidate = (manifest_dir / candidate).resolve()
    if manifest_candidate.exists():
        return manifest_candidate

    if media_root is not None:
        return (media_root / candidate).resolve()
    return manifest_candidate


def resolve_prefixed_media_path(
    raw_media_path: Any,
    *,
    manifest_dir: Path,
    media_root: Path | None,
    media_path_prefix: str | None,
    local_path_prefix: Path | None,
) -> tuple[Path | None, str | None]:
    if raw_media_path is None:
        return None, None
    if not isinstance(raw_media_path, str) or not raw_media_path.strip():
        return None, None

    rewritten_path = raw_media_path
    if media_path_prefix:
        if local_path_prefix is None:
            raise ValueError(
                "--local-path-prefix is required when --media-path-prefix is provided."
            )
        if raw_media_path.startswith(media_path_prefix):
            suffix = raw_media_path[len(media_path_prefix):].lstrip("/\\")
            rewritten_path = str(local_path_prefix.joinpath(*suffix.split("/")))

    resolved_path = resolve_media_path(
        rewritten_path,
        manifest_dir=manifest_dir,
        media_root=media_root,
    )
    return resolved_path, rewritten_path


def effective_media_prefix(
    specific_prefix: str | None,
    shared_prefix: str | None,
) -> str | None:
    return specific_prefix if specific_prefix is not None else shared_prefix


def effective_local_prefix(
    specific_prefix: Path | None,
    shared_prefix: Path | None,
) -> Path | None:
    return specific_prefix if specific_prefix is not None else shared_prefix


def path_matches_any(path: Path | None, patterns: Sequence[str]) -> bool:
    if path is None:
        return False
    path_text = str(path)
    return any(pattern and pattern in path_text for pattern in patterns)


def safe_filename_part(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "record"
